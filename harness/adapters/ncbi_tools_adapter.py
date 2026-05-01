"""Direct NCBI E-utilities client — no vendor dependency."""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import quote_plus

import httpx

from harness.adapter_base import AdapterBase

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class NCBIToolsAdapter(AdapterBase):
    name = "ncbi_tools"
    modality = "genomics"
    description = "Direct NCBI E-utilities client for gene info, ClinVar, dbSNP, PubMed, and BLAST."

    def __init__(self, config: dict | None = None, **kwargs: Any):
        self.api_key = (config or {}).get("api_key")
        self._http = httpx.AsyncClient(timeout=30, follow_redirects=True)

    def capabilities(self) -> list[str]:
        return [
            "gene_lookup",
            "variant_interpretation",
            "snp_info",
            "clinvar_lookup",
            "pubmed_search",
            "blast_sequence",
            "gene_disease_association",
        ]

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        ctx = context or {}
        results: list[str] = []
        evidence: list[str] = []

        genes = ctx.get("genes", [])
        variant = ctx.get("variant")
        rsid = ctx.get("rsid")
        sequence = ctx.get("sequence")

        tasks: list[asyncio.Task] = []

        # Gene info for any listed genes
        for gene in genes:
            tasks.append(asyncio.create_task(self.gene_info(gene)))

        # ClinVar for variant
        if variant:
            tasks.append(asyncio.create_task(self.clinvar_lookup(variant)))

        # dbSNP for rsid
        if rsid:
            tasks.append(asyncio.create_task(self.snp_info(rsid)))

        # BLAST for sequence
        if sequence:
            tasks.append(asyncio.create_task(self.blast_sequence(sequence)))

        # Always do a PubMed search on the query
        tasks.append(asyncio.create_task(self.pubmed_search(query)))

        # If no specific context, try to parse genes/variants from query
        if not genes and not variant and not rsid:
            tasks.append(asyncio.create_task(self._auto_search(query)))

        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        raw_parts: list[Any] = []

        for item in gathered:
            if isinstance(item, Exception):
                results.append(f"Error: {item}")
            elif isinstance(item, dict):
                raw_parts.append(item)
                if "summary" in item:
                    results.append(item["summary"])
                if "evidence" in item:
                    evidence.extend(item["evidence"])

        answer = "\n\n".join(results) if results else "No results from NCBI."
        return self.result(answer=answer, evidence=evidence, confidence=0.7, raw=raw_parts)

    # ------------------------------------------------------------------
    # NCBI API methods
    # ------------------------------------------------------------------

    async def gene_info(self, symbol: str) -> dict[str, Any]:
        """Fetch gene summary, aliases, location, and function."""
        search = await self._esearch("gene", f"{symbol}[Gene Name] AND Homo sapiens[Organism]")
        ids = search.get("ids", [])
        if not ids:
            return {"summary": f"Gene '{symbol}' not found.", "evidence": []}

        gene_id = ids[0]
        xml_text = await self._efetch("gene", gene_id, rettype="xml")
        root = ET.fromstring(xml_text)

        # Extract key fields from Entrezgene XML
        desc = self._xml_text(root, ".//Entrezgene_summary") or "No summary available."
        official = self._xml_text(root, ".//Gene-ref_locus") or symbol
        aliases_el = root.findall(".//Gene-ref_syn/Gene-ref_syn_E")
        aliases = [e.text for e in aliases_el if e.text] if aliases_el else []
        location = self._xml_text(root, ".//Gene-ref_maploc") or "unknown"

        summary = (
            f"**{official}** (ID: {gene_id})\n"
            f"Location: {location}\n"
            f"Aliases: {', '.join(aliases) if aliases else 'none'}\n"
            f"Summary: {desc}"
        )
        return {
            "summary": summary,
            "evidence": [f"NCBI Gene ID {gene_id}"],
            "gene_id": gene_id,
            "symbol": official,
        }

    async def clinvar_lookup(self, variant: str) -> dict[str, Any]:
        """Look up clinical significance of a variant in ClinVar."""
        search = await self._esearch("clinvar", variant)
        ids = search.get("ids", [])
        if not ids:
            return {"summary": f"Variant '{variant}' not found in ClinVar.", "evidence": []}

        var_id = ids[0]
        xml_text = await self._efetch("clinvar", var_id, rettype="xml")
        root = ET.fromstring(xml_text)

        clin_sig = self._xml_text(root, ".//ClinicalSignificance/Description") or "Unknown"
        review = self._xml_text(root, ".//ClinicalSignificance/ReviewStatus") or "Unknown"
        conditions = [
            e.text for e in root.findall(".//TraitSet/Trait/Name/ElementValue") if e.text
        ]

        summary = (
            f"**ClinVar: {variant}** (ID: {var_id})\n"
            f"Clinical significance: {clin_sig}\n"
            f"Review status: {review}\n"
            f"Associated conditions: {', '.join(conditions) if conditions else 'none listed'}"
        )
        return {
            "summary": summary,
            "evidence": [f"ClinVar ID {var_id}", f"Clinical significance: {clin_sig}"],
            "clinvar_id": var_id,
        }

    async def snp_info(self, rsid: str) -> dict[str, Any]:
        """Fetch SNP info from dbSNP."""
        rsid_clean = rsid.replace("rs", "")
        search = await self._esearch("snp", rsid_clean)
        ids = search.get("ids", [])
        if not ids:
            return {"summary": f"SNP '{rsid}' not found.", "evidence": []}

        snp_id = ids[0]
        xml_text = await self._efetch("snp", snp_id, rettype="xml")

        summary = f"**dbSNP: rs{snp_id}**\nRaw data retrieved. Parse for allele frequencies and clinical associations."
        return {
            "summary": summary,
            "evidence": [f"dbSNP rs{snp_id}"],
            "snp_id": snp_id,
            "raw_xml_length": len(xml_text),
        }

    async def gene_disease(self, gene_symbol: str) -> dict[str, Any]:
        """Find diseases associated with a gene via MedGen/ClinVar."""
        query = f"{gene_symbol}[Gene Name] AND clinical_significance_pathogenic[Filter]"
        search = await self._esearch("clinvar", query)
        ids = search.get("ids", [])[:10]

        if not ids:
            return {"summary": f"No pathogenic variants found for {gene_symbol}.", "evidence": []}

        summary = f"Found {len(ids)} pathogenic ClinVar entries for {gene_symbol}."
        return {"summary": summary, "evidence": [f"ClinVar pathogenic count: {len(ids)}"], "ids": ids}

    async def blast_sequence(self, sequence: str) -> dict[str, Any]:
        """Submit a BLAST search (simplified — real BLAST is async with polling)."""
        # NCBI BLAST requires PUT + polling; provide a stub that documents the flow.
        return {
            "summary": (
                "BLAST submission requires asynchronous polling (PUT to blast.ncbi.nlm.nih.gov/blast/Blast.cgi, "
                "then poll RID). This adapter provides a placeholder; implement full polling for production use."
            ),
            "evidence": [],
        }

    async def pubmed_search(self, query: str, max_results: int = 5) -> dict[str, Any]:
        """Search PubMed and return article titles + PMIDs."""
        search = await self._esearch("pubmed", query, retmax=max_results)
        ids = search.get("ids", [])
        if not ids:
            return {"summary": "No PubMed results.", "evidence": []}

        # Fetch summaries
        xml_text = await self._efetch("pubmed", ",".join(ids), rettype="xml")
        root = ET.fromstring(xml_text)
        articles = []
        for article in root.findall(".//PubmedArticle"):
            title = self._xml_text(article, ".//ArticleTitle") or "Untitled"
            pmid = self._xml_text(article, ".//PMID") or "?"
            articles.append(f"- PMID {pmid}: {title}")

        summary = f"**PubMed results ({len(articles)} articles):**\n" + "\n".join(articles)
        return {
            "summary": summary,
            "evidence": [f"PMID {pid}" for pid in ids],
        }

    # ------------------------------------------------------------------
    # Auto-search (extract terms from query)
    # ------------------------------------------------------------------

    async def _auto_search(self, query: str) -> dict[str, Any]:
        """Fallback: search NCBI Gene with the raw query."""
        search = await self._esearch("gene", f"({query}) AND Homo sapiens[Organism]", retmax=3)
        ids = search.get("ids", [])
        if not ids:
            return {"summary": "", "evidence": []}
        return {"summary": f"Auto-search found NCBI Gene IDs: {', '.join(ids)}", "evidence": []}

    # ------------------------------------------------------------------
    # Low-level E-utilities
    # ------------------------------------------------------------------

    async def _esearch(self, db: str, term: str, retmax: int = 10) -> dict[str, Any]:
        params: dict[str, str] = {
            "db": db,
            "term": term,
            "retmax": str(retmax),
            "retmode": "json",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        resp = await self._http.get(f"{EUTILS_BASE}/esearch.fcgi", params=params)
        resp.raise_for_status()
        data = resp.json()
        result = data.get("esearchresult", {})
        return {"ids": result.get("idlist", []), "count": int(result.get("count", 0))}

    async def _efetch(self, db: str, ids: str, rettype: str = "xml", retmode: str = "xml") -> str:
        params: dict[str, str] = {
            "db": db,
            "id": ids,
            "rettype": rettype,
            "retmode": retmode,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        resp = await self._http.get(f"{EUTILS_BASE}/efetch.fcgi", params=params)
        resp.raise_for_status()
        return resp.text

    @staticmethod
    def _xml_text(root: ET.Element, xpath: str) -> str | None:
        el = root.find(xpath)
        return el.text if el is not None and el.text else None
