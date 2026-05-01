"""Smoke test for ported OpenAI life-science skills.

One live API call per handler, asserts ``ok: true`` in the returned JSON
string (or at least a non-error response). Run with:

    .venv311/bin/python tests/smoke_olsp.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

from harness.tools.openai_ported.handlers import HANDLERS


# One representative arg set per skill. Chosen to be a cheap, stable query.
CASES: list[tuple[str, dict]] = [
    ("olsp_bgee_sparql", {"query": "ASK {}"}),
    ("olsp_bindingdb_ligands", {"mode": "pdb", "pdb": "1Q0L", "cutoff": 100,
                                  "identity": 90, "max_items": 3}),
    ("olsp_chebi_lookup", {"action": "search", "query": "caffeine", "size": 3,
                             "max_items": 3}),
    ("olsp_civic_graphql", {"query": "query { __typename }"}),
    ("olsp_efo_ontology", {"action": "search", "query": "asthma",
                             "ontology": "efo", "rows": 3, "max_items": 3}),
    ("olsp_gtex_eqtl", {"action": "gene",
                          "gencode_id": "ENSG00000135744.8",
                          "tissue": "Liver", "max_items": 3}),
    ("olsp_opentargets_graphql", {
        "query": "query { target(ensemblId: \"ENSG00000169083\") { id approvedSymbol } }"
    }),
    ("olsp_pharmgkb_lookup", {"action": "gene", "id": "PA124"}),
    ("olsp_reactome_query", {"action": "event",
                               "event_id": "R-HSA-199420"}),
    ("olsp_uniprot_lookup", {"action": "accession", "accession": "P04637"}),
    # Batch 1
    ("olsp_biorxiv_request", {"action": "details",
                                "server": "biorxiv",
                                "start": "2024-01-01",
                                "end": "2024-01-07",
                                "cursor": 0, "max_items": 3}),
    ("olsp_cbioportal_request", {"path": "studies",
                                   "params": {"pageSize": 3},
                                   "max_items": 3}),
    ("olsp_gnomad_graphql", {"query": "query { meta { clinvar_release_date } }"}),
    # BLAST submit would consume NCBI rate-budget; use 'status' on a
    # known-dead RID to verify the API shape without queueing work.
    ("olsp_ncbi_blast", {"action": "status", "rid": "NONEXISTENT_RID"}),
    ("olsp_ncbi_datasets", {"path": "taxonomy/taxon/9606"}),
    ("olsp_string_request", {"path": "get_string_ids",
                               "identifiers": "TP53", "species": 9606,
                               "limit": 1, "max_items": 3}),
    ("olsp_gwas_catalog_request", {"path": "metadata"}),
    ("olsp_human_protein_atlas", {"action": "gene",
                                    "ensg": "ENSG00000141510"}),
    # Batch 2
    ("olsp_quickgo_request", {"path": "ontology/go/terms/GO:0008150",
                                "max_items": 3}),
    ("olsp_rnacentral_request", {"path": "rna/URS0000000001/"}),
    ("olsp_encode_request", {"path": "search/",
                               "params": {"type": "Experiment",
                                            "limit": 3}, "max_items": 3}),
    ("olsp_rhea_request", {"query": "caffeine", "max_items": 3}),
    ("olsp_locus_to_gene_mapper", {"action": "l2g_for_gene_disease",
                                      "ensembl_id": "ENSG00000169083",
                                      "efo_id": "EFO_0000676"}),
    ("olsp_finngen_phewas", {"variant": "10:112998590-C-T",
                                "max_results": 3}),
    ("olsp_biobankjapan_phewas", {"variant": "10:114758349-C-T",
                                     "max_results": 3}),
    ("olsp_biostudies_request", {"path": "search",
                                    "params": {"query": "rna",
                                                 "pageSize": 3},
                                    "max_items": 3}),
    ("olsp_genebass_gene_burden", {"ensembl_gene_id": "ENSG00000173531",
                                      "burden_set": "pLoF",
                                      "max_results": 3}),
    ("olsp_epigraphdb_request", {"path": "ping"}),
    # Upstream /eqtl/api returns 500 for many paths; /eqtl/api/v2/ is current.
    # Our handler is path-verbatim so the caller drives the version prefix.
    ("olsp_eqtl_catalogue_request", {"path": "v2/datasets"}),
    # Batch 3
    ("olsp_ukb_topmed_phewas", {"variant": "10:112998590-C-T",
                                    "max_results": 3}),
    ("olsp_tpmi_phewas", {"variant": "6:160540105-T-C",
                              "max_results": 3}),
    ("olsp_hmdb_request", {"query": "serotonin",
                               "category": "metabolites",
                               "per_page": 3, "max_items": 3}),
    ("olsp_metabolights_request", {"path": "studies/MTBLS1"}),
    ("olsp_pride_request", {"path": "projects/PXD001357"}),
    ("olsp_proteomexchange_request", {"path": "datasets/PXD000001"}),
    # Batch 4
    ("olsp_mgnify_request", {"path": "biomes",
                                "params": {"page_size": 3},
                                "max_items": 3}),
    ("olsp_eva_request", {"path": "meta/species/list", "max_items": 3}),
    ("olsp_ipd_request", {"path": "allele",
                              "params": {"project": "HLA", "limit": 3},
                              "max_items": 3}),
    ("olsp_ncbi_clinicaltables", {"terms": "TP53", "count": 3,
                                     "max_items": 3}),
]


async def run_one(name: str, args: dict) -> tuple[str, bool, str]:
    fn = HANDLERS[name]
    t0 = time.time()
    try:
        result = await fn(args)
    except Exception as exc:
        return name, False, f"EXC {exc!r}"
    dt = time.time() - t0
    # Handler may truncate long outputs with "...<truncated>". For the
    # smoke check, we only care that the payload starts with ok:true and
    # names a source — look at the head of the string.
    ok = result.lstrip().startswith("{\"ok\": true")
    try:
        parsed = json.loads(result)
    except Exception:
        parsed = {"_raw_head": result[:220]}
    status = "PASS" if ok else "FAIL"
    preview = json.dumps(parsed, default=str)[:180]
    return name, ok, f"{status}  {dt:.2f}s  {preview}"


async def main() -> int:
    results = await asyncio.gather(*(run_one(n, a) for n, a in CASES))
    passes = sum(1 for _, ok, _ in results if ok)
    for name, ok, line in results:
        flag = "\u2713" if ok else "\u2717"
        print(f"{flag} {name:<30}  {line}")
    print(f"\n{passes}/{len(results)} passed")
    return 0 if passes == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
