"""TOOL_SPEC JSON-schema entries for ported OpenAI life-science skills.

Each spec matches the OpenAI/Gemini function-calling format used by
``harness/eval/function_calling_runner.py``.
"""

from __future__ import annotations

from typing import Any


OPENAI_PORTED_PREFIX = "olsp_"


OPENAI_PORTED_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "olsp_bgee_sparql",
            "description": (
                "Bgee gene-expression SPARQL endpoint "
                "(https://www.bgee.org/sparql/). Submit a SPARQL SELECT/ASK "
                "query and get bindings. Best for healthy wild-type expression "
                "metadata scoped by species/organ/stage. Keep queries small "
                "and add LIMIT. Returns JSON bindings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SPARQL query body"},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_bindingdb_ligands",
            "description": (
                "BindingDB REST lookup of ligand–target binding measurements "
                "(https://bindingdb.org/rest). Use mode='pdb' (with 'pdb'), "
                "mode='uniprot' (with 'uniprot'), or mode='smiles' (with "
                "'smiles' + similarity 'cutoff' 0-1). Returns measured Ki/Kd/IC50 rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["pdb", "uniprot", "smiles"],
                              "default": "smiles"},
                    "pdb": {"type": "string"},
                    "uniprot": {"type": "string"},
                    "smiles": {"type": "string"},
                    "cutoff": {"type": "number",
                                "description": "Similarity (smiles, 0-1) or identity % (pdb)"},
                    "identity": {"type": "integer", "description": "Identity for pdb mode (0-100)"},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_chebi_lookup",
            "description": (
                "ChEBI (EBI) small-molecule database "
                "(https://www.ebi.ac.uk/chebi/backend/api/public/). "
                "action='search' (free-text 'query') or action='compound' "
                "('chebi_id' like CHEBI:27732 or 27732). Returns compound "
                "metadata, formula, charge, IUPAC name, synonyms."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["search", "compound"],
                                "default": "search"},
                    "query": {"type": "string"},
                    "chebi_id": {"type": "string"},
                    "size": {"type": "integer", "default": 10},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_civic_graphql",
            "description": (
                "CIViC clinical interpretation of cancer variants via GraphQL "
                "(https://civicdb.org/api/graphql). Submit a GraphQL 'query' "
                "string (and optional 'variables'). Narrow selection sets and "
                "filters (e.g. variantId, geneId) are strongly preferred."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "variables": {"type": "object"},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_efo_ontology",
            "description": (
                "EBI Ontology Lookup Service (OLS4) for EFO and other "
                "ontologies (https://www.ebi.ac.uk/ols4/api). "
                "action='search' ('query', 'ontology' default 'efo') or "
                "action='term' ('iri' — full IRI string). Returns ontology "
                "term metadata and labels."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["search", "term"],
                                "default": "search"},
                    "query": {"type": "string"},
                    "iri": {"type": "string"},
                    "ontology": {"type": "string", "default": "efo"},
                    "rows": {"type": "integer", "default": 10},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_gtex_eqtl",
            "description": (
                "GTEx single-tissue eQTL associations "
                "(https://gtexportal.org/api/v2/association/singleTissueEqtl). "
                "action='variant' ('variant_id' in chrN_pos_ref_alt_b38 form) "
                "or action='gene' ('gencode_id'). Optional 'tissue' narrows "
                "by GTEx tissueSiteDetailId. Best for looking up which tissues "
                "show eQTL signal for a variant or gene."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["variant", "gene"],
                                "default": "variant"},
                    "variant_id": {"type": "string"},
                    "gencode_id": {"type": "string"},
                    "tissue": {"type": "string"},
                    "max_items": {"type": "integer", "default": 25},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_opentargets_graphql",
            "description": (
                "Open Targets Platform GraphQL "
                "(https://api.platform.opentargets.org/api/v4/graphql). "
                "Run a GraphQL 'query' (with optional 'variables') for "
                "target-disease associations, evidence, drug-target links, "
                "search, and schema lookup. Keep selection sets narrow."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "variables": {"type": "object"},
                    "max_items": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_pharmgkb_lookup",
            "description": (
                "PharmGKB pharmacogenomics API (https://api.pharmgkb.org/v1/data). "
                "action='gene'|'variant'|'chemical'|'disease' with 'id' "
                "(e.g. PA134865140 for a gene), or "
                "action='clinicalAnnotation' with 'chemical_id'/'gene_id' "
                "filters. Returns drug-gene-variant clinical annotations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string",
                                "enum": ["gene", "variant", "chemical",
                                        "disease", "clinicalAnnotation"],
                                "default": "gene"},
                    "id": {"type": "string"},
                    "chemical_id": {"type": "string"},
                    "gene_id": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_reactome_query",
            "description": (
                "Reactome ContentService (https://reactome.org/ContentService). "
                "action='event' ('event_id' e.g. R-HSA-199420) for a single "
                "event/pathway, 'pathways_for_entity' ('identifier' e.g. "
                "UniProt P38398) for pathway membership, or 'participants' "
                "('event_id') for pathway participants."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string",
                                "enum": ["event", "pathways_for_entity",
                                        "participants"],
                                "default": "pathways_for_entity"},
                    "event_id": {"type": "string"},
                    "identifier": {"type": "string"},
                    "species": {"type": "string", "default": "Homo sapiens"},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_uniprot_lookup",
            "description": (
                "UniProt REST (https://rest.uniprot.org). "
                "action='search' ('query' e.g. 'gene:TP53 AND organism_id:9606') "
                "or action='accession' ('accession' e.g. P04637). "
                "Returns protein metadata: names, sequence, features, "
                "cross-references, subcellular location."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["search", "accession"],
                                "default": "search"},
                    "query": {"type": "string"},
                    "accession": {"type": "string"},
                    "fields": {"type": "string"},
                    "size": {"type": "integer", "default": 10},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": [],
            },
        },
    },
    # ==================================================================
    # Batch 1 — high-value life-science skills (2026-04-18)
    # ==================================================================
    {
        "type": "function",
        "function": {
            "name": "olsp_biorxiv_request",
            "description": (
                "bioRxiv / medRxiv preprint metadata "
                "(https://api.biorxiv.org). action='details' or 'pubs'; "
                "'server' one of 'biorxiv' or 'medrxiv'. Supply 'doi' OR "
                "('start','end'[,'cursor']) date-range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string",
                                "enum": ["details", "pubs"],
                                "default": "details"},
                    "server": {"type": "string",
                                "enum": ["biorxiv", "medrxiv"],
                                "default": "biorxiv"},
                    "doi": {"type": "string"},
                    "start": {"type": "string",
                                "description": "YYYY-MM-DD"},
                    "end": {"type": "string",
                              "description": "YYYY-MM-DD"},
                    "cursor": {"type": "integer", "default": 0},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_cbioportal_request",
            "description": (
                "cBioPortal cancer genomics REST "
                "(https://www.cbioportal.org/api). Pass 'path' such as "
                "'studies', 'studies/{id}/molecular-profiles', or "
                "'molecular-profiles/{id}/mutations/fetch'. Use method='POST' "
                "+ 'json_body' for fetch-style endpoints."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "method": {"type": "string",
                                "enum": ["GET", "POST"], "default": "GET"},
                    "params": {"type": "object"},
                    "json_body": {"type": "object"},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_gnomad_graphql",
            "description": (
                "gnomAD variant-frequency / gene-constraint GraphQL "
                "(https://gnomad.broadinstitute.org/api). Submit a 'query' "
                "(optionally with 'variables'). Use dataset IDs like "
                "'gnomad_r4'. Keep selection sets narrow."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "variables": {"type": "object"},
                    "max_items": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_ncbi_blast",
            "description": (
                "NCBI BLAST Common URL API "
                "(https://blast.ncbi.nlm.nih.gov/Blast.cgi). action='submit' "
                "(needs 'program','database','query' FASTA; returns 'rid'), "
                "'status' (needs 'rid'), or 'fetch' (needs 'rid'). Respect "
                ">=60s between polls per RID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string",
                                "enum": ["submit", "status", "fetch"],
                                "default": "status"},
                    "program": {"type": "string",
                                  "enum": ["blastn", "blastp", "blastx",
                                            "tblastn", "tblastx"],
                                  "default": "blastp"},
                    "database": {"type": "string", "default": "swissprot"},
                    "query": {"type": "string",
                                "description": "FASTA sequence to search"},
                    "rid": {"type": "string"},
                    "hitlist_size": {"type": "integer", "default": 50},
                    "descriptions": {"type": "integer", "default": 5},
                    "alignments": {"type": "integer", "default": 5},
                    "result_format": {"type": "string",
                                        "enum": ["json2", "text"],
                                        "default": "json2"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_ncbi_datasets",
            "description": (
                "NCBI Datasets v2 "
                "(https://api.ncbi.nlm.nih.gov/datasets/v2). Pass 'path' "
                "like 'genome/accession/GCF_000001405.40/dataset_report', "
                "'genome/taxon/assembly_descriptors', or "
                "'taxonomy/taxon/9606'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "params": {"type": "object"},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_string_request",
            "description": (
                "STRING protein-protein interactions "
                "(https://string-db.org/api/json). path one of 'network', "
                "'interaction_partners', 'enrichment', 'get_string_ids', "
                "'homology', 'ppi_enrichment'. Identifiers '%0d' "
                "newline-separated (or one via 'identifier')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string",
                              "enum": ["network", "interaction_partners",
                                        "enrichment", "get_string_ids",
                                        "homology", "ppi_enrichment"],
                              "default": "network"},
                    "identifiers": {"type": "string"},
                    "identifier": {"type": "string"},
                    "species": {"type": "integer", "default": 9606},
                    "limit": {"type": "integer", "default": 10},
                    "required_score": {"type": "integer"},
                    "form": {"type": "object"},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_gwas_catalog_request",
            "description": (
                "GWAS Catalog REST v2 "
                "(https://www.ebi.ac.uk/gwas/rest/api/v2). Pass 'path' "
                "such as 'metadata', 'studies', 'studies/{acc}', "
                "'associations', 'snps', 'efoTraits', 'genes', 'loci'. "
                "'_embedded.<resource>' lists are auto-extracted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "params": {"type": "object"},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_human_protein_atlas",
            "description": (
                "Human Protein Atlas "
                "(https://www.proteinatlas.org). action='gene' ('ensg' e.g. "
                "ENSG00000141510), 'search_download' ('query', 'columns'), or "
                "'page_text' ('subpath' e.g. 'search/tissue/TP53' — returns "
                "capped HTML text head)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string",
                                "enum": ["gene", "search_download",
                                          "page_text"],
                                "default": "gene"},
                    "ensg": {"type": "string"},
                    "query": {"type": "string"},
                    "columns": {"type": "string"},
                    "format": {"type": "string", "default": "json"},
                    "subpath": {"type": "string"},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": [],
            },
        },
    },
    # ==================================================================
    # Batch 2 — medium-value life-science skills (2026-04-18)
    # ==================================================================
    {
        "type": "function",
        "function": {
            "name": "olsp_quickgo_request",
            "description": (
                "QuickGO GO terms / annotations "
                "(https://www.ebi.ac.uk/QuickGO/services). path e.g. "
                "'ontology/go/terms/GO:0008150', 'annotation/search'. "
                "Collections live in data.results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "params": {"type": "object"},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_rnacentral_request",
            "description": (
                "RNAcentral non-coding RNA "
                "(https://rnacentral.org/api/v1). path 'rna/', "
                "'rna/{URS}/' or 'rna/{URS}/xrefs/'. Trailing slash is "
                "required. Lists live in data.results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "params": {"type": "object"},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_encode_request",
            "description": (
                "ENCODE regulatory element atlas "
                "(https://www.encodeproject.org). path e.g. "
                "'biosamples/ENCBS000AAA/' or 'search/' with params "
                "type=Experiment. format=json is injected automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "params": {"type": "object"},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_rhea_request",
            "description": (
                "Rhea biochemical reactions "
                "(https://www.rhea-db.org/rhea). Free-text 'query' or "
                "'RHEA:<id>'. Returns reaction records."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_locus_to_gene_mapper",
            "description": (
                "Open Targets L2G / credible-set helper over the Platform "
                "GraphQL (wrapper around the upstream locus-to-gene-mapper "
                "orchestration). action='credibleSetsForVariant' "
                "('variant_id' e.g. '1_55516888_G_GA'), "
                "'credibleSetsForStudyLocus' ('study_locus_id'), or "
                "'l2g_for_gene_disease' ('ensembl_id' + 'efo_id'). Chain "
                "with olsp_opentargets_graphql / olsp_gwas_catalog_request / "
                "olsp_gtex_eqtl / olsp_genebass_gene_burden for full L2G."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string",
                                "enum": ["credibleSetsForVariant",
                                          "credibleSetsForStudyLocus",
                                          "l2g_for_gene_disease"],
                                "default": "credibleSetsForVariant"},
                    "variant_id": {"type": "string"},
                    "study_locus_id": {"type": "string"},
                    "ensembl_id": {"type": "string"},
                    "efo_id": {"type": "string"},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_finngen_phewas",
            "description": (
                "FinnGen single-variant PheWAS (GRCh38) "
                "(https://r12.finngen.fi). Supply 'variant' or 'grch38' "
                "as 'chr:pos-ref-alt' (e.g. '10:112998590-C-T'); rsID "
                "must be pre-resolved by the caller."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "variant": {"type": "string"},
                    "grch38": {"type": "string"},
                    "max_results": {"type": "integer", "default": 10},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_biobankjapan_phewas",
            "description": (
                "BioBank Japan single-variant PheWAS (GRCh37) "
                "(https://pheweb.jp). Supply 'variant' or 'grch37' "
                "as 'chr:pos-ref-alt' (e.g. '10:114758349-C-T'); rsID "
                "must be pre-resolved by the caller."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "variant": {"type": "string"},
                    "grch37": {"type": "string"},
                    "max_results": {"type": "integer", "default": 10},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_biostudies_request",
            "description": (
                "BioStudies / ArrayExpress functional genomics "
                "(https://www.ebi.ac.uk/biostudies/api/v1). path e.g. "
                "'search' + params query=rna, 'ArrayExpress/search', "
                "'studies/{accession}', or 'studies/{accession}/info'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "params": {"type": "object"},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_genebass_gene_burden",
            "description": (
                "Genebass gene-burden PheWAS "
                "(https://main.genebass.org/api). Needs 'ensembl_gene_id' "
                "(e.g. ENSG00000173531). burden_set one of 'pLoF', "
                "'missense|LC', 'synonymous'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ensembl_gene_id": {"type": "string"},
                    "burden_set": {"type": "string",
                                     "enum": ["pLoF", "missense|LC",
                                              "synonymous"],
                                     "default": "pLoF"},
                    "max_results": {"type": "integer", "default": 25},
                },
                "required": ["ensembl_gene_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_epigraphdb_request",
            "description": (
                "EpiGraphDB causal / MR graph "
                "(https://api.epigraphdb.org). path e.g. 'ping', "
                "'ontology/gwas-efo', 'gene/drugs', 'mr', 'literature/gwas'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "params": {"type": "object"},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_eqtl_catalogue_request",
            "description": (
                "eQTL Catalogue "
                "(https://www.ebi.ac.uk/eqtl/api). path e.g. "
                "'genes/{ensembl_id}/associations', "
                "'studies/{study}/associations', 'associations/{rsid}'. "
                "Upstream is fragile — supply filters like study/tissue/"
                "variant_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "params": {"type": "object"},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": ["path"],
            },
        },
    },
    # ==================================================================
    # Batch 3 — lower-value life-science skills (2026-04-18)
    # ==================================================================
    {
        "type": "function",
        "function": {
            "name": "olsp_ukb_topmed_phewas",
            "description": (
                "UKB-TOPMed joint PheWAS (GRCh38) "
                "(https://pheweb.org/UKB-TOPMed). Supply 'variant' or "
                "'grch38' as 'chr:pos-ref-alt'; rsID must be pre-resolved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "variant": {"type": "string"},
                    "grch38": {"type": "string"},
                    "max_results": {"type": "integer", "default": 10},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_tpmi_phewas",
            "description": (
                "Taiwan Precision Medicine Initiative PheWAS (GRCh38) "
                "(https://pheweb.ibms.sinica.edu.tw). Supply 'variant' or "
                "'grch38' as 'chr:pos-ref-alt'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "variant": {"type": "string"},
                    "grch38": {"type": "string"},
                    "max_results": {"type": "integer", "default": 10},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_hmdb_request",
            "description": (
                "Human Metabolome Database (https://hmdb.ca). Free-text "
                "'query' + 'category' (metabolites / proteins / diseases / "
                "pathways). Results live in data.<category>."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "category": {"type": "string",
                                   "enum": ["metabolites", "proteins",
                                            "diseases", "pathways"],
                                   "default": "metabolites"},
                    "per_page": {"type": "integer", "default": 10},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_metabolights_request",
            "description": (
                "MetaboLights metabolomics "
                "(https://www.ebi.ac.uk/metabolights/ws). path e.g. "
                "'studies' or 'studies/MTBLS1'. Lists live in data.content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "params": {"type": "object"},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_pride_request",
            "description": (
                "PRIDE Archive proteomics "
                "(https://www.ebi.ac.uk/pride/ws/archive/v2). path e.g. "
                "'projects' + params keyword=proteomics, "
                "'projects/{PXD accession}'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "params": {"type": "object"},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_proteomexchange_request",
            "description": (
                "ProteomeXchange PROXI "
                "(https://proteomecentral.proteomexchange.org/api/proxi/v0.1). "
                "path e.g. 'datasets', 'datasets/{PXD id}', 'spectra', "
                "'peptidoforms', 'proteins', 'usi_examples'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "params": {"type": "object"},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": ["path"],
            },
        },
    },
    # ==================================================================
    # Batch 4 — minimal-value life-science skills (2026-04-18)
    # ==================================================================
    {
        "type": "function",
        "function": {
            "name": "olsp_mgnify_request",
            "description": (
                "MGnify microbiome metagenomics "
                "(https://www.ebi.ac.uk/metagenomics/api/v1). path e.g. "
                "'studies', 'samples', 'biomes'. Uses JSON:API (records "
                "live in data.data)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "params": {"type": "object"},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_eva_request",
            "description": (
                "European Variation Archive "
                "(https://www.ebi.ac.uk/eva/webservices/rest/v1). path "
                "e.g. 'meta/species/list', 'variants/rs699'. Response is "
                "unwrapped from response[0].result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "params": {"type": "object"},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_ipd_request",
            "description": (
                "IPD (Immuno Polymorphism Database) REST "
                "(https://www.ebi.ac.uk/cgi-bin/ipd/api). path 'allele', "
                "'cell', 'allele/download'. Default params.project='HLA'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string",
                              "enum": ["allele", "cell", "allele/download"],
                              "default": "allele"},
                    "params": {"type": "object"},
                    "limit": {"type": "integer", "default": 10},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "olsp_ncbi_clinicaltables",
            "description": (
                "NCBI Clinical Tables gene autocomplete search "
                "(https://clinicaltables.nlm.nih.gov/api/ncbi_genes/v3/search). "
                "Needs 'terms' (e.g. 'TP53'). Optional params: 'df', 'ef', "
                "'sf', 'count', 'offset'. Returns [total, codes, extra, "
                "display] unwrapped."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "terms": {"type": "string"},
                    "params": {"type": "object"},
                    "count": {"type": "integer", "default": 10},
                    "max_items": {"type": "integer", "default": 10},
                },
                "required": ["terms"],
            },
        },
    },
]


OPENAI_PORTED_TOOL_NAMES: set[str] = {
    t["function"]["name"] for t in OPENAI_PORTED_TOOL_SPECS
}
