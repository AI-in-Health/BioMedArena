"""Category tags for TOOL_SPECS.

Each tool in ``harness/eval/function_calling_runner.py::TOOL_SPECS`` is
mapped here to one or more category tags. Benchmarks (see
``harness/benchmark_configs/registry.py``) can whitelist tools by
category instead of enumerating them one-by-one.

Tools not present in this map have no category tags — they will not
match any category-based whitelist. Callers wanting a superset should
use an explicit ``tool_whitelist`` on the benchmark config, or leave
both lists empty (current default → all tools advertised).
"""
from __future__ import annotations

from typing import Any


TOOL_CATEGORIES: dict[str, list[str]] = {
    # ---- Calculation & code exec ----
    "compute_calculator": ["calculation", "clinical"],
    "calculator_eval": ["calculation"],
    "python_exec": ["calculation", "code"],

    # ---- Literature search / PubMed ----
    "pubmed_search": ["literature", "search", "clinical"],
    "olsp_biorxiv_request": ["literature", "search"],

    # ---- Medical reference / drug labels ----
    "medlineplus_topic": ["clinical", "literature"],
    "rxnav_drug": ["clinical", "drug"],
    "dailymed_label": ["clinical", "drug"],
    "openfda_adverse": ["clinical", "drug"],
    "orphanet_lookup": ["clinical", "rare_disease"],
    "omim_lookup": ["clinical", "genetics"],

    # ---- Variants / clinical genetics ----
    "clinvar_lookup": ["genetics", "variant", "clinical"],
    "olsp_gnomad_graphql": ["genetics", "variant"],
    "olsp_gtex_eqtl": ["genetics", "gene_expression"],
    "olsp_civic_graphql": ["genetics", "variant", "cancer"],
    "olsp_eqtl_catalogue_request": ["genetics", "gene_expression"],
    "olsp_finngen_phewas": ["genetics", "variant", "gwas"],
    "olsp_biobankjapan_phewas": ["genetics", "variant", "gwas"],
    "olsp_ukb_topmed_phewas": ["genetics", "variant", "gwas"],
    "olsp_tpmi_phewas": ["genetics", "variant", "gwas"],
    "olsp_genebass_gene_burden": ["genetics", "gwas"],
    "olsp_gwas_catalog_request": ["genetics", "gwas"],
    "olsp_locus_to_gene_mapper": ["genetics", "gwas"],
    "olsp_epigraphdb_request": ["genetics", "gwas", "literature"],
    "olsp_eva_request": ["genetics", "variant"],

    # ---- Genes: sequence / info ----
    "gene_lookup": ["genetics"],
    "gget_info": ["genetics"],
    "gget_search": ["genetics", "search"],
    "gget_seq": ["genetics", "sequence"],
    "mygene_query": ["genetics"],
    "olsp_ncbi_datasets": ["genetics", "genomics"],
    "olsp_ncbi_blast": ["genetics", "sequence"],

    # ---- Protein / structure ----
    "alphafold_db_lookup": ["protein", "structure"],
    "olsp_uniprot_lookup": ["protein"],
    "olsp_string_request": ["protein", "network"],
    "olsp_human_protein_atlas": ["protein", "gene_expression"],
    "olsp_ipd_request": ["protein", "immunology"],

    # ---- Sequence utilities ----
    "dna_reverse_complement": ["sequence"],
    "dna_translate": ["sequence"],
    "fasta_parse": ["sequence"],
    "genbank_parse": ["sequence"],

    # ---- Chemistry / small-molecule ----
    "mol_descriptors": ["chemistry", "drug"],
    "mol_fingerprint": ["chemistry"],
    "mol_from_smiles": ["chemistry"],
    "mol_similarity": ["chemistry"],
    "mol_substructure_match": ["chemistry"],
    "molecular_property_predict": ["chemistry", "drug"],
    "molfeat_featurize": ["chemistry"],
    "admet_predict_native": ["chemistry", "drug"],
    "olsp_chebi_lookup": ["chemistry"],
    "olsp_bindingdb_ligands": ["chemistry", "drug"],
    "olsp_rhea_request": ["chemistry", "biochemistry"],

    # ---- Metabolomics / proteomics / multi-omics ----
    "olsp_hmdb_request": ["metabolomics"],
    "olsp_metabolights_request": ["metabolomics"],
    "olsp_pride_request": ["proteomics"],
    "olsp_proteomexchange_request": ["proteomics"],
    "olsp_mgnify_request": ["metagenomics"],

    # ---- Pathways / reactions ----
    "olsp_reactome_query": ["pathway"],

    # ---- Gene expression ----
    "olsp_bgee_sparql": ["gene_expression"],
    "olsp_encode_request": ["gene_expression", "regulation"],
    "olsp_rnacentral_request": ["gene_expression", "rna"],

    # ---- Ontology ----
    "olsp_efo_ontology": ["ontology"],
    "olsp_quickgo_request": ["ontology"],

    # ---- Pharmacogenomics / targets / drug ----
    "olsp_pharmgkb_lookup": ["pgx", "drug"],
    "olsp_opentargets_graphql": ["target", "drug"],

    # ---- Cancer ----
    "olsp_cbioportal_request": ["cancer", "genetics"],

    # ---- Clinical tables / coding ----
    "olsp_ncbi_clinicaltables": ["clinical", "ontology"],

    # ---- BioStudies / functional genomics ----
    "olsp_biostudies_request": ["genomics", "literature"],

    # ---- Imaging / DICOM ----
    "dicom_pixel_stats": ["imaging"],
    "medical_image_metadata": ["imaging"],
    "medical_image_normalize": ["imaging"],
    "read_dicom_metadata": ["imaging"],

    # ---- Code search (repo-aware) ----
    "code_search": ["code", "search"],

    # ---- Survival analysis ----
    "kaplan_meier_fit": ["stats", "survival"],
    "logrank_two_sample": ["stats", "survival"],

    # ---- TDC drug / ADMET ----
    "tdc_admet_lookup": ["chemistry", "drug"],
    "tdc_load_dataset_sample": ["chemistry", "drug"],
    "tdc_molecule_generation_sample": ["chemistry", "drug"],

    # ---- Web search (Serper + Jina) ----
    "serper_search": ["web", "search", "literature"],
    "jina_read_page": ["web", "search", "literature"],
}

OPTIONAL_TOOL_CATEGORY_ENTRIES: frozenset[str] = frozenset({
    "admet_predict_native",
    "medical_image_metadata",
    "medical_image_normalize",
    "molecular_property_predict",
    "molfeat_featurize",
})

ALL_CATEGORIES: tuple[str, ...] = tuple(
    sorted({category for categories in TOOL_CATEGORIES.values() for category in categories})
)


def _name(spec: dict[str, Any]) -> str:
    if "function" in spec and isinstance(spec["function"], dict):
        return str(spec["function"].get("name") or "")
    return str(spec.get("name") or "")


def get_tools_by_category(
    categories: list[str],
    all_tool_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter TOOL_SPECS to those tagged with any of the given categories."""
    wanted = set(categories or [])
    if not wanted:
        return []
    out: list[dict[str, Any]] = []
    for spec in all_tool_specs:
        tags = TOOL_CATEGORIES.get(_name(spec), [])
        if wanted.intersection(tags):
            out.append(spec)
    return out


def get_tools_by_whitelist(
    whitelist: list[str],
    all_tool_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter TOOL_SPECS to those whose name is in the whitelist."""
    wanted = set(whitelist or [])
    return [s for s in all_tool_specs if _name(s) in wanted]


def uncategorised_tools(all_tool_specs: list[dict[str, Any]]) -> list[str]:
    """Diagnostic: tool names present in TOOL_SPECS but missing from map."""
    return [_name(s) for s in all_tool_specs if _name(s) not in TOOL_CATEGORIES]
