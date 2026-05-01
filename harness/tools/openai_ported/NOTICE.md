# NOTICE — openai/plugins life-science-research port

Upstream repository: <https://github.com/openai/plugins>
Upstream path: `plugins/life-science-research/skills/`

## Scope

This package re-implements the public REST / GraphQL / SPARQL endpoints
documented in the upstream `SKILL.md` files as native Python handlers
for our function-calling runner. The upstream skills themselves are
Codex Markdown prompt instructions and small shell helpers — none of
that content is redistributed here.

## Ported skills

| Upstream skill | Our TOOL_SPEC |
| --- | --- |
| bgee-skill | `olsp_bgee_sparql` |
| bindingdb-skill | `olsp_bindingdb_ligands` |
| chebi-skill | `olsp_chebi_lookup` |
| civic-skill | `olsp_civic_graphql` |
| efo-ontology-skill | `olsp_efo_ontology` |
| gtex-eqtl-skill | `olsp_gtex_eqtl` |
| opentargets-skill | `olsp_opentargets_graphql` |
| pharmgkb-skill | `olsp_pharmgkb_lookup` |
| reactome-skill | `olsp_reactome_query` |
| uniprot-skill | `olsp_uniprot_lookup` |
| biorxiv-skill | `olsp_biorxiv_request` |
| cbioportal-skill | `olsp_cbioportal_request` |
| gnomad-graphql-skill | `olsp_gnomad_graphql` |
| ncbi-blast-skill | `olsp_ncbi_blast` |
| ncbi-datasets-skill | `olsp_ncbi_datasets` |
| string-skill | `olsp_string_request` |
| gwas-catalog-skill | `olsp_gwas_catalog_request` |
| human-protein-atlas-skill | `olsp_human_protein_atlas` |
| quickgo-skill | `olsp_quickgo_request` |
| rnacentral-skill | `olsp_rnacentral_request` |
| encode-skill | `olsp_encode_request` |
| rhea-skill | `olsp_rhea_request` |
| locus-to-gene-mapper-skill | `olsp_locus_to_gene_mapper` |
| finngen-phewas-skill | `olsp_finngen_phewas` |
| biobankjapan-phewas-skill | `olsp_biobankjapan_phewas` |
| biostudies-arrayexpress-skill | `olsp_biostudies_request` |
| genebass-gene-burden-skill | `olsp_genebass_gene_burden` |
| epigraphdb-skill | `olsp_epigraphdb_request` |
| eqtl-catalogue-skill | `olsp_eqtl_catalogue_request` |
| ukb-topmed-phewas-skill | `olsp_ukb_topmed_phewas` |
| tpmi-phewas-skill | `olsp_tpmi_phewas` |
| hmdb-skill | `olsp_hmdb_request` (Cloudflare 403 at port time) |
| metabolights-skill | `olsp_metabolights_request` |
| pride-skill | `olsp_pride_request` |
| proteomexchange-skill | `olsp_proteomexchange_request` |
| mgnify-skill | `olsp_mgnify_request` |
| eva-skill | `olsp_eva_request` |
| ipd-skill | `olsp_ipd_request` |
| ncbi-clinicaltables-skill | `olsp_ncbi_clinicaltables` |

**research-router-skill** is intentionally NOT ported — upstream is a
routing heuristic / orchestration meta-skill, not a distinct database.

## License

At port time the upstream repository did not publish a `LICENSE` file
at its root. We redistribute no upstream text or source code — only
publicly documented endpoint paths, parameter names, and payload
shapes (which are API surfaces of the underlying databases, not
OpenAI-authored content). If/when upstream adopts a license, update
this file to note it.
