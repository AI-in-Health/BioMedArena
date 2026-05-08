"""Registry for generic Hugging Face benchmark datasets.

The entries here are intentionally data-only. They are expanded into CLI
benchmark registrations and handled by ``bench_hf_benchmark``. Keep keys
stable and ASCII; they become public ``bioagent --benchmark`` names.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HFDatasetSpec:
    key: str
    repo: str
    task_type: str
    domain: str
    config: str | None = None
    split: str | None = None
    question_fields: tuple[str, ...] = ()
    answer_fields: tuple[str, ...] = ()
    text_fields: tuple[str, ...] = ()
    choice_fields: tuple[str, ...] = ()
    label_fields: tuple[str, ...] = ()
    input_fields: tuple[str, ...] = ()
    context_fields: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


def _spec(
    key: str,
    repo: str,
    task_type: str,
    domain: str,
    *,
    config: str | None = None,
    split: str | None = None,
    question_fields: tuple[str, ...] = (),
    answer_fields: tuple[str, ...] = (),
    text_fields: tuple[str, ...] = (),
    choice_fields: tuple[str, ...] = (),
    label_fields: tuple[str, ...] = (),
    input_fields: tuple[str, ...] = (),
    context_fields: tuple[str, ...] = (),
    **extra: Any,
) -> HFDatasetSpec:
    return HFDatasetSpec(
        key=key,
        repo=repo,
        config=config,
        split=split,
        task_type=task_type,
        domain=domain,
        question_fields=question_fields,
        answer_fields=answer_fields,
        text_fields=text_fields,
        choice_fields=choice_fields,
        label_fields=label_fields,
        input_fields=input_fields,
        context_fields=context_fields,
        extra=extra,
    )


HF_BENCHMARK_SPECS: dict[str, HFDatasetSpec] = {
    # Medical QA / reasoning / dialogue
    s.key: s for s in [
        _spec("hf_medquad", "keivalya/MedQuad-MedicalQnADataset", "qa", "medical"),
        _spec("hf_adaptllm_medicine_tasks", "AdaptLLM/medicine-tasks", "mcq", "medical", config="USMLE",
              derived_from="AdaptLLM-modified", recommend_canonical="medqa",
              warning="Reading comprehension format, not original benchmark format"),
        _spec("hf_medical_dialog", "UCSD26/medical_dialog", "qa", "medical"),
        _spec("hf_mts_dialogue_clinical_note", "har1/MTS_Dialogue-Clinical_Note", "summarization", "clinical"),
        _spec("hf_medical_question_pairs", "curaihealth/medical_questions_pairs", "pair_classification", "medical"),
        _spec("hf_medical_chronology", "Superinsight/medical-chronology-benchmark", "qa", "medical"),
        _spec("hf_chinese_medical_dialogue", "BillGPT/Chinese-medical-dialogue-data", "qa", "medical"),
        _spec("hf_industry_medicine_health_tcm", "BAAI/IndustryCorpus2_medicine_health_psychology_traditional_chinese_medicine", "text", "medical"),
        _spec("hf_bigbio_med_qa", "bigbio/med_qa", "mcq", "medical",
              parquet_config="med_qa_en_4options_bigbio_qa", parquet_split="test"),
        _spec("hf_bigbio_pubmed_qa", "bigbio/pubmed_qa", "classification", "medical",
              parquet_config="pubmed_qa_labeled_fold0_bigbio_qa", parquet_split="test",
              question_fields=("question",), context_fields=("context",),
              answer_fields=("final_decision",), scorer="mcq_exact_match",
              benchmark_task_type="mcq_qa"),
        _spec("hf_medqa_corpus_en", "cogbuji/medqa_corpus_en", "text", "medical"),
    ]
}


HF_BENCHMARK_SPECS.update({
    s.key: s for s in [
        # PubMed / biomedical NLP / retrieval-oriented text
        _spec("hf_medrag_pubmed", "MedRAG/pubmed", "retrieval", "biomedical", streaming=True),
        _spec("hf_ncbi_pubmed", "ncbi/pubmed", "text", "biomedical", streaming=True),
        _spec("hf_ccdv_pubmed_summarization", "ccdv/pubmed-summarization", "summarization", "biomedical"),
        _spec("hf_pubmed_20k_rct", "armanc/pubmed-rct20k", "classification", "biomedical",
              split="test", text_fields=("abstract",), answer_fields=("label",),
              deprecated_alias_for="hf_pubmed_200k_rct"),
        _spec("hf_pubmed_200k_rct", "pietrolesci/pubmed-200k-rct", "classification", "biomedical"),
        _spec("hf_pubmed_abstract", "uiyunkim-hub/pubmed-abstract", "text", "biomedical"),
        _spec("hf_multilingual_medical_corpus", "HiTZ/Multilingual-Medical-Corpus", "text", "medical"),
        _spec("hf_spaccc_tokenizer", "Biomedical-TeMU/SPACCC_Tokenizer", "classification", "biomedical"),
        _spec("hf_mteb_medical_qa", "mteb/medical_qa", "retrieval", "medical"),
        _spec("hf_mteb_medical_retrieval", "mteb/MedicalRetrieval", "retrieval", "medical"),
        _spec("hf_embedding_chatdoctor", "embedding-benchmark/ChatDoctor_HealthCareMagic", "retrieval", "medical"),
        _spec("hf_clinical_trials_data", "Dattito/clinical-trials-data", "text", "clinical"),
        _spec("hf_medicine_authorship", "michaelsyao/MedicineAuthorship", "classification", "medical"),
        _spec("hf_agentds_healthcare", "lainmn/AgentDS-Healthcare", "classification", "healthcare"),
        # Chemistry / molecule
        _spec("hf_xythick_chemistry", "XythicK/Chemistry", "text", "chemistry"),
        _spec("hf_gaianet_chemistry", "gaianet/chemistry", "text", "chemistry"),
        _spec("hf_chemistry_qa", "avaliev/ChemistryQA", "qa", "chemistry"),
        _spec("hf_chemistry_stackexchange", "jablonkagroup/chemistry_stackexchange", "text", "chemistry", config="completion_0"),
        _spec("hf_moleculenet_bace", "scikit-fingerprints/MoleculeNet_BACE", "molecule_property", "chemistry"),
        _spec("hf_moleculenet_bbbp", "scikit-fingerprints/MoleculeNet_BBBP", "molecule_property", "chemistry"),
        _spec("hf_moleculenet_hiv", "scikit-fingerprints/MoleculeNet_HIV", "molecule_property", "chemistry"),
        _spec("hf_moleculenet_pcba", "scikit-fingerprints/MoleculeNet_PCBA", "molecule_property", "chemistry"),
        _spec("hf_moleculenet_esol", "scikit-fingerprints/MoleculeNet_ESOL", "molecule_property", "chemistry"),
        _spec("hf_moleculenet_freesolv", "scikit-fingerprints/MoleculeNet_FreeSolv", "molecule_property", "chemistry"),
        _spec("hf_moleculenet_lipophilicity", "scikit-fingerprints/MoleculeNet_Lipophilicity", "molecule_property", "chemistry"),
        _spec("hf_moleculenet_toxcast", "scikit-fingerprints/MoleculeNet_ToxCast", "molecule_property", "chemistry"),
        _spec("hf_moleculenet_clintox", "scikit-fingerprints/MoleculeNet_ClinTox", "molecule_property", "chemistry"),
        _spec("hf_moleculenet_sider", "scikit-fingerprints/MoleculeNet_SIDER", "molecule_property", "chemistry"),
        _spec("hf_molecule3d", "maomlab/Molecule3D", "molecule_property", "chemistry", config="Molecule3D_random_split"),
        _spec("hf_molecule_property_instruction", "haitengzhao/molecule_property_instruction", "qa", "chemistry"),
        _spec("hf_lpm24_eval_molgen", "language-plus-molecules/LPM-24_eval-molgen", "qa", "chemistry",
              scorer="smiles_validity_plus_tanimoto", benchmark_task_type="smiles_generation"),
        _spec("hf_lpm24_eval_caption", "language-plus-molecules/LPM-24_eval-caption", "qa", "chemistry"),
        _spec("hf_moleculeace", "karina-zadorozhny/moleculeace", "molecule_property", "chemistry", config="CHEMBL1862_Ki"),
        _spec("hf_dolma_chemistry_only", "BASF-AI/dolma-chemistry-only", "text", "chemistry", streaming=True),
    ]
})


HF_BENCHMARK_SPECS.update({
    s.key: s for s in [
        # Stable config-level medical datasets from aggregated HF repos.
        _spec("hf_lavita_medmcqa", "lavita/medical-qa-datasets", "mcq", "medical", config="medmcqa", split="validation"),
        _spec("hf_lavita_usmle_step1", "lavita/medical-qa-datasets", "qa", "medical", config="usmle-self-assessment-step1"),
        _spec("hf_lavita_usmle_step2", "lavita/medical-qa-datasets", "qa", "medical", config="usmle-self-assessment-step2"),
        _spec("hf_lavita_usmle_step3", "lavita/medical-qa-datasets", "qa", "medical", config="usmle-self-assessment-step3"),
        _spec("hf_adaptllm_chemprot", "AdaptLLM/medicine-tasks", "mcq", "biomedical", config="ChemProt",
              derived_from="AdaptLLM-modified", recommend_canonical="hf_chemprot",
              warning="Reading comprehension format, not original benchmark format"),
        _spec("hf_adaptllm_mqp", "AdaptLLM/medicine-tasks", "mcq", "medical", config="MQP",
              derived_from="AdaptLLM-modified", recommend_canonical="hf_medical_question_pairs",
              warning="Reading comprehension format, not original benchmark format"),
        _spec("hf_adaptllm_rct", "AdaptLLM/medicine-tasks", "mcq", "biomedical", config="RCT",
              derived_from="AdaptLLM-modified", recommend_canonical="hf_pubmed_200k_rct",
              warning="Reading comprehension format, not original benchmark format"),
    ]
})


HF_BENCHMARK_SPECS.update({
    s.key: s for s in [
        # Stable config-level chemistry / molecule datasets.
        _spec("hf_katielink_moleculenet_bace", "katielink/moleculenet-benchmark", "molecule_property", "chemistry", config="bace"),
        _spec("hf_katielink_moleculenet_bbbp", "katielink/moleculenet-benchmark", "molecule_property", "chemistry", config="bbbp"),
        _spec("hf_katielink_moleculenet_clintox", "katielink/moleculenet-benchmark", "molecule_property", "chemistry", config="clintox"),
        _spec("hf_katielink_moleculenet_esol", "katielink/moleculenet-benchmark", "molecule_property", "chemistry", config="esol"),
        _spec("hf_katielink_moleculenet_freesolv", "katielink/moleculenet-benchmark", "molecule_property", "chemistry", config="freesolv"),
        _spec("hf_katielink_moleculenet_hiv", "katielink/moleculenet-benchmark", "molecule_property", "chemistry", config="hiv"),
        _spec("hf_katielink_moleculenet_lipo", "katielink/moleculenet-benchmark", "molecule_property", "chemistry",
              parquet_glob="hf://datasets/katielink/moleculenet-benchmark/lipo/lipo_test.csv",
              text_fields=("smiles",), answer_fields=("y",)),
        _spec("hf_katielink_moleculenet_sider", "katielink/moleculenet-benchmark", "molecule_property", "chemistry", config="sider"),
        _spec("hf_katielink_moleculenet_tox21", "katielink/moleculenet-benchmark", "molecule_property", "chemistry", config="tox21"),
        _spec("hf_moleculeace_chembl1871_ki", "karina-zadorozhny/moleculeace", "molecule_property", "chemistry", config="CHEMBL1871_Ki"),
        _spec("hf_moleculeace_chembl204_ki", "karina-zadorozhny/moleculeace", "molecule_property", "chemistry", config="CHEMBL204_Ki"),
        _spec("hf_moleculeace_chembl214_ki", "karina-zadorozhny/moleculeace", "molecule_property", "chemistry", config="CHEMBL214_Ki"),
        _spec("hf_moleculeace_chembl228_ki", "karina-zadorozhny/moleculeace", "molecule_property", "chemistry", config="CHEMBL228_Ki"),
        _spec("hf_moleculeace_chembl237_ec50", "karina-zadorozhny/moleculeace", "molecule_property", "chemistry", config="CHEMBL237_EC50"),
    ]
})


HF_BENCHMARK_SPECS.update({
    s.key: s for s in [
        # Protein/RNA config-level tasks.
        _spec("hf_proteinlmbench_uniprot_function", "tsynbio/ProteinLMBench", "mcq", "protein",
              config="UniProt_Function", question_fields=("question",), choice_fields=("options",),
              answer_fields=("answer",), scorer="mcq_exact_match", benchmark_task_type="mcq_qa"),
        _spec("hf_proteinlmbench_uniprot_induction", "tsynbio/ProteinLMBench", "mcq", "protein",
              config="UniProt_Induction", question_fields=("question",), choice_fields=("options",),
              answer_fields=("answer",), scorer="mcq_exact_match", benchmark_task_type="mcq_qa"),
        _spec("hf_proteinlmbench_uniprot_disease", "tsynbio/ProteinLMBench", "mcq", "protein",
              config="UniProt_Involvement in disease", question_fields=("question",), choice_fields=("options",),
              answer_fields=("answer",), scorer="mcq_exact_match", benchmark_task_type="mcq_qa"),
        _spec("hf_proteinlmbench_uniprot_ptm", "tsynbio/ProteinLMBench", "mcq", "protein",
              config="UniProt_Post-translational modification", question_fields=("question",), choice_fields=("options",),
              answer_fields=("answer",), scorer="mcq_exact_match", benchmark_task_type="mcq_qa"),
        _spec("hf_proteinlmbench_uniprot_subunit", "tsynbio/ProteinLMBench", "mcq", "protein",
              config="UniProt_Subunit structure", question_fields=("question",), choice_fields=("options",),
              answer_fields=("answer",), scorer="mcq_exact_match", benchmark_task_type="mcq_qa"),
        _spec("hf_proteinlmbench_uniprot_tissue", "tsynbio/ProteinLMBench", "mcq", "protein",
              config="UniProt_Tissue specificity", question_fields=("question",), choice_fields=("options",),
              answer_fields=("answer",), scorer="mcq_exact_match", benchmark_task_type="mcq_qa"),
        _spec("hf_proteinlmbench_enzyme_cot", "tsynbio/ProteinLMBench", "mcq", "protein",
              config="Enzyme_CoT", question_fields=("question",), choice_fields=("options",),
              answer_fields=("answer",), scorer="mcq_exact_match", benchmark_task_type="mcq_qa"),
        _spec("hf_rna_expression_hek", "genbio-ai/rna-downstream-tasks", "regression", "rna", config="expression_HEK"),
        _spec("hf_rna_expression_muscle", "genbio-ai/rna-downstream-tasks", "regression", "rna", config="expression_Muscle"),
        _spec("hf_rna_expression_pc3", "genbio-ai/rna-downstream-tasks", "regression", "rna", config="expression_pc3"),
        _spec("hf_rna_splice_site_acceptor", "genbio-ai/rna-downstream-tasks", "classification", "rna", config="splice_site_acceptor"),
        _spec("hf_rna_splice_site_donor", "genbio-ai/rna-downstream-tasks", "classification", "rna", config="splice_site_donor"),
        _spec("hf_rna_modification_site", "genbio-ai/rna-downstream-tasks", "classification", "rna", config="modification_site"),
        _spec("hf_rna_ncrna_family_bnoise0", "genbio-ai/rna-downstream-tasks", "classification", "rna", config="ncrna_family_bnoise0"),
        _spec("hf_rna_mean_ribosome_load", "genbio-ai/rna-downstream-tasks", "regression", "rna", config="mean_ribosome_load"),
    ]
})


HF_BENCHMARK_SPECS.update({
    s.key: s for s in [
        # Protein
        _spec("hf_protein_mpnn", "RosettaCommons/ProteinMPNN", "sequence", "protein"),
        _spec("hf_group_mpnn", "ProteinMPNN/group_mpnn", "sequence", "protein"),
        _spec("hf_proteingym_v1", "OATML-Markslab/ProteinGym_v1", "protein_fitness", "protein",
              parquet_glob="hf://datasets/OATML-Markslab/ProteinGym_v1/DMS_substitutions/*.parquet"),
        _spec("hf_proteingym_v01", "OATML-Markslab/ProteinGym_v0.1", "protein_fitness", "protein"),
        _spec("hf_icml2022_proteingym", "ICML2022/ProteinGym", "protein_fitness", "protein"),
        _spec("hf_genbio_proteingym_dms", "genbio-ai/ProteinGYM-DMS", "protein_fitness", "protein"),
        _spec("hf_proteinlmbench", "tsynbio/ProteinLMBench", "mcq", "protein",
              config="evaluation", split="train",
              question_fields=("question",), choice_fields=("options",), answer_fields=("answer",)),
        _spec("hf_protein_solubility", "proteinea/solubility", "classification", "protein"),
        _spec("hf_protein_fluorescence", "proteinea/fluorescence", "regression", "protein"),
        _spec("hf_protein_deeploc", "proteinea/deeploc", "classification", "protein"),
        _spec("hf_fluorescence_prediction", "proteinglm/fluorescence_prediction", "classification", "protein"),
        _spec("hf_protein_secondary_structure", "lamm-mit/protein_secondary_structure_from_PDB", "classification", "protein"),
        _spec("hf_pdb_protein_ligand", "jglaser/pdb_protein_ligand_complexes", "molecule_property", "protein"),
        _spec("hf_protein_binding_sequences", "ronig/protein_binding_sequences", "qa", "protein",
              text_fields=("receptor",), answer_fields=("peptide",),
              scorer="regression_spearman_auc", benchmark_task_type="protein_property"),
        _spec("hf_protein_stability", "SaProtHub/Dataset-Meta-scale-protein-stability", "regression", "protein"),
        _spec("hf_protein_conformational_states", "PDBEurope/protein_chain_conformational_states", "classification", "protein"),
        _spec("hf_protein_docs", "timodonnell/protein-docs", "text", "protein"),
        # Gene / genomics / DNA / RNA
        _spec("hf_genecorpus_30m", "ctheodoris/Genecorpus-30M", "sequence", "genomics", streaming=True),
        _spec("hf_geneexp", "xingyusu/GeneExp", "regression", "genomics"),
        _spec("hf_genomics_long_range", "InstaDeepAI/genomics-long-range-benchmark", "classification", "dna",
              loader_repo="InstaDeepAI/genomics-long-range-benchmark",
              parquet_glob="hf://datasets/InstaDeepAI/genomics-long-range-benchmark/regulatory_elements/enhancer_dataset_subset.csv"),
        _spec("hf_rna_downstream_tasks", "genbio-ai/rna-downstream-tasks", "classification", "rna",
              config="modification_site", split="test"),
        _spec("hf_bacbench_antibiotic_resistance_dna", "macwiatrak/bacbench-antibiotic-resistance-dna", "classification", "dna",
              parquet_config="default", parquet_split="partial-train"),
        _spec("hf_bacbench_phenotypic_traits_dna", "macwiatrak/bacbench-phenotypic-traits-dna", "classification", "dna",
              parquet_config="default", parquet_split="partial-train"),
        _spec("hf_pgs_catalog", "just-dna-seq/pgs-catalog", "classification", "dna"),
        _spec("hf_bacterial_intergenic_dna", "AllTheBacteria/Bac-Corpus-dna-intergenic-sequences-high-diversity", "sequence", "dna", streaming=True),
        _spec("hf_forensic_dnanet", "NetherlandsForensicInstitute/DNANet_2p5pMixture_PPF6C_2024", "classification", "dna"),
        _spec("hf_traitgym_mendelian_dna", "bolinas-dna/evals-traitgym_mendelian_v2_harness_255", "classification", "dna"),
        _spec("hf_animal_genomes_v5_5", "bolinas-dna/genomes-v5-genome_set-animals-intervals-v5_255_128", "sequence", "dna"),
        _spec("hf_animal_genomes_v5_1", "bolinas-dna/genomes-v5-genome_set-animals-intervals-v1_255_128", "sequence", "dna"),
        _spec("hf_animal_genomes_v5_15", "bolinas-dna/genomes-v5-genome_set-animals-intervals-v15_255_128", "sequence", "dna"),
        _spec("hf_rnagps", "introvoyz041/rnagps", "classification", "rna"),
    ]
})


HF_BENCHMARK_SPECS.update({
    s.key: s for s in [
        # Curated addable set: text/structured benchmarks that fit the
        # generic HF loader.
        _spec("hf_medconceptsqa", "ofir408/MedConceptsQA", "mcq", "medical",
              config="all", answer_fields=("answer_id", "answer"),
              choice_fields=("options", "option1", "option")),
        _spec("hf_meds_bench", "Henrychur/MedS-Bench", "qa", "medical"),
        _spec("hf_medexqa", "bluesky333/MedExQA", "mcq", "medical",
              configs=("biomedical_engineer", "clinical_laboratory_scientist",
                       "clinical_psychologist", "occupational_therapist",
                       "speech_pathologist")),
        _spec("hf_medcase_reasoning", "zou-lab/MedCaseReasoning", "qa", "clinical",
              split="test", question_fields=("case_prompt",), answer_fields=("final_diagnosis",)),
        _spec("hf_cmexam", "fzkuji/CMExam", "mcq", "medical",
              split="test", question_fields=("Question",),
              choice_fields=("Options",), answer_fields=("Answer",)),
        _spec("hf_cmb", "FreedomIntelligence/CMB", "mcq", "medical",
              config="CMB-Exam", split="test",
              choice_fields=("option", "options"), answer_fields=("answer",)),
        _spec("hf_headqa", "openlifescienceai/headqa", "mcq", "medical",
              split="test", question_fields=("qtext", "question"),
              choice_fields=("Options",), answer_fields=("Correct Option",)),
        _spec("hf_medmcqa_explanations", "openlifescienceai/medmcqa", "mcq", "medical",
              split="validation", answer_index_base=1),
        _spec("hf_liveqa_med", "hyesunyun/liveqa_medical_trec2017", "qa", "medical",
              split="test", question_fields=("NIST_PARAPHRASE", "NLM_SUMMARY"),
              answer_fields=("REFERENCE_ANSWERS",)),
        _spec("hf_medication_qa", "truehealth/medicationqa", "qa", "medical",
              split="train", question_fields=("Question",), answer_fields=("Answer",),
              context_fields=("Focus (Drug)", "Question Type", "Section Title")),
        _spec("hf_blurb", "EMBO/BLURB", "classification", "biomedical",
              split="test",
              blurb_ner_configs=("BC5CDR-chem-IOB", "BC5CDR-disease-IOB",
                                  "BC2GM-IOB", "NCBI-disease-IOB", "JNLPBA"),
              raw_base_url="https://github.com/cambridgeltl/MTL-Bioinformatics-2016/raw/master/data"),
        _spec("hf_bc5cdr", "EMBO/BLURB", "classification", "biomedical",
              split="test",
              blurb_ner_configs=("BC5CDR-chem-IOB", "BC5CDR-disease-IOB"),
              raw_base_url="https://github.com/cambridgeltl/MTL-Bioinformatics-2016/raw/master/data"),
        _spec("hf_ncbi_disease", "EMBO/BLURB", "classification", "biomedical",
              split="test", blurb_ner_configs=("NCBI-disease-IOB",),
              raw_base_url="https://github.com/cambridgeltl/MTL-Bioinformatics-2016/raw/master/data"),
        _spec("hf_ddi_corpus_2013", "OpenMed/DDI-Corpus-Processed", "classification", "biomedical",
              split="test", text_fields=("sentence",), answer_fields=("relation",),
              context_fields=("drug1", "drug2")),
        _spec("hf_mednli", "araag2/MedNLI", "classification", "clinical",
              config="processed", split="test", text_fields=("prompt",), answer_fields=("Label",)),
        _spec("hf_hallmarks_of_cancer", "bigbio/hallmarks_of_cancer", "classification", "biomedical",
              parquet_config="hallmarks_of_cancer_bigbio_text", parquet_split="train",
              text_fields=("text",), answer_fields=("labels",)),
        _spec("hf_litcovid", "ncats/litcovid", "classification", "biomedical",
              split="validation", text_fields=("text", "abstract"), answer_fields=("label",),
              loader_repo="KushT/LitCovid_BioCreative"),
        _spec("hf_ebm_nlp", "bigbio/ebm_pico", "qa", "biomedical",
              config="processed", split="test", question_fields=("prompt",),
              answer_fields=("Label", "completion"), loader_repo="araag2/EBM_NLP",
              scorer="pio_span_f1", output_format="structured_spans",
              benchmark_task_type="ner"),
        _spec("hf_biosses", "mteb/biosses-sts", "regression", "biomedical",
              split="test", answer_fields=("score",)),
        _spec("hf_gad", "bigbio/gad", "classification", "biomedical",
              parquet_config="gad_blurb_bigbio_text", parquet_split="test",
              text_fields=("text",), answer_fields=("labels",)),
        _spec("hf_biored", "bigbio/biored", "classification", "biomedical"),
        _spec("hf_ade_corpus_v2", "ade-benchmark-corpus/ade_corpus_v2", "classification", "biomedical",
              config="Ade_corpus_v2_classification", split="train",
              text_fields=("text",), answer_fields=("label",)),
        _spec("hf_jnlpba", "EMBO/BLURB", "classification", "biomedical",
              split="test", blurb_ner_configs=("JNLPBA",),
              raw_base_url="https://github.com/cambridgeltl/MTL-Bioinformatics-2016/raw/master/data"),
        _spec("hf_bc2gm", "spyysalo/bc2gm_corpus", "classification", "biomedical",
              config="bc2gm_corpus", split="test", text_fields=("tokens",), label_fields=("ner_tags",)),
        _spec("hf_ppi_benchmark", "bigbio/bioinfer", "classification", "protein",
              split="test",
              raw_urls={
                  "train": "https://github.com/metalrt/ppi-dataset/raw/master/csv_output/BioInfer-train.xml",
                  "test": "https://github.com/metalrt/ppi-dataset/raw/master/csv_output/BioInfer-test.xml",
              }),
        _spec("hf_chembench", "jablonkagroup/ChemBench", "qa", "chemistry",
              configs=("analytical_chemistry", "chemical_preference", "general_chemistry",
                       "inorganic_chemistry", "materials_science", "organic_chemistry",
                       "physical_chemistry", "technical_chemistry", "toxicity_and_safety")),
        _spec("hf_mollangbench", "ChemFM/MolLangBench", "qa", "chemistry",
              configs=("edit", "generation", "recognition")),
        _spec("hf_chemllmbench", "blc-org/chemllmbench", "qa", "chemistry",
              question_fields=("query",), answer_fields=("gt",),
              files=("chemllmbench/molecule_captioning/molecule_captioning.json",
                     "chemllmbench/molecule_design/molecule_design.json",
                     "chemllmbench/reaction_prediction/reaction_prediction.json",
                     "chemllmbench/reagent_selection/ligand.json",
                     "chemllmbench/reagent_selection/reactant.json",
                     "chemllmbench/reagent_selection/solvent.json",
                     "chemllmbench/retro/retro.json")),
        _spec("hf_fgbench", "xuan-liu/FGBench", "classification", "chemistry",
              split="test", text_fields=("question",), answer_fields=("answer",)),
        _spec("hf_uspto_reaction_prediction", "bing-yan/USPTO", "qa", "chemistry",
              split="test", question_fields=("source",), answer_fields=("target",),
              scorer="smiles_topk_canonical_match", benchmark_task_type="smiles_generation"),
        _spec("hf_mol_instructions_pubchemqa", "zjunlp/Mol-Instructions", "qa", "chemistry",
              archive="data/Biomolecular_Text_Instructions.zip",
              files=("Biomolecular_Text_Instructions/open_question.json",
                     "Biomolecular_Text_Instructions/multi_choice_question.json",
                     "Biomolecular_Text_Instructions/true_or_false_question.json")),
        _spec("hf_smiles_caption_mol2text", "zjunlp/Mol-Instructions", "qa", "chemistry",
              archive="data/Molecule-oriented_Instructions.zip",
              files=("Molecule-oriented_Instructions/molecular_description_generation.json",
                     "Molecule-oriented_Instructions/description_guided_molecule_design.json")),
        _spec("hf_clinical_trials_eligibility_nlp", "bigbio/n2c2_2018_track1", "classification", "clinical"),
        _spec("hf_ms2", "allenai/mslr2022", "summarization", "biomedical",
              split="validation", text_fields=("abstract",), answer_fields=("target",),
              loader_repo="allenai/ms2_sparse_max"),
        _spec("hf_meqsum", "albertvillanova/meqsum", "summarization", "medical",
              split="train", text_fields=("CHQ",), answer_fields=("Summary",)),
        _spec("hf_anatem", "bigbio/anat_em", "classification", "biomedical",
              split="test", text_fields=("tokens",), label_fields=("ner_tags",),
              loader_repo="disi-unibo-nlp/AnatEM"),
        _spec("hf_evidence_inference", "hpi-dhc/evidence-inference-simple", "classification", "biomedical",
              split="test", text_fields=("text",), answer_fields=("label",),
              scorer="delta_ei", benchmark_task_type="classification"),
        _spec("hf_nlmchem", "jablonkagroup/nlmchem", "qa", "biomedical",
              config="instruction_0", split="test", question_fields=("input", "text"),
              answer_fields=("output",)),
        _spec("hf_pgr", "lasigeBioTM/PGR", "classification", "biomedical",
              split="test",
              scorer="custom_re_with_silver_warning", benchmark_task_type="relation_extraction",
              raw_urls={
                  "train": "https://raw.githubusercontent.com/lasigeBioTM/PGR/master/corpora/10_12_2018_corpus/train.tsv",
                  "test": "https://raw.githubusercontent.com/lasigeBioTM/PGR/master/corpora/10_12_2018_corpus/test.tsv",
              }),
        _spec("hf_craft", "bigbio/craft", "classification", "biomedical"),
        _spec("hf_cord19_qa", "allenai/cord19", "qa", "biomedical"),
        _spec("hf_longhealth", "tonychenxyz/longhealth", "mcq", "medical",
              config="plain", split="test"),
        _spec("hf_pubmed_abstract_classification", "uiyunkim-hub/pubmed-abstract", "classification", "biomedical",
              streaming=True),
        _spec("hf_biocreative_viii_biored", "bigbio/biored", "classification", "biomedical"),
        _spec("hf_meddialogqa", "UCSD26/medical_dialog", "qa", "medical"),
        _spec("hf_healthqa", "nlplabtdtu/health_qa", "qa", "medical"),
        _spec("hf_raredis", "guan-wang/ReDis-QA", "mcq", "medical",
              split="test", question_fields=("question",),
              answer_fields=("cop",), answer_index_base=1),
        _spec("hf_clicr", "bigbio/clicr", "qa", "clinical"),
        _spec("hf_medqa_taiwan", "xuxuxuxuxu/MedQA_Taiwan_test", "mcq", "medical"),
        _spec("hf_geneturing", "vladimire/geneturing", "qa", "genomics",
              config="all", split="test", question_fields=("question",),
              answer_fields=("goldstandard",)),
        _spec("hf_biomedbench", "biomedbench/BioMedBench", "qa", "biomedical"),
        _spec("hf_medpalm_eval_set", "katielink/healthsearchqa", "qa", "medical",
              config="140_question_subset", split="train", question_fields=("question",)),
        _spec("hf_careqa", "HPAI-BSC/CareQA", "mcq", "medical",
              config="CareQA_en", split="test", answer_fields=("cop",),
              answer_index_base=1),
        _spec("hf_discoverybench_biomedical", "allenai/discoverybench", "qa", "biomedical",
              split="train"),
        _spec("hf_medpub_qa", "qiaojin/PubMedQA", "mcq", "biomedical",
              config="pqa_labeled", split="train",
              question_fields=("question",), context_fields=("context",),
              choice_fields=("choices",), answer_fields=("final_decision",)),
    ]
})


_EXCEL_ADDABLE_BENCHMARK_KEYS = frozenset({
    "hf_ade_corpus_v2",
    "hf_anatem",
    "hf_bc2gm",
    "hf_bc5cdr",
    "hf_biocreative_viii_biored",
    "hf_biomedbench",
    "hf_biored",
    "hf_biosses",
    "hf_blurb",
    "hf_blue_benchmark",
    "hf_careqa",
    "hf_chembench",
    "hf_chemllmbench",
    "hf_chinese_medbench",
    "hf_clinical_trials_eligibility_nlp",
    "hf_clicr",
    "hf_cmexam",
    "hf_cmb",
    "hf_cord19_qa",
    "hf_craft",
    "hf_ddi_corpus_2013",
    "hf_discoverybench_biomedical",
    "hf_ebm_nlp",
    "hf_evidence_inference",
    "hf_fgbench",
    "hf_gad",
    "hf_geneturing",
    "hf_hallmarks_of_cancer",
    "hf_headqa",
    "hf_healthqa",
    "hf_jnlpba",
    "hf_litcovid",
    "hf_liveqa_med",
    "hf_longhealth",
    "hf_meddialogqa",
    "hf_medcase_reasoning",
    "hf_medconceptsqa",
    "hf_medexqa",
    "hf_medication_qa",
    "hf_medmcqa_explanations",
    "hf_mednli",
    "hf_mednli_augmented",
    "hf_medpalm_eval_set",
    "hf_medpub_qa",
    "hf_medqa_taiwan",
    "hf_meds_bench",
    "hf_meqsum",
    "hf_mol_instructions_pubchemqa",
    "hf_mollangbench",
    "hf_ms2",
    "hf_ncbi_disease",
    "hf_nlmchem",
    "hf_openddi",
    "hf_pgr",
    "hf_ppi_benchmark",
    "hf_pubmed_abstract_classification",
    "hf_raredis",
    "hf_smiles_caption_mol2text",
    "hf_usmle_step_series",
    "hf_uspto_reaction_prediction",
})


HF_VERIFIED_BENCHMARK_KEYS = frozenset({
    "hf_adaptllm_chemprot",
    "hf_adaptllm_medicine_tasks",
    "hf_adaptllm_mqp",
    "hf_adaptllm_rct",
    "hf_bacbench_antibiotic_resistance_dna",
    "hf_bacbench_phenotypic_traits_dna",
    "hf_bigbio_med_qa",
    "hf_bigbio_pubmed_qa",
    "hf_ccdv_pubmed_summarization",
    "hf_chemistry_qa",
    "hf_fluorescence_prediction",
    "hf_gaianet_chemistry",
    "hf_genbio_proteingym_dms",
    "hf_genomics_long_range",
    "hf_icml2022_proteingym",
    "hf_katielink_moleculenet_bace",
    "hf_katielink_moleculenet_bbbp",
    "hf_katielink_moleculenet_clintox",
    "hf_katielink_moleculenet_esol",
    "hf_katielink_moleculenet_freesolv",
    "hf_katielink_moleculenet_hiv",
    "hf_katielink_moleculenet_lipo",
    "hf_katielink_moleculenet_sider",
    "hf_katielink_moleculenet_tox21",
    "hf_lavita_medmcqa",
    "hf_lavita_usmle_step1",
    "hf_lavita_usmle_step2",
    "hf_lavita_usmle_step3",
    "hf_lpm24_eval_caption",
    "hf_lpm24_eval_molgen",
    "hf_medical_question_pairs",
    "hf_medquad",
    "hf_moleculeace",
    "hf_moleculeace_chembl1871_ki",
    "hf_moleculeace_chembl204_ki",
    "hf_moleculeace_chembl214_ki",
    "hf_moleculeace_chembl228_ki",
    "hf_moleculeace_chembl237_ec50",
    "hf_moleculenet_bace",
    "hf_moleculenet_bbbp",
    "hf_moleculenet_clintox",
    "hf_moleculenet_esol",
    "hf_moleculenet_freesolv",
    "hf_moleculenet_hiv",
    "hf_moleculenet_lipophilicity",
    "hf_moleculenet_pcba",
    "hf_moleculenet_sider",
    "hf_moleculenet_toxcast",
    "hf_mteb_medical_qa",
    "hf_mteb_medical_retrieval",
    "hf_mts_dialogue_clinical_note",
    "hf_protein_binding_sequences",
    "hf_protein_deeploc",
    "hf_protein_fluorescence",
    "hf_protein_secondary_structure",
    "hf_protein_solubility",
    "hf_protein_stability",
    "hf_proteingym_v01",
    "hf_proteingym_v1",
    "hf_proteinlmbench",
    "hf_proteinlmbench_enzyme_cot",
    "hf_proteinlmbench_uniprot_disease",
    "hf_proteinlmbench_uniprot_function",
    "hf_proteinlmbench_uniprot_induction",
    "hf_proteinlmbench_uniprot_ptm",
    "hf_proteinlmbench_uniprot_subunit",
    "hf_proteinlmbench_uniprot_tissue",
    "hf_pubmed_200k_rct",
    "hf_pubmed_20k_rct",
    "hf_pubmed_rct20k",
    "hf_rna_downstream_tasks",
    "hf_rna_expression_hek",
    "hf_rna_expression_muscle",
    "hf_rna_expression_pc3",
    "hf_rna_mean_ribosome_load",
    "hf_rna_modification_site",
    "hf_rna_ncrna_family_bnoise0",
    "hf_rna_splice_site_acceptor",
    "hf_rna_splice_site_donor",
    "hf_traitgym_mendelian_dna",
}) | _EXCEL_ADDABLE_BENCHMARK_KEYS


TRAINING_DATA_SPECS: dict[str, HFDatasetSpec] = {
    s.key: s for s in [
        _spec("hf_asclepius_clinical_notes", "starmpcc/Asclepius-Synthetic-Clinical-Notes",
              "summarization", "clinical", is_training_only=True,
              warning="Instruction-tuning data for building clinical LLMs; not a benchmark."),
        _spec("hf_augmented_clinical_notes", "AGBonnet/augmented-clinical-notes",
              "summarization", "clinical", is_training_only=True,
              warning="Training data used for MediNote models; not a benchmark."),
        _spec("hf_icliniq_10k", "lavita/ChatDoctor-HealthCareMagic-100k",
              "qa", "medical", is_training_only=True,
              warning="ChatDoctor SFT training corpus; not a benchmark."),
        _spec("hf_healthcare_data", "Nicolybgs/healthcare_data",
              "classification", "healthcare", is_training_only=True,
              warning="Small personal dataset without standard evaluation protocol."),
    ]
}


HF_DEPRECATED_ALIASES: dict[str, str] = {
    "hf_chinese_medbench": "hf_cmb",
    "hf_mednli_augmented": "hf_mednli",
    "hf_pubmed_20k_rct": "hf_pubmed_200k_rct",
    "hf_pubmed_rct20k": "hf_pubmed_200k_rct",
    "hf_blue_benchmark": "hf_blurb",
    "hf_openddi": "hf_ddi_corpus_2013",
    "hf_lavita_medmcqa": "medmcqa",
    "hf_lavita_usmle_step1": "medqa",
    "hf_lavita_usmle_step2": "medqa",
    "hf_lavita_usmle_step3": "medqa",
    "hf_usmle_step_series": "medqa",
}

HF_REMOVED_NONBENCHMARK_KEYS = frozenset(TRAINING_DATA_SPECS) | {
    "hf_medsts",
    "hf_clinical_sts",
}

HF_VERIFIED_BENCHMARK_KEYS = frozenset(
    key for key in HF_VERIFIED_BENCHMARK_KEYS
    if key in HF_BENCHMARK_SPECS
    and key not in HF_DEPRECATED_ALIASES
    and key not in HF_REMOVED_NONBENCHMARK_KEYS
)


def hf_benchmark_cli_entries() -> dict[str, dict[str, Any]]:
    entries = {
        key: {
            "loader": "load_hf_benchmark_tasks",
            "benchmark_key": key,
            "kwargs": {"dataset_key": key},
        }
        for key in HF_VERIFIED_BENCHMARK_KEYS
    }
    for alias, canonical in HF_DEPRECATED_ALIASES.items():
        if canonical in HF_VERIFIED_BENCHMARK_KEYS:
            entries[alias] = {
                "loader": "load_hf_benchmark_tasks",
                "benchmark_key": canonical,
                "kwargs": {"dataset_key": canonical},
                "deprecated_alias_for": canonical,
            }
    return entries


BENCHMARK_TASK_TYPES = {
    "mcq": "mcq_qa",
    "qa": "open_qa",
    "summarization": "summarization",
    "retrieval": "open_qa",
    "text": "open_qa",
    "sequence": "open_qa",
    "classification": "classification",
    "pair_classification": "classification",
    "molecule_property": "molecule_property",
    "protein_fitness": "protein_fitness",
    "regression": "sts",
}


_TASK_TO_ANSWER_TYPE = {
    "mcq": "multipleChoice",
    "qa": "openText",
    "retrieval": "openText",
    "summarization": "openText",
    "text": "openText",
    "sequence": "openText",
    "classification": "exactMatch",
    "pair_classification": "exactMatch",
    "molecule_property": "exactMatch",
    "protein_fitness": "exactNumeric",
    "regression": "exactNumeric",
}


def hf_spec_metadata(spec: HFDatasetSpec) -> dict[str, Any]:
    """Return machine-readable release metadata for a HF benchmark spec.

    Most HF entries use conservative defaults: first run requires network,
    later runs can use the HuggingFace datasets cache, and all current HF
    entries are text/structured rather than multimodal.
    """
    task_type = spec.task_type
    answer_type = _TASK_TO_ANSWER_TYPE.get(task_type, "openText")
    return {
        "key": spec.key,
        "source": spec.repo,
        "source_url": f"https://huggingface.co/datasets/{spec.repo}",
        "config": spec.config,
        "split": spec.split or "default",
        "count": spec.extra.get("count", "unknown"),
        "domain": spec.domain,
        "task_type": task_type,
        "benchmark_task_type": spec.extra.get("benchmark_task_type", BENCHMARK_TASK_TYPES.get(task_type, task_type)),
        "answer_type": answer_type,
        "input_type": spec.extra.get("input_type", "text"),
        "scorer": spec.extra.get("scorer", answer_type),
        "derived_from": spec.extra.get("derived_from"),
        "recommend_canonical": spec.extra.get("recommend_canonical"),
        "warning": spec.extra.get("warning"),
        "deprecated_alias_for": spec.extra.get("deprecated_alias_for"),
        "gated": bool(spec.extra.get("gated", False)),
        "needs_network": bool(spec.extra.get("needs_network", True)),
        "offline_cache": bool(spec.extra.get("offline_cache", True)),
        "multimodal": bool(spec.extra.get("multimodal", False)),
        "license": spec.extra.get("license", "unknown"),
        "revision": spec.extra.get("revision", "main"),
        "streaming": bool(spec.extra.get("streaming", False)),
        "status": "verified" if spec.key in HF_VERIFIED_BENCHMARK_KEYS else "registered",
    }


def hf_verified_metadata() -> dict[str, dict[str, Any]]:
    """Metadata table for the public HF benchmark registrations."""
    return {
        key: hf_spec_metadata(HF_BENCHMARK_SPECS[key])
        for key in sorted(HF_VERIFIED_BENCHMARK_KEYS)
    }


def validate_hf_metadata() -> list[str]:
    """Return release-gate metadata problems for verified HF benchmarks."""
    required = {
        "key", "source", "source_url", "count", "domain", "task_type",
        "answer_type", "input_type", "scorer", "gated", "needs_network",
        "offline_cache", "multimodal", "license", "revision", "status",
    }
    problems: list[str] = []
    for key in sorted(HF_VERIFIED_BENCHMARK_KEYS):
        spec = HF_BENCHMARK_SPECS.get(key)
        if spec is None:
            problems.append(f"{key}: missing spec")
            continue
        meta = hf_spec_metadata(spec)
        missing = sorted(name for name in required if name not in meta)
        if missing:
            problems.append(f"{key}: missing metadata fields {missing}")
        if meta["status"] != "verified":
            problems.append(f"{key}: status is not verified")
        if not str(meta["source_url"]).startswith("https://huggingface.co/datasets/"):
            problems.append(f"{key}: invalid HuggingFace source_url")
        if not meta["task_type"] or not meta["answer_type"]:
            problems.append(f"{key}: missing task/scorer mapping")
    return problems


__all__ = [
    "HFDatasetSpec",
    "HF_BENCHMARK_SPECS",
    "HF_DEPRECATED_ALIASES",
    "HF_REMOVED_NONBENCHMARK_KEYS",
    "TRAINING_DATA_SPECS",
    "HF_VERIFIED_BENCHMARK_KEYS",
    "hf_spec_metadata",
    "hf_verified_metadata",
    "validate_hf_metadata",
    "hf_benchmark_cli_entries",
]
