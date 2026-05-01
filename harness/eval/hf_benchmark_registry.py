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
        _spec("hf_medqa_usmle_4_options", "GBaker/MedQA-USMLE-4-options", "mcq", "medical"),
        _spec("hf_medqa_usmle_4_options_hf", "GBaker/MedQA-USMLE-4-options-hf", "mcq", "medical"),
        _spec("hf_medical_flashcards", "medalpaca/medical_meadow_medical_flashcards", "qa", "medical"),
        _spec("hf_medical_meadow_medqa", "medalpaca/medical_meadow_medqa", "qa", "medical"),
        _spec("hf_medical_meadow_wikidoc", "medalpaca/medical_meadow_wikidoc", "qa", "medical"),
        _spec("hf_medquad", "keivalya/MedQuad-MedicalQnADataset", "qa", "medical"),
        _spec("hf_lavita_medical_qa_datasets", "lavita/medical-qa-datasets", "qa", "medical", config="all-processed"),
        _spec("hf_malikeh_medical_qa", "Malikeh1375/medical-question-answering-datasets", "qa", "medical", config="all-processed"),
        _spec("hf_medical_o1_reasoning_sft", "FreedomIntelligence/medical-o1-reasoning-SFT", "qa", "medical", config="en"),
        _spec("hf_medical_o1_verifiable", "FreedomIntelligence/medical-o1-verifiable-problem", "qa", "medical"),
        _spec("hf_medical_r1_distill", "FreedomIntelligence/Medical-R1-Distill-Data", "qa", "medical"),
        _spec("hf_openmed_reasoning_sft", "OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B", "qa", "medical"),
        _spec("hf_adaptllm_medicine_tasks", "AdaptLLM/medicine-tasks", "mcq", "medical", config="USMLE"),
        _spec("hf_chatdoctor_healthcaremagic", "lavita/ChatDoctor-HealthCareMagic-100k", "qa", "medical"),
        _spec("hf_medical_dialog", "UCSD26/medical_dialog", "qa", "medical"),
        _spec("hf_ai_medical_chatbot", "ruslanmv/ai-medical-chatbot", "qa", "medical"),
        _spec("hf_augmented_clinical_notes", "AGBonnet/augmented-clinical-notes", "summarization", "clinical"),
        _spec("hf_asclepius_clinical_notes", "starmpcc/Asclepius-Synthetic-Clinical-Notes", "summarization", "clinical"),
        _spec("hf_mts_dialogue_clinical_note", "har1/MTS_Dialogue-Clinical_Note", "summarization", "clinical"),
        _spec("hf_medical_question_pairs", "curaihealth/medical_questions_pairs", "pair_classification", "medical"),
        _spec("hf_medical_chronology", "Superinsight/medical-chronology-benchmark", "qa", "medical"),
        _spec("hf_arabic_medical_consultations", "Ahmed-Selem/Shifaa_Arabic_Medical_Consultations", "qa", "medical"),
        _spec("hf_chinese_medical_dialogue", "BillGPT/Chinese-medical-dialogue-data", "qa", "medical"),
        _spec("hf_huatuo_medical_qa", "shibing624/huatuo_medical_qa_sharegpt", "qa", "medical"),
        _spec("hf_tcm_sft", "SylvanL/Traditional-Chinese-Medicine-Dataset-SFT", "qa", "medical"),
        _spec("hf_tcm_pretrain", "SylvanL/Traditional-Chinese-Medicine-Dataset-Pretrain", "text", "medical"),
        _spec("hf_industry_medicine_health_tcm", "BAAI/IndustryCorpus2_medicine_health_psychology_traditional_chinese_medicine", "text", "medical"),
        _spec("hf_openlifescience_medqa", "openlifescienceai/medqa", "mcq", "medical"),
        _spec("hf_openlifescience_pubmedqa", "openlifescienceai/pubmedqa", "mcq", "medical"),
        _spec("hf_bigbio_med_qa", "bigbio/med_qa", "mcq", "medical"),
        _spec("hf_bigbio_pubmed_qa", "bigbio/pubmed_qa", "mcq", "medical"),
        _spec("hf_medqa_corpus_en", "cogbuji/medqa_corpus_en", "text", "medical"),
        _spec("hf_xuxu_medqa_us_test", "xuxuxuxuxu/MedQA_US_test", "mcq", "medical"),
        _spec("hf_xuxu_medqa_mainland_test", "xuxuxuxuxu/MedQA_Mainland_test", "mcq", "medical"),
        _spec("hf_xuxu_medqa_taiwan_test", "xuxuxuxuxu/MedQA_Taiwan_test", "mcq", "medical"),
    ]
}


HF_BENCHMARK_SPECS.update({
    s.key: s for s in [
        # PubMed / biomedical NLP / retrieval-oriented text
        _spec("hf_medrag_pubmed", "MedRAG/pubmed", "retrieval", "biomedical", streaming=True),
        _spec("hf_ncbi_pubmed", "ncbi/pubmed", "text", "biomedical", streaming=True),
        _spec("hf_ccdv_pubmed_summarization", "ccdv/pubmed-summarization", "summarization", "biomedical"),
        _spec("hf_pubmed_rct20k", "armanc/pubmed-rct20k", "classification", "biomedical"),
        _spec("hf_pubmed_20k_rct", "pietrolesci/pubmed-20k-rct", "classification", "biomedical"),
        _spec("hf_pubmed_200k_rct", "pietrolesci/pubmed-200k-rct", "classification", "biomedical"),
        _spec("hf_abdelmo_pubmed_dataset", "abdelmo/pubmed-dataset", "text", "biomedical", streaming=True),
        _spec("hf_abdelmo_pubmed_ds", "abdelmo/pubmed-ds", "text", "biomedical", streaming=True),
        _spec("hf_common_pile_pubmed", "common-pile/pubmed", "text", "biomedical", streaming=True),
        _spec("hf_common_pile_pubmed_filtered", "common-pile/pubmed_filtered", "text", "biomedical", streaming=True),
        _spec("hf_pubmed_abstract", "uiyunkim-hub/pubmed-abstract", "text", "biomedical"),
        _spec("hf_multilingual_medical_corpus", "HiTZ/Multilingual-Medical-Corpus", "text", "medical"),
        _spec("hf_spaccc_tokenizer", "Biomedical-TeMU/SPACCC_Tokenizer", "classification", "biomedical"),
        _spec("hf_mteb_medical_qa", "mteb/medical_qa", "retrieval", "medical"),
        _spec("hf_mteb_medical_retrieval", "mteb/MedicalRetrieval", "retrieval", "medical"),
        _spec("hf_embedding_chatdoctor", "embedding-benchmark/ChatDoctor_HealthCareMagic", "retrieval", "medical"),
        _spec("hf_clinical_trials_data", "Dattito/clinical-trials-data", "text", "clinical"),
        _spec("hf_healthcare_data", "Nicolybgs/healthcare_data", "classification", "healthcare", answer_fields=("Stay (in days)",)),
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
        _spec("hf_lpm24_train", "language-plus-molecules/LPM-24_train", "qa", "chemistry"),
        _spec("hf_lpm24_eval_molgen", "language-plus-molecules/LPM-24_eval-molgen", "qa", "chemistry"),
        _spec("hf_lpm24_eval_caption", "language-plus-molecules/LPM-24_eval-caption", "qa", "chemistry"),
        _spec("hf_moleculestm", "chao1224/MoleculeSTM", "text", "chemistry"),
        _spec("hf_smiles_molecules_chembl", "antoinebcx/smiles-molecules-chembl", "text", "chemistry"),
        _spec("hf_moleculenet_benchmark", "katielink/moleculenet-benchmark", "molecule_property", "chemistry", config="bace"),
        _spec("hf_moleculeace", "karina-zadorozhny/moleculeace", "molecule_property", "chemistry", config="CHEMBL1862_Ki"),
        _spec("hf_dolma_chemistry_only", "BASF-AI/dolma-chemistry-only", "text", "chemistry", streaming=True),
    ]
})


HF_BENCHMARK_SPECS.update({
    s.key: s for s in [
        # Stable config-level medical datasets from aggregated HF repos.
        _spec("hf_lavita_medqa_4options", "lavita/medical-qa-datasets", "qa", "medical", config="med-qa-en-4options-source"),
        _spec("hf_lavita_medqa_5options", "lavita/medical-qa-datasets", "qa", "medical", config="med-qa-en-5options-source"),
        _spec("hf_lavita_medmcqa", "lavita/medical-qa-datasets", "mcq", "medical", config="medmcqa", split="validation"),
        _spec("hf_lavita_pubmedqa", "lavita/medical-qa-datasets", "qa", "medical", config="pubmed-qa"),
        _spec("hf_lavita_mmmlu_anatomy", "lavita/medical-qa-datasets", "qa", "medical", config="mmmlu-anatomy"),
        _spec("hf_lavita_mmmlu_clinical_knowledge", "lavita/medical-qa-datasets", "qa", "medical", config="mmmlu-clinical-knowledge"),
        _spec("hf_lavita_mmmlu_college_biology", "lavita/medical-qa-datasets", "qa", "medical", config="mmmlu-college-biology"),
        _spec("hf_lavita_mmmlu_college_medicine", "lavita/medical-qa-datasets", "qa", "medical", config="mmmlu-college-medicine"),
        _spec("hf_lavita_mmmlu_medical_genetics", "lavita/medical-qa-datasets", "qa", "medical", config="mmmlu-medical-genetics"),
        _spec("hf_lavita_mmmlu_professional_medicine", "lavita/medical-qa-datasets", "qa", "medical", config="mmmlu-professional-medicine"),
        _spec("hf_lavita_usmle_step1", "lavita/medical-qa-datasets", "qa", "medical", config="usmle-self-assessment-step1"),
        _spec("hf_lavita_usmle_step2", "lavita/medical-qa-datasets", "qa", "medical", config="usmle-self-assessment-step2"),
        _spec("hf_lavita_usmle_step3", "lavita/medical-qa-datasets", "qa", "medical", config="usmle-self-assessment-step3"),
        _spec("hf_malikeh_chatdoctor_healthcaremagic", "Malikeh1375/medical-question-answering-datasets", "qa", "medical", config="chatdoctor_healthcaremagic"),
        _spec("hf_malikeh_chatdoctor_icliniq", "Malikeh1375/medical-question-answering-datasets", "qa", "medical", config="chatdoctor_icliniq"),
        _spec("hf_malikeh_medical_flashcards", "Malikeh1375/medical-question-answering-datasets", "qa", "medical", config="medical_meadow_medical_flashcards"),
        _spec("hf_malikeh_medqa", "Malikeh1375/medical-question-answering-datasets", "qa", "medical", config="medical_meadow_medqa"),
        _spec("hf_malikeh_mmmlu", "Malikeh1375/medical-question-answering-datasets", "qa", "medical", config="medical_meadow_mmmlu"),
        _spec("hf_malikeh_pubmed_causal", "Malikeh1375/medical-question-answering-datasets", "qa", "medical", config="medical_meadow_pubmed_causal"),
        _spec("hf_malikeh_wikidoc", "Malikeh1375/medical-question-answering-datasets", "qa", "medical", config="medical_meadow_wikidoc"),
        _spec("hf_malikeh_wikidoc_patient_information", "Malikeh1375/medical-question-answering-datasets", "qa", "medical", config="medical_meadow_wikidoc_patient_information"),
        _spec("hf_adaptllm_chemprot", "AdaptLLM/medicine-tasks", "mcq", "biomedical", config="ChemProt"),
        _spec("hf_adaptllm_mqp", "AdaptLLM/medicine-tasks", "mcq", "medical", config="MQP"),
        _spec("hf_adaptllm_pubmedqa", "AdaptLLM/medicine-tasks", "mcq", "medical", config="PubMedQA"),
        _spec("hf_adaptllm_rct", "AdaptLLM/medicine-tasks", "mcq", "biomedical", config="RCT"),
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
        _spec("hf_katielink_moleculenet_lipo", "katielink/moleculenet-benchmark", "molecule_property", "chemistry", config="lipophilicity"),
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
        _spec("hf_proteinlmbench_uniprot_function", "tsynbio/ProteinLMBench", "qa", "protein", config="UniProt_Function"),
        _spec("hf_proteinlmbench_uniprot_induction", "tsynbio/ProteinLMBench", "qa", "protein", config="UniProt_Induction"),
        _spec("hf_proteinlmbench_uniprot_disease", "tsynbio/ProteinLMBench", "qa", "protein", config="UniProt_Involvement in disease"),
        _spec("hf_proteinlmbench_uniprot_ptm", "tsynbio/ProteinLMBench", "qa", "protein", config="UniProt_Post-translational modification"),
        _spec("hf_proteinlmbench_uniprot_subunit", "tsynbio/ProteinLMBench", "qa", "protein", config="UniProt_Subunit structure"),
        _spec("hf_proteinlmbench_uniprot_tissue", "tsynbio/ProteinLMBench", "qa", "protein", config="UniProt_Tissue specificity"),
        _spec("hf_proteinlmbench_enzyme_cot", "tsynbio/ProteinLMBench", "qa", "protein", config="Enzyme_CoT"),
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
        _spec("hf_metanova_proteins", "Metanova/Proteins", "sequence", "protein"),
        _spec("hf_proteingym_v1", "OATML-Markslab/ProteinGym_v1", "protein_fitness", "protein"),
        _spec("hf_proteingym_v01", "OATML-Markslab/ProteinGym_v0.1", "protein_fitness", "protein"),
        _spec("hf_icml2022_proteingym", "ICML2022/ProteinGym", "protein_fitness", "protein"),
        _spec("hf_genbio_proteingym_dms", "genbio-ai/ProteinGYM-DMS", "protein_fitness", "protein"),
        _spec("hf_proteinlmbench", "tsynbio/ProteinLMBench", "qa", "protein"),
        _spec("hf_protein_solubility", "proteinea/solubility", "classification", "protein"),
        _spec("hf_protein_fluorescence", "proteinea/fluorescence", "regression", "protein"),
        _spec("hf_protein_deeploc", "proteinea/deeploc", "classification", "protein"),
        _spec("hf_fluorescence_prediction", "proteinglm/fluorescence_prediction", "classification", "protein"),
        _spec("hf_protein_secondary_structure", "lamm-mit/protein_secondary_structure_from_PDB", "classification", "protein"),
        _spec("hf_pdb_protein_ligand", "jglaser/pdb_protein_ligand_complexes", "molecule_property", "protein"),
        _spec("hf_protein_binding_sequences", "ronig/protein_binding_sequences", "classification", "protein"),
        _spec("hf_protein_stability", "SaProtHub/Dataset-Meta-scale-protein-stability", "regression", "protein"),
        _spec("hf_protein_conformational_states", "PDBEurope/protein_chain_conformational_states", "classification", "protein"),
        _spec("hf_protein_docs", "timodonnell/protein-docs", "text", "protein"),
        # Gene / genomics / DNA / RNA
        _spec("hf_genecorpus_104m", "theodoris-lab/Genecorpus-104M", "sequence", "genomics", streaming=True),
        _spec("hf_genecorpus_30m", "ctheodoris/Genecorpus-30M", "sequence", "genomics", streaming=True),
        _spec("hf_geneexp", "xingyusu/GeneExp", "regression", "genomics"),
        _spec("hf_dna_gen", "xingyusu/DNA_Gen", "sequence", "dna"),
        _spec("hf_genomics_long_range", "InstaDeepAI/genomics-long-range-benchmark", "sequence", "dna"),
        _spec("hf_rna_downstream_tasks", "genbio-ai/rna-downstream-tasks", "classification", "rna"),
        _spec("hf_bacbench_antibiotic_resistance_dna", "macwiatrak/bacbench-antibiotic-resistance-dna", "classification", "dna"),
        _spec("hf_bacbench_phenotypic_traits_dna", "macwiatrak/bacbench-phenotypic-traits-dna", "classification", "dna"),
        _spec("hf_pgs_catalog", "just-dna-seq/pgs-catalog", "classification", "dna"),
        _spec("hf_dna_llm_aligned_seqs", "DNA-LLM/aligned_seqs", "sequence", "dna"),
        _spec("hf_bacterial_intergenic_dna", "AllTheBacteria/Bac-Corpus-dna-intergenic-sequences-high-diversity", "sequence", "dna", streaming=True),
        _spec("hf_forensic_dnanet", "NetherlandsForensicInstitute/DNANet_2p5pMixture_PPF6C_2024", "classification", "dna"),
        _spec("hf_traitgym_mendelian_dna", "bolinas-dna/evals-traitgym_mendelian_v2_harness_255", "classification", "dna"),
        _spec("hf_genomes_v5_validation_5", "bolinas-dna/genomes-v5-validation-intervals-v5_255_255", "sequence", "dna"),
        _spec("hf_genomes_v5_validation_1", "bolinas-dna/genomes-v5-validation-intervals-v1_255_255", "sequence", "dna"),
        _spec("hf_genomes_v5_validation_15", "bolinas-dna/genomes-v5-validation-intervals-v15_255_255", "sequence", "dna"),
        _spec("hf_animal_genomes_v5_5", "bolinas-dna/genomes-v5-genome_set-animals-intervals-v5_255_128", "sequence", "dna"),
        _spec("hf_animal_genomes_v5_1", "bolinas-dna/genomes-v5-genome_set-animals-intervals-v1_255_128", "sequence", "dna"),
        _spec("hf_animal_genomes_v5_15", "bolinas-dna/genomes-v5-genome_set-animals-intervals-v15_255_128", "sequence", "dna"),
        _spec("hf_rnagps", "introvoyz041/rnagps", "classification", "rna"),
    ]
})


HF_VERIFIED_BENCHMARK_KEYS = frozenset({
    "hf_abdelmo_pubmed_dataset",
    "hf_abdelmo_pubmed_ds",
    "hf_adaptllm_chemprot",
    "hf_adaptllm_medicine_tasks",
    "hf_adaptllm_mqp",
    "hf_adaptllm_pubmedqa",
    "hf_adaptllm_rct",
    "hf_ai_medical_chatbot",
    "hf_arabic_medical_consultations",
    "hf_asclepius_clinical_notes",
    "hf_augmented_clinical_notes",
    "hf_ccdv_pubmed_summarization",
    "hf_chatdoctor_healthcaremagic",
    "hf_chemistry_qa",
    "hf_common_pile_pubmed",
    "hf_common_pile_pubmed_filtered",
    "hf_dna_gen",
    "hf_dna_llm_aligned_seqs",
    "hf_fluorescence_prediction",
    "hf_gaianet_chemistry",
    "hf_genbio_proteingym_dms",
    "hf_genecorpus_104m",
    "hf_genomes_v5_validation_1",
    "hf_genomes_v5_validation_15",
    "hf_genomes_v5_validation_5",
    "hf_healthcare_data",
    "hf_huatuo_medical_qa",
    "hf_katielink_moleculenet_bace",
    "hf_katielink_moleculenet_bbbp",
    "hf_katielink_moleculenet_clintox",
    "hf_katielink_moleculenet_esol",
    "hf_katielink_moleculenet_freesolv",
    "hf_katielink_moleculenet_hiv",
    "hf_katielink_moleculenet_sider",
    "hf_katielink_moleculenet_tox21",
    "hf_lavita_medical_qa_datasets",
    "hf_lavita_medmcqa",
    "hf_lavita_medqa_4options",
    "hf_lavita_medqa_5options",
    "hf_lavita_mmmlu_anatomy",
    "hf_lavita_mmmlu_clinical_knowledge",
    "hf_lavita_mmmlu_college_biology",
    "hf_lavita_mmmlu_college_medicine",
    "hf_lavita_mmmlu_medical_genetics",
    "hf_lavita_mmmlu_professional_medicine",
    "hf_lavita_pubmedqa",
    "hf_lavita_usmle_step1",
    "hf_lavita_usmle_step2",
    "hf_lavita_usmle_step3",
    "hf_lpm24_eval_caption",
    "hf_lpm24_eval_molgen",
    "hf_lpm24_train",
    "hf_malikeh_chatdoctor_healthcaremagic",
    "hf_malikeh_chatdoctor_icliniq",
    "hf_malikeh_medical_flashcards",
    "hf_malikeh_medical_qa",
    "hf_malikeh_medqa",
    "hf_malikeh_mmmlu",
    "hf_malikeh_pubmed_causal",
    "hf_malikeh_wikidoc",
    "hf_malikeh_wikidoc_patient_information",
    "hf_medical_flashcards",
    "hf_medical_meadow_medqa",
    "hf_medical_meadow_wikidoc",
    "hf_medical_o1_reasoning_sft",
    "hf_medical_o1_verifiable",
    "hf_medical_question_pairs",
    "hf_medical_r1_distill",
    "hf_medqa_usmle_4_options",
    "hf_medqa_usmle_4_options_hf",
    "hf_medquad",
    "hf_metanova_proteins",
    "hf_moleculeace",
    "hf_moleculeace_chembl1871_ki",
    "hf_moleculeace_chembl204_ki",
    "hf_moleculeace_chembl214_ki",
    "hf_moleculeace_chembl228_ki",
    "hf_moleculeace_chembl237_ec50",
    "hf_moleculenet_bace",
    "hf_moleculenet_bbbp",
    "hf_moleculenet_benchmark",
    "hf_moleculenet_clintox",
    "hf_moleculenet_esol",
    "hf_moleculenet_freesolv",
    "hf_moleculenet_hiv",
    "hf_moleculenet_lipophilicity",
    "hf_moleculenet_sider",
    "hf_moleculenet_toxcast",
    "hf_moleculestm",
    "hf_mts_dialogue_clinical_note",
    "hf_openlifescience_medqa",
    "hf_openlifescience_pubmedqa",
    "hf_openmed_reasoning_sft",
    "hf_protein_fluorescence",
    "hf_protein_solubility",
    "hf_protein_stability",
    "hf_proteinlmbench_enzyme_cot",
    "hf_proteinlmbench_uniprot_disease",
    "hf_proteinlmbench_uniprot_function",
    "hf_proteinlmbench_uniprot_induction",
    "hf_proteinlmbench_uniprot_ptm",
    "hf_proteinlmbench_uniprot_subunit",
    "hf_proteinlmbench_uniprot_tissue",
    "hf_pubmed_200k_rct",
    "hf_pubmed_rct20k",
    "hf_rna_expression_hek",
    "hf_rna_expression_muscle",
    "hf_rna_expression_pc3",
    "hf_rna_mean_ribosome_load",
    "hf_rna_modification_site",
    "hf_rna_ncrna_family_bnoise0",
    "hf_rna_splice_site_acceptor",
    "hf_rna_splice_site_donor",
    "hf_smiles_molecules_chembl",
    "hf_tcm_pretrain",
    "hf_tcm_sft",
    "hf_traitgym_mendelian_dna",
    "hf_xuxu_medqa_mainland_test",
    "hf_xuxu_medqa_taiwan_test",
    "hf_xuxu_medqa_us_test",
})


def hf_benchmark_cli_entries() -> dict[str, dict[str, Any]]:
    return {
        key: {
            "loader": "load_hf_benchmark_tasks",
            "benchmark_key": key,
            "kwargs": {"dataset_key": key},
        }
        for key in HF_VERIFIED_BENCHMARK_KEYS
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
        "answer_type": answer_type,
        "input_type": spec.extra.get("input_type", "text"),
        "scorer": spec.extra.get("scorer", answer_type),
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
    "HF_VERIFIED_BENCHMARK_KEYS",
    "hf_spec_metadata",
    "hf_verified_metadata",
    "validate_hf_metadata",
    "hf_benchmark_cli_entries",
]
