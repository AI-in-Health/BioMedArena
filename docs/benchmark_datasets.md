# Benchmark Dataset Inventory

This file is generated from the current CLI registry and loader metadata. It is the public source-of-truth for what the repository supports at release time.

- CLI benchmark registrations: **147**
- Core/non-HF registrations: **27**
- Generic HuggingFace registrations exposed in CLI: **120**
- CLI modes: **4**

Count semantics: core benchmark counts are the default loader scope where the code pins one; HF counts use registry metadata when present. `unknown upstream split size` means the loader follows the official HuggingFace split but the repository does not pin a static count, so users should inspect the current upstream dataset card or run a source audit in their environment.

## Core Benchmarks

| Benchmark | Source | Count | Content | Input type | Task type | Answer/scorer | Gated | Needs network | Offline cache | Multimodal | Loader |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `aa_lcr` | ArtificialAnalysis/AA-LCR | upstream split size | long-context reasoning over documents | text, optional retrieved documents | long-context QA | openText | no | yes on first run | yes | no | `load_aa_lcr_tasks` |
| `agentclinic` | AgentClinic official release | loader default | doctor-patient diagnostic scenarios | text dialogue | clinical simulation | openText | no | source-dependent | yes | no | `load_agentclinic_tasks` |
| `bioasq` | BioASQ official/local source | loader default | factoid/list/yes-no biomedical questions | text | biomedical QA | openText | no | source-dependent | yes | no | `load_bioasq_tasks` |
| `bioprobench` | BioProBench official data | official benchmark scope | biological protocol understanding and repair | text | protocol QA | mixed | no | yes on first run | yes | no | `load_bioprobench_tasks` |
| `bixbench` | futurehouse/BixBench | 205 | closed-book biomedical information tasks | text | bioinformatics QA | openText | no | yes | yes | no | `load_bixbench_tasks` |
| `genotex` | GenoTEX official data | loader default | genomics text reasoning | text/genomics | genomics QA | openText | no | source-dependent | yes | no | `load_genotex_tasks` |
| `gpqa_bio` | Idavidrein/gpqa:gpqa_diamond/train | 198 | graduate-level biology/chemistry/medicine questions | text | graduate science MCQ | multipleChoice | yes | yes | yes | no | `load_gpqa_bio_tasks` |
| `healthbench` | OpenAI HealthBench official data | official benchmark scope | consumer-health answer quality and safety | text conversation | health conversation | openText | source-dependent | yes on first run | yes | no | `load_healthbench_tasks` |
| `hle_gold` | futurehouse/hle-gold-bio-chem:train | 149 | HLE Gold bio/chem subset | text | expert QA | mixed | yes | yes | yes | no | `load_hle_gold_tasks` |
| `labbench` | futurehouse/lab-bench | loader default subsets | LitQA, cloning, protocol tasks | text | biomedical agent QA | mixed | yes | yes | yes | no | `load_labbench_tasks` |
| `labbench2` | EdisonScientific/labbench2 text-only subsets | 821 | LAB-Bench 2 text-only evaluation subset | text | literature/database/patent QA | openText | yes | yes | yes | no by default | `load_labbench2_tasks` |
| `labbench2_821` | EdisonScientific/labbench2 text-only subsets | 821 | Alias for the 821-row LAB-Bench 2 text subset | text | literature/database/patent QA | openText | yes | yes | yes | no by default | `load_labbench2_tasks` |
| `medagentbench` | MedAgentBench official data | loader default | clinical workflow and EHR tasks | text/EHR | medical agent workflow | mixed | no | source-dependent | yes | no | `load_medagentbench_tasks` |
| `medcalc` | ncbi/MedCalc-Bench-v1.2:test | 1100 | medical calculator word problems | text | clinical calculation | exactNumeric | no | yes | yes | no | `load_medcalc_tasks` |
| `medhelm` | MedHELM official/public sources | official benchmark scope | medical QA, safety, and scenario tasks | text | medical HELM tasks | mixed | source-dependent | yes on first run | yes | no | `load_medhelm_tasks` |
| `medmcqa` | openlifescienceai/medmcqa | loader default split | medical entrance-exam questions | text | medical MCQ | multipleChoice | no | yes | yes | no | `load_medical_qa_tasks` |
| `medqa` | GBaker/MedQA-USMLE-4-options | loader default split | USMLE-style questions | text | USMLE MCQ | multipleChoice | no | yes | yes | no | `load_medical_qa_tasks` |
| `medxpertqa` | TsinghuaC3I/MedXpertQA | loader default Text subset | expert medical reasoning questions | text | expert medical MCQ | multipleChoice | no | yes | yes | no | `load_medxpertqa_tasks` |
| `medxpertqa_mm` | TsinghuaC3I/MedXpertQA-MM | official multimodal subset | expert medical multimodal questions | text with optional images | medical VQA/MCQ | multipleChoice | no | yes | yes | yes; text fallback by default | `load_medxpertqa_mm_tasks` |
| `mmlu` | MMLU medical/biology subjects | loader default subjects | MMLU anatomy, medicine, biology, genetics subjects | text | academic MCQ | multipleChoice | no | yes | yes | no | `load_mmlu_tasks` |
| `pathvqa` | PathVQA official/HF source | loader default split | pathology image questions | text+image | pathology VQA | openText | no | yes | yes | yes | `load_pathvqa_tasks` |
| `pubmedqa` | qiaojin/PubMedQA or OpenLifeScience mirror | loader default split | yes/no/maybe biomedical literature questions | text abstract | PubMed abstract QA | multipleChoice | no | yes | yes | no | `load_medical_qa_tasks` |
| `quick_suite` | built-in repository fixtures | 20 | 5 MCQ, 5 exact, 5 numeric, 5 open-text scorer checks | text | offline smoke | mixed | no | no | not needed | no | `load_quick_suite_tasks` |
| `rag_essential` | built-in RAG essential tasks | 12 | tasks designed to reward retrieval/tool use | text | retrieval/tool-use QA | openText | no | no | not needed | no | `load_rag_essential_tasks` |
| `super_chemistry` | ZehuaZhao/SUPERChem:SUPERChem-500.parquet | 500 text rows by default; 500 official rows total | advanced chemistry questions | text, optional images | chemistry MCQ | multipleChoice | no | yes | yes | yes; text fallback by default | `load_super_chemistry_tasks` |
| `superchem` | SuperChem official data | loader default | chemistry evaluation tasks | text | chemistry QA | mixed | source-dependent | yes on first run | yes | no | `load_superchem_tasks` |
| `supergpqa` | SuperGPQA official data | loader default | graduate-level science questions | text | science MCQ | multipleChoice | source-dependent | yes on first run | yes | no | `load_supergpqa_tasks` |

## HuggingFace Benchmarks

All `hf_*` entries load from the official HuggingFace dataset repo listed below via `harness.eval.bench_hf_benchmark.load_hf_benchmark_tasks`. First run needs network access; subsequent runs can use the HuggingFace datasets cache. The current generic HF loader normalizes rows to text/structured tasks and does not register multimodal HF datasets.

| Benchmark | Source | Config | Split | Count | Domain | Task type | Answer/scorer | Gated | Network | Offline cache | Multimodal |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `hf_abdelmo_pubmed_dataset` | [`abdelmo/pubmed-dataset`](https://huggingface.co/datasets/abdelmo/pubmed-dataset) |  | default | unknown upstream split size | biomedical | text | openText | no | yes | yes | no |
| `hf_abdelmo_pubmed_ds` | [`abdelmo/pubmed-ds`](https://huggingface.co/datasets/abdelmo/pubmed-ds) |  | default | unknown upstream split size | biomedical | text | openText | no | yes | yes | no |
| `hf_adaptllm_chemprot` | [`AdaptLLM/medicine-tasks`](https://huggingface.co/datasets/AdaptLLM/medicine-tasks) | ChemProt | default | unknown upstream split size | biomedical | mcq | multipleChoice | no | yes | yes | no |
| `hf_adaptllm_medicine_tasks` | [`AdaptLLM/medicine-tasks`](https://huggingface.co/datasets/AdaptLLM/medicine-tasks) | USMLE | default | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |
| `hf_adaptllm_mqp` | [`AdaptLLM/medicine-tasks`](https://huggingface.co/datasets/AdaptLLM/medicine-tasks) | MQP | default | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |
| `hf_adaptllm_pubmedqa` | [`AdaptLLM/medicine-tasks`](https://huggingface.co/datasets/AdaptLLM/medicine-tasks) | PubMedQA | default | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |
| `hf_adaptllm_rct` | [`AdaptLLM/medicine-tasks`](https://huggingface.co/datasets/AdaptLLM/medicine-tasks) | RCT | default | unknown upstream split size | biomedical | mcq | multipleChoice | no | yes | yes | no |
| `hf_ai_medical_chatbot` | [`ruslanmv/ai-medical-chatbot`](https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot) |  | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_arabic_medical_consultations` | [`Ahmed-Selem/Shifaa_Arabic_Medical_Consultations`](https://huggingface.co/datasets/Ahmed-Selem/Shifaa_Arabic_Medical_Consultations) |  | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_asclepius_clinical_notes` | [`starmpcc/Asclepius-Synthetic-Clinical-Notes`](https://huggingface.co/datasets/starmpcc/Asclepius-Synthetic-Clinical-Notes) |  | default | unknown upstream split size | clinical | summarization | openText | no | yes | yes | no |
| `hf_augmented_clinical_notes` | [`AGBonnet/augmented-clinical-notes`](https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes) |  | default | unknown upstream split size | clinical | summarization | openText | no | yes | yes | no |
| `hf_ccdv_pubmed_summarization` | [`ccdv/pubmed-summarization`](https://huggingface.co/datasets/ccdv/pubmed-summarization) |  | default | unknown upstream split size | biomedical | summarization | openText | no | yes | yes | no |
| `hf_chatdoctor_healthcaremagic` | [`lavita/ChatDoctor-HealthCareMagic-100k`](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) |  | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_chemistry_qa` | [`avaliev/ChemistryQA`](https://huggingface.co/datasets/avaliev/ChemistryQA) |  | default | unknown upstream split size | chemistry | qa | openText | no | yes | yes | no |
| `hf_common_pile_pubmed` | [`common-pile/pubmed`](https://huggingface.co/datasets/common-pile/pubmed) |  | default | unknown upstream split size | biomedical | text | openText | no | yes | yes | no |
| `hf_common_pile_pubmed_filtered` | [`common-pile/pubmed_filtered`](https://huggingface.co/datasets/common-pile/pubmed_filtered) |  | default | unknown upstream split size | biomedical | text | openText | no | yes | yes | no |
| `hf_dna_gen` | [`xingyusu/DNA_Gen`](https://huggingface.co/datasets/xingyusu/DNA_Gen) |  | default | unknown upstream split size | dna | sequence | openText | no | yes | yes | no |
| `hf_dna_llm_aligned_seqs` | [`DNA-LLM/aligned_seqs`](https://huggingface.co/datasets/DNA-LLM/aligned_seqs) |  | default | unknown upstream split size | dna | sequence | openText | no | yes | yes | no |
| `hf_fluorescence_prediction` | [`proteinglm/fluorescence_prediction`](https://huggingface.co/datasets/proteinglm/fluorescence_prediction) |  | default | unknown upstream split size | protein | classification | exactMatch | no | yes | yes | no |
| `hf_gaianet_chemistry` | [`gaianet/chemistry`](https://huggingface.co/datasets/gaianet/chemistry) |  | default | unknown upstream split size | chemistry | text | openText | no | yes | yes | no |
| `hf_genbio_proteingym_dms` | [`genbio-ai/ProteinGYM-DMS`](https://huggingface.co/datasets/genbio-ai/ProteinGYM-DMS) |  | default | unknown upstream split size | protein | protein_fitness | exactNumeric | no | yes | yes | no |
| `hf_genecorpus_104m` | [`theodoris-lab/Genecorpus-104M`](https://huggingface.co/datasets/theodoris-lab/Genecorpus-104M) |  | default | unknown upstream split size | genomics | sequence | openText | no | yes | yes | no |
| `hf_genomes_v5_validation_1` | [`bolinas-dna/genomes-v5-validation-intervals-v1_255_255`](https://huggingface.co/datasets/bolinas-dna/genomes-v5-validation-intervals-v1_255_255) |  | default | unknown upstream split size | dna | sequence | openText | no | yes | yes | no |
| `hf_genomes_v5_validation_15` | [`bolinas-dna/genomes-v5-validation-intervals-v15_255_255`](https://huggingface.co/datasets/bolinas-dna/genomes-v5-validation-intervals-v15_255_255) |  | default | unknown upstream split size | dna | sequence | openText | no | yes | yes | no |
| `hf_genomes_v5_validation_5` | [`bolinas-dna/genomes-v5-validation-intervals-v5_255_255`](https://huggingface.co/datasets/bolinas-dna/genomes-v5-validation-intervals-v5_255_255) |  | default | unknown upstream split size | dna | sequence | openText | no | yes | yes | no |
| `hf_healthcare_data` | [`Nicolybgs/healthcare_data`](https://huggingface.co/datasets/Nicolybgs/healthcare_data) |  | default | unknown upstream split size | healthcare | classification | exactMatch | no | yes | yes | no |
| `hf_huatuo_medical_qa` | [`shibing624/huatuo_medical_qa_sharegpt`](https://huggingface.co/datasets/shibing624/huatuo_medical_qa_sharegpt) |  | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_katielink_moleculenet_bace` | [`katielink/moleculenet-benchmark`](https://huggingface.co/datasets/katielink/moleculenet-benchmark) | bace | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_katielink_moleculenet_bbbp` | [`katielink/moleculenet-benchmark`](https://huggingface.co/datasets/katielink/moleculenet-benchmark) | bbbp | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_katielink_moleculenet_clintox` | [`katielink/moleculenet-benchmark`](https://huggingface.co/datasets/katielink/moleculenet-benchmark) | clintox | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_katielink_moleculenet_esol` | [`katielink/moleculenet-benchmark`](https://huggingface.co/datasets/katielink/moleculenet-benchmark) | esol | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_katielink_moleculenet_freesolv` | [`katielink/moleculenet-benchmark`](https://huggingface.co/datasets/katielink/moleculenet-benchmark) | freesolv | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_katielink_moleculenet_hiv` | [`katielink/moleculenet-benchmark`](https://huggingface.co/datasets/katielink/moleculenet-benchmark) | hiv | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_katielink_moleculenet_sider` | [`katielink/moleculenet-benchmark`](https://huggingface.co/datasets/katielink/moleculenet-benchmark) | sider | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_katielink_moleculenet_tox21` | [`katielink/moleculenet-benchmark`](https://huggingface.co/datasets/katielink/moleculenet-benchmark) | tox21 | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_lavita_medical_qa_datasets` | [`lavita/medical-qa-datasets`](https://huggingface.co/datasets/lavita/medical-qa-datasets) | all-processed | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_lavita_medmcqa` | [`lavita/medical-qa-datasets`](https://huggingface.co/datasets/lavita/medical-qa-datasets) | medmcqa | validation | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |
| `hf_lavita_medqa_4options` | [`lavita/medical-qa-datasets`](https://huggingface.co/datasets/lavita/medical-qa-datasets) | med-qa-en-4options-source | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_lavita_medqa_5options` | [`lavita/medical-qa-datasets`](https://huggingface.co/datasets/lavita/medical-qa-datasets) | med-qa-en-5options-source | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_lavita_mmmlu_anatomy` | [`lavita/medical-qa-datasets`](https://huggingface.co/datasets/lavita/medical-qa-datasets) | mmmlu-anatomy | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_lavita_mmmlu_clinical_knowledge` | [`lavita/medical-qa-datasets`](https://huggingface.co/datasets/lavita/medical-qa-datasets) | mmmlu-clinical-knowledge | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_lavita_mmmlu_college_biology` | [`lavita/medical-qa-datasets`](https://huggingface.co/datasets/lavita/medical-qa-datasets) | mmmlu-college-biology | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_lavita_mmmlu_college_medicine` | [`lavita/medical-qa-datasets`](https://huggingface.co/datasets/lavita/medical-qa-datasets) | mmmlu-college-medicine | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_lavita_mmmlu_medical_genetics` | [`lavita/medical-qa-datasets`](https://huggingface.co/datasets/lavita/medical-qa-datasets) | mmmlu-medical-genetics | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_lavita_mmmlu_professional_medicine` | [`lavita/medical-qa-datasets`](https://huggingface.co/datasets/lavita/medical-qa-datasets) | mmmlu-professional-medicine | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_lavita_pubmedqa` | [`lavita/medical-qa-datasets`](https://huggingface.co/datasets/lavita/medical-qa-datasets) | pubmed-qa | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_lavita_usmle_step1` | [`lavita/medical-qa-datasets`](https://huggingface.co/datasets/lavita/medical-qa-datasets) | usmle-self-assessment-step1 | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_lavita_usmle_step2` | [`lavita/medical-qa-datasets`](https://huggingface.co/datasets/lavita/medical-qa-datasets) | usmle-self-assessment-step2 | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_lavita_usmle_step3` | [`lavita/medical-qa-datasets`](https://huggingface.co/datasets/lavita/medical-qa-datasets) | usmle-self-assessment-step3 | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_lpm24_eval_caption` | [`language-plus-molecules/LPM-24_eval-caption`](https://huggingface.co/datasets/language-plus-molecules/LPM-24_eval-caption) |  | default | unknown upstream split size | chemistry | qa | openText | no | yes | yes | no |
| `hf_lpm24_eval_molgen` | [`language-plus-molecules/LPM-24_eval-molgen`](https://huggingface.co/datasets/language-plus-molecules/LPM-24_eval-molgen) |  | default | unknown upstream split size | chemistry | qa | openText | no | yes | yes | no |
| `hf_lpm24_train` | [`language-plus-molecules/LPM-24_train`](https://huggingface.co/datasets/language-plus-molecules/LPM-24_train) |  | default | unknown upstream split size | chemistry | qa | openText | no | yes | yes | no |
| `hf_malikeh_chatdoctor_healthcaremagic` | [`Malikeh1375/medical-question-answering-datasets`](https://huggingface.co/datasets/Malikeh1375/medical-question-answering-datasets) | chatdoctor_healthcaremagic | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_malikeh_chatdoctor_icliniq` | [`Malikeh1375/medical-question-answering-datasets`](https://huggingface.co/datasets/Malikeh1375/medical-question-answering-datasets) | chatdoctor_icliniq | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_malikeh_medical_flashcards` | [`Malikeh1375/medical-question-answering-datasets`](https://huggingface.co/datasets/Malikeh1375/medical-question-answering-datasets) | medical_meadow_medical_flashcards | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_malikeh_medical_qa` | [`Malikeh1375/medical-question-answering-datasets`](https://huggingface.co/datasets/Malikeh1375/medical-question-answering-datasets) | all-processed | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_malikeh_medqa` | [`Malikeh1375/medical-question-answering-datasets`](https://huggingface.co/datasets/Malikeh1375/medical-question-answering-datasets) | medical_meadow_medqa | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_malikeh_mmmlu` | [`Malikeh1375/medical-question-answering-datasets`](https://huggingface.co/datasets/Malikeh1375/medical-question-answering-datasets) | medical_meadow_mmmlu | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_malikeh_pubmed_causal` | [`Malikeh1375/medical-question-answering-datasets`](https://huggingface.co/datasets/Malikeh1375/medical-question-answering-datasets) | medical_meadow_pubmed_causal | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_malikeh_wikidoc` | [`Malikeh1375/medical-question-answering-datasets`](https://huggingface.co/datasets/Malikeh1375/medical-question-answering-datasets) | medical_meadow_wikidoc | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_malikeh_wikidoc_patient_information` | [`Malikeh1375/medical-question-answering-datasets`](https://huggingface.co/datasets/Malikeh1375/medical-question-answering-datasets) | medical_meadow_wikidoc_patient_information | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_medical_flashcards` | [`medalpaca/medical_meadow_medical_flashcards`](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards) |  | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_medical_meadow_medqa` | [`medalpaca/medical_meadow_medqa`](https://huggingface.co/datasets/medalpaca/medical_meadow_medqa) |  | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_medical_meadow_wikidoc` | [`medalpaca/medical_meadow_wikidoc`](https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc) |  | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_medical_o1_reasoning_sft` | [`FreedomIntelligence/medical-o1-reasoning-SFT`](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) | en | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_medical_o1_verifiable` | [`FreedomIntelligence/medical-o1-verifiable-problem`](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-verifiable-problem) |  | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_medical_question_pairs` | [`curaihealth/medical_questions_pairs`](https://huggingface.co/datasets/curaihealth/medical_questions_pairs) |  | default | unknown upstream split size | medical | pair_classification | exactMatch | no | yes | yes | no |
| `hf_medical_r1_distill` | [`FreedomIntelligence/Medical-R1-Distill-Data`](https://huggingface.co/datasets/FreedomIntelligence/Medical-R1-Distill-Data) |  | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_medqa_usmle_4_options` | [`GBaker/MedQA-USMLE-4-options`](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) |  | default | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |
| `hf_medqa_usmle_4_options_hf` | [`GBaker/MedQA-USMLE-4-options-hf`](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf) |  | default | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |
| `hf_medquad` | [`keivalya/MedQuad-MedicalQnADataset`](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset) |  | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_metanova_proteins` | [`Metanova/Proteins`](https://huggingface.co/datasets/Metanova/Proteins) |  | default | unknown upstream split size | protein | sequence | openText | no | yes | yes | no |
| `hf_moleculeace` | [`karina-zadorozhny/moleculeace`](https://huggingface.co/datasets/karina-zadorozhny/moleculeace) | CHEMBL1862_Ki | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculeace_chembl1871_ki` | [`karina-zadorozhny/moleculeace`](https://huggingface.co/datasets/karina-zadorozhny/moleculeace) | CHEMBL1871_Ki | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculeace_chembl204_ki` | [`karina-zadorozhny/moleculeace`](https://huggingface.co/datasets/karina-zadorozhny/moleculeace) | CHEMBL204_Ki | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculeace_chembl214_ki` | [`karina-zadorozhny/moleculeace`](https://huggingface.co/datasets/karina-zadorozhny/moleculeace) | CHEMBL214_Ki | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculeace_chembl228_ki` | [`karina-zadorozhny/moleculeace`](https://huggingface.co/datasets/karina-zadorozhny/moleculeace) | CHEMBL228_Ki | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculeace_chembl237_ec50` | [`karina-zadorozhny/moleculeace`](https://huggingface.co/datasets/karina-zadorozhny/moleculeace) | CHEMBL237_EC50 | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculenet_bace` | [`scikit-fingerprints/MoleculeNet_BACE`](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_BACE) |  | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculenet_bbbp` | [`scikit-fingerprints/MoleculeNet_BBBP`](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_BBBP) |  | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculenet_benchmark` | [`katielink/moleculenet-benchmark`](https://huggingface.co/datasets/katielink/moleculenet-benchmark) | bace | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculenet_clintox` | [`scikit-fingerprints/MoleculeNet_ClinTox`](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_ClinTox) |  | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculenet_esol` | [`scikit-fingerprints/MoleculeNet_ESOL`](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_ESOL) |  | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculenet_freesolv` | [`scikit-fingerprints/MoleculeNet_FreeSolv`](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_FreeSolv) |  | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculenet_hiv` | [`scikit-fingerprints/MoleculeNet_HIV`](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_HIV) |  | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculenet_lipophilicity` | [`scikit-fingerprints/MoleculeNet_Lipophilicity`](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_Lipophilicity) |  | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculenet_sider` | [`scikit-fingerprints/MoleculeNet_SIDER`](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_SIDER) |  | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculenet_toxcast` | [`scikit-fingerprints/MoleculeNet_ToxCast`](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_ToxCast) |  | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculestm` | [`chao1224/MoleculeSTM`](https://huggingface.co/datasets/chao1224/MoleculeSTM) |  | default | unknown upstream split size | chemistry | text | openText | no | yes | yes | no |
| `hf_mts_dialogue_clinical_note` | [`har1/MTS_Dialogue-Clinical_Note`](https://huggingface.co/datasets/har1/MTS_Dialogue-Clinical_Note) |  | default | unknown upstream split size | clinical | summarization | openText | no | yes | yes | no |
| `hf_openlifescience_medqa` | [`openlifescienceai/medqa`](https://huggingface.co/datasets/openlifescienceai/medqa) |  | default | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |
| `hf_openlifescience_pubmedqa` | [`openlifescienceai/pubmedqa`](https://huggingface.co/datasets/openlifescienceai/pubmedqa) |  | default | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |
| `hf_openmed_reasoning_sft` | [`OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B`](https://huggingface.co/datasets/OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B) |  | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_protein_fluorescence` | [`proteinea/fluorescence`](https://huggingface.co/datasets/proteinea/fluorescence) |  | default | unknown upstream split size | protein | regression | exactNumeric | no | yes | yes | no |
| `hf_protein_solubility` | [`proteinea/solubility`](https://huggingface.co/datasets/proteinea/solubility) |  | default | unknown upstream split size | protein | classification | exactMatch | no | yes | yes | no |
| `hf_protein_stability` | [`SaProtHub/Dataset-Meta-scale-protein-stability`](https://huggingface.co/datasets/SaProtHub/Dataset-Meta-scale-protein-stability) |  | default | unknown upstream split size | protein | regression | exactNumeric | no | yes | yes | no |
| `hf_proteinlmbench_enzyme_cot` | [`tsynbio/ProteinLMBench`](https://huggingface.co/datasets/tsynbio/ProteinLMBench) | Enzyme_CoT | default | unknown upstream split size | protein | qa | openText | no | yes | yes | no |
| `hf_proteinlmbench_uniprot_disease` | [`tsynbio/ProteinLMBench`](https://huggingface.co/datasets/tsynbio/ProteinLMBench) | UniProt_Involvement in disease | default | unknown upstream split size | protein | qa | openText | no | yes | yes | no |
| `hf_proteinlmbench_uniprot_function` | [`tsynbio/ProteinLMBench`](https://huggingface.co/datasets/tsynbio/ProteinLMBench) | UniProt_Function | default | unknown upstream split size | protein | qa | openText | no | yes | yes | no |
| `hf_proteinlmbench_uniprot_induction` | [`tsynbio/ProteinLMBench`](https://huggingface.co/datasets/tsynbio/ProteinLMBench) | UniProt_Induction | default | unknown upstream split size | protein | qa | openText | no | yes | yes | no |
| `hf_proteinlmbench_uniprot_ptm` | [`tsynbio/ProteinLMBench`](https://huggingface.co/datasets/tsynbio/ProteinLMBench) | UniProt_Post-translational modification | default | unknown upstream split size | protein | qa | openText | no | yes | yes | no |
| `hf_proteinlmbench_uniprot_subunit` | [`tsynbio/ProteinLMBench`](https://huggingface.co/datasets/tsynbio/ProteinLMBench) | UniProt_Subunit structure | default | unknown upstream split size | protein | qa | openText | no | yes | yes | no |
| `hf_proteinlmbench_uniprot_tissue` | [`tsynbio/ProteinLMBench`](https://huggingface.co/datasets/tsynbio/ProteinLMBench) | UniProt_Tissue specificity | default | unknown upstream split size | protein | qa | openText | no | yes | yes | no |
| `hf_pubmed_200k_rct` | [`pietrolesci/pubmed-200k-rct`](https://huggingface.co/datasets/pietrolesci/pubmed-200k-rct) |  | default | unknown upstream split size | biomedical | classification | exactMatch | no | yes | yes | no |
| `hf_pubmed_rct20k` | [`armanc/pubmed-rct20k`](https://huggingface.co/datasets/armanc/pubmed-rct20k) |  | default | unknown upstream split size | biomedical | classification | exactMatch | no | yes | yes | no |
| `hf_rna_expression_hek` | [`genbio-ai/rna-downstream-tasks`](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks) | expression_HEK | default | unknown upstream split size | rna | regression | exactNumeric | no | yes | yes | no |
| `hf_rna_expression_muscle` | [`genbio-ai/rna-downstream-tasks`](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks) | expression_Muscle | default | unknown upstream split size | rna | regression | exactNumeric | no | yes | yes | no |
| `hf_rna_expression_pc3` | [`genbio-ai/rna-downstream-tasks`](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks) | expression_pc3 | default | unknown upstream split size | rna | regression | exactNumeric | no | yes | yes | no |
| `hf_rna_mean_ribosome_load` | [`genbio-ai/rna-downstream-tasks`](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks) | mean_ribosome_load | default | unknown upstream split size | rna | regression | exactNumeric | no | yes | yes | no |
| `hf_rna_modification_site` | [`genbio-ai/rna-downstream-tasks`](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks) | modification_site | default | unknown upstream split size | rna | classification | exactMatch | no | yes | yes | no |
| `hf_rna_ncrna_family_bnoise0` | [`genbio-ai/rna-downstream-tasks`](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks) | ncrna_family_bnoise0 | default | unknown upstream split size | rna | classification | exactMatch | no | yes | yes | no |
| `hf_rna_splice_site_acceptor` | [`genbio-ai/rna-downstream-tasks`](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks) | splice_site_acceptor | default | unknown upstream split size | rna | classification | exactMatch | no | yes | yes | no |
| `hf_rna_splice_site_donor` | [`genbio-ai/rna-downstream-tasks`](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks) | splice_site_donor | default | unknown upstream split size | rna | classification | exactMatch | no | yes | yes | no |
| `hf_smiles_molecules_chembl` | [`antoinebcx/smiles-molecules-chembl`](https://huggingface.co/datasets/antoinebcx/smiles-molecules-chembl) |  | default | unknown upstream split size | chemistry | text | openText | no | yes | yes | no |
| `hf_tcm_pretrain` | [`SylvanL/Traditional-Chinese-Medicine-Dataset-Pretrain`](https://huggingface.co/datasets/SylvanL/Traditional-Chinese-Medicine-Dataset-Pretrain) |  | default | unknown upstream split size | medical | text | openText | no | yes | yes | no |
| `hf_tcm_sft` | [`SylvanL/Traditional-Chinese-Medicine-Dataset-SFT`](https://huggingface.co/datasets/SylvanL/Traditional-Chinese-Medicine-Dataset-SFT) |  | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_traitgym_mendelian_dna` | [`bolinas-dna/evals-traitgym_mendelian_v2_harness_255`](https://huggingface.co/datasets/bolinas-dna/evals-traitgym_mendelian_v2_harness_255) |  | default | unknown upstream split size | dna | classification | exactMatch | no | yes | yes | no |
| `hf_xuxu_medqa_mainland_test` | [`xuxuxuxuxu/MedQA_Mainland_test`](https://huggingface.co/datasets/xuxuxuxuxu/MedQA_Mainland_test) |  | default | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |
| `hf_xuxu_medqa_taiwan_test` | [`xuxuxuxuxu/MedQA_Taiwan_test`](https://huggingface.co/datasets/xuxuxuxuxu/MedQA_Taiwan_test) |  | default | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |
| `hf_xuxu_medqa_us_test` | [`xuxuxuxuxu/MedQA_US_test`](https://huggingface.co/datasets/xuxuxuxuxu/MedQA_US_test) |  | default | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |

## Verification

Use the offline gate first:

```bash
python3 scripts/run_quick_suite.py
python3 scripts/release_gate.py --strict
```

For live source checks, run a tiny loader audit in an environment with HuggingFace access and accepted gated dataset terms:

```bash
python3 scripts/verify_benchmark_sources.py --benchmarks all
```
