# Benchmark Dataset Inventory

This file is generated from the current CLI registry and loader metadata. It is the public source-of-truth for what the repository supports at release time.

- CLI benchmark registrations: **166**
- Core/non-HF registrations: **26**
- Generic HuggingFace registrations exposed in CLI: **140**
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

All `hf_*` entries below are exposed by the CLI and load from the HuggingFace dataset repo listed via `harness.eval.bench_hf_benchmark.load_hf_benchmark_tasks`. Deprecated aliases remain selectable for compatibility but redirect to the canonical benchmark shown in the first column. Training-only corpora and removed non-benchmark rows are intentionally excluded from this public inventory.

| Benchmark | Source | Config | Split | Count | Domain | Task type | Answer/scorer | Gated | Network | Offline cache | Multimodal |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `hf_adaptllm_chemprot` | [`AdaptLLM/medicine-tasks`](https://huggingface.co/datasets/AdaptLLM/medicine-tasks) | ChemProt | default | unknown upstream split size | biomedical | mcq | multipleChoice | no | yes | yes | no |
| `hf_adaptllm_medicine_tasks` | [`AdaptLLM/medicine-tasks`](https://huggingface.co/datasets/AdaptLLM/medicine-tasks) | USMLE | default | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |
| `hf_adaptllm_mqp` | [`AdaptLLM/medicine-tasks`](https://huggingface.co/datasets/AdaptLLM/medicine-tasks) | MQP | default | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |
| `hf_adaptllm_rct` | [`AdaptLLM/medicine-tasks`](https://huggingface.co/datasets/AdaptLLM/medicine-tasks) | RCT | default | unknown upstream split size | biomedical | mcq | multipleChoice | no | yes | yes | no |
| `hf_ade_corpus_v2` | [`ade-benchmark-corpus/ade_corpus_v2`](https://huggingface.co/datasets/ade-benchmark-corpus/ade_corpus_v2) | Ade_corpus_v2_classification | train | unknown upstream split size | biomedical | classification | exactMatch | no | yes | yes | no |
| `hf_anatem` | [`bigbio/anat_em`](https://huggingface.co/datasets/bigbio/anat_em) |  | test | unknown upstream split size | biomedical | classification | exactMatch | no | yes | yes | no |
| `hf_bacbench_antibiotic_resistance_dna` | [`macwiatrak/bacbench-antibiotic-resistance-dna`](https://huggingface.co/datasets/macwiatrak/bacbench-antibiotic-resistance-dna) |  | default | unknown upstream split size | dna | classification | exactMatch | no | yes | yes | no |
| `hf_bacbench_phenotypic_traits_dna` | [`macwiatrak/bacbench-phenotypic-traits-dna`](https://huggingface.co/datasets/macwiatrak/bacbench-phenotypic-traits-dna) |  | default | unknown upstream split size | dna | classification | exactMatch | no | yes | yes | no |
| `hf_bc2gm` | [`spyysalo/bc2gm_corpus`](https://huggingface.co/datasets/spyysalo/bc2gm_corpus) | bc2gm_corpus | test | unknown upstream split size | biomedical | classification | exactMatch | no | yes | yes | no |
| `hf_bc5cdr` | [`EMBO/BLURB`](https://huggingface.co/datasets/EMBO/BLURB) |  | test | unknown upstream split size | biomedical | classification | exactMatch | no | yes | yes | no |
| `hf_bigbio_med_qa` | [`bigbio/med_qa`](https://huggingface.co/datasets/bigbio/med_qa) |  | default | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |
| `hf_bigbio_pubmed_qa` | [`bigbio/pubmed_qa`](https://huggingface.co/datasets/bigbio/pubmed_qa) |  | default | unknown upstream split size | medical | classification | exactMatch | no | yes | yes | no |
| `hf_biocreative_viii_biored` | [`bigbio/biored`](https://huggingface.co/datasets/bigbio/biored) |  | default | unknown upstream split size | biomedical | classification | exactMatch | no | yes | yes | no |
| `hf_biomedbench` | [`biomedbench/BioMedBench`](https://huggingface.co/datasets/biomedbench/BioMedBench) |  | default | unknown upstream split size | biomedical | qa | openText | no | yes | yes | no |
| `hf_biored` | [`bigbio/biored`](https://huggingface.co/datasets/bigbio/biored) |  | default | unknown upstream split size | biomedical | classification | exactMatch | no | yes | yes | no |
| `hf_biosses` | [`mteb/biosses-sts`](https://huggingface.co/datasets/mteb/biosses-sts) |  | test | unknown upstream split size | biomedical | regression | exactNumeric | no | yes | yes | no |
| `hf_blurb` | [`EMBO/BLURB`](https://huggingface.co/datasets/EMBO/BLURB) |  | test | unknown upstream split size | biomedical | classification | exactMatch | no | yes | yes | no |
| `hf_careqa` | [`HPAI-BSC/CareQA`](https://huggingface.co/datasets/HPAI-BSC/CareQA) | CareQA_en | test | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |
| `hf_ccdv_pubmed_summarization` | [`ccdv/pubmed-summarization`](https://huggingface.co/datasets/ccdv/pubmed-summarization) |  | default | unknown upstream split size | biomedical | summarization | openText | no | yes | yes | no |
| `hf_chembench` | [`jablonkagroup/ChemBench`](https://huggingface.co/datasets/jablonkagroup/ChemBench) |  | default | unknown upstream split size | chemistry | qa | openText | no | yes | yes | no |
| `hf_chemistry_qa` | [`avaliev/ChemistryQA`](https://huggingface.co/datasets/avaliev/ChemistryQA) |  | default | unknown upstream split size | chemistry | qa | openText | no | yes | yes | no |
| `hf_chemllmbench` | [`blc-org/chemllmbench`](https://huggingface.co/datasets/blc-org/chemllmbench) |  | default | unknown upstream split size | chemistry | qa | openText | no | yes | yes | no |
| `hf_clicr` | [`bigbio/clicr`](https://huggingface.co/datasets/bigbio/clicr) |  | default | unknown upstream split size | clinical | qa | openText | no | yes | yes | no |
| `hf_clinical_trials_eligibility_nlp` | [`bigbio/n2c2_2018_track1`](https://huggingface.co/datasets/bigbio/n2c2_2018_track1) |  | default | unknown upstream split size | clinical | classification | exactMatch | no | yes | yes | no |
| `hf_cmb` | [`FreedomIntelligence/CMB`](https://huggingface.co/datasets/FreedomIntelligence/CMB) | CMB-Exam | test | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |
| `hf_cmexam` | [`fzkuji/CMExam`](https://huggingface.co/datasets/fzkuji/CMExam) |  | test | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |
| `hf_cord19_qa` | [`allenai/cord19`](https://huggingface.co/datasets/allenai/cord19) |  | default | unknown upstream split size | biomedical | qa | openText | no | yes | yes | no |
| `hf_craft` | [`bigbio/craft`](https://huggingface.co/datasets/bigbio/craft) |  | default | unknown upstream split size | biomedical | classification | exactMatch | no | yes | yes | no |
| `hf_ddi_corpus_2013` | [`OpenMed/DDI-Corpus-Processed`](https://huggingface.co/datasets/OpenMed/DDI-Corpus-Processed) |  | test | unknown upstream split size | biomedical | classification | exactMatch | no | yes | yes | no |
| `hf_discoverybench_biomedical` | [`allenai/discoverybench`](https://huggingface.co/datasets/allenai/discoverybench) |  | train | unknown upstream split size | biomedical | qa | openText | no | yes | yes | no |
| `hf_ebm_nlp` | [`bigbio/ebm_pico`](https://huggingface.co/datasets/bigbio/ebm_pico) | processed | test | unknown upstream split size | biomedical | qa | openText | no | yes | yes | no |
| `hf_evidence_inference` | [`hpi-dhc/evidence-inference-simple`](https://huggingface.co/datasets/hpi-dhc/evidence-inference-simple) |  | test | unknown upstream split size | biomedical | classification | exactMatch | no | yes | yes | no |
| `hf_fgbench` | [`xuan-liu/FGBench`](https://huggingface.co/datasets/xuan-liu/FGBench) |  | test | unknown upstream split size | chemistry | classification | exactMatch | no | yes | yes | no |
| `hf_fluorescence_prediction` | [`proteinglm/fluorescence_prediction`](https://huggingface.co/datasets/proteinglm/fluorescence_prediction) |  | default | unknown upstream split size | protein | classification | exactMatch | no | yes | yes | no |
| `hf_gad` | [`bigbio/gad`](https://huggingface.co/datasets/bigbio/gad) |  | default | unknown upstream split size | biomedical | classification | exactMatch | no | yes | yes | no |
| `hf_gaianet_chemistry` | [`gaianet/chemistry`](https://huggingface.co/datasets/gaianet/chemistry) |  | default | unknown upstream split size | chemistry | text | openText | no | yes | yes | no |
| `hf_genbio_proteingym_dms` | [`genbio-ai/ProteinGYM-DMS`](https://huggingface.co/datasets/genbio-ai/ProteinGYM-DMS) |  | default | unknown upstream split size | protein | protein_fitness | exactNumeric | no | yes | yes | no |
| `hf_geneturing` | [`vladimire/geneturing`](https://huggingface.co/datasets/vladimire/geneturing) | all | test | unknown upstream split size | genomics | qa | openText | no | yes | yes | no |
| `hf_genomics_long_range` | [`InstaDeepAI/genomics-long-range-benchmark`](https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark) |  | default | unknown upstream split size | dna | classification | exactMatch | no | yes | yes | no |
| `hf_hallmarks_of_cancer` | [`bigbio/hallmarks_of_cancer`](https://huggingface.co/datasets/bigbio/hallmarks_of_cancer) |  | default | unknown upstream split size | biomedical | classification | exactMatch | no | yes | yes | no |
| `hf_headqa` | [`openlifescienceai/headqa`](https://huggingface.co/datasets/openlifescienceai/headqa) |  | test | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |
| `hf_healthqa` | [`nlplabtdtu/health_qa`](https://huggingface.co/datasets/nlplabtdtu/health_qa) |  | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_icml2022_proteingym` | [`ICML2022/ProteinGym`](https://huggingface.co/datasets/ICML2022/ProteinGym) |  | default | unknown upstream split size | protein | protein_fitness | exactNumeric | no | yes | yes | no |
| `hf_jnlpba` | [`EMBO/BLURB`](https://huggingface.co/datasets/EMBO/BLURB) |  | test | unknown upstream split size | biomedical | classification | exactMatch | no | yes | yes | no |
| `hf_katielink_moleculenet_bace` | [`katielink/moleculenet-benchmark`](https://huggingface.co/datasets/katielink/moleculenet-benchmark) | bace | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_katielink_moleculenet_bbbp` | [`katielink/moleculenet-benchmark`](https://huggingface.co/datasets/katielink/moleculenet-benchmark) | bbbp | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_katielink_moleculenet_clintox` | [`katielink/moleculenet-benchmark`](https://huggingface.co/datasets/katielink/moleculenet-benchmark) | clintox | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_katielink_moleculenet_esol` | [`katielink/moleculenet-benchmark`](https://huggingface.co/datasets/katielink/moleculenet-benchmark) | esol | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_katielink_moleculenet_freesolv` | [`katielink/moleculenet-benchmark`](https://huggingface.co/datasets/katielink/moleculenet-benchmark) | freesolv | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_katielink_moleculenet_hiv` | [`katielink/moleculenet-benchmark`](https://huggingface.co/datasets/katielink/moleculenet-benchmark) | hiv | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_katielink_moleculenet_lipo` | [`katielink/moleculenet-benchmark`](https://huggingface.co/datasets/katielink/moleculenet-benchmark) |  | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_katielink_moleculenet_sider` | [`katielink/moleculenet-benchmark`](https://huggingface.co/datasets/katielink/moleculenet-benchmark) | sider | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_katielink_moleculenet_tox21` | [`katielink/moleculenet-benchmark`](https://huggingface.co/datasets/katielink/moleculenet-benchmark) | tox21 | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_litcovid` | [`ncats/litcovid`](https://huggingface.co/datasets/ncats/litcovid) |  | validation | unknown upstream split size | biomedical | classification | exactMatch | no | yes | yes | no |
| `hf_liveqa_med` | [`hyesunyun/liveqa_medical_trec2017`](https://huggingface.co/datasets/hyesunyun/liveqa_medical_trec2017) |  | test | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_longhealth` | [`tonychenxyz/longhealth`](https://huggingface.co/datasets/tonychenxyz/longhealth) | plain | test | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |
| `hf_lpm24_eval_caption` | [`language-plus-molecules/LPM-24_eval-caption`](https://huggingface.co/datasets/language-plus-molecules/LPM-24_eval-caption) |  | default | unknown upstream split size | chemistry | qa | openText | no | yes | yes | no |
| `hf_lpm24_eval_molgen` | [`language-plus-molecules/LPM-24_eval-molgen`](https://huggingface.co/datasets/language-plus-molecules/LPM-24_eval-molgen) |  | default | unknown upstream split size | chemistry | qa | openText | no | yes | yes | no |
| `hf_medcase_reasoning` | [`zou-lab/MedCaseReasoning`](https://huggingface.co/datasets/zou-lab/MedCaseReasoning) |  | test | unknown upstream split size | clinical | qa | openText | no | yes | yes | no |
| `hf_medconceptsqa` | [`ofir408/MedConceptsQA`](https://huggingface.co/datasets/ofir408/MedConceptsQA) | all | default | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |
| `hf_meddialogqa` | [`UCSD26/medical_dialog`](https://huggingface.co/datasets/UCSD26/medical_dialog) |  | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_medexqa` | [`bluesky333/MedExQA`](https://huggingface.co/datasets/bluesky333/MedExQA) |  | default | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |
| `hf_medical_question_pairs` | [`curaihealth/medical_questions_pairs`](https://huggingface.co/datasets/curaihealth/medical_questions_pairs) |  | default | unknown upstream split size | medical | pair_classification | exactMatch | no | yes | yes | no |
| `hf_medication_qa` | [`truehealth/medicationqa`](https://huggingface.co/datasets/truehealth/medicationqa) |  | train | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_medmcqa_explanations` | [`openlifescienceai/medmcqa`](https://huggingface.co/datasets/openlifescienceai/medmcqa) |  | validation | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |
| `hf_mednli` | [`araag2/MedNLI`](https://huggingface.co/datasets/araag2/MedNLI) | processed | test | unknown upstream split size | clinical | classification | exactMatch | no | yes | yes | no |
| `hf_medpalm_eval_set` | [`katielink/healthsearchqa`](https://huggingface.co/datasets/katielink/healthsearchqa) | 140_question_subset | train | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_medpub_qa` | [`qiaojin/PubMedQA`](https://huggingface.co/datasets/qiaojin/PubMedQA) | pqa_labeled | train | unknown upstream split size | biomedical | mcq | multipleChoice | no | yes | yes | no |
| `hf_medqa_taiwan` | [`xuxuxuxuxu/MedQA_Taiwan_test`](https://huggingface.co/datasets/xuxuxuxuxu/MedQA_Taiwan_test) |  | default | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |
| `hf_medquad` | [`keivalya/MedQuad-MedicalQnADataset`](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset) |  | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_meds_bench` | [`Henrychur/MedS-Bench`](https://huggingface.co/datasets/Henrychur/MedS-Bench) |  | default | unknown upstream split size | medical | qa | openText | no | yes | yes | no |
| `hf_meqsum` | [`albertvillanova/meqsum`](https://huggingface.co/datasets/albertvillanova/meqsum) |  | train | unknown upstream split size | medical | summarization | openText | no | yes | yes | no |
| `hf_mol_instructions_pubchemqa` | [`zjunlp/Mol-Instructions`](https://huggingface.co/datasets/zjunlp/Mol-Instructions) |  | default | unknown upstream split size | chemistry | qa | openText | no | yes | yes | no |
| `hf_moleculeace` | [`karina-zadorozhny/moleculeace`](https://huggingface.co/datasets/karina-zadorozhny/moleculeace) | CHEMBL1862_Ki | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculeace_chembl1871_ki` | [`karina-zadorozhny/moleculeace`](https://huggingface.co/datasets/karina-zadorozhny/moleculeace) | CHEMBL1871_Ki | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculeace_chembl204_ki` | [`karina-zadorozhny/moleculeace`](https://huggingface.co/datasets/karina-zadorozhny/moleculeace) | CHEMBL204_Ki | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculeace_chembl214_ki` | [`karina-zadorozhny/moleculeace`](https://huggingface.co/datasets/karina-zadorozhny/moleculeace) | CHEMBL214_Ki | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculeace_chembl228_ki` | [`karina-zadorozhny/moleculeace`](https://huggingface.co/datasets/karina-zadorozhny/moleculeace) | CHEMBL228_Ki | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculeace_chembl237_ec50` | [`karina-zadorozhny/moleculeace`](https://huggingface.co/datasets/karina-zadorozhny/moleculeace) | CHEMBL237_EC50 | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculenet_bace` | [`scikit-fingerprints/MoleculeNet_BACE`](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_BACE) |  | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculenet_bbbp` | [`scikit-fingerprints/MoleculeNet_BBBP`](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_BBBP) |  | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculenet_clintox` | [`scikit-fingerprints/MoleculeNet_ClinTox`](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_ClinTox) |  | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculenet_esol` | [`scikit-fingerprints/MoleculeNet_ESOL`](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_ESOL) |  | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculenet_freesolv` | [`scikit-fingerprints/MoleculeNet_FreeSolv`](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_FreeSolv) |  | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculenet_hiv` | [`scikit-fingerprints/MoleculeNet_HIV`](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_HIV) |  | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculenet_lipophilicity` | [`scikit-fingerprints/MoleculeNet_Lipophilicity`](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_Lipophilicity) |  | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculenet_pcba` | [`scikit-fingerprints/MoleculeNet_PCBA`](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_PCBA) |  | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculenet_sider` | [`scikit-fingerprints/MoleculeNet_SIDER`](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_SIDER) |  | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_moleculenet_toxcast` | [`scikit-fingerprints/MoleculeNet_ToxCast`](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_ToxCast) |  | default | unknown upstream split size | chemistry | molecule_property | exactMatch | no | yes | yes | no |
| `hf_mollangbench` | [`ChemFM/MolLangBench`](https://huggingface.co/datasets/ChemFM/MolLangBench) |  | default | unknown upstream split size | chemistry | qa | openText | no | yes | yes | no |
| `hf_ms2` | [`allenai/mslr2022`](https://huggingface.co/datasets/allenai/mslr2022) |  | validation | unknown upstream split size | biomedical | summarization | openText | no | yes | yes | no |
| `hf_mteb_medical_qa` | [`mteb/medical_qa`](https://huggingface.co/datasets/mteb/medical_qa) |  | default | unknown upstream split size | medical | retrieval | openText | no | yes | yes | no |
| `hf_mteb_medical_retrieval` | [`mteb/MedicalRetrieval`](https://huggingface.co/datasets/mteb/MedicalRetrieval) |  | default | unknown upstream split size | medical | retrieval | openText | no | yes | yes | no |
| `hf_mts_dialogue_clinical_note` | [`har1/MTS_Dialogue-Clinical_Note`](https://huggingface.co/datasets/har1/MTS_Dialogue-Clinical_Note) |  | default | unknown upstream split size | clinical | summarization | openText | no | yes | yes | no |
| `hf_ncbi_disease` | [`EMBO/BLURB`](https://huggingface.co/datasets/EMBO/BLURB) |  | test | unknown upstream split size | biomedical | classification | exactMatch | no | yes | yes | no |
| `hf_nlmchem` | [`jablonkagroup/nlmchem`](https://huggingface.co/datasets/jablonkagroup/nlmchem) | instruction_0 | test | unknown upstream split size | biomedical | qa | openText | no | yes | yes | no |
| `hf_pgr` | [`lasigeBioTM/PGR`](https://huggingface.co/datasets/lasigeBioTM/PGR) |  | test | unknown upstream split size | biomedical | classification | exactMatch | no | yes | yes | no |
| `hf_ppi_benchmark` | [`bigbio/bioinfer`](https://huggingface.co/datasets/bigbio/bioinfer) |  | test | unknown upstream split size | protein | classification | exactMatch | no | yes | yes | no |
| `hf_protein_binding_sequences` | [`ronig/protein_binding_sequences`](https://huggingface.co/datasets/ronig/protein_binding_sequences) |  | default | unknown upstream split size | protein | qa | openText | no | yes | yes | no |
| `hf_protein_deeploc` | [`proteinea/deeploc`](https://huggingface.co/datasets/proteinea/deeploc) |  | default | unknown upstream split size | protein | classification | exactMatch | no | yes | yes | no |
| `hf_protein_fluorescence` | [`proteinea/fluorescence`](https://huggingface.co/datasets/proteinea/fluorescence) |  | default | unknown upstream split size | protein | regression | exactNumeric | no | yes | yes | no |
| `hf_protein_secondary_structure` | [`lamm-mit/protein_secondary_structure_from_PDB`](https://huggingface.co/datasets/lamm-mit/protein_secondary_structure_from_PDB) |  | default | unknown upstream split size | protein | classification | exactMatch | no | yes | yes | no |
| `hf_protein_solubility` | [`proteinea/solubility`](https://huggingface.co/datasets/proteinea/solubility) |  | default | unknown upstream split size | protein | classification | exactMatch | no | yes | yes | no |
| `hf_protein_stability` | [`SaProtHub/Dataset-Meta-scale-protein-stability`](https://huggingface.co/datasets/SaProtHub/Dataset-Meta-scale-protein-stability) |  | default | unknown upstream split size | protein | regression | exactNumeric | no | yes | yes | no |
| `hf_proteingym_v01` | [`OATML-Markslab/ProteinGym_v0.1`](https://huggingface.co/datasets/OATML-Markslab/ProteinGym_v0.1) |  | default | unknown upstream split size | protein | protein_fitness | exactNumeric | no | yes | yes | no |
| `hf_proteingym_v1` | [`OATML-Markslab/ProteinGym_v1`](https://huggingface.co/datasets/OATML-Markslab/ProteinGym_v1) |  | default | unknown upstream split size | protein | protein_fitness | exactNumeric | no | yes | yes | no |
| `hf_proteinlmbench` | [`tsynbio/ProteinLMBench`](https://huggingface.co/datasets/tsynbio/ProteinLMBench) | evaluation | train | unknown upstream split size | protein | mcq | multipleChoice | no | yes | yes | no |
| `hf_proteinlmbench_enzyme_cot` | [`tsynbio/ProteinLMBench`](https://huggingface.co/datasets/tsynbio/ProteinLMBench) | Enzyme_CoT | default | unknown upstream split size | protein | mcq | multipleChoice | no | yes | yes | no |
| `hf_proteinlmbench_uniprot_disease` | [`tsynbio/ProteinLMBench`](https://huggingface.co/datasets/tsynbio/ProteinLMBench) | UniProt_Involvement in disease | default | unknown upstream split size | protein | mcq | multipleChoice | no | yes | yes | no |
| `hf_proteinlmbench_uniprot_function` | [`tsynbio/ProteinLMBench`](https://huggingface.co/datasets/tsynbio/ProteinLMBench) | UniProt_Function | default | unknown upstream split size | protein | mcq | multipleChoice | no | yes | yes | no |
| `hf_proteinlmbench_uniprot_induction` | [`tsynbio/ProteinLMBench`](https://huggingface.co/datasets/tsynbio/ProteinLMBench) | UniProt_Induction | default | unknown upstream split size | protein | mcq | multipleChoice | no | yes | yes | no |
| `hf_proteinlmbench_uniprot_ptm` | [`tsynbio/ProteinLMBench`](https://huggingface.co/datasets/tsynbio/ProteinLMBench) | UniProt_Post-translational modification | default | unknown upstream split size | protein | mcq | multipleChoice | no | yes | yes | no |
| `hf_proteinlmbench_uniprot_subunit` | [`tsynbio/ProteinLMBench`](https://huggingface.co/datasets/tsynbio/ProteinLMBench) | UniProt_Subunit structure | default | unknown upstream split size | protein | mcq | multipleChoice | no | yes | yes | no |
| `hf_proteinlmbench_uniprot_tissue` | [`tsynbio/ProteinLMBench`](https://huggingface.co/datasets/tsynbio/ProteinLMBench) | UniProt_Tissue specificity | default | unknown upstream split size | protein | mcq | multipleChoice | no | yes | yes | no |
| `hf_pubmed_200k_rct` | [`pietrolesci/pubmed-200k-rct`](https://huggingface.co/datasets/pietrolesci/pubmed-200k-rct) |  | default | unknown upstream split size | biomedical | classification | exactMatch | no | yes | yes | no |
| `hf_pubmed_abstract_classification` | [`uiyunkim-hub/pubmed-abstract`](https://huggingface.co/datasets/uiyunkim-hub/pubmed-abstract) |  | default | unknown upstream split size | biomedical | classification | exactMatch | no | yes | yes | no |
| `hf_raredis` | [`guan-wang/ReDis-QA`](https://huggingface.co/datasets/guan-wang/ReDis-QA) |  | test | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |
| `hf_rna_downstream_tasks` | [`genbio-ai/rna-downstream-tasks`](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks) | modification_site | test | unknown upstream split size | rna | classification | exactMatch | no | yes | yes | no |
| `hf_rna_expression_hek` | [`genbio-ai/rna-downstream-tasks`](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks) | expression_HEK | default | unknown upstream split size | rna | regression | exactNumeric | no | yes | yes | no |
| `hf_rna_expression_muscle` | [`genbio-ai/rna-downstream-tasks`](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks) | expression_Muscle | default | unknown upstream split size | rna | regression | exactNumeric | no | yes | yes | no |
| `hf_rna_expression_pc3` | [`genbio-ai/rna-downstream-tasks`](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks) | expression_pc3 | default | unknown upstream split size | rna | regression | exactNumeric | no | yes | yes | no |
| `hf_rna_mean_ribosome_load` | [`genbio-ai/rna-downstream-tasks`](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks) | mean_ribosome_load | default | unknown upstream split size | rna | regression | exactNumeric | no | yes | yes | no |
| `hf_rna_modification_site` | [`genbio-ai/rna-downstream-tasks`](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks) | modification_site | default | unknown upstream split size | rna | classification | exactMatch | no | yes | yes | no |
| `hf_rna_ncrna_family_bnoise0` | [`genbio-ai/rna-downstream-tasks`](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks) | ncrna_family_bnoise0 | default | unknown upstream split size | rna | classification | exactMatch | no | yes | yes | no |
| `hf_rna_splice_site_acceptor` | [`genbio-ai/rna-downstream-tasks`](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks) | splice_site_acceptor | default | unknown upstream split size | rna | classification | exactMatch | no | yes | yes | no |
| `hf_rna_splice_site_donor` | [`genbio-ai/rna-downstream-tasks`](https://huggingface.co/datasets/genbio-ai/rna-downstream-tasks) | splice_site_donor | default | unknown upstream split size | rna | classification | exactMatch | no | yes | yes | no |
| `hf_smiles_caption_mol2text` | [`zjunlp/Mol-Instructions`](https://huggingface.co/datasets/zjunlp/Mol-Instructions) |  | default | unknown upstream split size | chemistry | qa | openText | no | yes | yes | no |
| `hf_traitgym_mendelian_dna` | [`bolinas-dna/evals-traitgym_mendelian_v2_harness_255`](https://huggingface.co/datasets/bolinas-dna/evals-traitgym_mendelian_v2_harness_255) |  | default | unknown upstream split size | dna | classification | exactMatch | no | yes | yes | no |
| `hf_uspto_reaction_prediction` | [`bing-yan/USPTO`](https://huggingface.co/datasets/bing-yan/USPTO) |  | test | unknown upstream split size | chemistry | qa | openText | no | yes | yes | no |

### Deprecated aliases

| Alias | Source | Config | Split | Count | Domain | Task type | Answer/scorer | Gated | Network | Offline cache | Multimodal |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `hf_blue_benchmark` -> `hf_blurb` | [`EMBO/BLURB`](https://huggingface.co/datasets/EMBO/BLURB) |  | test | unknown upstream split size | biomedical | classification | exactMatch | no | yes | yes | no |
| `hf_chinese_medbench` -> `hf_cmb` | [`FreedomIntelligence/CMB`](https://huggingface.co/datasets/FreedomIntelligence/CMB) | CMB-Exam | test | unknown upstream split size | medical | mcq | multipleChoice | no | yes | yes | no |
| `hf_lavita_medmcqa` -> `medmcqa` | core benchmark alias |  | default | loader default | medical | mcq | multipleChoice | no | source-dependent | yes | no |
| `hf_lavita_usmle_step1` -> `medqa` | core benchmark alias |  | default | loader default | medical | mcq | multipleChoice | no | source-dependent | yes | no |
| `hf_lavita_usmle_step2` -> `medqa` | core benchmark alias |  | default | loader default | medical | mcq | multipleChoice | no | source-dependent | yes | no |
| `hf_lavita_usmle_step3` -> `medqa` | core benchmark alias |  | default | loader default | medical | mcq | multipleChoice | no | source-dependent | yes | no |
| `hf_mednli_augmented` -> `hf_mednli` | [`araag2/MedNLI`](https://huggingface.co/datasets/araag2/MedNLI) | processed | test | unknown upstream split size | clinical | classification | exactMatch | no | yes | yes | no |
| `hf_openddi` -> `hf_ddi_corpus_2013` | [`OpenMed/DDI-Corpus-Processed`](https://huggingface.co/datasets/OpenMed/DDI-Corpus-Processed) |  | test | unknown upstream split size | biomedical | classification | exactMatch | no | yes | yes | no |
| `hf_pubmed_20k_rct` -> `hf_pubmed_200k_rct` | [`pietrolesci/pubmed-200k-rct`](https://huggingface.co/datasets/pietrolesci/pubmed-200k-rct) |  | default | unknown upstream split size | biomedical | classification | exactMatch | no | yes | yes | no |
| `hf_pubmed_rct20k` -> `hf_pubmed_200k_rct` | [`pietrolesci/pubmed-200k-rct`](https://huggingface.co/datasets/pietrolesci/pubmed-200k-rct) |  | default | unknown upstream split size | biomedical | classification | exactMatch | no | yes | yes | no |
| `hf_usmle_step_series` -> `medqa` | core benchmark alias |  | default | loader default | medical | mcq | multipleChoice | no | source-dependent | yes | no |
