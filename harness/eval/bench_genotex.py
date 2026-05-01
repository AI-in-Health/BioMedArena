"""Genomic QA loader — converts gene expression analysis tasks to QA format.

We convert trait-gene association tasks into QA: "What genes are associated
with [trait]?" evaluated against the curated related_genes lists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Representative gene-trait association tasks covering multiple disease
# categories. Each question asks about gene associations, pathway mechanisms,
# or expression analysis interpretation.
GENOTEX_TASKS: list[dict[str, Any]] = [
    # --- Cardiovascular ---
    {
        "id": "gtx_001", "category": "Genomics/Cardiovascular",
        "answer_type": "multipleChoice",
        "question": "In a genome-wide association study of coronary artery disease, which gene locus on chromosome 9p21.3 has been most consistently identified as the strongest common genetic risk factor?\nA) CDKN2A/CDKN2B (encoding p16/p15 tumor suppressors)\nB) APOE (apolipoprotein E)\nC) PCSK9 (proprotein convertase subtilisin/kexin type 9)\nD) LDLR (LDL receptor)\nE) MTHFR (methylenetetrahydrofolate reductase)",
        "answer": "A",
    },
    {
        "id": "gtx_002", "category": "Genomics/Cardiovascular",
        "answer_type": "exactMatch",
        "question": "Gene expression analysis of cardiac tissue from heart failure patients consistently shows upregulation of natriuretic peptide genes. Which gene encodes BNP (B-type natriuretic peptide), the protein whose blood levels are used as a clinical biomarker for heart failure?",
        "answer": "NPPB",
    },
    {
        "id": "gtx_003", "category": "Genomics/Cardiovascular",
        "answer_type": "multipleChoice",
        "question": "Familial hypercholesterolemia follows autosomal dominant inheritance and is primarily caused by mutations in three genes. Which combination is correct?\nA) LDLR, APOB, PCSK9\nB) LDLR, HMGCR, APOE\nC) APOB, CETP, LCAT\nD) PCSK9, NPC1L1, ABCG5",
        "answer": "A",
    },
    # --- Cancer/Oncology ---
    {
        "id": "gtx_010", "category": "Genomics/Cancer",
        "answer_type": "exactMatch",
        "question": "In differential gene expression analysis of breast cancer vs. normal tissue, which estrogen receptor gene shows the highest overexpression in luminal A subtype tumors and is the primary target for tamoxifen therapy?",
        "answer": "ESR1",
    },
    {
        "id": "gtx_011", "category": "Genomics/Cancer",
        "answer_type": "multipleChoice",
        "question": "A gene expression profiling study of colorectal cancer identified a significantly mutated oncogene present in ~40% of cases. Activating mutations in this gene (codons 12, 13, 61) predict resistance to anti-EGFR therapy. Which gene?\nA) KRAS\nB) BRAF\nC) PIK3CA\nD) APC\nE) TP53",
        "answer": "A",
    },
    {
        "id": "gtx_012", "category": "Genomics/Cancer",
        "answer_type": "exactMatch",
        "question": "In chronic myeloid leukemia, the Philadelphia chromosome translocation t(9;22) creates a fusion gene. What is the name of this fusion gene?",
        "answer": "BCR-ABL1",
    },
    {
        "id": "gtx_013", "category": "Genomics/Cancer",
        "answer_type": "multipleChoice",
        "question": "Gene expression analysis of non-small cell lung cancer identified a subset with activating mutations in a receptor tyrosine kinase, particularly common in non-smoking Asian females with adenocarcinoma. These patients respond to gefitinib/erlotinib. Which gene?\nA) EGFR\nB) ALK\nC) ROS1\nD) MET\nE) HER2",
        "answer": "A",
    },
    # --- Neurological ---
    {
        "id": "gtx_020", "category": "Genomics/Neurology",
        "answer_type": "exactMatch",
        "question": "The APOE gene has three common alleles: ε2, ε3, and ε4. Which APOE genotype confers the highest risk for late-onset Alzheimer's disease?",
        "answer": "APOE e4/e4",
    },
    {
        "id": "gtx_021", "category": "Genomics/Neurology",
        "answer_type": "multipleChoice",
        "question": "Huntington's disease is caused by a CAG trinucleotide repeat expansion. In which gene, and how many repeats are needed for full penetrance?\nA) HTT gene, ≥40 repeats\nB) FMR1 gene, ≥200 repeats\nC) DMPK gene, ≥50 repeats\nD) ATN1 gene, ≥48 repeats",
        "answer": "A",
    },
    {
        "id": "gtx_022", "category": "Genomics/Neurology",
        "answer_type": "exactMatch",
        "question": "Transcriptomic analysis of Parkinson's disease substantia nigra shows decreased expression of a gene encoding the rate-limiting enzyme in dopamine synthesis. What is this gene?",
        "answer": "TH",
    },
    # --- Immunology/Autoimmune ---
    {
        "id": "gtx_030", "category": "Genomics/Immunology",
        "answer_type": "multipleChoice",
        "question": "HLA gene association studies of autoimmune diseases show the strongest known genetic association in medicine is between ankylosing spondylitis and which HLA allele?\nA) HLA-B27\nB) HLA-DR4\nC) HLA-DQ2\nD) HLA-B51\nE) HLA-DR3",
        "answer": "A",
    },
    {
        "id": "gtx_031", "category": "Genomics/Immunology",
        "answer_type": "exactMatch",
        "question": "Celiac disease has a strong HLA association. What are the two HLA class II molecules that together account for nearly all genetic susceptibility to celiac disease?",
        "answer": "HLA-DQ2 and HLA-DQ8",
    },
    # --- Metabolic/Endocrine ---
    {
        "id": "gtx_040", "category": "Genomics/Metabolic",
        "answer_type": "exactMatch",
        "question": "In MODY (maturity-onset diabetes of the young), the most common form (MODY3) is caused by mutations in which transcription factor gene?",
        "answer": "HNF1A",
    },
    {
        "id": "gtx_041", "category": "Genomics/Metabolic",
        "answer_type": "multipleChoice",
        "question": "Genome-wide association studies have identified >400 loci for Type 2 diabetes. Which gene, encoding a zinc transporter in pancreatic beta cells, was one of the first T2D-associated genes discovered via GWAS?\nA) SLC30A8\nB) TCF7L2\nC) KCNJ11\nD) PPARG\nE) IRS1",
        "answer": "A",
    },
    # --- Pharmacogenomics ---
    {
        "id": "gtx_050", "category": "Genomics/Pharmacogenomics",
        "answer_type": "exactMatch",
        "question": "The FDA requires HLA-B*57:01 testing before prescribing which antiretroviral drug, due to risk of severe hypersensitivity reaction?",
        "answer": "abacavir",
    },
    {
        "id": "gtx_051", "category": "Genomics/Pharmacogenomics",
        "answer_type": "multipleChoice",
        "question": "CYP2D6 poor metabolizers have significantly reduced efficacy of the breast cancer drug tamoxifen because CYP2D6 is required to convert tamoxifen to its active metabolite. What is this active metabolite?\nA) Endoxifen\nB) 4-hydroxytamoxifen\nC) N-desmethyltamoxifen\nD) Norendoxifen",
        "answer": "A",
    },
    # --- Rare Genetic Diseases ---
    {
        "id": "gtx_060", "category": "Genomics/Rare Disease",
        "answer_type": "exactMatch",
        "question": "Cystic fibrosis is caused by mutations in the CFTR gene. What is the most common mutation, found in ~70% of CF alleles in European populations?",
        "answer": "F508del",
    },
    {
        "id": "gtx_061", "category": "Genomics/Rare Disease",
        "answer_type": "multipleChoice",
        "question": "Sickle cell disease is caused by a specific point mutation in the HBB gene. Which amino acid substitution occurs?\nA) Glutamic acid to valine at position 6 (E6V)\nB) Glutamic acid to lysine at position 6 (E6K)\nC) Valine to glutamic acid at position 6 (V6E)\nD) Glutamic acid to aspartic acid at position 6 (E6D)",
        "answer": "A",
    },
]


def load_genotex_tasks(
    vendor_path: str = "vendors/GenoTEX",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load genomic QA tasks.

    Tries vendor repo first (metadata/task_info.json), converting trait-gene
    associations to QA format. Falls back to built-in tasks. `limit` caps
    the number of returned tasks, matching the signature convention used
    by other bench_*_tasks loaders.
    """
    vendor = Path(vendor_path)
    task_info = vendor / "metadata" / "task_info.json"

    if task_info.exists():
        try:
            raw = json.loads(task_info.read_text())
            tasks = []
            for i, (trait, info) in enumerate(raw.items()):
                genes = info.get("related_genes", [])
                if not genes:
                    continue
                # Convert to QA: ask for top associated genes
                top_genes = genes[:5]
                gene_str = ", ".join(top_genes)
                tasks.append({
                    "id": f"gtx_{i:03d}",
                    "question": (
                        f"In gene expression analysis and GWAS studies, which genes "
                        f"are most strongly associated with {trait}? "
                        f"Name the top {min(5, len(genes))} genes."
                    ),
                    "answer": gene_str,
                    "answer_type": "exactMatch",
                    "category": "Genomics/Expression",
                    "context": {"genes": top_genes, "trait": trait},
                })
            if tasks:
                return tasks[:limit] if limit else tasks
        except Exception:
            pass

    return GENOTEX_TASKS[:limit] if limit else list(GENOTEX_TASKS)
