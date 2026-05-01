"""RAG-Essential benchmark — questions where pure-LLM knowledge is insufficient.

Two task families:

1. RECENCY: Questions whose answers depend on information published AFTER
   common LLM training cutoffs (2024-onward). Without retrieval, the model
   must guess. With retrieval (PubMed, OpenFDA, FDA news), retrieval should
   provide the answer.

2. ORACLE: Questions where pure recall is unreliable but a structured database
   gives a definitive answer (specific drug interactions, exact gene-disease
   associations from OMIM, SNOMED/ICD codes, dosing).

These tasks specifically reward agents that USE TOOLS over those that don't.
"""

from __future__ import annotations

from typing import Any


# Hand-curated question set. Keep concise; LLM-as-judge will do scoring.
_RAG_ESSENTIAL_TASKS: list[dict[str, Any]] = [
    # ====================== RECENCY (2024+ info) ======================
    {
        "id": "rag_rec_001",
        "category": "RAG/Recency",
        "answer_type": "openText",
        "question": (
            "What is the FDA-approved drug donanemab's mechanism of action, "
            "and what year was it approved for Alzheimer's disease? "
            "Be specific about the year."
        ),
        "answer": (
            "Donanemab is a humanized monoclonal antibody that targets aggregated "
            "amyloid beta in the brain (specifically a modified form called N3pG). "
            "It was approved by the FDA in July 2024 for early symptomatic "
            "Alzheimer's disease."
        ),
    },
    {
        "id": "rag_rec_002",
        "category": "RAG/Recency",
        "answer_type": "openText",
        "question": (
            "What 2024 FDA-approved gene therapy treats sickle cell disease using "
            "CRISPR-Cas9 editing? Name the product and approval year."
        ),
        "answer": (
            "Casgevy (exagamglogene autotemcel, exa-cel), developed by Vertex/CRISPR "
            "Therapeutics. Approved by the FDA in December 2023 (for sickle cell "
            "disease) — the first CRISPR-based therapy approved in the US."
        ),
    },
    {
        "id": "rag_rec_003",
        "category": "RAG/Recency",
        "answer_type": "openText",
        "question": (
            "What new class of obesity drugs gained major regulatory expansion in "
            "2024 for cardiovascular benefit beyond weight loss? Name the class "
            "and a representative drug, and the new indication added."
        ),
        "answer": (
            "GLP-1 receptor agonists (or dual GIP/GLP-1 agonists). Semaglutide "
            "(brand: Wegovy / Ozempic) — in March 2024 the FDA approved Wegovy "
            "specifically for reducing major adverse cardiovascular events "
            "(MACE) in adults with obesity and established cardiovascular disease."
        ),
    },
    {
        "id": "rag_rec_004",
        "category": "RAG/Recency",
        "answer_type": "openText",
        "question": (
            "Which 2024-published large multimodal model from Google DeepMind "
            "showed promising medical question-answering performance? "
            "Provide the model name and its key reported capability."
        ),
        "answer": (
            "Med-Gemini (or Gemini-Med) from Google DeepMind. Reported state-of-"
            "the-art performance on MedQA (USMLE) and was the first system to "
            "exceed 90% on USMLE-style questions, with strong multimodal "
            "(image + text) medical reasoning."
        ),
    },

    # ====================== ORACLE (database-precise) ======================
    {
        "id": "rag_orc_001",
        "category": "RAG/Oracle",
        "answer_type": "openText",
        "question": (
            "A patient is on warfarin (target INR 2-3) and develops cellulitis. "
            "Which oral antibiotic class is most likely to cause a CLINICALLY "
            "SIGNIFICANT INR increase (commonly raising INR >4) due to CYP2C9 "
            "inhibition? Provide the class and the mechanism."
        ),
        "answer": (
            "Trimethoprim-sulfamethoxazole (sulfa) — and to a lesser extent "
            "fluoroquinolones (e.g., ciprofloxacin) and macrolides (clarithromycin/"
            "erythromycin). The mechanism is CYP2C9 inhibition reducing warfarin "
            "metabolism, plus displacement from albumin and reduced vitamin K "
            "synthesis by gut flora suppression."
        ),
    },
    {
        "id": "rag_orc_002",
        "category": "RAG/Oracle",
        "answer_type": "openText",
        "question": (
            "What is the Orphanet ORPHA code for Huntington disease, and "
            "what gene/mutation is responsible (gene name + repeat type + "
            "pathogenic threshold)?"
        ),
        "answer": (
            "Orphanet code is ORPHA:399 (Huntington disease). Caused by a "
            "CAG trinucleotide repeat expansion in the HTT gene on chromosome "
            "4p16.3. Pathogenic threshold is ≥40 CAG repeats (full penetrance); "
            "36-39 is reduced penetrance; <27 is normal."
        ),
    },
    {
        "id": "rag_orc_003",
        "category": "RAG/Oracle",
        "answer_type": "openText",
        "question": (
            "What is the renal dose adjustment for apixaban in a patient with "
            "atrial fibrillation, age 80, weight 55 kg, and serum creatinine "
            "1.6 mg/dL? State the specific dose."
        ),
            "answer": (
            "Apixaban 2.5 mg orally twice daily. The patient meets two of the "
            "three apixaban dose-reduction criteria (age ≥80, weight ≤60 kg, "
            "creatinine ≥1.5 mg/dL) — having two or more requires reduction "
            "from the standard 5 mg BID to 2.5 mg BID."
        ),
    },
    {
        "id": "rag_orc_004",
        "category": "RAG/Oracle",
        "answer_type": "openText",
        "question": (
            "What is the OMIM (MIM) number for cystic fibrosis, and what is the "
            "most common pathogenic CFTR variant worldwide?"
        ),
        "answer": (
            "Cystic fibrosis MIM number is 219700. The most common pathogenic "
            "variant in the CFTR gene is F508del (also written as p.Phe508del or "
            "c.1521_1523delCTT) — a 3-base deletion removing phenylalanine at "
            "position 508. Found in roughly 70% of CF alleles in European-"
            "ancestry populations."
        ),
    },
    {
        "id": "rag_orc_005",
        "category": "RAG/Oracle",
        "answer_type": "openText",
        "question": (
            "A 65-year-old patient with eGFR of 35 mL/min/1.73 m² is being "
            "started on metformin. What is the maximum daily dose recommended, "
            "and at what eGFR threshold should metformin be discontinued?"
        ),
        "answer": (
            "At eGFR 30-45, metformin can be continued but the dose should be "
            "halved — maximum 1000 mg/day (commonly given as 500 mg twice daily). "
            "Metformin should be DISCONTINUED if eGFR drops below 30 mL/min/1.73 m². "
            "Initiation is contraindicated below eGFR 45."
        ),
    },
    {
        "id": "rag_orc_006",
        "category": "RAG/Oracle",
        "answer_type": "openText",
        "question": (
            "What pharmacogenomic variant (gene + star allele) is associated with "
            "severe abacavir hypersensitivity, and what is the FDA labelling "
            "requirement before prescribing abacavir?"
        ),
        "answer": (
            "HLA-B*57:01 (HLA-B gene, 5701 allele). FDA black box warning: "
            "screening for the HLA-B*57:01 allele is REQUIRED before initiating "
            "abacavir; positive patients should not receive abacavir due to the "
            "risk of severe, potentially fatal hypersensitivity reaction."
        ),
    },
    {
        "id": "rag_orc_007",
        "category": "RAG/Oracle",
        "answer_type": "openText",
        "question": (
            "A patient on clarithromycin develops new ventricular tachycardia. "
            "Their other medications include simvastatin, amlodipine, and "
            "rivaroxaban. Which co-administered drug is most likely contributing "
            "via a CYP3A4 interaction with clarithromycin, and what is the "
            "clinical risk?"
        ),
        "answer": (
            "Simvastatin. Clarithromycin is a strong CYP3A4 inhibitor and "
            "dramatically increases simvastatin exposure (5-fold or more), "
            "leading to risk of rhabdomyolysis with myoglobinuria and acute "
            "kidney injury. Concomitant use of clarithromycin with simvastatin "
            "is contraindicated; simvastatin should be held while on the macrolide."
        ),
    },
    {
        "id": "rag_orc_008",
        "category": "RAG/Oracle",
        "answer_type": "openText",
        "question": (
            "What is the SNOMED CT code (or describe the standard concept) for "
            "the disorder 'acute myocardial infarction'? Provide the SNOMED "
            "concept ID if known."
        ),
        "answer": (
            "Acute myocardial infarction is SNOMED CT concept ID 57054005 "
            "(also written as SCTID 57054005). It is part of the SNOMED CT "
            "hierarchy under 'Disorder of the cardiovascular system'."
        ),
    },
]


def load_rag_essential_tasks(
    limit: int | None = None,
    families: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load RAG-essential tasks.

    Args:
        limit: cap number of tasks
        families: filter by ['Recency'] or ['Oracle'] or both
    """
    tasks = list(_RAG_ESSENTIAL_TASKS)
    if families:
        tasks = [
            t for t in tasks
            if any(fam.lower() in t["category"].lower() for fam in families)
        ]
    if limit:
        tasks = tasks[:limit]
    return tasks
