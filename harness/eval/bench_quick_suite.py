"""BioMedArena-Eval quick suite.

Twenty tiny, deterministic, offline tasks for install-time smoke tests.
The suite intentionally avoids HuggingFace, external vendors, network
tools, and API keys so users can validate loader/scorer wiring in under
two minutes.
"""
from __future__ import annotations

from typing import Any


_TASKS: tuple[dict[str, Any], ...] = (
    {
        "id": "quick_mcq_001",
        "question": (
            "Which molecule stores hereditary genetic information in most "
            "living organisms?\n\nA. DNA\nB. ATP\nC. Glucose\nD. Hemoglobin"
        ),
        "answer": "A",
        "answer_type": "multipleChoice",
        "category": "quick_suite/biology",
        "raw_subject": "biology",
    },
    {
        "id": "quick_mcq_002",
        "question": (
            "Which organelle is the primary site of oxidative phosphorylation?\n\n"
            "A. Lysosome\nB. Mitochondrion\nC. Golgi apparatus\nD. Nucleus"
        ),
        "answer": "B",
        "answer_type": "multipleChoice",
        "category": "quick_suite/cell_biology",
        "raw_subject": "cell_biology",
    },
    {
        "id": "quick_mcq_003",
        "question": (
            "Which base pairs with adenine in DNA?\n\n"
            "A. Cytosine\nB. Guanine\nC. Thymine\nD. Uracil"
        ),
        "answer": "C",
        "answer_type": "multipleChoice",
        "category": "quick_suite/genetics",
        "raw_subject": "genetics",
    },
    {
        "id": "quick_mcq_004",
        "question": (
            "A competitive inhibitor primarily changes which Michaelis-Menten "
            "parameter?\n\nA. Decreases Km\nB. Increases apparent Km\n"
            "C. Increases Vmax\nD. Decreases enzyme concentration"
        ),
        "answer": "B",
        "answer_type": "multipleChoice",
        "category": "quick_suite/biochemistry",
        "raw_subject": "biochemistry",
    },
    {
        "id": "quick_mcq_005",
        "question": (
            "Which immune cell type produces antibodies?\n\n"
            "A. B cell\nB. Neutrophil\nC. Erythrocyte\nD. Platelet"
        ),
        "answer": "A",
        "answer_type": "multipleChoice",
        "category": "quick_suite/immunology",
        "raw_subject": "immunology",
    },
    {
        "id": "quick_exact_001",
        "question": "What three-letter codon commonly starts translation?",
        "answer": "AUG",
        "answer_type": "exactMatch",
        "category": "quick_suite/molecular_biology",
        "raw_subject": "molecular_biology",
    },
    {
        "id": "quick_exact_002",
        "question": "Name the blood protein that transports oxygen.",
        "answer": "hemoglobin",
        "answer_type": "exactMatch",
        "category": "quick_suite/physiology",
        "raw_subject": "physiology",
    },
    {
        "id": "quick_exact_003",
        "question": "What is the one-letter amino acid code for glycine?",
        "answer": "G",
        "answer_type": "exactMatch",
        "category": "quick_suite/protein",
        "raw_subject": "protein",
    },
    {
        "id": "quick_exact_004",
        "question": "Which nucleotide base is present in RNA but not DNA?",
        "answer": "uracil",
        "answer_type": "exactMatch",
        "category": "quick_suite/rna",
        "raw_subject": "rna",
    },
    {
        "id": "quick_exact_005",
        "question": "What macromolecule class are enzymes usually made of?",
        "answer": "protein",
        "answer_type": "exactMatch",
        "category": "quick_suite/biochemistry",
        "raw_subject": "biochemistry",
    },
    {
        "id": "quick_numeric_001",
        "question": "A solution has pH 7. What is [H+] in mol/L?",
        "answer": "1e-7",
        "answer_type": "exactNumeric",
        "category": "quick_suite/chemistry",
        "raw_subject": "chemistry",
    },
    {
        "id": "quick_numeric_002",
        "question": "A patient weighs 70 kg. Convert this weight to grams.",
        "answer": "70000",
        "answer_type": "exactNumeric",
        "category": "quick_suite/clinical_calculation",
        "raw_subject": "clinical_calculation",
    },
    {
        "id": "quick_numeric_003",
        "question": "If 2 mL contains 10 mg drug, what is the concentration in mg/mL?",
        "answer": "5",
        "answer_type": "exactNumeric",
        "category": "quick_suite/clinical_calculation",
        "raw_subject": "clinical_calculation",
    },
    {
        "id": "quick_numeric_004",
        "question": "What decimal is equal to one half?",
        "answer": "0.5",
        "answer_type": "exactNumeric",
        "category": "quick_suite/math",
        "raw_subject": "math",
    },
    {
        "id": "quick_numeric_005",
        "question": "A DNA fragment has 30 adenines and 30 thymines. How many A-T base pairs is that?",
        "answer": "30",
        "answer_type": "exactNumeric",
        "category": "quick_suite/genetics",
        "raw_subject": "genetics",
    },
    {
        "id": "quick_open_001",
        "question": "Briefly define PCR.",
        "answer": "polymerase chain reaction amplifies DNA",
        "answer_type": "openText",
        "category": "quick_suite/molecular_biology",
        "raw_subject": "molecular_biology",
    },
    {
        "id": "quick_open_002",
        "question": "What is the main purpose of a negative control in an experiment?",
        "answer": "detect background signal or contamination",
        "answer_type": "openText",
        "category": "quick_suite/experimental_design",
        "raw_subject": "experimental_design",
    },
    {
        "id": "quick_open_003",
        "question": "What does an odds ratio greater than 1 usually indicate?",
        "answer": "higher odds of the outcome in the exposed group",
        "answer_type": "openText",
        "category": "quick_suite/statistics",
        "raw_subject": "statistics",
    },
    {
        "id": "quick_open_004",
        "question": "What is the central dogma of molecular biology?",
        "answer": "DNA is transcribed to RNA and RNA is translated to protein",
        "answer_type": "openText",
        "category": "quick_suite/molecular_biology",
        "raw_subject": "molecular_biology",
    },
    {
        "id": "quick_open_005",
        "question": "What does ELISA commonly measure?",
        "answer": "antigen or antibody concentration",
        "answer_type": "openText",
        "category": "quick_suite/assay",
        "raw_subject": "assay",
    },
)


def load_quick_suite_tasks(
    limit: int | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic offline smoke-test tasks.

    Args:
        limit: optional cap after category filtering.
        category: optional exact category suffix, e.g. ``"chemistry"`` or
            full category ``"quick_suite/chemistry"``.
    """
    wanted = None
    if category:
        wanted = category if category.startswith("quick_suite/") else f"quick_suite/{category}"

    tasks: list[dict[str, Any]] = []
    for task in _TASKS:
        if wanted and task["category"] != wanted:
            continue
        item = dict(task)
        item["context"] = {
            "source": "BioMedArena-Eval quick suite",
            "offline": True,
            "requires_network": False,
            "requires_token": False,
            "input_type": "text",
        }
        item["metadata"] = {
            "source": "quick_suite_builtin",
            "license": "MIT",
            "smoke_test": True,
        }
        tasks.append(item)
        if limit and len(tasks) >= limit:
            break
    return tasks
