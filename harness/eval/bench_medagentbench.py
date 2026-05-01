"""Clinical EHR task loader — FHIR-based workflow tasks.

Vendor data: vendors/MedAgentBench/data/medagentbench/test_data_v2.json (300 tasks)

10 task categories × 30 variants:
- task1: MRN lookup (patient name + DOB → MRN)  [has 'sol' field]
- task2: Calculate patient age
- task3: Record vital signs (POST Observation)
- task4: Query lab results
- task5: Lab-based clinical decision making
- task6: Lab averages over time
- task7: Query medication orders
- task8: Place referrals
- task9: Query procedures
- task10: Other clinical task

For tasks beyond task1, evaluation typically requires a running FHIR server
(Docker). We expose them in OFFLINE mode as text-QA: ask the LLM what API
calls it would make + reasoning. Only task1 has a deterministic 'sol' answer
that can be scored exactly without FHIR.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# Path to the full 300-task JSON inside the cloned vendor
_FULL_TASKS_FILE = Path("vendors/MedAgentBench/data/medagentbench/test_data_v2.json")
_FUNCS_FILE = Path("vendors/MedAgentBench/data/medagentbench/funcs_v1.json")




# Small built-in smoke sample used when the optional vendor dataset is absent.
_REPRESENTATIVE_TASKS: list[dict[str, Any]] = [
    {
        "id": "mab_001", "category": "EHR/Patient Info",
        "answer_type": "exactMatch",
        "question": "A 67-year-old male patient (MRN: 12345) presents to the ED. His chart shows: BP 158/92, HR 88, Temp 37.2°C, SpO2 96%. His active medications include lisinopril 20mg daily, metformin 1000mg BID, and atorvastatin 40mg daily. What is this patient's most likely primary chronic condition based on his medication profile?",
        "answer": "type 2 diabetes",
    },
]


def load_medagentbench_tasks(
    vendor_path: str = "vendors/MedAgentBench",
    use_full_dataset: bool = True,
    task_types: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load clinical EHR tasks.

    Args:
        vendor_path: path to cloned vendor repo
        use_full_dataset: if True, load all 300 tasks from test_data_v2.json
        task_types: optional filter, e.g. ["task1", "task5"]
        limit: cap number of tasks returned
    """
    vendor_dir = Path(vendor_path)
    full_file = vendor_dir / "data" / "medagentbench" / "test_data_v2.json"

    if use_full_dataset and full_file.exists():
        return _load_full_300(full_file, task_types=task_types, limit=limit)

    # Fall back to representative subset
    if not _REPRESENTATIVE_TASKS:
        return []
    return _REPRESENTATIVE_TASKS[:limit] if limit else _REPRESENTATIVE_TASKS


def _load_full_300(
    full_file: Path,
    task_types: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    raw = json.loads(full_file.read_text(encoding="utf-8"))
    tasks: list[dict[str, Any]] = []

    for item in raw:
        task_id = item.get("id", "")
        # task_id format: "task1_5" → category "task1"
        category = task_id.split("_")[0] if "_" in task_id else "task_unknown"
        if task_types and category not in task_types:
            continue

        instruction = item.get("instruction", "")
        context_text = item.get("context", "")
        eval_mrn = item.get("eval_MRN", "")
        sol = item.get("sol", [])

        # Compose question
        question_parts = []
        if context_text:
            question_parts.append(f"Context: {context_text}")
        question_parts.append(f"Task: {instruction}")
        question_parts.append(
            "\nAnswer with the specific value/identifier requested. "
            "End with: The answer is [value]."
        )
        question = "\n\n".join(question_parts)

        # Determine answer
        if sol:
            answer = str(sol[0]) if isinstance(sol, list) else str(sol)
            answer_type = "exactMatch"
        else:
            # No deterministic solution — needs FHIR server to verify
            # In offline mode, use the eval_MRN as a weak signal
            answer = eval_mrn or ""
            answer_type = "openText"

        tasks.append({
            "id": task_id,
            "question": question,
            "answer": answer,
            "answer_type": answer_type,
            "category": f"MedAgentBench/{category}",
            "raw_subject": category,
            "context": {
                "eval_MRN": eval_mrn,
                "task_type": category,
                "has_sol": bool(sol),
                "raw_instruction": instruction,
            },
        })

        if limit and len(tasks) >= limit:
            break

    return tasks
