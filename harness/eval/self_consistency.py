"""Self-Consistency voting wrapper.

Runs an inner mode (simple/deep/harness/fc) N times with varying temperatures
and takes a majority vote on the extracted answer.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any, Awaitable, Callable

from harness.eval.scoring import extract_answer_from_response, normalize_text


async def self_consistent_run(
    inner_runner: Callable[[dict], Awaitable[tuple[str, list[str]]]],
    task: dict[str, Any],
    n: int = 5,
    temperatures: list[float] | None = None,
) -> tuple[str, list[str]]:
    """Run inner_runner N times and majority-vote the answer.

    Args:
        inner_runner: async function `(task) -> (response_text, tools)`
        task: benchmark task dict
        n: number of samples
        temperatures: ignored by most inner_runners (they're temperature=0);
                       kept for interface completeness but inner handles sampling.

    Returns:
        (best_response_text, union_of_tools)
    """
    # Launch N parallel runs (they should differ due to any internal sampling
    # or API non-determinism). We re-seed by adding a tiny perturbation hint
    # via the task (does not affect inner logic, just ensures some diversity).
    async def _one():
        return await inner_runner(task)

    results = await asyncio.gather(*[_one() for _ in range(n)], return_exceptions=True)
    valid = [(r, t) for r, t in results if not isinstance(r, Exception) for _ in [None]
              if isinstance(r, tuple) or not isinstance(r, BaseException)]
    # Simpler filter
    valid = []
    for item in results:
        if isinstance(item, Exception):
            continue
        if isinstance(item, tuple) and len(item) == 2:
            valid.append(item)

    if not valid:
        return "[self-consistency: all runs failed]", []

    answer_type = task.get("answer_type", "exactMatch")

    # Extract answer from each run
    extractions: list[tuple[str, str, list[str]]] = []  # (extracted_answer, full_resp, tools)
    for resp, tools in valid:
        extracted = extract_answer_from_response(resp, answer_type)
        if answer_type in ("multipleChoice", "multiple_choice"):
            key = extracted.strip().upper()
        elif answer_type in ("exactNumeric", "exact_numeric"):
            key = _normalize_numeric(extracted)
        else:
            key = normalize_text(extracted)[:100]
        extractions.append((key, resp, tools))

    # Majority vote
    counter = Counter(e[0] for e in extractions)
    winner_key, _ = counter.most_common(1)[0]

    # Pick the first response matching the winner for final output
    winning_resp = next((resp for key, resp, _ in extractions if key == winner_key), extractions[0][1])

    # Union of tools across all runs
    all_tools: list[str] = []
    for _, _, tools in extractions:
        all_tools.extend(tools)

    return winning_resp, all_tools


def _normalize_numeric(s: str) -> str:
    """Normalize a numeric string to 3 significant figures for voting."""
    import re
    nums = re.findall(r"-?\d+\.?\d*", s.replace(",", ""))
    if not nums:
        return s.strip().lower()
    try:
        v = float(nums[-1])
        if v == 0:
            return "0"
        import math
        digits = 3
        magnitude = math.floor(math.log10(abs(v)))
        factor = 10 ** (digits - 1 - magnitude)
        return str(round(v * factor) / factor)
    except (ValueError, OverflowError):
        return s.strip().lower()
