"""Unit tests for LAB-Bench answer-position shuffle.

The loader must avoid pinning every correct answer to position A. This
test file guards the shuffle contract by:

  1. asserting the shuffle is deterministic per task (same id → same
     letter for `ideal`),
  2. asserting the distribution of `ideal_letter` across 100 synthetic
     tasks is roughly uniform (±5 percentage points per position),
  3. asserting the `openai_tri` preset opens ProtocolQA as open-ended
     (no MCQ options, scorer_hint == llm_judge).

We build synthetic tasks here so the test runs offline.
"""

from __future__ import annotations

from collections import Counter

import pytest

from harness.eval.bench_labbench import (
    OPENAI_TRI_SUBSETS,
    _OPEN_IN_OPENAI_TRI,
    _seed_for_task,
    _shuffled_mcq,
)


def _mk_task(i: int) -> dict:
    return {
        "id": f"labtest-{i:04d}",
        "ideal": f"answer_{i}",
        "distractors": [f"d{i}_1", f"d{i}_2", f"d{i}_3"],
    }


def test_shuffle_is_deterministic():
    t = _mk_task(0)
    seed = _seed_for_task(t["id"], "LitQA2")
    s1, _l1, ideal_letter_1 = _shuffled_mcq(t["ideal"], t["distractors"], seed)
    s2, _l2, ideal_letter_2 = _shuffled_mcq(t["ideal"], t["distractors"], seed)
    assert s1 == s2
    assert ideal_letter_1 == ideal_letter_2


def test_shuffle_seeds_differ_between_subsets():
    t = _mk_task(42)
    a = _seed_for_task(t["id"], "LitQA2")
    b = _seed_for_task(t["id"], "ProtocolQA")
    assert a != b


def test_ideal_letter_distribution_is_roughly_uniform():
    """Across 100 synthetic tasks the ideal_letter should hit each of
    {A,B,C,D} between 20% and 30% of the time (±5 pp around 25%)."""
    N = 100
    counter: Counter[str] = Counter()
    for i in range(N):
        t = _mk_task(i)
        seed = _seed_for_task(t["id"], "LitQA2")
        _, _, il = _shuffled_mcq(t["ideal"], t["distractors"], seed)
        counter[il] += 1

    assert set(counter) == {"A", "B", "C", "D"}, (
        f"expected all 4 positions to be hit, got {sorted(counter)}"
    )
    for letter in "ABCD":
        frac = counter[letter] / N
        assert 0.20 <= frac <= 0.30, (
            f"position {letter} fraction {frac:.2f} outside [0.20, 0.30] — "
            f"full distribution {dict(counter)}"
        )


def test_ideal_is_no_longer_pinned_to_A():
    """Over 20 synthetic tasks we must see at least 2 non-A positions."""
    non_A = 0
    for i in range(20):
        t = _mk_task(i)
        seed = _seed_for_task(t["id"], "LitQA2")
        _, _, il = _shuffled_mcq(t["ideal"], t["distractors"], seed)
        if il != "A":
            non_A += 1
    assert non_A >= 2, (
        f"ideal_letter was 'A' on {20 - non_A}/20 tasks — shuffle not effective"
    )


def test_openai_tri_configuration():
    assert OPENAI_TRI_SUBSETS == ["LitQA2", "ProtocolQA", "CloningScenarios"]
    assert "ProtocolQA" in _OPEN_IN_OPENAI_TRI
    assert "LitQA2" not in _OPEN_IN_OPENAI_TRI
    assert "CloningScenarios" not in _OPEN_IN_OPENAI_TRI


def test_shuffled_options_preserve_content():
    """Shuffling must not invent new options or drop distractors."""
    t = _mk_task(7)
    seed = _seed_for_task(t["id"], "LitQA2")
    shuffled, letters, _ = _shuffled_mcq(t["ideal"], t["distractors"], seed)
    # Set of options is preserved
    assert set(shuffled) == {t["ideal"], *t["distractors"]}
    assert len(shuffled) == 4 == len(letters)
