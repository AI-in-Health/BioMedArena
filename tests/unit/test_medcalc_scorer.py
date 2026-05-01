"""Unit tests for the MedCalc numeric extractor and tolerance scorer."""

from __future__ import annotations

import pytest

from harness.eval.scoring import (
    extract_numeric_answer,
    score_numeric_with_tolerance,
)


# ---------------------------------------------------------------------------
# Extractor behaviour
# ---------------------------------------------------------------------------


class TestExtractor:
    def test_primary_trailer_plain(self):
        v, reason = extract_numeric_answer(
            "Work not shown. The answer is 127.718 mL/min/1.73 m²."
        )
        assert v == pytest.approx(127.718)
        assert reason == "primary_trailer"

    def test_primary_trailer_with_brackets(self):
        v, reason = extract_numeric_answer("The answer is [42]")
        assert v == 42.0
        assert reason == "primary_trailer"

    def test_answer_prefix(self):
        v, reason = extract_numeric_answer(
            "Computed via CKD-EPI.\n\nAnswer: 83.072"
        )
        assert v == pytest.approx(83.072)
        assert reason == "answer_prefix"

    def test_boxed_answer(self):
        v, reason = extract_numeric_answer(
            r"Per formula, $\text{GFR} = \boxed{12.086}$ mL/min."
        )
        assert v == pytest.approx(12.086)
        assert reason == "boxed"

    def test_fallback_skips_1_73_constant(self):
        """CKD-EPI ends in `mL/min/1.73 m²`. Scorer must
        return the real GFR (130.65), not 1.73."""
        text = (
            "The patient's Glomerular Filtration Rate (GFR) is approximately "
            "130.65 mL/min/1.73 m²."
        )
        v, reason = extract_numeric_answer(text)
        assert v == pytest.approx(130.65)
        assert reason == "fallback_last_filtered"

    def test_fallback_skips_100_and_1_constants(self):
        text = "Percent change 1.0 × 100 in the end, final value 47."
        v, reason = extract_numeric_answer(text)
        assert v == 47.0
        assert reason == "fallback_last_filtered"

    def test_fallback_skips_year_like_integer(self):
        # "2021 CKD-EPI" must not be selected as the answer over
        # the real scalar that follows it.
        text = (
            "The patient's Glomerular Filtration Rate using the 2021 "
            "CKD-EPI Creatinine equation is approximately 11.25 "
            "mL/min/1.73 m²."
        )
        v, reason = extract_numeric_answer(text)
        assert v == pytest.approx(11.25)
        assert reason == "fallback_last_filtered"

    def test_extraction_failed_on_empty(self):
        v, reason = extract_numeric_answer("")
        assert v is None
        assert reason == "extraction_failed"

    def test_extraction_failed_on_constants_only(self):
        # If every number in the tail is a known constant, report
        # extraction_failed rather than silently returning e.g. 1.73.
        v, reason = extract_numeric_answer("The GFR normalises to 1.73 m² 100 %.")
        assert v is None
        assert reason == "extraction_failed"


# ---------------------------------------------------------------------------
# Scorer behaviour for CKD-EPI-style answers with unit constants.
# ---------------------------------------------------------------------------


CELL2_PREDS = [
    # (id,        gold,     prediction,                                        expected_correct)
    ("mcb_0021", 127.718,
     "The patient's Glomerular Filtration Rate (GFR) is approximately 130.65 mL/min/1.73 m².",
     True),
    ("mcb_0022", 12.086,
     "The patient's Glomerular Filtration Rate (GFR) using the 2021 CKD-EPI Creatinine equation is approximately 11.25 mL/min/1.73 m².",
     True),
    ("mcb_0023", 83.072,
     "The patient's Glomerular Filtration Rate (GFR) is approximately 82.78 mL/min/1.73 m².",
     True),
    ("mcb_0024", 88.199,
     "The patient's Glomerular Filtration Rate (GFR) is approximately 85.8 mL/min/1.73 m².",
     True),
    ("mcb_0025", 80.852,
     "The patient's Glomerular Filtration Rate (GFR) using the 2021 CKD-EPI Creatinine equation is approximately 94.47 mL/min/1.73 m².",
     False),  # 13.6 diff, gold×10% = 8.08 → outside tolerance
]


@pytest.mark.parametrize("qid,gold,pred,want", CELL2_PREDS)
def test_cell2_regression_passes_4_of_5(qid, gold, pred, want):
    got = score_numeric_with_tolerance(pred, str(gold), rel_tol=0.10)
    assert got is want, (
        f"{qid}: gold={gold} pred={pred!r} "
        f"want_correct={want} got_correct={got}"
    )


def test_cell2_aggregate():
    """Sanity: the 5-pred aggregate is 4/5, matching the manual re-score
    in EXPANDED_MINI_RESULTS.md."""
    passed = sum(
        1 for _qid, g, p, _ in CELL2_PREDS
        if score_numeric_with_tolerance(p, str(g), rel_tol=0.10)
    )
    assert passed == 4
