from harness.eval.scoring import score_question


def test_exact_numeric_extracts_decimal_after_answer_phrase():
    assert score_question("The answer is 1.3010312.", "1.3010312", "exactNumeric") is True


def test_exact_numeric_extracts_leading_decimal_and_scientific_notation():
    assert score_question("The answer is -.5.", "-0.5", "exactNumeric") is True
    assert score_question("The answer is 1e-3.", "0.001", "exactNumeric") is True


def test_exact_match_does_not_accept_numeric_substrings():
    assert score_question("10", "0", "exactMatch") is False


def test_exact_match_does_not_accept_label_substrings():
    assert score_question("inactive", "active", "exactMatch") is False


def test_exact_match_does_not_accept_short_label_inside_longer_answer():
    assert score_question("yes and no", "yes", "exactMatch") is False


def test_exact_match_still_accepts_direct_normalized_match():
    assert score_question(" The Answer Is ACTIVE. ", "active", "exactMatch") is True


def test_exact_match_allows_long_phrase_in_final_sentence():
    assert score_question(
        "Final label: severe adverse event",
        "severe adverse event",
        "exactMatch",
    ) is True


def test_public_answer_type_aliases_route_to_same_scorers():
    assert score_question("The answer is B.", "B", "mcq") is True
    assert score_question("The answer is 1e-3.", "0.001", "numeric") is True
    assert score_question(" The Answer Is ACTIVE. ", "active", "exact") is True
