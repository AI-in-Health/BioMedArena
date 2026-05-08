from harness.eval.scoring import (
    extract_answer_from_response,
    extract_numeric_answer,
    multilabel_f1,
    relation_set_f1,
    retrieval_hit,
    score_multiple_choice,
    score_question,
    token_label_f1,
)


def test_multiple_choice_extraction_supports_multi_select_answers():
    assert extract_answer_from_response("答案是ABC", "multipleChoice") == "ABC"
    assert extract_answer_from_response("Answer: A, C", "multipleChoice") == "AC"
    assert score_multiple_choice("CA", "AC")


def test_multiple_choice_extraction_avoids_obvious_answer_word():
    assert extract_answer_from_response("The answer is BECAUSE", "multipleChoice") == "B"


def test_token_label_f1_scores_iob_spans_not_exact_strings():
    expected = '["O", "B-Chemical", "I-Chemical", "O", "B-Disease"]'
    assert token_label_f1(expected, expected) == 1.0
    assert token_label_f1('["O", "B-Chemical", "O", "O", "B-Disease"]', expected) < 1.0
    assert score_question(expected, expected, "tokenSequence", {"scorer_kind": "token_f1"})


def test_multilabel_f1_scores_vectors_and_label_sets():
    expected = '["1", "0", "1"]'
    assert multilabel_f1(expected, expected) == 1.0
    assert multilabel_f1('["0", "0", "1"]', expected) < 1.0
    assert score_question(expected, expected, "multiLabel", {"scorer_kind": "multilabel_f1"})


def test_numeric_extraction_accepts_direct_scalar_one():
    assert extract_numeric_answer("1") == (1.0, "direct_scalar")
    assert score_question("1", "1", "exactNumeric", {"scorer_kind": "exact"})


def test_relation_set_f1_scores_json_relations():
    expected = '[{"type": "PPI", "arg1_id": "e1", "arg2_id": "e2"}]'
    predicted = '[{"type": "PPI", "arg1_id": "e1", "arg2_id": "e2"}]'
    wrong = '[{"type": "PPI", "arg1_id": "e1", "arg2_id": "e3"}]'
    assert relation_set_f1(predicted, expected) == 1.0
    assert relation_set_f1(wrong, expected) == 0.0
    assert score_question(predicted, expected, "relationSet", {"scorer_kind": "relation_f1"})


def test_retrieval_ranking_scores_when_relevant_id_is_present():
    expected = '["doc1"]'
    predicted = '["doc3", "doc1", "doc2"]'
    assert retrieval_hit(predicted, expected)
    assert score_question(predicted, expected, "ranking", {"scorer_kind": "retrieval_hit"})


def test_smiles_scorers_accept_identical_answers():
    assert score_question("CCO", "CCO", "exactMatch", {"scorer_kind": "smiles_topk_canonical_match"})
    assert score_question("CCO", "CCO", "exactMatch", {"scorer_kind": "smiles_validity_plus_tanimoto"})
