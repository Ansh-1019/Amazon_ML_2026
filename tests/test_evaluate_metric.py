import pandas as pd

from evaluate import evaluate, f05_score


def test_f05_penalizes_false_merge():
    true_groups = {"A": {"r1", "r2"}, "B": {"r3"}}
    pred_groups = {"p1": {"r1", "r2", "r3"}}

    score = evaluate(true_groups, pred_groups)

    assert 0.0 <= score < 1.0
    assert score < 1.0


def test_macro_average_uses_all_s1_entities():
    true_groups = {"A": {"r1", "r2"}, "B": {"r3"}}
    pred_groups = {"p1": {"r1"}, "p2": {"r3"}}

    score = evaluate(true_groups, pred_groups)

    expected_a = f05_score({"r1", "r2"}, {"r1"})
    expected_b = f05_score({"r3"}, {"r3"})
    expected = (expected_a + expected_b) / 2.0

    assert abs(score - expected) < 1e-12


def test_singleton_is_handled_like_other_s1_entities():
    true_groups = {"A": {"r1"}, "B": {"r2"}, "C": {"r3"}}
    pred_groups = {"p1": {"r1", "r2"}, "p2": {"r3"}}

    score = evaluate(true_groups, pred_groups)

    assert score < 1.0
    assert score > 0.0


def test_dataframe_input_supports_s1_and_cluster_columns():
    true_df = pd.DataFrame(
        {
            "record_id": ["r1", "r2", "r3"],
            "s1_id": ["A", "A", "B"],
        }
    )
    pred_df = pd.DataFrame(
        {
            "record_id": ["r1", "r2", "r3"],
            "cluster_id": ["p1", "p1", "p2"],
        }
    )

    score = evaluate(true_df, pred_df, record_col="record_id", entity_col="s1_id", pred_col="cluster_id")

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
