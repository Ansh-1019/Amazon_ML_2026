import numpy as np
import pandas as pd
import pytest

from src.decision.threshold import (
    choose_threshold,
    entity_decision,
    keep_match,
    aggregate_entity_matches,
    _f05_score,
)


def test_choose_threshold_uses_probabilities_with_labels():
    labels = [0, 0, 1, 1]
    probs = [0.1, 0.45, 0.55, 0.9]

    threshold = choose_threshold(labels, probs)
    assert 0.45 <= threshold <= 0.9


def test_choose_threshold_handles_edge_cases():
    assert choose_threshold([], []) == 0.5
    assert choose_threshold(None, None) == 0.5
    with pytest.raises(ValueError):
        choose_threshold([1, 0], [0.5])


def test_f05_score_calculation():
    # When precision = 1.0, recall = 1.0 -> F0.5 = 1.0
    assert pytest.approx(_f05_score(1.0, 1.0)) == 1.0
    # When precision = 0.0, recall = 0.0 -> F0.5 = 0.0
    assert _f05_score(0.0, 0.0) == 0.0
    # F0.5 weights precision higher than recall
    # Precision 0.8, Recall 0.4:
    # beta^2 = 0.25 -> (1.25 * 0.8 * 0.4) / (0.25 * 0.8 + 0.4) = 0.4 / 0.6 = 0.6667
    assert pytest.approx(_f05_score(0.8, 0.4), rel=1e-3) == 0.6667


def test_keep_match_uses_threshold_on_probability():
    assert keep_match(0.51, 0.5) is True
    assert keep_match(0.49, 0.5) is False
    assert keep_match(None, 0.5) is False
    assert keep_match(np.nan, 0.5) is False
    assert keep_match("invalid", 0.5) is False
    assert keep_match(0.1, 0.5, singleton=True) is True


def test_entity_decision_keeps_singletons_and_uses_max_pair_score():
    assert entity_decision([], singleton=True) is True
    assert entity_decision([0.2, 0.3], threshold=0.5) is False
    assert entity_decision([0.2, 0.8], threshold=0.5) is True
    assert entity_decision(None) is False


def test_aggregate_entity_matches():
    df = pd.DataFrame([
        {"source1_entity_id": "s1_01", "candidate_entity_id": "s2_01", "probability": 0.85},
        {"source1_entity_id": "s1_01", "candidate_entity_id": "s3_01", "probability": 0.40},
        {"source1_entity_id": "s1_02", "candidate_entity_id": "s2_02", "probability": 0.20},
    ])
    res = aggregate_entity_matches(df, threshold=0.5)
    assert res["s1_01"] == ["s2_01"]
    assert res["s1_02"] == []
