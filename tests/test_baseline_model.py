import pandas as pd

from business_entity_resolution.src.features import build_pair_features
from business_entity_resolution.src.model import (
    compare_validation_results,
    train_baseline_models,
)


def _make_pairs():
    rows = [
        {
            "left_name": "Acme Corp",
            "right_name": "Acme Corporation",
            "left_city": "Boston",
            "right_city": "Boston",
            "left_zip": 10001,
            "right_zip": 10001,
            "label": 1,
        },
        {
            "left_name": "Acme Corp",
            "right_name": "Beta LLC",
            "left_city": "Boston",
            "right_city": "New York",
            "left_zip": 10001,
            "right_zip": 20002,
            "label": 0,
        },
        {
            "left_name": "Green Energy",
            "right_name": "Green Energy Inc",
            "left_city": "Austin",
            "right_city": "Austin",
            "left_zip": 73301,
            "right_zip": 73301,
            "label": 1,
        },
        {
            "left_name": "Green Energy",
            "right_name": "Red Solar",
            "left_city": "Austin",
            "right_city": "Denver",
            "left_zip": 73301,
            "right_zip": 80202,
            "label": 0,
        },
    ]
    return pd.DataFrame(rows)


def test_build_pair_features_returns_numeric_matrix():
    pairs = _make_pairs()
    features = build_pair_features(pairs)
    assert not features.empty
    assert features.shape[0] == len(pairs)
    assert features.select_dtypes(include="number").shape[1] > 0


def test_train_baseline_models_returns_comparison():
    pairs = _make_pairs()
    result = train_baseline_models(pairs, target_col="label")
    assert set(result.keys()) == {"catboost", "lightgbm", "comparison"}
    assert result["comparison"]["catboost"]["val_auc"] >= 0.0
    assert result["comparison"]["lightgbm"]["val_auc"] >= 0.0


def test_compare_validation_results_uses_probability_output():
    pairs = _make_pairs()
    comparison = compare_validation_results(pairs, target_col="label")
    assert isinstance(comparison, dict)
    assert "catboost" in comparison
    assert "lightgbm" in comparison
    assert comparison["catboost"]["val_auc"] >= 0.0
