import numpy as np
import pandas as pd
import pytest

from src.features.features import build_pair_features
from src.modeling.model import (
    compare_validation_results,
    generate_hard_negatives,
    train_baseline_models,
    train_catboost_model,
    train_lightgbm_model,
    predict_pair_probabilities,
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
            "left_address": "123 Main St",
            "right_address": "123 Main Street",
            "left_country": "US",
            "right_country": "US",
            "label": 1,
        },
        {
            "left_name": "Acme Corp",
            "right_name": "Beta LLC",
            "left_city": "Boston",
            "right_city": "New York",
            "left_zip": 10001,
            "right_zip": 20002,
            "left_address": "123 Main St",
            "right_address": "456 Oak Ave",
            "left_country": "US",
            "right_country": "US",
            "label": 0,
        },
        {
            "left_name": "Green Energy",
            "right_name": "Green Energy Inc",
            "left_city": "Austin",
            "right_city": "Austin",
            "left_zip": 73301,
            "right_zip": 73301,
            "left_address": "100 Solar Way",
            "right_address": "100 Solar Way",
            "left_country": "US",
            "right_country": "US",
            "label": 1,
        },
        {
            "left_name": "Green Energy",
            "right_name": "Red Solar",
            "left_city": "Austin",
            "right_city": "Denver",
            "left_zip": 73301,
            "right_zip": 80202,
            "left_address": "100 Solar Way",
            "right_address": "200 Wind Rd",
            "left_country": "US",
            "right_country": "US",
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


def test_generate_hard_negatives_creates_ambiguous_false_matches():
    pairs = _make_pairs()
    hard_negatives = generate_hard_negatives(pairs, target_col="label")

    assert not hard_negatives.empty
    assert set(hard_negatives["label"].unique()) == {0}
    assert len(hard_negatives) > 0
    assert {"left_name", "right_name"}.issubset(hard_negatives.columns)


def test_predict_pair_probabilities_standard_format():
    pairs = _make_pairs()
    pairs["source1_entity_id"] = ["s1_1", "s1_1", "s1_2", "s1_2"]
    pairs["candidate_entity_id"] = ["s2_1", "s2_2", "s2_3", "s2_4"]
    pairs["target_source"] = ["s2", "s2", "s2", "s2"]

    features = build_pair_features(pairs, target_col="label")
    y = features.pop("label")
    features["source1_entity_id"] = pairs["source1_entity_id"]
    features["candidate_entity_id"] = pairs["candidate_entity_id"]
    features["target_source"] = pairs["target_source"]

    # Train a model
    num_cols = [c for c in features.select_dtypes(include=[np.number]).columns]
    X = features[num_cols]
    model = train_catboost_model(X, y, X, y, iterations=10)

    preds_df = predict_pair_probabilities(model, features)
    assert not preds_df.empty
    assert list(preds_df.columns) == ["source1_entity_id", "candidate_entity_id", "target_source", "probability"]
    assert len(preds_df) == len(pairs)
    assert preds_df["probability"].between(0.0, 1.0).all()
