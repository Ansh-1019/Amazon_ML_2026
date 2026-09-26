"""Unit tests for Anmol's pairwise feature extraction module."""
import numpy as np
import pandas as pd
import pytest

from src.features.features import build_pair_features


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
    assert not features.isna().any().any()


def test_build_pair_features_has_advanced_name_and_address_signals():
    pairs = pd.DataFrame(
        [
            {
                "left_name": "Acme Corporation",
                "right_name": "Acme Corp",
                "left_address": "123 Main Street, Boston, MA",
                "right_address": "123 Main St, Boston, MA",
                "left_country": "US",
                "right_country": "US",
                "label": 1,
            },
            {
                "left_name": "Green Solar",
                "right_name": "Red Energy",
                "left_address": "10 Park Avenue, New York, NY",
                "right_address": "20 Lake Road, Chicago, IL",
                "left_country": "US",
                "right_country": "US",
                "label": 0,
            },
        ]
    )

    features = build_pair_features(pairs, target_col="label")

    assert "name_rare_token_overlap" in features.columns
    assert "name_char_similarity" in features.columns
    assert "address_char_similarity" in features.columns
    assert "country_exact" in features.columns

    assert features["name_rare_token_overlap"].between(0.0, 1.0).all()
    assert features["name_char_similarity"].between(0.0, 1.0).all()
    assert features["address_char_similarity"].between(0.0, 1.0).all()
    assert not features.isna().any().any()


def test_build_pair_features_empty_dataframe_raises():
    with pytest.raises(ValueError):
        build_pair_features(pd.DataFrame())


def test_build_pair_features_no_nan_or_inf_on_missing_values():
    pairs = pd.DataFrame([
        {
            "left_name": None,
            "right_name": "Acme Corp",
            "left_address": float("nan"),
            "right_address": None,
            "left_country": "",
            "right_country": "US",
        }
    ])
    features = build_pair_features(pairs)
    assert not features.isna().any().any()
    assert np.isfinite(features.select_dtypes(include=[np.number]).values).all()
