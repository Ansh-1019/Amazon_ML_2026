"""Integration tests connecting Anmol model predictions to Ansh EntityResolver.

Tests the 9 required resolution behaviors:
1. Zero matches
2. One match
3. Multiple matches
4. S2 match
5. S3 match
6. S2 + S3 matches
7. Duplicate candidates
8. Below-threshold candidates
9. Invalid candidates
"""
import numpy as np
import pandas as pd
import pytest

from src.decision.entity_resolver import EntityResolver
from src.decision.adapter import adapt_model_predictions_for_resolver


@pytest.fixture
def base_config():
    return {
        "decision": {
            "threshold": 0.5,
            "threshold_s2": None,
            "threshold_s3": None,
            "top_margin": None,
        }
    }


def test_1_zero_matches(base_config):
    """Test S1 entity with all candidates below threshold produces zero matches."""
    preds_df = pd.DataFrame([
        {"source1_entity_id": "s1_01", "candidate_entity_id": "s2_01", "target_source": "s2", "probability": 0.20},
        {"source1_entity_id": "s1_01", "candidate_entity_id": "s3_01", "target_source": "s3", "probability": 0.45},
    ])
    adapted = adapt_model_predictions_for_resolver(preds_df)
    resolver = EntityResolver(threshold=0.5)
    res = resolver.resolve(adapted, config=base_config)

    assert res.empty
    assert list(res.columns) == ["s1_id", "target_id", "match_score", "source"]


def test_2_one_match(base_config):
    """Test S1 entity with a single candidate above threshold."""
    preds_df = pd.DataFrame([
        {"source1_entity_id": "s1_01", "candidate_entity_id": "s2_01", "target_source": "s2", "probability": 0.95},
        {"source1_entity_id": "s1_01", "candidate_entity_id": "s2_02", "target_source": "s2", "probability": 0.10},
    ])
    adapted = adapt_model_predictions_for_resolver(preds_df)
    resolver = EntityResolver(threshold=0.5)
    res = resolver.resolve(adapted, config=base_config)

    assert len(res) == 1
    assert res.iloc[0]["s1_id"] == "s1_01"
    assert res.iloc[0]["target_id"] == "s2_01"
    assert res.iloc[0]["match_score"] == 0.95


def test_3_multiple_matches_no_forced_single_match(base_config):
    """Test S1 entity with multiple candidates above threshold (does not force 1-to-1)."""
    preds_df = pd.DataFrame([
        {"source1_entity_id": "s1_01", "candidate_entity_id": "s2_01", "target_source": "s2", "probability": 0.95},
        {"source1_entity_id": "s1_01", "candidate_entity_id": "s2_02", "target_source": "s2", "probability": 0.88},
        {"source1_entity_id": "s1_01", "candidate_entity_id": "s2_03", "target_source": "s2", "probability": 0.75},
        {"source1_entity_id": "s1_01", "candidate_entity_id": "s2_04", "target_source": "s2", "probability": 0.30},
    ])
    adapted = adapt_model_predictions_for_resolver(preds_df)
    resolver = EntityResolver(threshold=0.5)
    res = resolver.resolve(adapted, config=base_config)

    # All 3 above 0.5 must be retained, not just the max
    assert len(res) == 3
    assert set(res["target_id"]) == {"s2_01", "s2_02", "s2_03"}
    assert "s2_04" not in res["target_id"].values


def test_4_s2_match(base_config):
    """Test resolution of S2 target source."""
    preds_df = pd.DataFrame([
        {"source1_entity_id": "s1_01", "candidate_entity_id": "s2_100", "target_source": "s2", "probability": 0.92},
    ])
    adapted = adapt_model_predictions_for_resolver(preds_df)
    resolver = EntityResolver(threshold=0.5)
    res = resolver.resolve(adapted, config=base_config, s2_ids={"s2_100"})

    assert len(res) == 1
    assert res.iloc[0]["source"] == "s2"
    assert res.iloc[0]["target_id"] == "s2_100"


def test_5_s3_match(base_config):
    """Test resolution of S3 target source."""
    preds_df = pd.DataFrame([
        {"source1_entity_id": "s1_01", "candidate_entity_id": "s3_200", "target_source": "s3", "probability": 0.89},
    ])
    adapted = adapt_model_predictions_for_resolver(preds_df)
    resolver = EntityResolver(threshold=0.5)
    res = resolver.resolve(adapted, config=base_config, s3_ids={"s3_200"})

    assert len(res) == 1
    assert res.iloc[0]["source"] == "s3"
    assert res.iloc[0]["target_id"] == "s3_200"


def test_6_s2_and_s3_matches(base_config):
    """Test S1 entity matched to both S2 and S3 simultaneously."""
    preds_df = pd.DataFrame([
        {"source1_entity_id": "s1_01", "candidate_entity_id": "s2_100", "target_source": "s2", "probability": 0.90},
        {"source1_entity_id": "s1_01", "candidate_entity_id": "s3_200", "target_source": "s3", "probability": 0.85},
    ])
    adapted = adapt_model_predictions_for_resolver(preds_df)
    resolver = EntityResolver(threshold=0.5)
    res = resolver.resolve(adapted, config=base_config, s2_ids={"s2_100"}, s3_ids={"s3_200"})

    assert len(res) == 2
    sources = set(res["source"])
    assert "s2" in sources
    assert "s3" in sources
    assert set(res["target_id"]) == {"s2_100", "s3_200"}


def test_7_duplicate_candidates(base_config):
    """Test that duplicate pairs are deduplicated, keeping the highest score."""
    preds_df = pd.DataFrame([
        {"source1_entity_id": "s1_01", "candidate_entity_id": "s2_01", "target_source": "s2", "probability": 0.60},
        {"source1_entity_id": "s1_01", "candidate_entity_id": "s2_01", "target_source": "s2", "probability": 0.95},
        {"source1_entity_id": "s1_01", "candidate_entity_id": "s2_01", "target_source": "s2", "probability": 0.70},
    ])
    adapted = adapt_model_predictions_for_resolver(preds_df)
    resolver = EntityResolver(threshold=0.5)
    res = resolver.resolve(adapted, config=base_config)

    assert len(res) == 1
    assert res.iloc[0]["target_id"] == "s2_01"
    assert res.iloc[0]["match_score"] == 0.95


def test_8_below_threshold_candidates(base_config):
    """Test candidate filtering across multiple entities with configurable thresholds."""
    preds_df = pd.DataFrame([
        {"source1_entity_id": "s1_01", "candidate_entity_id": "s2_01", "target_source": "s2", "probability": 0.80},
        {"source1_entity_id": "s1_01", "candidate_entity_id": "s2_02", "target_source": "s2", "probability": 0.35},
        {"source1_entity_id": "s1_02", "candidate_entity_id": "s2_03", "target_source": "s2", "probability": 0.49},
        {"source1_entity_id": "s1_02", "candidate_entity_id": "s3_01", "target_source": "s3", "probability": 0.65},
    ])
    adapted = adapt_model_predictions_for_resolver(preds_df)

    # Global threshold 0.5
    resolver = EntityResolver(threshold=0.5)
    res = resolver.resolve(adapted, config={"decision": {"threshold": 0.5}})
    assert set(res["target_id"]) == {"s2_01", "s3_01"}

    # Higher threshold 0.7
    res_high = resolver.resolve(adapted, config={"decision": {"threshold": 0.7}})
    assert set(res_high["target_id"]) == {"s2_01"}


def test_9_invalid_candidates(base_config):
    """Test safe handling of None, NaN, empty strings, and missing scores."""
    preds_df = pd.DataFrame([
        {"source1_entity_id": "s1_01", "candidate_entity_id": None, "target_source": "s2", "probability": 0.90},
        {"source1_entity_id": None, "candidate_entity_id": "s2_01", "target_source": "s2", "probability": 0.90},
        {"source1_entity_id": "s1_02", "candidate_entity_id": "s2_02", "target_source": "s2", "probability": np.nan},
        {"source1_entity_id": "s1_03", "candidate_entity_id": "s2_03", "target_source": "s2", "probability": 0.85},
    ])
    adapted = adapt_model_predictions_for_resolver(preds_df)
    resolver = EntityResolver(threshold=0.5)
    res = resolver.resolve(adapted, config=base_config)

    # Only s1_03 -> s2_03 is valid and above threshold
    assert len(res) == 1
    assert res.iloc[0]["s1_id"] == "s1_03"
    assert res.iloc[0]["target_id"] == "s2_03"
    assert res.iloc[0]["match_score"] == 0.85
