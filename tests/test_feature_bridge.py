"""Integration tests for FeatureExtractor bridge and Anmol pairwise feature extraction."""
import numpy as np
import pandas as pd
import pytest

from src.data.normalizer import DataNormalizer
from src.blocking.candidate_gen import CandidateGenerator
from src.features.pairwise import FeatureExtractor


@pytest.fixture
def sample_sources():
    s1 = pd.DataFrame([
        ("s1_01", "Acme Corporation", "123 Main Street, New York, NY", "US"),
        ("s1_02", "Beta Solutions", "456 Oak Avenue, Los Angeles, CA", "US"),
        ("s1_03", "Mu Enterprises SARL", "15 Rue de Rivoli, Paris", "France"),
    ], columns=["entity_id", "business_name", "business_address", "country"])

    s2 = pd.DataFrame([
        ("s2_01", "Acme Corp", "123 Main St, New York, NY", "US"),
        ("s2_02", "Beta Solns Inc", "456 Oak Ave, Los Angeles, CA", "US"),
    ], columns=["entity_id", "business_name", "business_address", "country"])

    s3 = pd.DataFrame([
        ("s3_01", "Acme International", "123 Main Street, NY", "US"),
        ("s3_03", "Mu Entreprises SARL", "15 Rue de Rivoli, Paris 75001", "France"),
    ], columns=["entity_id", "business_name", "business_address", "country"])

    return s1, s2, s3


def test_feature_extractor_end_to_end_from_candidates(sample_sources):
    s1, s2, s3 = sample_sources
    norm = DataNormalizer()
    s1_n = norm.normalize_dataframe(s1)
    s2_n = norm.normalize_dataframe(s2)
    s3_n = norm.normalize_dataframe(s3)

    gen = CandidateGenerator(top_k=20)
    cand_df = gen.generate_candidates(s1_n, s2_n, s3_n)

    extractor = FeatureExtractor()
    features_df = extractor.extract_features(cand_df, s1_n, s2_n, s3_n)

    assert not features_df.empty
    assert len(features_df) == len(cand_df)

    # 1. Metadata identifier columns must remain attached
    assert "source1_entity_id" in features_df.columns
    assert "candidate_entity_id" in features_df.columns
    assert "target_source" in features_df.columns
    assert "s1_id" in features_df.columns

    # 2. Both S1 <-> S2 and S1 <-> S3 must be present
    sources = set(features_df["target_source"].unique())
    assert "s2" in sources
    assert "s3" in sources

    # 3. Numeric columns must have no NaN or infinite values
    num_cols = features_df.select_dtypes(include=[np.number]).columns
    assert len(num_cols) > 0
    assert not features_df[num_cols].isna().any().any()
    assert np.isfinite(features_df[num_cols].values).all()

    # 4. Critical signal features present
    assert "name_text_similarity" in features_df.columns
    assert "name_char_similarity" in features_df.columns
    assert "address_char_similarity" in features_df.columns
    assert "country_exact" in features_df.columns
    assert "pair_quality_score" in features_df.columns

    # 5. No ground truth labels in input features
    for label_col in ["label", "is_match", "ground_truth", "matched"]:
        assert label_col not in features_df.columns


def test_feature_extractor_deterministic(sample_sources):
    s1, s2, s3 = sample_sources
    cand_df = pd.DataFrame([
        {"s1_id": "s1_01", "s2_id": "s2_01", "s3_id": None, "target_id": "s2_01", "source": "s2"},
        {"s1_id": "s1_03", "s2_id": None, "s3_id": "s3_03", "target_id": "s3_03", "source": "s3"},
    ])

    extractor = FeatureExtractor()
    f1 = extractor.extract_features(cand_df, s1, s2, s3)
    f2 = extractor.extract_features(cand_df, s1, s2, s3)

    num_cols = f1.select_dtypes(include=[np.number]).columns
    np.testing.assert_array_equal(f1[num_cols].values, f2[num_cols].values)


def test_feature_extractor_empty_candidates():
    extractor = FeatureExtractor()
    res = extractor.extract_features(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert res.empty


def test_jaccard_similarity_utility():
    assert FeatureExtractor.jaccard_similarity("acme corp", "acme corp") == 1.0
    assert FeatureExtractor.jaccard_similarity("acme corp", "beta llc") == 0.0
    assert 0.0 < FeatureExtractor.jaccard_similarity("acme corporation", "acme corp") <= 1.0
    assert FeatureExtractor.jaccard_similarity("", "acme") == 0.0
    assert FeatureExtractor.jaccard_similarity(None, "acme") == 0.0
