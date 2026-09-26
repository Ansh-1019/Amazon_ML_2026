"""Integration test for Maithili's blocking system through CandidateGenerator adapter.

Tests the full flow:
    DataLoader / Normalized Data
         ↓
    Maithili CandidatePipeline (Exact, Token, Char-Ngram)
         ↓
    IdAdapter (Reversible ID mapping)
         ↓
    Ansh-compatible canonical candidate DataFrame

Verifies:
  - total candidate pairs
  - mean candidates/S1
  - median candidates/S1
  - p95 candidates/S1
  - zero-candidate rate
  - overall blocking recall on ground truth
  - S2 blocking recall
  - S3 blocking recall
  - Reversibility of IDs (never corrupts official IDs)
  - Canonical schema compliance (s1_id, s2_id, s3_id, block_reasons, blocking_score)
"""
import numpy as np
import pandas as pd
import pytest

from src.data.normalizer import DataNormalizer
from src.blocking.candidate_gen import CandidateGenerator, IdAdapter
from src.pipeline import candidates_df_to_map


@pytest.fixture
def synthetic_data():
    s1 = pd.DataFrame([
        ("s1_01", "Acme Corporation", "123 Main Street, New York, NY 10001", "US"),
        ("s1_02", "Beta Solutions International", "456 Oak Avenue, Los Angeles, CA", "US"),
        ("s1_03", "Gamma & Enterprises LLC", "789 Pine Road, Chicago, IL 60601", "US"),
        ("s1_04", "Delta Technologies Group", "101 Elm Street, Houston, TX 77001", "US"),
        ("s1_05", "Epsilon Global Services", "202 Maple Drive, Phoenix, AZ 85001", "US"),
        ("s1_06", "Zeta Industries Ltd", "303 Cedar Street, Dallas, TX 75201", "US"),
        ("s1_07", "Eta Manufacturing Corp", "404 Birch Lane, San Jose, CA 95101", "US"),
        ("s1_08", "Theta Franchise Inc", "500 Commerce Blvd, Miami, FL 33101", "US"),
        ("s1_09", "Iota Ventures LLC", "999 Unknown Rd, Unknown City", "US"),
        ("s1_10", "Kappa Solutions Corp", "456 Oak Avenue, Los Angeles, CA", "US"),
        ("s1_11", "Lambda UK Ltd", "10 Downing Street, London", "UK"),
        ("s1_12", "Mu Enterprises SARL", "15 Rue de Rivoli, Paris", "France"),
    ], columns=["entity_id", "business_name", "business_address", "country"])

    s2 = pd.DataFrame([
        ("s2_01", "Acme Corporation", "123 Main Street, New York, NY 10001", "US"),
        ("s2_02", "Beta Solns Intl", "456 Oak Ave, Los Angeles, CA", "US"),
        ("s2_03", "Gamma Enterprises LLC", "789 Pine Road, Chicago, IL 60601", "US"),
        ("s2_04", "Group Technologies Delta", "101 Elm St, Houston, TX 77001", "US"),
        ("s2_05", "Epsilom Global Services", "202 Maple Drive, Phoenix, AZ 85001", "US"),
        ("s2_06", "Zeta Industries Ltd", "303 Cedar St, Dallas, TX 75201", "US"),
        ("s2_07", "Eta Manufacturing Corp", "404 Birch Lane, San Jose, CA", "US"),
        ("s2_08a", "Theta Franchise Inc", "500 Commerce Blvd, Miami, FL 33101", "US"),
        ("s2_08b", "Theta Franchise Incorporated", "500 Commerce Blvd, Miami, FL", "US"),
        ("s2_09", "Completely Different Company", "001 Nowhere Ave, Nowhere", "US"),
        ("s2_10", "Kappa Solutions Corp", "456 Oak Avenue, Los Angeles, CA", "US"),
        ("s2_11", "Lambda UK Limited", "10 Downing St, London", "UK"),
    ], columns=["entity_id", "business_name", "business_address", "country"])

    s3 = pd.DataFrame([
        ("s3_01", "Acme Corp", "123 Main St, New York, 10001", "US"),
        ("s3_03", "Gamma Enterprises", "789 Pine Rd, Chicago, IL", "US"),
        ("s3_12", "Mu Entreprises SARL", "15 Rue de Rivoli, Paris 75001", "France"),
        ("s3_xx", "Unrelated Company X", "1 Random Street, Somewhere", "US"),
    ], columns=["entity_id", "business_name", "business_address", "country"])

    ground_truth = {
        "s1_01": {"s2_01", "s3_01"},
        "s1_02": {"s2_02"},
        "s1_03": {"s2_03", "s3_03"},
        "s1_04": {"s2_04"},
        "s1_05": {"s2_05"},
        "s1_06": {"s2_06"},
        "s1_07": {"s2_07"},
        "s1_08": {"s2_08a", "s2_08b"},
        "s1_09": set(),
        "s1_10": set(),
        "s1_11": {"s2_11"},
        "s1_12": {"s3_12"},
    }

    return s1, s2, s3, ground_truth


def test_id_adapter_roundtrip():
    adapter = IdAdapter()
    orig_s2 = ["101", "s2_abc", "S2-999"]
    orig_s3 = ["101", "s3_xyz", "S3-888"]

    internal_s2 = adapter.register_s2(orig_s2)
    internal_s3 = adapter.register_s3(orig_s3)

    # Internal IDs must all start with S2- / S3-
    for iid in internal_s2:
        assert iid.startswith("S2-")
    for iid in internal_s3:
        assert iid.startswith("S3-")

    # Round-trip decode must be exact
    for iid, expected in zip(internal_s2, orig_s2):
        decoded_id, src = adapter.decode(iid)
        assert decoded_id == expected
        assert src == "s2"

    for iid, expected in zip(internal_s3, orig_s3):
        decoded_id, src = adapter.decode(iid)
        assert decoded_id == expected
        assert src == "s3"


def test_candidate_generator_integration_and_metrics(synthetic_data):
    s1, s2, s3, gt = synthetic_data
    norm = DataNormalizer()
    s1_n = norm.normalize_dataframe(s1)
    s2_n = norm.normalize_dataframe(s2)
    s3_n = norm.normalize_dataframe(s3)

    gen = CandidateGenerator(top_k=20)
    cand_df = gen.generate_candidates(s1_n, s2_n, s3_n)

    # 1. Schema check
    expected_cols = ["s1_id", "s2_id", "s3_id", "target_id", "source", "block_reasons", "blocking_score"]
    for col in expected_cols:
        assert col in cand_df.columns

    # 2. Convert to candidate map
    all_s1_ids = s1["entity_id"].tolist()
    cand_map = candidates_df_to_map(cand_df, all_s1_ids=all_s1_ids)

    # 3. Compute distribution and volume metrics
    candidate_counts = [len(cand_map[s1_id]) for s1_id in all_s1_ids]
    total_candidate_pairs = sum(candidate_counts)
    mean_candidates = float(np.mean(candidate_counts))
    median_candidates = float(np.median(candidate_counts))
    p95_candidates = float(np.percentile(candidate_counts, 95))
    zero_candidates = sum(1 for c in candidate_counts if c == 0)
    zero_candidate_rate = zero_candidates / len(all_s1_ids)

    assert total_candidate_pairs > 0
    assert mean_candidates > 0
    assert p95_candidates >= median_candidates
    assert 0.0 <= zero_candidate_rate <= 1.0

    # 4. Compute Recall metrics
    total_gt = 0
    captured_gt = 0
    s2_gt = 0
    s2_captured = 0
    s3_gt = 0
    s3_captured = 0

    valid_s2_ids = set(s2["entity_id"])
    valid_s3_ids = set(s3["entity_id"])

    for s1_id, gt_targets in gt.items():
        cands = set(cand_map[s1_id])
        for target in gt_targets:
            total_gt += 1
            if target in valid_s2_ids:
                s2_gt += 1
                if target in cands:
                    s2_captured += 1
            elif target in valid_s3_ids:
                s3_gt += 1
                if target in cands:
                    s3_captured += 1

            if target in cands:
                captured_gt += 1

    overall_recall = captured_gt / total_gt if total_gt else 1.0
    s2_recall = s2_captured / s2_gt if s2_gt else 1.0
    s3_recall = s3_captured / s3_gt if s3_gt else 1.0

    # Critical requirement: Recall must be 1.0 (100%) on synthetic ground truth
    assert overall_recall == 1.0, f"Overall blocking recall dropped to {overall_recall:.4f}"
    assert s2_recall == 1.0, f"S2 blocking recall dropped to {s2_recall:.4f}"
    assert s3_recall == 1.0, f"S3 blocking recall dropped to {s3_recall:.4f}"

    # 5. Provenance check
    reasons = set(cand_df["block_reasons"].dropna().tolist())
    assert any("exact_name" in r for r in reasons)
    assert any("token" in r for r in reasons)
    assert any("char_ngram" in r for r in reasons)


def test_empty_source1_handling():
    gen = CandidateGenerator()
    res = gen.generate_candidates(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert res.empty
    assert "s1_id" in res.columns
    assert "block_reasons" in res.columns
