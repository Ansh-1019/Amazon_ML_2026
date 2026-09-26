"""Unit and integration tests for Unified Evaluation Layer across Blocking, Pairwise, and Entity metrics."""
import pytest
import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    EntityEvaluator,
    compute_single_entity_metrics,
    compute_f_beta,
    f05_score,
    compute_macro_f05,
    parse_id_set,
)
from src.validation.harness import ValidationHarness, ValidationReport


def test_empty_cases_boundary_conditions():
    """Verify official competition boundary conditions for empty match sets."""
    # 1. GT={} + prediction={} -> F0.5=1.0, Precision=1.0, Recall=1.0
    m1 = compute_single_entity_metrics(set(), set(), beta=0.5)
    assert m1["f_0.5"] == 1.0
    assert m1["precision"] == 1.0
    assert m1["recall"] == 1.0
    assert f05_score(set(), set()) == 1.0
    assert f05_score([], []) == 1.0
    assert f05_score(None, None) == 1.0

    # 2. GT={} + prediction={A} -> F0.5=0.0, Precision=0.0, Recall=0.0
    m2 = compute_single_entity_metrics(set(), {"A"}, beta=0.5)
    assert m2["f_0.5"] == 0.0
    assert m2["precision"] == 0.0
    assert m2["recall"] == 0.0
    assert f05_score(set(), {"A"}) == 0.0

    # 3. GT={A} + prediction={} -> F0.5=0.0, Precision=0.0, Recall=0.0
    m3 = compute_single_entity_metrics({"A"}, set(), beta=0.5)
    assert m3["f_0.5"] == 0.0
    assert m3["precision"] == 0.0
    assert m3["recall"] == 0.0
    assert f05_score({"A"}, set()) == 0.0


def test_macro_f05_across_all_s1_entities():
    """Ensure Macro F0.5 is averaged per-entity across all S1 entities, not global micro."""
    gt = {
        "s1_01": {"s2_01"},  # exact match -> F0.5 = 1.0
        "s1_02": {"s2_02"},  # exact match -> F0.5 = 1.0
        "s1_03": set(),       # empty match -> F0.5 = 1.0
        "s1_04": {"s2_04"},  # missed match -> F0.5 = 0.0
    }
    pred = {
        "s1_01": {"s2_01"},
        "s1_02": {"s2_01", "s2_02"},  # TP=1, FP=1, P=0.5, R=1.0 -> F0.5 = 5/9 ≈ 0.55556
        "s1_03": set(),
        "s1_04": set(),
    }

    evaluator = EntityEvaluator(beta=0.5)
    res = evaluator.evaluate(gt, pred, all_s1_ids=["s1_01", "s1_02", "s1_03", "s1_04"])

    # Expected entity scores:
    # s1_01: 1.0
    # s1_02: 5/9 ≈ 0.555556
    # s1_03: 1.0
    # s1_04: 0.0
    # Macro F0.5 = (1.0 + 5/9 + 1.0 + 0.0) / 4 = 2.555556 / 4 = 0.63889
    expected_macro = (1.0 + 5.0 / 9.0 + 1.0 + 0.0) / 4.0
    assert pytest.approx(res["macro_f0.5"], abs=1e-4) == expected_macro
    assert res["number_of_entities"] == 4
    assert res["correctly_empty"] == 1
    assert res["false_positive_empty"] == 0
    assert res["zero_match_count"] == 2  # s1_03 and s1_04
    assert res["one_match_count"] == 1   # s1_01
    assert res["multi_match_count"] == 1 # s1_02


def test_unified_validation_report_three_levels():
    """Verify ValidationReport contains all required keys across Blocking, Pairwise, and Entity levels."""
    report_data = {
        "run_info": {
            "experiment_id": "test_exp",
            "timestamp": "2026-09-27T00:00:00",
            "elapsed_seconds": 1.23,
            "threshold": 0.5,
        },
        "blocking": {
            "blocking_recall": 0.95,
            "S2_blocking_recall": 0.96,
            "S3_blocking_recall": 0.94,
            "mean_candidates": 12.5,
            "median_candidates": 10.0,
            "p95_candidates": 20.0,
            "zero_candidate_rate": 0.02,
        },
        "pairwise": {
            "training_pairs": 1500,
            "positive_pairs": 120,
            "negative_pairs": 1380,
            "hard_negative_pairs": 240,
        },
        "entity": {
            "macro_f0.5": 0.925,
            "macro_precision": 0.950,
            "macro_recall": 0.910,
            "exact_match_rate": 0.880,
            "correctly_empty": 15,
            "false_positive_empty": 1,
            "average_matches_per_entity": 0.98,
            "zero_match_count": 18,
            "one_match_count": 75,
            "multi_match_count": 7,
        },
    }

    report = ValidationReport(report_data)

    # 1. Blocking Level
    blk = report.blocking
    assert blk["blocking_recall"] == 0.95
    assert blk["S2_blocking_recall"] == 0.96
    assert blk["S3_blocking_recall"] == 0.94
    assert blk["mean_candidates"] == 12.5
    assert blk["median_candidates"] == 10.0
    assert blk["p95_candidates"] == 20.0
    assert blk["zero_candidate_rate"] == 0.02

    # 2. Pairwise Level
    pw = report.pairwise
    assert pw["training_pairs"] == 1500
    assert pw["positive_pairs"] == 120
    assert pw["negative_pairs"] == 1380
    assert pw["hard_negative_pairs"] == 240

    # 3. Entity Level
    ent = report.entity
    assert ent["macro_f0.5"] == 0.925
    assert ent["macro_precision"] == 0.950
    assert ent["macro_recall"] == 0.910
    assert ent["exact_match_rate"] == 0.880
    assert ent["correctly_empty"] == 15
    assert ent["false_positive_empty"] == 1
    assert ent["average_matches_per_entity"] == 0.98
    assert ent["zero_match_count"] == 18
    assert ent["one_match_count"] == 75
    assert ent["multi_match_count"] == 7

    # Check direct index access
    assert report["blocking"]["blocking_recall"] == 0.95
    assert report["pairwise"]["training_pairs"] == 1500
    assert report["entity"]["macro_f0.5"] == 0.925
