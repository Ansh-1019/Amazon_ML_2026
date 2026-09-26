"""
Tests for the redesigned Entity-Level Decision Layer.

Covers all 8 required cases plus edge-case/integration scenarios:

Case 1:  One true match (exactly one candidate above threshold)
Case 2:  Multiple true matches (several candidates above threshold)
Case 3:  Zero true matches (all candidates below threshold)
Case 4:  One high-probability + several low-probability candidates
Case 5:  Multiple high-probability candidates (e.g., twins/franchises)
Case 6:  Candidate prediction contains an invalid target ID
Case 7:  Duplicate candidate IDs in input
Case 8:  S1 entity with zero candidates

Additional cases:
  - Per-source threshold: S2 and S3 thresholds treated independently
  - top_margin pruning: entity-local margin applied after global threshold
  - Empty input DataFrame
  - Config-driven thresholds override constructor defaults
  - Long-form output format (s1_id, target_id, match_score, source)
"""

import unittest
from typing import Dict, List, Optional, Set

import pandas as pd
import numpy as np

from src.decision.entity_resolver import EntityResolver


# ─────────────────────────────────────────────────────────────────────────────
# Helper to build a candidates DataFrame quickly
# ─────────────────────────────────────────────────────────────────────────────

def make_candidates(*rows) -> pd.DataFrame:
    """
    Each row is a dict with keys: s1_id, target_col ('s2_id' or 's3_id'), target_val, score.
    Returns a wide-format DataFrame like the pipeline generates.
    """
    data = []
    for r in rows:
        row = {"s1_id": r["s1"], "s2_id": None, "s3_id": None, "match_score": r["score"]}
        if r.get("s2"):
            row["s2_id"] = r["s2"]
        if r.get("s3"):
            row["s3_id"] = r["s3"]
        data.append(row)
    return pd.DataFrame(data)


def resolve(
    df: pd.DataFrame,
    threshold: float = 0.5,
    threshold_s2: Optional[float] = None,
    threshold_s3: Optional[float] = None,
    top_margin: Optional[float] = None,
    s2_ids: Optional[Set[str]] = None,
    s3_ids: Optional[Set[str]] = None,
) -> Dict[str, List[str]]:
    """Convenience: resolve and return {s1_id: [target_ids]} dict."""
    resolver = EntityResolver(
        threshold=threshold,
        threshold_s2=threshold_s2,
        threshold_s3=threshold_s3,
        top_margin=top_margin,
    )
    config = {"decision": {}}
    result_df = resolver.resolve(df, config=config, s2_ids=s2_ids, s3_ids=s3_ids)
    if result_df.empty:
        return {}
    out: Dict[str, List[str]] = {}
    for _, row in result_df.iterrows():
        s1 = row["s1_id"]
        out.setdefault(s1, [])
        if row["target_id"] not in out[s1]:
            out[s1].append(row["target_id"])
    return out


# ─────────────────────────────────────────────────────────────────────────────
class TestEntityResolverCases(unittest.TestCase):

    def setUp(self):
        self.resolver = EntityResolver(threshold=0.5)
        self.empty_config = {"decision": {}}

    # ── Case 1: One true match ──────────────────────────────────────────────
    def test_case_1_one_true_match(self):
        """One candidate above threshold → exactly one ID in result."""
        df = make_candidates(
            {"s1": "s1_1", "s2": "s2_101", "score": 0.92},
        )
        out = resolve(df, threshold=0.5)
        self.assertEqual(out.get("s1_1"), ["s2_101"])

    # ── Case 2: Multiple true matches ──────────────────────────────────────
    def test_case_2_multiple_true_matches(self):
        """
        Three candidates, all above threshold.
        Old behavior: only the top-scoring one would be kept (groupby.first).
        New behavior: ALL three must be kept.
        """
        df = make_candidates(
            {"s1": "s1_A", "s2": "s2_001", "score": 0.97},
            {"s1": "s1_A", "s2": "s2_002", "score": 0.95},
            {"s1": "s1_A", "s3": "s3_301", "score": 0.88},
        )
        out = resolve(df, threshold=0.5)
        matched = sorted(out.get("s1_A", []))
        self.assertIn("s2_001", matched)
        self.assertIn("s2_002", matched)
        self.assertIn("s3_301", matched)
        self.assertEqual(len(matched), 3)

    # ── Case 3: Zero true matches ───────────────────────────────────────────
    def test_case_3_zero_matches_all_below_threshold(self):
        """All candidates below threshold → entity resolves to empty."""
        df = make_candidates(
            {"s1": "s1_1", "s2": "s2_200", "score": 0.10},
            {"s1": "s1_1", "s3": "s3_300", "score": 0.25},
        )
        out = resolve(df, threshold=0.5)
        # Entity should not appear (pipeline's resolved_df_to_map fills [] for missing s1s)
        self.assertEqual(out.get("s1_1", []), [])

    # ── Case 4: One high + several low ─────────────────────────────────────
    def test_case_4_one_high_several_low(self):
        """Only the high-probability candidate clears threshold; low ones are dropped."""
        df = make_candidates(
            {"s1": "s1_B", "s2": "s2_010", "score": 0.91},
            {"s1": "s1_B", "s2": "s2_011", "score": 0.22},
            {"s1": "s1_B", "s3": "s3_310", "score": 0.18},
        )
        out = resolve(df, threshold=0.5)
        self.assertEqual(out.get("s1_B"), ["s2_010"])

    # ── Case 5: Multiple high-probability candidates ────────────────────────
    def test_case_5_multiple_high_probability_candidates(self):
        """
        Scenario: two genuine counterparts (e.g., franchise with two records).
        Both must survive; no artificial singleton constraint.
        """
        df = make_candidates(
            {"s1": "s1_C", "s2": "s2_100", "score": 0.98},
            {"s1": "s1_C", "s2": "s2_101", "score": 0.97},
            {"s1": "s1_C", "s3": "s3_400", "score": 0.96},
        )
        out = resolve(df, threshold=0.9)
        matched = sorted(out.get("s1_C", []))
        self.assertEqual(len(matched), 3, "All three high-confidence candidates must be kept")
        self.assertIn("s2_100", matched)
        self.assertIn("s2_101", matched)
        self.assertIn("s3_400", matched)

    # ── Case 6: Invalid target ID (not in valid set) ────────────────────────
    def test_case_6_invalid_target_id_not_in_valid_set(self):
        """
        The resolver itself does not validate against a valid-ID set (that's
        resolved_df_to_map's job). But target IDs from non-existent columns
        or null values must be skipped.
        """
        # Row with s2_id=None (NaN) and valid s3_id → only s3_id should appear
        df = pd.DataFrame([
            {"s1_id": "s1_D", "s2_id": None, "s3_id": "s3_500", "match_score": 0.8},
        ])
        out = resolve(df, threshold=0.5)
        self.assertEqual(out.get("s1_D"), ["s3_500"])

    # ── Case 7: Duplicate candidate IDs in input ───────────────────────────
    def test_case_7_duplicate_candidate_ids_deduplicated(self):
        """
        Same (s1_id, target_id) appears twice with different scores.
        Only one entry should appear in the output (the higher score is kept).
        """
        df = pd.DataFrame([
            {"s1_id": "s1_E", "s2_id": "s2_999", "s3_id": None, "match_score": 0.85},
            {"s1_id": "s1_E", "s2_id": "s2_999", "s3_id": None, "match_score": 0.60},  # duplicate
        ])
        out = resolve(df, threshold=0.5)
        matched = out.get("s1_E", [])
        self.assertEqual(matched.count("s2_999"), 1, "Duplicate target IDs must be deduplicated")
        self.assertEqual(len(matched), 1)

    # ── Case 8: S1 entity with zero candidates ─────────────────────────────
    def test_case_8_s1_entity_zero_candidates(self):
        """
        Empty input → resolver returns empty DataFrame.
        The pipeline's resolved_df_to_map then assigns [] to all_s1_ids members.
        """
        df = pd.DataFrame(columns=["s1_id", "s2_id", "s3_id", "match_score"])
        result_df = self.resolver.resolve(df, config=self.empty_config)
        self.assertTrue(result_df.empty)

    # ── Per-source threshold ────────────────────────────────────────────────
    def test_per_source_threshold_s2_stricter(self):
        """
        S2 threshold = 0.90, S3 threshold = 0.50.
        A S2 candidate at 0.75 should be dropped.
        A S3 candidate at 0.75 should be kept.
        """
        s2_ids = {"s2_010", "s2_011"}
        s3_ids = {"s3_010"}
        df = make_candidates(
            {"s1": "s1_F", "s2": "s2_010", "score": 0.75},   # below s2-threshold
            {"s1": "s1_F", "s3": "s3_010", "score": 0.75},   # above s3-threshold
        )
        out = resolve(df, threshold=0.5, threshold_s2=0.90, threshold_s3=0.50,
                      s2_ids=s2_ids, s3_ids=s3_ids)
        matched = out.get("s1_F", [])
        self.assertNotIn("s2_010", matched, "s2 candidate below s2-threshold must be excluded")
        self.assertIn("s3_010", matched, "s3 candidate above s3-threshold must be kept")

    def test_per_source_threshold_s3_stricter(self):
        """
        S2 threshold = 0.50, S3 threshold = 0.90.
        S3 candidate at 0.75 must be dropped; S2 candidate at 0.75 must be kept.
        """
        s2_ids = {"s2_020"}
        s3_ids = {"s3_020"}
        df = make_candidates(
            {"s1": "s1_G", "s2": "s2_020", "score": 0.75},
            {"s1": "s1_G", "s3": "s3_020", "score": 0.75},
        )
        out = resolve(df, threshold=0.5, threshold_s2=0.50, threshold_s3=0.90,
                      s2_ids=s2_ids, s3_ids=s3_ids)
        matched = out.get("s1_G", [])
        self.assertIn("s2_020", matched)
        self.assertNotIn("s3_020", matched)

    # ── top_margin pruning ──────────────────────────────────────────────────
    def test_top_margin_keeps_close_candidates(self):
        """
        top_margin=0.10: best=0.95 → cutoff=0.85.
        Candidate at 0.92 (>= 0.85) must be kept.
        Candidate at 0.80 (< 0.85) must be dropped.
        """
        df = make_candidates(
            {"s1": "s1_H", "s2": "s2_101", "score": 0.95},
            {"s1": "s1_H", "s2": "s2_102", "score": 0.92},
            {"s1": "s1_H", "s3": "s3_201", "score": 0.80},
        )
        out = resolve(df, threshold=0.5, top_margin=0.10)
        matched = out.get("s1_H", [])
        self.assertIn("s2_101", matched)
        self.assertIn("s2_102", matched)
        self.assertNotIn("s3_201", matched, "s3_201 is 0.15 below best; exceeds margin=0.10")

    def test_top_margin_none_keeps_all_above_threshold(self):
        """top_margin=None (default) → no margin pruning, all above threshold kept."""
        df = make_candidates(
            {"s1": "s1_I", "s2": "s2_200", "score": 0.95},
            {"s1": "s1_I", "s2": "s2_201", "score": 0.55},
        )
        out = resolve(df, threshold=0.5, top_margin=None)
        matched = out.get("s1_I", [])
        self.assertIn("s2_200", matched)
        self.assertIn("s2_201", matched)

    def test_top_margin_zero_keeps_only_top_score(self):
        """
        top_margin=0.0 → only candidates EXACTLY equal to the best score are kept.
        With distinct floating-point scores this reduces to exactly one match.
        """
        df = make_candidates(
            {"s1": "s1_J", "s2": "s2_300", "score": 0.95},
            {"s1": "s1_J", "s2": "s2_301", "score": 0.90},
        )
        out = resolve(df, threshold=0.5, top_margin=0.0)
        matched = out.get("s1_J", [])
        self.assertIn("s2_300", matched)
        # s2_301 has score 0.90, best is 0.95 → 0.90 < 0.95 - 0.0 → dropped
        self.assertNotIn("s2_301", matched)

    # ── Config-driven thresholds override constructor ───────────────────────
    def test_config_overrides_constructor_threshold(self):
        """Config decision.threshold takes precedence over constructor default."""
        resolver = EntityResolver(threshold=0.9)  # high constructor default
        df = make_candidates({"s1": "s1_K", "s2": "s2_400", "score": 0.75})
        # Config says 0.5 → the candidate should pass
        result_df = resolver.resolve(df, config={"decision": {"threshold": 0.5}})
        self.assertFalse(result_df.empty)
        self.assertIn("s2_400", result_df["target_id"].tolist())

    # ── Empty input ─────────────────────────────────────────────────────────
    def test_empty_input_returns_empty_dataframe(self):
        df = pd.DataFrame()
        result_df = self.resolver.resolve(df, config=self.empty_config)
        self.assertTrue(result_df.empty)

    # ── Multi-entity: different outcomes ────────────────────────────────────
    def test_multiple_s1_entities_independent_decisions(self):
        """
        s1_1 → 2 matches (both above threshold)
        s1_2 → 1 match
        s1_3 → 0 matches (all below threshold)
        """
        df = pd.DataFrame([
            {"s1_id": "s1_1", "s2_id": "s2_A", "s3_id": None,   "match_score": 0.91},
            {"s1_id": "s1_1", "s2_id": None,   "s3_id": "s3_A", "match_score": 0.88},
            {"s1_id": "s1_2", "s2_id": "s2_B", "s3_id": None,   "match_score": 0.77},
            {"s1_id": "s1_3", "s2_id": "s2_C", "s3_id": None,   "match_score": 0.12},
        ])
        out = resolve(df, threshold=0.5)

        self.assertEqual(sorted(out.get("s1_1", [])), ["s2_A", "s3_A"])
        self.assertEqual(out.get("s1_2"), ["s2_B"])
        self.assertEqual(out.get("s1_3", []), [])  # nothing above threshold

    # ── Output schema ────────────────────────────────────────────────────────
    def test_output_columns_schema(self):
        """Resolved DataFrame must have exactly [s1_id, target_id, match_score, source]."""
        df = make_candidates({"s1": "s1_Z", "s2": "s2_999", "score": 0.9})
        result_df = self.resolver.resolve(df, config=self.empty_config)
        self.assertFalse(result_df.empty)
        for col in ("s1_id", "target_id", "match_score", "source"):
            self.assertIn(col, result_df.columns, f"Missing column: {col}")

    def test_no_duplicate_target_ids_in_output(self):
        """Even if the same (s1, target) appears multiple times, output is deduplicated."""
        df = pd.DataFrame([
            {"s1_id": "s1_M", "s2_id": "s2_dup", "s3_id": None, "match_score": 0.9},
            {"s1_id": "s1_M", "s2_id": "s2_dup", "s3_id": None, "match_score": 0.7},
            {"s1_id": "s1_M", "s2_id": "s2_dup", "s3_id": None, "match_score": 0.6},
        ])
        result_df = self.resolver.resolve(df, config=self.empty_config)
        s1_m_rows = result_df[result_df["s1_id"] == "s1_M"]
        self.assertEqual(len(s1_m_rows), 1)
        self.assertEqual(s1_m_rows.iloc[0]["target_id"], "s2_dup")

    # ── Compatibility with resolved_df_to_map ───────────────────────────────
    def test_compatible_with_resolved_df_to_map(self):
        """
        The resolved DataFrame must work with the pipeline's resolved_df_to_map.
        """
        from src.pipeline import resolved_df_to_map

        df = pd.DataFrame([
            {"s1_id": "s1_1", "s2_id": "s2_A", "s3_id": None,   "match_score": 0.9},
            {"s1_id": "s1_1", "s2_id": None,   "s3_id": "s3_A", "match_score": 0.85},
            {"s1_id": "s1_2", "s2_id": "s2_B", "s3_id": None,   "match_score": 0.3},  # below
        ])
        result_df = self.resolver.resolve(df, config=self.empty_config)
        all_s1 = ["s1_1", "s1_2", "s1_3"]
        cand_map = {
            "s1_1": ["s2_A", "s3_A"],
            "s1_2": ["s2_B"],
            "s1_3": [],
        }
        match_map = resolved_df_to_map(result_df, all_s1_ids=all_s1, candidate_map=cand_map)

        self.assertIn("s1_1", match_map)
        self.assertIn("s1_2", match_map)
        self.assertIn("s1_3", match_map)

        self.assertIn("s2_A", match_map["s1_1"])
        self.assertIn("s3_A", match_map["s1_1"])
        self.assertEqual(match_map["s1_2"], [])  # 0.3 below threshold
        self.assertEqual(match_map["s1_3"], [])  # no candidates at all


if __name__ == "__main__":
    unittest.main()
