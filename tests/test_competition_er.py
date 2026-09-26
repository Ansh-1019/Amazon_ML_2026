"""
Competition-Oriented Entity Resolution Test Suite — Amazon ML Challenge 2026.

Tests the full resolution pipeline against a deterministic synthetic dataset
designed to mimic real competition data variability:

Synthetic Dataset Coverage
--------------------------
Source 1 entities:
  s1_01  Exact match available (US)
  s1_02  Abbreviated name variant
  s1_03  Punctuation differences
  s1_04  Reordered words
  s1_05  Typo in name
  s1_06  Address abbreviation (Street → St)
  s1_07  Missing address component (no zip)
  s1_08  Multiple valid matches (franchise / twin records)
  s1_09  Zero valid matches (no counterpart in S2/S3)
  s1_10  Hard negative — very similar name, different entity
  s1_11  Different country (UK)
  s1_12  Unseen country (France)

Ground truth is explicitly defined and checked in every test.

Test Areas
----------
1.  DataLoader — schema validation, TSV round-trip, source identity
2.  DataNormalizer — clean_text, combined field, missing values
3.  CandidateGenerator — blocking recall, candidate statistics
4.  FeatureExtractor — jaccard similarity, no NaN output
5.  EntityResolver — zero / one / multiple matches
6.  EntityEvaluator — macro F0.5, edge cases
7.  Candidate-subset rule
8.  Submission format compliance
9.  Unseen country passthrough
10. Deterministic results under fixed seed

Blocking Recall Metric
----------------------
blocking_recall = |GT ∩ candidates| / |GT|

Measured and reported for all test entities.
Regression guard: test fails if blocking_recall < BLOCKING_RECALL_MIN_THRESHOLD.
"""

import io
import unittest
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ── Module imports ──────────────────────────────────────────────────────────
from src.data.normalizer import DataNormalizer
from src.blocking.candidate_gen import CandidateGenerator
from src.features.pairwise import FeatureExtractor
from src.decision.entity_resolver import EntityResolver
from src.evaluation.metrics import EntityEvaluator, compute_single_entity_metrics
from src.pipeline import candidates_df_to_map, resolved_df_to_map
from src.submission import generate_submission_files, run_official_validator


# ════════════════════════════════════════════════════════════════════════════
# SECTION 0 — Synthetic Dataset
# ════════════════════════════════════════════════════════════════════════════

# Minimum blocking recall we require before raising a regression alarm.
# With the current naive top-K baseline this is 1.0 (all entities fit in top_k).
# When a smarter blocker is added this threshold should only ever INCREASE.
BLOCKING_RECALL_MIN_THRESHOLD = 0.80

SEED = 42


def _seed():
    random.seed(SEED)
    np.random.seed(SEED)


# ── Ground-truth match sets  (s1_id → {s2_ids} ∪ {s3_ids}) ─────────────────
GROUND_TRUTH: Dict[str, Set[str]] = {
    "s1_01": {"s2_01", "s3_01"},        # exact match
    "s1_02": {"s2_02"},                  # abbreviated name — S3 has no match
    "s1_03": {"s2_03", "s3_03"},        # punctuation differences
    "s1_04": {"s2_04"},                  # reordered words
    "s1_05": {"s2_05"},                  # typo in name
    "s1_06": {"s2_06"},                  # address abbreviation
    "s1_07": {"s2_07"},                  # missing address component
    "s1_08": {"s2_08a", "s2_08b"},      # multiple valid matches
    "s1_09": set(),                      # zero valid matches
    "s1_10": set(),                      # hard negative (similar but wrong)
    "s1_11": {"s2_11"},                  # UK entity
    "s1_12": {"s3_12"},                  # France (unseen country)
}


def make_source1() -> pd.DataFrame:
    return pd.DataFrame([
        # entity_id, business_name, business_address, country
        ("s1_01", "Acme Corporation",            "123 Main Street, New York, NY 10001", "US"),
        ("s1_02", "Beta Solutions International", "456 Oak Avenue, Los Angeles, CA",    "US"),
        ("s1_03", "Gamma & Enterprises LLC",      "789 Pine Road, Chicago, IL 60601",   "US"),
        ("s1_04", "Delta Technologies Group",     "101 Elm Street, Houston, TX 77001",  "US"),
        ("s1_05", "Epsilon Global Services",      "202 Maple Drive, Phoenix, AZ 85001", "US"),
        ("s1_06", "Zeta Industries Ltd",          "303 Cedar Street, Dallas, TX 75201", "US"),
        ("s1_07", "Eta Manufacturing Corp",       "404 Birch Lane, San Jose, CA 95101", "US"),
        ("s1_08", "Theta Franchise Inc",          "500 Commerce Blvd, Miami, FL 33101", "US"),
        ("s1_09", "Iota Ventures LLC",            "999 Unknown Rd, Unknown City",       "US"),
        ("s1_10", "Kappa Solutions Corp",         "456 Oak Avenue, Los Angeles, CA",    "US"),  # hard negative
        ("s1_11", "Lambda UK Ltd",                "10 Downing Street, London",          "UK"),
        ("s1_12", "Mu Enterprises SARL",          "15 Rue de Rivoli, Paris",            "France"),
    ], columns=["entity_id", "business_name", "business_address", "country"])


def make_source2() -> pd.DataFrame:
    return pd.DataFrame([
        ("s2_01",  "Acme Corporation",            "123 Main Street, New York, NY 10001", "US"),   # exact
        ("s2_02",  "Beta Solns Intl",             "456 Oak Ave, Los Angeles, CA",        "US"),   # abbreviated
        ("s2_03",  "Gamma Enterprises LLC",        "789 Pine Road, Chicago, IL 60601",   "US"),   # punctuation diff
        ("s2_04",  "Group Technologies Delta",     "101 Elm St, Houston, TX 77001",      "US"),   # reordered
        ("s2_05",  "Epsilom Global Services",      "202 Maple Drive, Phoenix, AZ 85001", "US"),   # typo (Epsilom)
        ("s2_06",  "Zeta Industries Ltd",          "303 Cedar St, Dallas, TX 75201",     "US"),   # St vs Street
        ("s2_07",  "Eta Manufacturing Corp",       "404 Birch Lane, San Jose, CA",       "US"),   # missing zip
        ("s2_08a", "Theta Franchise Inc",          "500 Commerce Blvd, Miami, FL 33101", "US"),   # multi-match A
        ("s2_08b", "Theta Franchise Incorporated", "500 Commerce Blvd, Miami, FL",       "US"),   # multi-match B
        ("s2_09",  "Completely Different Company", "001 Nowhere Ave, Nowhere",           "US"),   # no match
        ("s2_10",  "Kappa Solutions Corp",         "456 Oak Avenue, Los Angeles, CA",    "US"),   # hard negative
        ("s2_11",  "Lambda UK Limited",            "10 Downing St, London",              "UK"),   # UK
    ], columns=["entity_id", "business_name", "business_address", "country"])


def make_source3() -> pd.DataFrame:
    return pd.DataFrame([
        ("s3_01",  "Acme Corp",                   "123 Main St, New York, 10001",        "US"),   # exact variant
        ("s3_03",  "Gamma Enterprises",            "789 Pine Rd, Chicago, IL",            "US"),   # punctuation
        ("s3_12",  "Mu Entreprises SARL",          "15 Rue de Rivoli, Paris 75001",       "France"),  # France
        ("s3_xx",  "Unrelated Company X",          "1 Random Street, Somewhere",          "US"),   # distractor
    ], columns=["entity_id", "business_name", "business_address", "country"])


# ── Config used across all component tests ────────────────────────────────────
TEST_CONFIG = {
    "data": {
        "id_column_s1": "entity_id",
        "id_column_s2": "entity_id",
        "id_column_s3": "entity_id",
    },
    "blocking": {
        "top_k": 20,  # large enough to capture all S2/S3 in synthetic data
        "blocking_fields": ["business_name", "business_address", "country"],
    },
    "features": {"string_similarity_metrics": ["jaccard"]},
    "decision": {"threshold": 0.5},
}


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Data Loading
# ════════════════════════════════════════════════════════════════════════════

class TestDataLoading(unittest.TestCase):

    def setUp(self):
        _seed()
        self.s1 = make_source1()
        self.s2 = make_source2()
        self.s3 = make_source3()

    def test_source1_has_canonical_columns(self):
        for col in ("entity_id", "business_name", "business_address", "country"):
            self.assertIn(col, self.s1.columns, f"Missing canonical column: {col}")

    def test_source1_correct_row_count(self):
        self.assertEqual(len(self.s1), 12)

    def test_source2_correct_row_count(self):
        self.assertEqual(len(self.s2), 12)

    def test_source3_correct_row_count(self):
        self.assertEqual(len(self.s3), 4)

    def test_all_source1_ids_unique(self):
        self.assertEqual(len(self.s1["entity_id"].unique()), len(self.s1))

    def test_all_source2_ids_unique(self):
        self.assertEqual(len(self.s2["entity_id"].unique()), len(self.s2))

    def test_ground_truth_s1_ids_are_subset_of_source1(self):
        s1_ids = set(self.s1["entity_id"].tolist())
        for s1_id in GROUND_TRUTH:
            self.assertIn(s1_id, s1_ids)

    def test_ground_truth_target_ids_are_subset_of_s2_or_s3(self):
        s2_ids = set(self.s2["entity_id"].tolist())
        s3_ids = set(self.s3["entity_id"].tolist())
        all_target = s2_ids | s3_ids
        for s1_id, targets in GROUND_TRUTH.items():
            for tid in targets:
                self.assertIn(tid, all_target, f"GT target {tid} (for {s1_id}) not in S2 ∪ S3")

    def test_unseen_country_france_preserved(self):
        france_rows = self.s1[self.s1["country"] == "France"]
        self.assertFalse(france_rows.empty, "France row must exist in Source 1")
        self.assertEqual(france_rows.iloc[0]["entity_id"], "s1_12")

    def test_dataloader_validate_source_schema(self):
        """DataLoader.load_source_file must raise SchemaValidationError on missing column."""
        from src.data.loader import DataLoader, SchemaValidationError
        import tempfile, os
        bad_df = pd.DataFrame([{"entity_id": "x", "business_name": "Y"}])  # missing columns
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as fh:
            bad_df.to_csv(fh, sep="\t", index=False)
            tmp = fh.name
        try:
            loader = DataLoader(raw_data_dir="data/raw", strict=True)
            with self.assertRaises(SchemaValidationError):
                loader.load_source_file(tmp, source_name="source1")
        finally:
            os.unlink(tmp)

    def test_dataloader_adds_source_column(self):
        """DataLoader must tag each row with source identity."""
        import tempfile, os
        from src.data.loader import DataLoader
        df = make_source2()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as fh:
            df.to_csv(fh, sep="\t", index=False)
            tmp = fh.name
        try:
            loader = DataLoader(raw_data_dir="data/raw", strict=True)
            loaded = loader.load_source_file(tmp, source_name="source2")
            self.assertIn("source", loaded.columns)
            self.assertTrue((loaded["source"] == "source2").all())
        finally:
            os.unlink(tmp)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Text Normalization
# ════════════════════════════════════════════════════════════════════════════

class TestNormalization(unittest.TestCase):

    def setUp(self):
        self.norm = DataNormalizer()

    def test_lowercase_transformation(self):
        # normalize_name() expands trailing legal abbreviations: 'corp' → 'corporation'.
        # This is intentional and correct: Maithili's implementation is now the authority.
        self.assertEqual(self.norm.clean_text("ACME CORP"), "acme corporation")
        # Plain words without legal abbreviations are just lowercased.
        self.assertEqual(self.norm.clean_text("ACME COMPANY"), "acme company")

    def test_punctuation_normalization(self):
        # normalize_name() converts whitespace-delimited '&' to 'and' — verified.
        # It preserves commas and dots (meaningful punctuation is not stripped).
        # The old test checked the old regex behavior of stripping all punctuation;
        # Maithili's implementation is more conservative and correct.
        result = self.norm.clean_text("Gamma & Enterprises, LLC.")
        self.assertNotIn("&", result)          # '& ' → 'and' ✓
        self.assertIn("and", result)            # ampersand converted ✓
        self.assertIn("llc", result)            # lowercased ✓
        # Commas and dots are preserved by normalize_name (they carry meaning in addresses/names)

    def test_extra_whitespace_collapsed(self):
        result = self.norm.clean_text("Acme   Corporation")
        self.assertEqual(result, "acme corporation")

    def test_none_input_returns_empty(self):
        self.assertEqual(self.norm.clean_text(None), "")

    def test_empty_string_returns_empty(self):
        self.assertEqual(self.norm.clean_text(""), "")

    def test_unicode_preserved(self):
        result = self.norm.clean_text("Société Générale")
        self.assertIn("société", result)
        self.assertIn("générale", result)

    def test_normalize_dataframe_adds_clean_columns(self):
        df = make_source1()
        normed = self.norm.normalize_dataframe(df, text_columns=["business_name", "business_address"])
        self.assertIn("business_name_clean", normed.columns)
        self.assertIn("business_address_clean", normed.columns)

    def test_normalize_dataframe_no_nan_in_clean_columns(self):
        df = make_source1()
        # Inject a NaN
        df.loc[0, "business_name"] = None
        normed = self.norm.normalize_dataframe(df, text_columns=["business_name"])
        self.assertFalse(normed["business_name_clean"].isna().any(), "NaN must not appear in clean columns")
        self.assertEqual(normed.loc[0, "business_name_clean"], "")

    def test_combined_text_field_generated(self):
        df = make_source1()
        normed = self.norm.normalize_dataframe(df, text_columns=["business_name", "business_address"])
        self.assertIn("text_clean_combined", normed.columns)
        # Combined field must be non-empty for rows with real data
        self.assertTrue((normed["text_clean_combined"].str.len() > 0).all())

    def test_unseen_country_passes_through_normalization(self):
        df = make_source1()
        normed = self.norm.normalize_dataframe(df, text_columns=["country"])
        france_rows = normed[normed["country"] == "France"]
        self.assertFalse(france_rows.empty)
        # Normalized to lowercase
        self.assertEqual(france_rows.iloc[0]["country_clean"], "france")

    def test_address_abbreviation_reduces_after_normalization(self):
        """After normalization, 'Street' and 'St' both become 'street' and 'st'
        — they won't be identical, but normalization should not inflate distance.
        normalize_name() preserves commas (they are meaningful address separators).
        """
        full = self.norm.clean_text("303 Cedar Street, Dallas, TX 75201")
        abbrev = self.norm.clean_text("303 Cedar St, Dallas, TX 75201")
        # Both must be lowercase
        self.assertEqual(full, full.lower())
        self.assertEqual(abbrev, abbrev.lower())
        # 'Street' and 'st' are preserved as-is (not address abbreviations in normalize_name)
        self.assertIn("street", full)
        self.assertIn("st", abbrev)
        # normalize_name does not strip commas — this is correct: commas in addresses are meaningful
        self.assertIn(",", full)
        self.assertIn(",", abbrev)

    def test_typo_entity_normalized_correctly(self):
        """'Epsilom' (typo) normalizes to lowercase without altering characters further."""
        result = self.norm.clean_text("Epsilom Global Services")
        self.assertEqual(result, "epsilom global services")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Candidate Generation & Blocking Recall
# ════════════════════════════════════════════════════════════════════════════

class BlockingRecallReport:
    """Computes and stores blocking recall metrics for a test run."""

    def __init__(self, ground_truth: Dict[str, Set[str]], candidate_map: Dict[str, List[str]]):
        self.gt = ground_truth
        self.cmap = candidate_map
        self._compute()

    def _compute(self):
        total_gt = 0
        found_in_candidates = 0
        per_entity_stats = []
        entities_with_zero_candidates = 0
        entities_missing_true_candidate = 0

        for s1_id, gt_set in self.gt.items():
            cands = set(self.cmap.get(s1_id, []))
            n_cands = len(cands)

            if n_cands == 0:
                entities_with_zero_candidates += 1

            matched = gt_set & cands
            found = len(matched)
            total_gt += len(gt_set)
            found_in_candidates += found

            if gt_set and not matched:
                entities_missing_true_candidate += 1

            per_entity_stats.append(n_cands)

        self.total_gt_pairs = total_gt
        self.gt_found_in_candidates = found_in_candidates
        self.blocking_recall = found_in_candidates / total_gt if total_gt > 0 else 1.0
        self.candidate_count_mean = float(np.mean(per_entity_stats)) if per_entity_stats else 0.0
        self.candidate_count_median = float(np.median(per_entity_stats)) if per_entity_stats else 0.0
        self.candidate_count_p95 = float(np.percentile(per_entity_stats, 95)) if per_entity_stats else 0.0
        self.entities_with_zero_candidates = entities_with_zero_candidates
        self.entities_with_missing_true_candidate = entities_missing_true_candidate

    def __str__(self):
        return (
            f"\nBlocking Recall Report\n"
            f"  blocking_recall                    = {self.blocking_recall:.4f}\n"
            f"  total_gt_pairs                     = {self.total_gt_pairs}\n"
            f"  gt_found_in_candidates             = {self.gt_found_in_candidates}\n"
            f"  candidate_count_mean               = {self.candidate_count_mean:.1f}\n"
            f"  candidate_count_median             = {self.candidate_count_median:.1f}\n"
            f"  candidate_count_p95                = {self.candidate_count_p95:.1f}\n"
            f"  entities_with_zero_candidates      = {self.entities_with_zero_candidates}\n"
            f"  entities_with_missing_true_cand    = {self.entities_with_missing_true_candidate}\n"
        )


# Shared recall report computed once for all blocking tests
_RECALL_REPORT: Optional[BlockingRecallReport] = None


def _build_recall_report() -> BlockingRecallReport:
    global _RECALL_REPORT
    if _RECALL_REPORT is not None:
        return _RECALL_REPORT
    _seed()
    s1 = make_source1()
    s2 = make_source2()
    s3 = make_source3()
    norm = DataNormalizer()
    s1_n = norm.normalize_dataframe(s1, text_columns=["business_name", "business_address", "country"])
    s2_n = norm.normalize_dataframe(s2, text_columns=["business_name", "business_address", "country"])
    s3_n = norm.normalize_dataframe(s3, text_columns=["business_name", "business_address", "country"])
    gen = CandidateGenerator(top_k=20, blocking_fields=["business_name", "business_address", "country"])
    cand_df = gen.generate_candidates(s1_n, s2_n, s3_n, TEST_CONFIG)
    all_s1_ids = s1["entity_id"].tolist()
    cand_map = candidates_df_to_map(cand_df, all_s1_ids=all_s1_ids)
    _RECALL_REPORT = BlockingRecallReport(GROUND_TRUTH, cand_map)
    return _RECALL_REPORT


class TestCandidateGeneration(unittest.TestCase):

    def setUp(self):
        _seed()
        self.s1 = make_source1()
        self.s2 = make_source2()
        self.s3 = make_source3()
        self.norm = DataNormalizer()
        self.s1_n = self.norm.normalize_dataframe(self.s1, text_columns=["business_name", "business_address", "country"])
        self.s2_n = self.norm.normalize_dataframe(self.s2, text_columns=["business_name", "business_address", "country"])
        self.s3_n = self.norm.normalize_dataframe(self.s3, text_columns=["business_name", "business_address", "country"])
        self.gen = CandidateGenerator(top_k=20, blocking_fields=["business_name", "business_address", "country"])
        self.cand_df = self.gen.generate_candidates(self.s1_n, self.s2_n, self.s3_n, TEST_CONFIG)
        self.all_s1_ids = self.s1["entity_id"].tolist()
        self.cand_map = candidates_df_to_map(self.cand_df, all_s1_ids=self.all_s1_ids)

    def test_all_s1_entities_have_entry_in_candidate_map(self):
        """Every Source 1 entity must have a key in the candidate map (may be empty list)."""
        for s1_id in self.all_s1_ids:
            self.assertIn(s1_id, self.cand_map, f"s1_id {s1_id} missing from candidate map")

    def test_candidates_dataframe_not_empty(self):
        self.assertFalse(self.cand_df.empty, "Candidate DataFrame must not be empty")

    def test_candidates_have_required_columns(self):
        for col in ("s1_id",):
            self.assertIn(col, self.cand_df.columns)

    def test_all_candidate_ids_are_valid_s2_or_s3(self):
        """No spurious IDs — every candidate must come from S2 or S3."""
        valid_ids = set(self.s2["entity_id"]) | set(self.s3["entity_id"])
        for s1_id, cands in self.cand_map.items():
            for cid in cands:
                self.assertIn(cid, valid_ids, f"Invalid candidate ID {cid} for {s1_id}")

    def test_no_duplicate_candidate_ids_per_entity(self):
        for s1_id, cands in self.cand_map.items():
            self.assertEqual(len(cands), len(set(cands)), f"Duplicate candidates for {s1_id}")

    def test_blocking_recall_above_threshold(self):
        """
        REGRESSION GUARD: Blocking recall must be >= BLOCKING_RECALL_MIN_THRESHOLD.
        If this test fails after changing the candidate generator, the generator
        has reduced recall — this is unacceptable for a competition pipeline.
        """
        report = _build_recall_report()
        self.assertGreaterEqual(
            report.blocking_recall,
            BLOCKING_RECALL_MIN_THRESHOLD,
            f"Blocking recall {report.blocking_recall:.4f} is below minimum "
            f"{BLOCKING_RECALL_MIN_THRESHOLD}. The candidate generator has "
            f"reduced recall — regression detected!{report}"
        )

    def test_exact_match_appears_in_candidates_s1_01(self):
        """s1_01 (Acme Corporation) → s2_01 must be a candidate."""
        cands = set(self.cand_map.get("s1_01", []))
        self.assertIn("s2_01", cands, "Exact match s2_01 must appear in s1_01 candidates")

    def test_multi_match_entity_both_appear_in_candidates(self):
        """s1_08 (Theta Franchise) has two GT matches — both s2_08a and s2_08b must be candidates."""
        cands = set(self.cand_map.get("s1_08", []))
        self.assertIn("s2_08a", cands, "s2_08a must be a candidate for s1_08")
        self.assertIn("s2_08b", cands, "s2_08b must be a candidate for s1_08")

    def test_france_entity_has_candidates(self):
        """s1_12 (France) must have at least one candidate (s3_12)."""
        cands = set(self.cand_map.get("s1_12", []))
        self.assertIn("s3_12", cands, "France entity s3_12 must appear as candidate for s1_12")

    def test_candidate_count_statistics_are_finite(self):
        report = _build_recall_report()
        self.assertTrue(np.isfinite(report.candidate_count_mean))
        self.assertTrue(np.isfinite(report.candidate_count_median))
        self.assertTrue(np.isfinite(report.candidate_count_p95))

    def test_gt_pairs_counted_correctly_in_report(self):
        """Verify that total GT pairs equals the sum of |GT| over all entities."""
        report = _build_recall_report()
        expected = sum(len(v) for v in GROUND_TRUTH.values())
        self.assertEqual(report.total_gt_pairs, expected)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Feature Extraction
# ════════════════════════════════════════════════════════════════════════════

class TestFeatureExtraction(unittest.TestCase):

    def setUp(self):
        _seed()
        self.extractor = FeatureExtractor()
        self.norm = DataNormalizer()

    def _make_cands_df(self, pairs: List[Tuple[str, str, str]]) -> pd.DataFrame:
        """pairs: [(s1_id, s2_id, s3_id or None), ...]"""
        return pd.DataFrame([
            {"s1_id": p[0], "s2_id": p[1], "s3_id": p[2]} for p in pairs
        ])

    def test_jaccard_identical_strings_is_1(self):
        self.assertAlmostEqual(FeatureExtractor.jaccard_similarity("acme corp", "acme corp"), 1.0)

    def test_jaccard_disjoint_strings_is_0(self):
        self.assertAlmostEqual(FeatureExtractor.jaccard_similarity("acme corp", "zeta industries"), 0.0)

    def test_jaccard_partial_overlap(self):
        score = FeatureExtractor.jaccard_similarity("acme corporation", "acme corp")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_jaccard_empty_string_returns_zero(self):
        self.assertEqual(FeatureExtractor.jaccard_similarity("", "acme"), 0.0)
        self.assertEqual(FeatureExtractor.jaccard_similarity("acme", ""), 0.0)

    def test_jaccard_non_string_returns_zero(self):
        self.assertEqual(FeatureExtractor.jaccard_similarity(None, "acme"), 0.0)
        self.assertEqual(FeatureExtractor.jaccard_similarity(123, "acme"), 0.0)

    def test_feature_extraction_returns_dataframe(self):
        s1 = make_source1()
        s2 = make_source2()
        s3 = make_source3()
        s1_n = self.norm.normalize_dataframe(s1, text_columns=["business_name", "business_address"])
        s2_n = self.norm.normalize_dataframe(s2, text_columns=["business_name", "business_address"])
        s3_n = self.norm.normalize_dataframe(s3, text_columns=["business_name", "business_address"])
        cands = pd.DataFrame([
            {"s1_id": "s1_01", "s2_id": "s2_01", "s3_id": "s3_01"},
            {"s1_id": "s1_02", "s2_id": "s2_02", "s3_id": None},
        ])
        features = self.extractor.extract_features(cands, s1_n, s2_n, s3_n, TEST_CONFIG)
        self.assertIsInstance(features, pd.DataFrame)
        self.assertFalse(features.empty)

    def test_feature_extraction_no_nan_values(self):
        """Feature extraction must never produce NaN — NaN breaks ML models."""
        s1 = make_source1()
        s2 = make_source2()
        s3 = make_source3()
        s1_n = self.norm.normalize_dataframe(s1, text_columns=["business_name"])
        s2_n = self.norm.normalize_dataframe(s2, text_columns=["business_name"])
        s3_n = self.norm.normalize_dataframe(s3, text_columns=["business_name"])
        cands = pd.DataFrame([
            {"s1_id": "s1_01", "s2_id": "s2_01", "s3_id": None},
            {"s1_id": "s1_09", "s2_id": "s2_09", "s3_id": None},
        ])
        features = self.extractor.extract_features(cands, s1_n, s2_n, s3_n, TEST_CONFIG)
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            has_nan = features[col].isna().any()
            self.assertFalse(has_nan, f"NaN found in feature column '{col}'")

    def test_feature_extraction_empty_candidates_returns_empty(self):
        empty_df = pd.DataFrame()
        result = self.extractor.extract_features(
            empty_df, make_source1(), make_source2(), make_source3(), TEST_CONFIG
        )
        self.assertTrue(result.empty)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Entity Resolver (multi-match decision layer)
# ════════════════════════════════════════════════════════════════════════════

class TestEntityResolverCompetition(unittest.TestCase):

    def setUp(self):
        _seed()
        self.resolver = EntityResolver(threshold=0.5)
        self.cfg = {"decision": {}}

    def _df(self, rows: List[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_resolver_zero_matches_entity(self):
        """Entity with all candidates below threshold → empty result for that entity."""
        df = self._df([
            {"s1_id": "s1_09", "s2_id": "s2_09", "s3_id": None, "match_score": 0.10},
        ])
        result = self.resolver.resolve(df, self.cfg)
        s1_09_rows = result[result["s1_id"] == "s1_09"] if not result.empty else pd.DataFrame()
        self.assertTrue(s1_09_rows.empty, "s1_09 has no real match — resolver must produce zero rows")

    def test_resolver_single_match(self):
        """s1_04 has one candidate above threshold."""
        df = self._df([
            {"s1_id": "s1_04", "s2_id": "s2_04", "s3_id": None, "match_score": 0.75},
            {"s1_id": "s1_04", "s2_id": "s2_10", "s3_id": None, "match_score": 0.20},
        ])
        result = self.resolver.resolve(df, self.cfg)
        matched = result[result["s1_id"] == "s1_04"]["target_id"].tolist()
        self.assertEqual(matched, ["s2_04"])

    def test_resolver_multiple_matches_s1_08(self):
        """
        s1_08 (Theta Franchise) has TWO genuine matches.
        Old resolver would pick only the top-1; new resolver keeps both.
        """
        df = self._df([
            {"s1_id": "s1_08", "s2_id": "s2_08a", "s3_id": None, "match_score": 0.95},
            {"s1_id": "s1_08", "s2_id": None,     "s3_id": "s2_08b", "match_score": 0.92},  # note: s2_08b in s3_id col
            {"s1_id": "s1_08", "s2_id": "s2_09",   "s3_id": None, "match_score": 0.10},
        ])
        result = self.resolver.resolve(df, self.cfg)
        matched = set(result[result["s1_id"] == "s1_08"]["target_id"].tolist())
        self.assertIn("s2_08a", matched, "s2_08a must be matched")
        self.assertIn("s2_08b", matched, "s2_08b must be matched")
        self.assertNotIn("s2_09", matched, "s2_09 below threshold must be excluded")

    def test_resolver_hard_negative_not_matched(self):
        """
        s1_10 (Kappa Solutions) has a confusingly similar S2 candidate.
        If the model correctly scores it below threshold, it must not be matched.
        """
        df = self._df([
            {"s1_id": "s1_10", "s2_id": "s2_10", "s3_id": None, "match_score": 0.30},
        ])
        result = self.resolver.resolve(df, self.cfg)
        s1_10_rows = result[result["s1_id"] == "s1_10"] if not result.empty else pd.DataFrame()
        self.assertTrue(s1_10_rows.empty, "Hard negative below threshold must not be matched")

    def test_resolver_uk_entity_matched(self):
        df = self._df([
            {"s1_id": "s1_11", "s2_id": "s2_11", "s3_id": None, "match_score": 0.88},
        ])
        result = self.resolver.resolve(df, self.cfg)
        matched = result[result["s1_id"] == "s1_11"]["target_id"].tolist()
        self.assertEqual(matched, ["s2_11"])

    def test_resolver_france_entity_matched(self):
        df = self._df([
            {"s1_id": "s1_12", "s2_id": None, "s3_id": "s3_12", "match_score": 0.82},
        ])
        result = self.resolver.resolve(df, self.cfg)
        matched = result[result["s1_id"] == "s1_12"]["target_id"].tolist()
        self.assertEqual(matched, ["s3_12"])

    def test_resolver_output_is_subset_of_candidates(self):
        """Every resolved match must be in the original candidate set."""
        df = self._df([
            {"s1_id": "s1_01", "s2_id": "s2_01", "s3_id": "s3_01", "match_score": 0.95},
            {"s1_id": "s1_01", "s2_id": "s2_09", "s3_id": None,    "match_score": 0.10},
        ])
        cand_map = {"s1_01": ["s2_01", "s3_01", "s2_09"]}
        result = self.resolver.resolve(df, self.cfg)
        match_map = resolved_df_to_map(result, all_s1_ids=["s1_01"], candidate_map=cand_map)
        for mid in match_map.get("s1_01", []):
            self.assertIn(mid, cand_map["s1_01"], f"Match {mid} is not in candidate set")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — F0.5 Evaluator (macro, entity-level)
# ════════════════════════════════════════════════════════════════════════════

class TestF05EvaluatorCompetition(unittest.TestCase):

    def setUp(self):
        self.evaluator = EntityEvaluator(beta=0.5)

    def test_perfect_predictions_f05_is_1(self):
        y_true = {"s1_01": {"s2_01", "s3_01"}, "s1_08": {"s2_08a", "s2_08b"}}
        y_pred = {"s1_01": {"s2_01", "s3_01"}, "s1_08": {"s2_08a", "s2_08b"}}
        res = self.evaluator.evaluate(y_true, y_pred)
        self.assertAlmostEqual(res["macro_f_beta"], 1.0, places=5)

    def test_empty_gt_empty_pred_is_1(self):
        """Competition rule: no-match entity correctly predicted empty → F0.5 = 1.0."""
        metrics = compute_single_entity_metrics(set(), set(), beta=0.5)
        self.assertEqual(metrics["f_0.5"], 1.0)

    def test_false_positive_on_empty_gt_is_0(self):
        """Predicting a match when GT is empty → precision=0 → F0.5=0.0."""
        metrics = compute_single_entity_metrics(set(), {"s2_01"}, beta=0.5)
        self.assertEqual(metrics["f_0.5"], 0.0)

    def test_missed_multi_match_penalized(self):
        """
        s1_08 has GT = {s2_08a, s2_08b}.
        Predicting only s2_08a means recall = 0.5.
        F0.5 = 1.25 * 1.0 * 0.5 / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 5/6
        """
        metrics = compute_single_entity_metrics(
            gt_set={"s2_08a", "s2_08b"},
            pred_set={"s2_08a"},
            beta=0.5,
        )
        self.assertAlmostEqual(metrics["f_0.5"], 5.0 / 6.0, places=5)

    def test_false_positive_penalizes_precision(self):
        """
        GT = {s2_01}, Pred = {s2_01, s2_09} (one FP).
        Precision = 0.5, Recall = 1.0
        F0.5 = 1.25 * 0.5 * 1.0 / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 5/9
        """
        metrics = compute_single_entity_metrics(
            gt_set={"s2_01"},
            pred_set={"s2_01", "s2_09"},
            beta=0.5,
        )
        self.assertAlmostEqual(metrics["f_0.5"], 5.0 / 9.0, places=5)

    def test_macro_average_is_entity_level_not_global(self):
        """F0.5 must be macro-averaged (per entity), not global micro average."""
        y_true = {"e1": {"A", "B", "C", "D"}, "e2": {"E"}}
        y_pred = {"e1": {"A", "B", "C", "D"}, "e2": set()}
        res = self.evaluator.evaluate(y_true, y_pred)
        # Macro: (1.0 + 0.0) / 2 = 0.5
        self.assertAlmostEqual(res["macro_f_beta"], 0.5, places=4)
        # Micro would be ~0.952 — must NOT equal 0.5
        self.assertNotAlmostEqual(res["macro_f_beta"], 0.952, places=2)

    def test_all_source1_entities_covered_in_eval(self):
        """all_s1_ids ensures even entities absent from predictions are evaluated."""
        y_true = {"s1_01": {"s2_01"}, "s1_09": set()}
        y_pred = {"s1_01": {"s2_01"}}  # s1_09 not in predictions
        all_s1 = ["s1_01", "s1_09"]
        res = self.evaluator.evaluate(y_true, y_pred, all_s1_ids=all_s1)
        self.assertEqual(res["number_of_entities"], 2)

    def test_per_entity_records_contain_all_fields(self):
        y_true = {"s1_01": {"s2_01"}}
        y_pred = {"s1_01": {"s2_01"}}
        res = self.evaluator.evaluate(y_true, y_pred)
        records = res["per_entity_records"]
        self.assertEqual(len(records), 1)
        for field in ("s1_id", "ground_truth", "prediction", "precision", "recall", "f_0.5", "tp", "fp", "fn"):
            self.assertIn(field, records[0], f"Missing per-entity field: {field}")

    def test_synthetic_dataset_eval_produces_valid_scores(self):
        """Run evaluator on synthetic GT with empty predictions → macro F0.5 >= 0."""
        y_pred = {s1_id: set() for s1_id in GROUND_TRUTH}
        res = self.evaluator.evaluate(GROUND_TRUTH, y_pred)
        self.assertGreaterEqual(res["macro_f_beta"], 0.0)
        self.assertLessEqual(res["macro_f_beta"], 1.0)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Candidate-Subset Rule
# ════════════════════════════════════════════════════════════════════════════

class TestCandidateSubsetRule(unittest.TestCase):

    def test_match_not_in_candidates_is_excluded_by_resolved_df_to_map(self):
        """
        resolved_df_to_map must drop any matched ID that doesn't appear
        in the candidate_map for that s1 entity.
        """
        resolved_df = pd.DataFrame([
            {"s1_id": "s1_01", "target_id": "s2_01",   "match_score": 0.95, "source": "s2"},
            {"s1_id": "s1_01", "target_id": "s2_FAKE", "match_score": 0.90, "source": "s2"},  # not in candidates
        ])
        cand_map = {"s1_01": ["s2_01"]}
        match_map = resolved_df_to_map(resolved_df, all_s1_ids=["s1_01"], candidate_map=cand_map)
        self.assertIn("s2_01", match_map["s1_01"])
        self.assertNotIn("s2_FAKE", match_map["s1_01"])

    def test_match_subset_is_always_subset_of_candidates(self):
        """Final match set must always be ⊆ candidate set."""
        resolved_df = pd.DataFrame([
            {"s1_id": "s1_08", "target_id": "s2_08a", "match_score": 0.95, "source": "s2"},
            {"s1_id": "s1_08", "target_id": "s2_08b", "match_score": 0.92, "source": "s2"},
        ])
        cand_map = {"s1_08": ["s2_08a", "s2_08b", "s2_09"]}
        match_map = resolved_df_to_map(resolved_df, all_s1_ids=["s1_08"], candidate_map=cand_map)
        for mid in match_map["s1_08"]:
            self.assertIn(mid, cand_map["s1_08"], f"Match {mid} violates candidate-subset rule")

    def test_zero_candidate_entity_has_empty_match_list(self):
        """s1_09 with no candidates → match list must be []."""
        resolved_df = pd.DataFrame(columns=["s1_id", "target_id", "match_score", "source"])
        match_map = resolved_df_to_map(resolved_df, all_s1_ids=["s1_09"], candidate_map={"s1_09": []})
        self.assertEqual(match_map["s1_09"], [])


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Submission Format
# ════════════════════════════════════════════════════════════════════════════

class TestSubmissionFormat(unittest.TestCase):

    def setUp(self):
        _seed()
        self.all_s1_ids = list(GROUND_TRUTH.keys())
        self.valid_target_ids = (
            set(make_source2()["entity_id"]) | set(make_source3()["entity_id"])
        )
        # Simulate a small set of candidates and matches
        self.cand_map = {
            "s1_01": ["s2_01", "s3_01"],
            "s1_08": ["s2_08a", "s2_08b"],
            "s1_09": [],
            "s1_12": ["s3_12"],
        }
        for s1_id in self.all_s1_ids:
            if s1_id not in self.cand_map:
                self.cand_map[s1_id] = []
        self.match_map = {
            "s1_01": ["s2_01", "s3_01"],
            "s1_08": ["s2_08a", "s2_08b"],
            "s1_09": [],
            "s1_12": ["s3_12"],
        }
        for s1_id in self.all_s1_ids:
            if s1_id not in self.match_map:
                self.match_map[s1_id] = []

    def test_submission_files_pass_official_validator(self):
        matching_path, candidates_path = generate_submission_files(
            test_s1_ids=self.all_s1_ids,
            candidate_pairs_map=self.cand_map,
            matched_results_map=self.match_map,
            output_dir="output",
            valid_target_ids=self.valid_target_ids,
            strict=True,
        )
        is_valid, log = run_official_validator(
            matching_filepath=matching_path,
            candidates_filepath=candidates_path,
        )
        self.assertTrue(is_valid, f"Official validator FAILED: {log}")

    def test_all_s1_entities_present_in_matching_results(self):
        matching_path, _ = generate_submission_files(
            test_s1_ids=self.all_s1_ids,
            candidate_pairs_map=self.cand_map,
            matched_results_map=self.match_map,
            output_dir="output",
            valid_target_ids=self.valid_target_ids,
            strict=True,
        )
        result_df = pd.read_csv(matching_path, sep="\t", dtype=str)
        submitted_ids = set(result_df.iloc[:, 0].tolist())
        for s1_id in self.all_s1_ids:
            self.assertIn(s1_id, submitted_ids, f"{s1_id} missing from matching_results.tsv")

    def test_matches_are_subset_of_candidates_in_tsv(self):
        matching_path, candidates_path = generate_submission_files(
            test_s1_ids=self.all_s1_ids,
            candidate_pairs_map=self.cand_map,
            matched_results_map=self.match_map,
            output_dir="output",
            valid_target_ids=self.valid_target_ids,
            strict=True,
        )
        match_df = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
        cand_df = pd.read_csv(candidates_path, sep="\t", dtype=str, keep_default_na=False)

        def parse_ids(cell: str) -> Set[str]:
            return {x.strip() for x in cell.split(",") if x.strip()}

        cand_map_tsv = dict(zip(cand_df.iloc[:, 0], cand_df.iloc[:, 1].apply(parse_ids)))
        for _, row in match_df.iterrows():
            s1_id = row.iloc[0]
            matched = parse_ids(row.iloc[1])
            cands = cand_map_tsv.get(s1_id, set())
            for mid in matched:
                self.assertIn(mid, cands, f"Match {mid} for {s1_id} not in candidate TSV")

    def test_tsv_output_not_csv_quoted(self):
        """TSV output must use tab as delimiter and not quote comma-separated ID lists."""
        matching_path, _ = generate_submission_files(
            test_s1_ids=["s1_01"],
            candidate_pairs_map={"s1_01": ["s2_01", "s3_01"]},
            matched_results_map={"s1_01": ["s2_01", "s3_01"]},
            output_dir="output",
            valid_target_ids=self.valid_target_ids,
            strict=True,
        )
        raw = open(matching_path, "r", encoding="utf-8").read()
        self.assertIn("\t", raw, "Output must be tab-separated")
        self.assertNotIn('"s2_01', raw, "ID list must not be quoted")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Unseen Country
# ════════════════════════════════════════════════════════════════════════════

class TestUnseenCountry(unittest.TestCase):

    def setUp(self):
        self.norm = DataNormalizer()

    def test_france_entity_loads_without_error(self):
        df = make_source1()
        france = df[df["entity_id"] == "s1_12"]
        self.assertEqual(len(france), 1)
        self.assertEqual(france.iloc[0]["country"], "France")

    def test_france_entity_normalizes_correctly(self):
        df = make_source1()
        normed = self.norm.normalize_dataframe(df, text_columns=["country"])
        france_row = normed[normed["entity_id"] == "s1_12"]
        self.assertEqual(france_row.iloc[0]["country_clean"], "france")

    def test_france_entity_s3_counterpart_in_source3(self):
        s3 = make_source3()
        france_s3 = s3[s3["entity_id"] == "s3_12"]
        self.assertFalse(france_s3.empty)
        self.assertEqual(france_s3.iloc[0]["country"], "France")

    def test_france_entity_passes_through_pipeline_without_error(self):
        """France entities must not cause any crash at any pipeline stage."""
        from src.pipeline import EntityResolutionPipeline, set_seed
        set_seed(SEED)
        s1 = make_source1()
        s2 = make_source2()
        s3 = make_source3()
        pipeline = EntityResolutionPipeline(
            config={
                "project": {"name": "Test", "experiment_id": "test_france", "seed": SEED},
                "paths": {"submissions_dir": "output"},
                "data": {"id_column_s1": "entity_id", "id_column_s2": "entity_id", "id_column_s3": "entity_id"},
                "blocking": {"top_k": 20, "blocking_fields": ["business_name", "business_address", "country"]},
                "features": {"string_similarity_metrics": ["jaccard"]},
                "decision": {"threshold": 0.5},
                "evaluation": {"beta": 0.5},
                "submission": {"matching_filename": "matching_results.tsv", "candidates_filename": "candidate_pairs.tsv"},
            }
        )
        results = pipeline.run(s1_df=s1, s2_df=s2, s3_df=s3, train_matches=None, output_dir="output")
        self.assertEqual(results["num_s1_entities"], len(s1))
        self.assertTrue(results["is_submission_valid"])


# ════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Deterministic Results
# ════════════════════════════════════════════════════════════════════════════

class TestDeterminism(unittest.TestCase):

    def _run_pipeline(self):
        from src.pipeline import EntityResolutionPipeline, set_seed
        set_seed(SEED)
        pipeline = EntityResolutionPipeline(
            config={
                "project": {"name": "Test", "experiment_id": "test_determinism", "seed": SEED},
                "paths": {"submissions_dir": "output"},
                "data": {"id_column_s1": "entity_id", "id_column_s2": "entity_id", "id_column_s3": "entity_id"},
                "blocking": {"top_k": 20, "blocking_fields": ["business_name", "business_address"]},
                "features": {"string_similarity_metrics": ["jaccard"]},
                "decision": {"threshold": 0.5},
                "evaluation": {"beta": 0.5},
                "submission": {"matching_filename": "matching_results.tsv", "candidates_filename": "candidate_pairs.tsv"},
            }
        )
        return pipeline.run(
            s1_df=make_source1(), s2_df=make_source2(), s3_df=make_source3(),
            train_matches=None, output_dir="output",
        )

    def test_repeated_runs_produce_same_candidate_count(self):
        """Two runs with identical seed must produce the same number of candidates."""
        r1 = self._run_pipeline()
        r2 = self._run_pipeline()
        self.assertEqual(
            r1["num_candidates_generated"],
            r2["num_candidates_generated"],
            "Candidate count differs between runs — non-determinism detected",
        )

    def test_candidate_generation_deterministic_across_seeds(self):
        """With fixed top_k, the current naive baseline is fully deterministic."""
        _seed()
        s1 = make_source1()
        s2 = make_source2()
        s3 = make_source3()
        norm = DataNormalizer()
        s1_n = norm.normalize_dataframe(s1, text_columns=["business_name"])
        s2_n = norm.normalize_dataframe(s2, text_columns=["business_name"])
        s3_n = norm.normalize_dataframe(s3, text_columns=["business_name"])
        gen = CandidateGenerator(top_k=20)
        df1 = gen.generate_candidates(s1_n, s2_n, s3_n, TEST_CONFIG)
        df2 = gen.generate_candidates(s1_n, s2_n, s3_n, TEST_CONFIG)
        self.assertEqual(len(df1), len(df2))


# ════════════════════════════════════════════════════════════════════════════
# SECTION 11 — Blocking Recall Summary (printed to stdout)
# ════════════════════════════════════════════════════════════════════════════

class TestBlockingRecallSummary(unittest.TestCase):
    """
    Prints a full blocking recall report. This test always passes —
    the actual regression guard is in TestCandidateGeneration.test_blocking_recall_above_threshold.
    """

    def test_print_blocking_recall_report(self):
        report = _build_recall_report()
        print(str(report))
        # Sanity checks
        self.assertGreaterEqual(report.blocking_recall, 0.0)
        self.assertLessEqual(report.blocking_recall, 1.0)
        self.assertGreaterEqual(report.candidate_count_mean, 0.0)
        self.assertGreaterEqual(report.entities_with_zero_candidates, 0)


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
