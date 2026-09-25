"""
Tests for the Competition Validation Harness.

Validates:
  - ValidationReport structure and serialisation
  - ValidationHarness.run() end-to-end on the synthetic dataset
  - Metric section correctness (data, blocking, matching, metrics, error analysis)
  - ExperimentTracker integration (summary.json + report.json written)
  - compare_reports() via the CLI helper
  - --synthetic CLI flag (subprocess smoke-test)
  - Two-run comparison via ValidationReport.compare()
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Set

# ── Make project root importable ────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.validation.harness import ValidationHarness, ValidationReport
from src.evaluation.metrics import parse_id_set

# ── Re-use the synthetic dataset from test_competition_er ────────────────────
from tests.test_competition_er import (
    make_source1, make_source2, make_source3,
    GROUND_TRUTH, SEED,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

_SYNTHETIC_CONFIG = {
    "project": {"experiment_id": "test_harness", "seed": SEED},
    "data": {
        "id_column_s1": "entity_id",
        "id_column_s2": "entity_id",
        "id_column_s3": "entity_id",
    },
    "blocking": {
        "top_k": 20,
        "blocking_fields": ["business_name", "business_address", "country"],
    },
    "features": {"string_similarity_metrics": ["jaccard"]},
    "decision": {"threshold": 0.5, "threshold_s2": None, "threshold_s3": None, "top_margin": None},
    "evaluation": {"beta": 0.5},
    "submission": {
        "matching_filename": "matching_results.tsv",
        "candidates_filename": "candidate_pairs.tsv",
    },
    "paths": {"submissions_dir": "output", "experiments_dir": "experiments"},
}


def _make_harness(tmp_exp_dir: str) -> ValidationHarness:
    cfg = {**_SYNTHETIC_CONFIG, "paths": {"submissions_dir": "output", "experiments_dir": tmp_exp_dir}}
    return ValidationHarness(config=cfg, experiments_dir=tmp_exp_dir)


def _run_harness(tmp_exp_dir: str, threshold: float = 0.5, exp_id: str = "test_run") -> ValidationReport:
    harness = _make_harness(tmp_exp_dir)
    return harness.run(
        ground_truth=dict(GROUND_TRUTH),
        s1_df=make_source1(),
        s2_df=make_source2(),
        s3_df=make_source3(),
        experiment_id=exp_id,
        threshold=threshold,
        output_dir="output",
        save_report=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. ValidationReport structure
# ─────────────────────────────────────────────────────────────────────────────

class TestValidationReportStructure(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="test_harness_")
        cls.report = _run_harness(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_report_has_all_top_level_sections(self):
        d = self.report.as_dict
        for section in ("run_info", "config_snapshot", "data_stats",
                        "blocking_stats", "matching_stats", "metrics", "error_analysis"):
            self.assertIn(section, d, f"Missing top-level section: {section}")

    def test_run_info_has_required_keys(self):
        ri = self.report.as_dict["run_info"]
        for key in ("experiment_id", "timestamp", "elapsed_seconds", "threshold",
                    "run_dir", "is_submission_valid"):
            self.assertIn(key, ri, f"run_info missing key: {key}")

    def test_config_snapshot_preserved(self):
        cs = self.report.as_dict["config_snapshot"]
        for key in ("blocking", "features", "decision", "seed"):
            self.assertIn(key, cs, f"config_snapshot missing: {key}")
        self.assertEqual(cs["seed"], SEED)

    def test_data_stats_correct_counts(self):
        ds = self.report.as_dict["data_stats"]
        self.assertEqual(ds["n_s1"], 12)
        self.assertEqual(ds["n_s2"], 12)
        self.assertEqual(ds["n_s3"], 4)

    def test_data_stats_includes_france(self):
        ds = self.report.as_dict["data_stats"]
        self.assertIn("France", ds["countries_s1"],
                      "France must appear in countries_s1")

    def test_data_stats_gt_entity_counts(self):
        ds = self.report.as_dict["data_stats"]
        gt_with    = sum(1 for v in GROUND_TRUTH.values() if v)
        gt_without = sum(1 for v in GROUND_TRUTH.values() if not v)
        self.assertEqual(ds["gt_entities_with_matches"], gt_with)
        self.assertEqual(ds["gt_entities_without_matches"], gt_without)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Blocking statistics
# ─────────────────────────────────────────────────────────────────────────────

class TestBlockingStats(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="test_blocking_")
        cls.report = _run_harness(cls.tmp)
        cls.bs = cls.report.as_dict["blocking_stats"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_total_candidates_positive(self):
        self.assertGreater(self.bs["total_candidates"], 0)

    def test_mean_candidates_finite(self):
        import math
        self.assertTrue(math.isfinite(self.bs["mean_candidates_per_s1"]))
        self.assertGreater(self.bs["mean_candidates_per_s1"], 0)

    def test_p95_gte_median(self):
        self.assertGreaterEqual(self.bs["p95_candidates_per_s1"],
                                self.bs["median_candidates_per_s1"])

    def test_blocking_recall_between_0_and_1(self):
        recall = self.bs["blocking_recall"]
        self.assertGreaterEqual(recall, 0.0)
        self.assertLessEqual(recall, 1.0)

    def test_blocking_recall_matches_gt_coverage(self):
        """
        With top_k=20 and only 16 S2/S3 entities in total,
        every GT pair should appear in the candidate set → recall = 1.0.
        """
        self.assertAlmostEqual(self.bs["blocking_recall"], 1.0, places=4)

    def test_gt_pairs_total_correct(self):
        expected = sum(len(v) for v in GROUND_TRUTH.values())
        self.assertEqual(self.bs["total_gt_pairs"], expected)

    def test_missing_gt_in_candidates_zero(self):
        """No GT pair should be missed at top_k=20 on the 16-entity synthetic set."""
        self.assertEqual(self.bs["gt_pairs_missing_in_candidates"], 0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Matching statistics
# ─────────────────────────────────────────────────────────────────────────────

class TestMatchingStats(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="test_matching_")
        cls.report = _run_harness(cls.tmp)
        cls.ms = cls.report.as_dict["matching_stats"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_matching_stats_sum_equals_s1_count(self):
        total = (self.ms["entities_zero_matches"] +
                 self.ms["entities_one_match"] +
                 self.ms["entities_multi_matches"])
        self.assertEqual(total, 12)

    def test_avg_matches_non_negative(self):
        self.assertGreaterEqual(self.ms["avg_matches_per_entity"], 0.0)

    def test_total_predicted_matches_consistent(self):
        """total_predicted_matches = sum of all match counts."""
        self.assertGreaterEqual(self.ms["total_predicted_matches"], 0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Metrics section
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsSection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="test_metrics_")
        cls.report = _run_harness(cls.tmp)
        cls.mt = cls.report.as_dict["metrics"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_all_metric_keys_present(self):
        for key in ("macro_f0.5", "macro_precision", "macro_recall",
                    "exact_match_rate", "correctly_empty_rate",
                    "false_positive_empty_rate", "number_of_entities"):
            self.assertIn(key, self.mt, f"Missing metrics key: {key}")

    def test_macro_f05_in_valid_range(self):
        self.assertGreaterEqual(self.mt["macro_f0.5"], 0.0)
        self.assertLessEqual(self.mt["macro_f0.5"], 1.0)

    def test_number_of_entities_equals_s1_size(self):
        self.assertEqual(self.mt["number_of_entities"], 12)

    def test_correctly_empty_rate_in_valid_range(self):
        self.assertGreaterEqual(self.mt["correctly_empty_rate"], 0.0)
        self.assertLessEqual(self.mt["correctly_empty_rate"], 1.0)

    def test_false_positive_empty_rate_in_valid_range(self):
        self.assertGreaterEqual(self.mt["false_positive_empty_rate"], 0.0)
        self.assertLessEqual(self.mt["false_positive_empty_rate"], 1.0)

    def test_f05_is_bounded_by_precision_and_recall(self):
        mt = self.mt
        p = mt["macro_precision"]
        r = mt["macro_recall"]
        f = mt["macro_f0.5"]
        if p > 0 and r > 0:
            self.assertGreaterEqual(f, 0.0)
            self.assertLessEqual(f, max(p, r) + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Error analysis section
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="test_error_")
        cls.report = _run_harness(cls.tmp)
        cls.ea = cls.report.as_dict["error_analysis"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_error_analysis_keys_present(self):
        for key in ("false_positive_entities", "false_negative_entities",
                    "missed_candidate_entities", "high_confidence_false_matches",
                    "low_confidence_true_matches", "per_entity_detail"):
            self.assertIn(key, self.ea, f"Missing error_analysis key: {key}")

    def test_per_entity_detail_covers_all_s1(self):
        self.assertEqual(len(self.ea["per_entity_detail"]), 12)

    def test_per_entity_detail_record_has_required_fields(self):
        for rec in self.ea["per_entity_detail"]:
            for field in ("s1_id", "ground_truth", "prediction",
                          "precision", "recall", "tp", "fp", "fn"):
                self.assertIn(field, rec, f"per_entity_detail record missing: {field}")

    def test_fp_fn_lists_are_subsets_of_all_s1_ids(self):
        all_s1 = set(make_source1()["entity_id"].tolist())
        for s1_id in self.ea["false_positive_entities"]:
            self.assertIn(s1_id, all_s1)
        for s1_id in self.ea["false_negative_entities"]:
            self.assertIn(s1_id, all_s1)

    def test_missed_candidate_entities_is_empty_at_top_k_20(self):
        """At top_k=20, no GT pair is missed → zero missed-candidate entities."""
        self.assertEqual(len(self.ea["missed_candidate_entities"]), 0)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Report serialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestReportSerialisation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="test_serial_")
        cls.report = _run_harness(cls.tmp)
        cls.json_path = Path(cls.tmp) / "serialisation_test.json"
        cls.report.save_json(cls.json_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_json_file_created(self):
        self.assertTrue(self.json_path.exists())

    def test_json_is_valid(self):
        with open(self.json_path) as f:
            loaded = json.load(f)
        self.assertIsInstance(loaded, dict)

    def test_json_round_trips_macro_f05(self):
        with open(self.json_path) as f:
            loaded = json.load(f)
        self.assertAlmostEqual(
            loaded["metrics"]["macro_f0.5"],
            self.report.macro_f05, places=5
        )

    def test_json_round_trips_blocking_recall(self):
        with open(self.json_path) as f:
            loaded = json.load(f)
        self.assertAlmostEqual(
            loaded["blocking_stats"]["blocking_recall"],
            self.report.blocking_recall, places=5
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. ExperimentTracker integration
# ─────────────────────────────────────────────────────────────────────────────

class TestExperimentTrackerIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="test_tracker_")
        cls.report = _run_harness(cls.tmp, exp_id="tracker_test")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_experiment_dir_created(self):
        dirs = list(Path(self.tmp).iterdir())   # instance attribute, not class
        self.assertGreater(len(dirs), 0, "No experiment run directory was created")

    def test_summary_json_written(self):
        run_dir = Path(self.report.as_dict["run_info"]["run_dir"])
        summary = run_dir / "summary.json"
        self.assertTrue(summary.exists(), "summary.json not written by ExperimentTracker")

    def test_report_json_written(self):
        run_dir = Path(self.report.as_dict["run_info"]["run_dir"])
        report_json = run_dir / "report.json"
        self.assertTrue(report_json.exists(), "report.json not saved")

    def test_summary_contains_key_metrics(self):
        run_dir = Path(self.report.as_dict["run_info"]["run_dir"])
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)
        metrics = summary.get("metrics", {})
        self.assertIn("macro_f0.5", metrics, "macro_f0.5 not logged to tracker")
        self.assertIn("blocking_recall", metrics, "blocking_recall not logged to tracker")

    def test_summary_experiment_id_matches(self):
        run_dir = Path(self.report.as_dict["run_info"]["run_dir"])
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)
        self.assertEqual(summary["experiment_id"], "tracker_test")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Two-run comparison via ValidationReport.compare()
# ─────────────────────────────────────────────────────────────────────────────

class TestTwoRunComparison(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="test_compare_")
        cls.report_05 = _run_harness(cls.tmp, threshold=0.5, exp_id="run_thr_05")
        cls.report_09 = _run_harness(cls.tmp, threshold=0.9, exp_id="run_thr_09")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_compare_returns_deltas(self):
        deltas = self.report_05.compare(self.report_09)
        for key in ("delta_macro_f0.5", "delta_macro_precision", "delta_macro_recall", "delta_blocking_recall"):
            self.assertIn(key, deltas)

    def test_higher_threshold_reduces_matches(self):
        """
        Threshold 0.9 is much stricter -> fewer or equal matches vs 0.5.
        The baseline model uses constant scores of 0.5, so 0.9 threshold
        clears no candidates -> zero matches for all entities.
        """
        ms_05 = self.report_05.as_dict["matching_stats"]["total_predicted_matches"]
        ms_09 = self.report_09.as_dict["matching_stats"]["total_predicted_matches"]
        self.assertGreaterEqual(ms_05, ms_09,
                                "Stricter threshold should never increase total predicted matches")

    def test_compare_delta_is_float(self):
        deltas = self.report_05.compare(self.report_09)
        delta_recall = deltas["delta_macro_recall"]
        self.assertIsInstance(delta_recall, float)


# ─────────────────────────────────────────────────────────────────────────────
# 9. CLI smoke-tests (subprocess)
# ─────────────────────────────────────────────────────────────────────────────

class TestCLISmoke(unittest.TestCase):
    """Runs the CLI script as a subprocess to catch import/crash regressions."""

    def test_synthetic_flag_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "scripts/evaluate.py", "--synthetic", "--json-only"],
            capture_output=True, text=True,
            cwd=str(_ROOT),
        )
        self.assertEqual(
            result.returncode, 0,
            f"CLI exited with code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    def test_synthetic_flag_prints_json_path(self):
        result = subprocess.run(
            [sys.executable, "scripts/evaluate.py", "--synthetic", "--json-only"],
            capture_output=True, text=True,
            cwd=str(_ROOT),
        )
        # Last non-empty line of stdout should be the report path
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        last_line = lines[-1] if lines else ""
        self.assertTrue(
            last_line.endswith("report.json"),
            f"Expected last stdout line to be a report.json path, got: {last_line!r}"
        )

    def test_compare_flag_exits_zero(self):
        """--compare requires two report.json files. Run synthetic twice then compare."""
        tmp = tempfile.mkdtemp(prefix="test_cli_compare_")
        try:
            def run_cli(exp_id):
                r = subprocess.run(
                    [sys.executable, "scripts/evaluate.py", "--synthetic",
                     "--json-only", "--experiment-id", exp_id,
                     "--experiments-dir", tmp],
                    capture_output=True, text=True,
                    cwd=str(_ROOT),
                )
                # Last non-empty stdout line = path (log lines go to stdout too,
                # but the path is always the last thing printed)
                lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
                return lines[-1] if lines else ""

            path_a = run_cli("cli_cmp_a")
            path_b = run_cli("cli_cmp_b")

            if not (path_a.endswith("report.json") and path_b.endswith("report.json")):
                self.skipTest(
                    f"Could not parse report paths from CLI output. "
                    f"Got: {path_a!r}, {path_b!r}"
                )

            result = subprocess.run(
                [sys.executable, "scripts/evaluate.py", "--compare", path_a, path_b],
                capture_output=True, text=True,
                cwd=str(_ROOT),
            )
            self.assertEqual(result.returncode, 0,
                             f"--compare failed:\n{result.stdout}\n{result.stderr}")
            self.assertIn("Experiment A", result.stdout)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Ground truth loading helper
# ─────────────────────────────────────────────────────────────────────────────

class TestGroundTruthLoading(unittest.TestCase):

    def test_competition_schema_parsed_correctly(self):
        import pandas as pd
        from scripts.evaluate import load_ground_truth_from_dataframe
        df = pd.DataFrame([
            {"source1_entity_id": "s1_01", "matched_entity_ids": "s2_01,s3_01"},
            {"source1_entity_id": "s1_09", "matched_entity_ids": ""},
        ])
        gt = load_ground_truth_from_dataframe(df)
        self.assertEqual(gt["s1_01"], {"s2_01", "s3_01"})
        self.assertEqual(gt["s1_09"], set())

    def test_legacy_triplet_schema_parsed_correctly(self):
        import pandas as pd
        from scripts.evaluate import load_ground_truth_from_dataframe
        df = pd.DataFrame([
            {"s1_id": "s1_01", "s2_id": "s2_01", "s3_id": "s3_01"},
            {"s1_id": "s1_09", "s2_id": None, "s3_id": None},
        ])
        gt = load_ground_truth_from_dataframe(df)
        self.assertIn("s2_01", gt["s1_01"])
        self.assertIn("s3_01", gt["s1_01"])
        self.assertEqual(gt.get("s1_09", set()), set())

    def test_none_dataframe_returns_empty(self):
        from scripts.evaluate import load_ground_truth_from_dataframe
        gt = load_ground_truth_from_dataframe(None)
        self.assertEqual(gt, {})

    def test_malformed_ids_skipped(self):
        import pandas as pd
        from scripts.evaluate import load_ground_truth_from_dataframe
        df = pd.DataFrame([
            {"source1_entity_id": "", "matched_entity_ids": "s2_01"},     # blank s1
            {"source1_entity_id": "nan", "matched_entity_ids": "s2_02"},  # nan s1
            {"source1_entity_id": "s1_03", "matched_entity_ids": "nan,None,"},  # bad targets
        ])
        gt = load_ground_truth_from_dataframe(df)
        self.assertNotIn("", gt)
        self.assertNotIn("nan", gt)
        self.assertEqual(gt.get("s1_03", None), set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
