"""
Unit and Mathematical Validation Tests for Entity-Level Macro F_0.5 Evaluation.

Tests:
  - All 8 competition edge cases (empty GT, empty Pred, partial matches, false positives, etc.)
  - Mathematical proof test: Global Micro F_0.5 != Competition Macro F_0.5
  - Entity ordering invariance
  - ID deduplication
  - Complete Source 1 coverage (including zero-match entities)
  - Full diagnostic metrics and per-entity records verification
"""

import unittest
import numpy as np
from src.evaluation.metrics import (
    EntityEvaluator,
    compute_f_beta,
    compute_single_entity_metrics,
    parse_id_set,
)


class TestEntityEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = EntityEvaluator(beta=0.5)

    def test_case_1_empty_gt_empty_pred(self):
        """Case 1: GT = {}, Prediction = {} => F0.5 = 1.0, Precision = 1.0, Recall = 1.0."""
        metrics = compute_single_entity_metrics(gt_set=set(), pred_set=set(), beta=0.5)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f_0.5"], 1.0)
        self.assertTrue(metrics["is_exact_match"])

    def test_case_2_nonempty_gt_empty_pred(self):
        """Case 2: GT = {'A'}, Prediction = {} => F0.5 = 0.0, Precision = 0.0, Recall = 0.0."""
        metrics = compute_single_entity_metrics(gt_set={"A"}, pred_set=set(), beta=0.5)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f_0.5"], 0.0)
        self.assertFalse(metrics["is_exact_match"])

    def test_case_3_empty_gt_nonempty_pred(self):
        """Case 3: GT = {}, Prediction = {'A'} => F0.5 = 0.0, Precision = 0.0, Recall = 0.0."""
        metrics = compute_single_entity_metrics(gt_set=set(), pred_set={"A"}, beta=0.5)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f_0.5"], 0.0)
        self.assertFalse(metrics["is_exact_match"])

    def test_case_4_single_match_exact(self):
        """Case 4: GT = {'A'}, Prediction = {'A'} => F0.5 = 1.0, Precision = 1.0, Recall = 1.0."""
        metrics = compute_single_entity_metrics(gt_set={"A"}, pred_set={"A"}, beta=0.5)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f_0.5"], 1.0)
        self.assertTrue(metrics["is_exact_match"])

    def test_case_5_partial_match_higher_precision(self):
        """
        Case 5: GT = {'A', 'B'}, Prediction = {'A'}
        TP = 1, FP = 0, FN = 1
        Precision = 1.0, Recall = 0.5
        F0.5 = (1 + 0.25) * 1.0 * 0.5 / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 5/6 ≈ 0.83333
        """
        metrics = compute_single_entity_metrics(gt_set={"A", "B"}, pred_set={"A"}, beta=0.5)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f_0.5"], 5.0 / 6.0, places=5)
        self.assertFalse(metrics["is_exact_match"])

    def test_case_6_partial_match_lower_precision(self):
        """
        Case 6: GT = {'A'}, Prediction = {'A', 'B'}
        TP = 1, FP = 1, FN = 0
        Precision = 0.5, Recall = 1.0
        F0.5 = (1 + 0.25) * 0.5 * 1.0 / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 5/9 ≈ 0.55556
        """
        metrics = compute_single_entity_metrics(gt_set={"A"}, pred_set={"A", "B"}, beta=0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f_0.5"], 5.0 / 9.0, places=5)
        self.assertFalse(metrics["is_exact_match"])

    def test_case_7_multi_match_exact(self):
        """Case 7: GT = {'A', 'B'}, Prediction = {'A', 'B'} => F0.5 = 1.0."""
        metrics = compute_single_entity_metrics(gt_set={"A", "B"}, pred_set={"A", "B"}, beta=0.5)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f_0.5"], 1.0)
        self.assertTrue(metrics["is_exact_match"])

    def test_case_8_complete_mismatch(self):
        """Case 8: GT = {'A'}, Prediction = {'B'} => TP=0, FP=1, FN=1 => F0.5 = 0.0."""
        metrics = compute_single_entity_metrics(gt_set={"A"}, pred_set={"B"}, beta=0.5)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f_0.5"], 0.0)
        self.assertFalse(metrics["is_exact_match"])

    def test_macro_averaging_across_entities(self):
        """Tests macro averaging F_0.5 across multiple entities with mixed outcomes."""
        y_true = {
            "s1_1": {"s2_101", "s3_201"},  # exact match -> F0.5 = 1.0
            "s1_2": {"s2_102"},            # pred empty -> F0.5 = 0.0
            "s1_3": set(),                 # both empty -> F0.5 = 1.0
            "s1_4": {"s2_104", "s3_204"},  # pred only s2_104 -> F0.5 = 5/6 ≈ 0.83333
        }
        y_pred = {
            "s1_1": {"s2_101", "s3_201"},
            "s1_2": set(),
            "s1_3": set(),
            "s1_4": {"s2_104"},
        }

        expected_macro_f05 = (1.0 + 0.0 + 1.0 + (5.0 / 6.0)) / 4.0  # ≈ 0.70833
        res = self.evaluator.evaluate(y_true, y_pred)

        self.assertAlmostEqual(res["macro_f_beta"], expected_macro_f05, places=5)
        self.assertEqual(res["number_of_entities"], 4)
        self.assertEqual(res["exact_match_rate"], 2 / 4)
        self.assertEqual(res["empty_gt_count"], 1)
        self.assertEqual(res["correctly_empty_count"], 1)
        self.assertEqual(res["false_positive_empty_count"], 0)

    def test_micro_f05_differs_from_competition_macro_f05(self):
        """
        Mathematical proof test:
        Demonstrates that Global Micro F_0.5 != Competition Macro F_0.5
        when entity match set sizes differ.
        
        Entity 1 (size 4): GT={A, B, C, D}, Pred={A, B, C, D} => F0.5 = 1.0 (TP=4, FP=0, FN=0)
        Entity 2 (size 1): GT={E}, Pred={} => F0.5 = 0.0 (TP=0, FP=0, FN=1)
        
        Macro F0.5 = (1.0 + 0.0) / 2 = 0.50000
        
        Global Micro:
          Total TP = 4, Total FP = 0, Total FN = 1
          Micro Prec = 4/4 = 1.0
          Micro Rec = 4/5 = 0.8
          Micro F0.5 = 1.25 * 1.0 * 0.8 / (0.25 * 1.0 + 0.8) = 1.0 / 1.05 ≈ 0.95238
        """
        y_true = {
            "e1": {"A", "B", "C", "D"},
            "e2": {"E"},
        }
        y_pred = {
            "e1": {"A", "B", "C", "D"},
            "e2": set(),
        }

        macro_res = self.evaluator.evaluate(y_true, y_pred)
        macro_f05 = macro_res["macro_f_beta"]
        self.assertAlmostEqual(macro_f05, 0.50000, places=4)

        # Compute Micro F0.5
        total_tp = 4
        total_fp = 0
        total_fn = 1
        micro_prec = total_tp / (total_tp + total_fp)
        micro_rec = total_tp / (total_tp + total_fn)
        micro_f05 = compute_f_beta(micro_prec, micro_rec, beta=0.5)

        self.assertAlmostEqual(micro_f05, 1.0 / 1.05, places=4)
        self.assertNotEqual(round(macro_f05, 4), round(micro_f05, 4))
        self.assertGreater(micro_f05, macro_f05)

    def test_entity_ordering_invariance(self):
        """Tests that dictionary insertion / iteration order does not affect the macro score."""
        y_true_1 = {"e1": {"A"}, "e2": {"B"}, "e3": set()}
        y_pred_1 = {"e1": {"A"}, "e2": set(), "e3": set()}

        y_true_2 = {"e3": set(), "e1": {"A"}, "e2": {"B"}}
        y_pred_2 = {"e3": set(), "e2": set(), "e1": {"A"}}

        res1 = self.evaluator.evaluate(y_true_1, y_pred_1)
        res2 = self.evaluator.evaluate(y_true_2, y_pred_2)

        self.assertEqual(res1["macro_f_beta"], res2["macro_f_beta"])
        self.assertEqual(res1["macro_precision"], res2["macro_precision"])
        self.assertEqual(res1["macro_recall"], res2["macro_recall"])

    def test_id_deduplication(self):
        """Tests that duplicate IDs in string or list predictions are deduplicated before scoring."""
        gt = {"s1": ["A", "B"]}
        pred_with_dups = {"s1": "A, A, B, B"}  # duplicate strings

        res = self.evaluator.evaluate(gt, pred_with_dups)
        self.assertEqual(res["macro_f_beta"], 1.0)
        self.assertEqual(res["macro_precision"], 1.0)
        self.assertEqual(res["macro_recall"], 1.0)

    def test_per_entity_records_diagnostics(self):
        """Tests that per-entity records contain complete diagnostic fields for error analysis."""
        y_true = {"e1": {"A", "B"}, "e2": set()}
        y_pred = {"e1": {"A", "C"}, "e2": {"D"}}

        res = self.evaluator.evaluate(y_true, y_pred)
        records = res["per_entity_records"]

        self.assertEqual(len(records), 2)
        r1 = next(r for r in records if r["s1_id"] == "e1")
        self.assertEqual(r1["tp"], 1)
        self.assertEqual(r1["fp"], 1)
        self.assertEqual(r1["fn"], 1)
        self.assertEqual(r1["precision"], 0.5)
        self.assertEqual(r1["recall"], 0.5)
        self.assertAlmostEqual(r1["f_0.5"], 0.5, places=4)

        r2 = next(r for r in records if r["s1_id"] == "e2")
        self.assertEqual(r2["tp"], 0)
        self.assertEqual(r2["fp"], 1)
        self.assertEqual(r2["fn"], 0)
        self.assertEqual(r2["f_0.5"], 0.0)

    def test_complete_s1_coverage_enforcement(self):
        """Tests that providing all_s1_ids evaluates all entities even if absent from predictions."""
        all_s1 = ["e1", "e2", "e3", "e4"]
        y_true = {"e1": {"A"}, "e2": {"B"}}
        y_pred = {"e1": {"A"}}  # e2, e3, e4 missing from predictions

        res = self.evaluator.evaluate(y_true, y_pred, all_s1_ids=all_s1)
        self.assertEqual(res["number_of_entities"], 4)
        # e1: GT={A}, Pred={A} => 1.0
        # e2: GT={B}, Pred={}  => 0.0
        # e3: GT={}, Pred={}   => 1.0
        # e4: GT={}, Pred={}   => 1.0
        # Macro F0.5 = (1.0 + 0.0 + 1.0 + 1.0) / 4 = 0.75
        self.assertEqual(res["macro_f_beta"], 0.75)


if __name__ == "__main__":
    unittest.main()
