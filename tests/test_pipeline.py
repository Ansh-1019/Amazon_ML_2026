"""
Integration Tests for Entity Resolution Pipeline & Predict Modules.

Validates:
  1. Full end-to-end pipeline execution (train/val evaluation mode & inference mode)
  2. Candidate generation & candidate pairs logging
  3. Feature extraction & model scoring stages
  4. Entity decision with configurable thresholds
  5. Official submission validator compliance
  6. Standalone src.predict inference interface
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from src.pipeline import (
    EntityResolutionPipeline,
    candidates_df_to_map,
    resolved_df_to_map,
    set_seed,
)
from src.predict import run_inference
from src.submission import run_official_validator


class TestPipelineIntegration(unittest.TestCase):

    def setUp(self):
        set_seed(42)
        # Synthetic Source 1 (Reference)
        self.s1_df = pd.DataFrame({
            "s1_id": ["s1_1", "s1_2", "s1_3", "s1_4", "s1_5"],
            "name": ["Acme Corp", "Beta Solutions", "Gamma Enterprises", "Delta Tech", "Epsilon Global"],
            "address": ["123 Main St", "456 Oak Ave", "789 Pine Rd", "101 Elm St", "202 Maple Dr"],
            "phone": ["555-0101", "555-0102", "555-0103", "555-0104", "555-0105"],
        })

        # Synthetic Source 2
        self.s2_df = pd.DataFrame({
            "s2_id": ["s2_101", "s2_102", "s2_103", "s2_104"],
            "name": ["Acme Corporation", "Beta Solns", "Gamma Ent", "Zeta Industries"],
            "address": ["123 Main Street", "456 Oak Avenue", "789 Pine Road", "999 Unknown Way"],
            "phone": ["555-0101", "555-0102", "555-0103", "555-9999"],
        })

        # Synthetic Source 3
        self.s3_df = pd.DataFrame({
            "s3_id": ["s3_201", "s3_202", "s3_203"],
            "name": ["Acme", "Beta Solutions Inc", "Gamma Enterprises LLC"],
            "address": ["123 Main", "456 Oak", "789 Pine"],
            "phone": ["555-0101", "555-0102", "555-0103"],
        })

        # Synthetic Ground Truth Matches
        self.train_matches = pd.DataFrame({
            "s1_id": ["s1_1", "s1_2", "s1_3"],
            "s2_id": ["s2_101", "s2_102", "s2_103"],
            "s3_id": ["s3_201", "s3_202", "s3_203"],
        })

    def test_pipeline_end_to_end_with_evaluation(self):
        """Tests full pipeline run with evaluation metrics and submission generation."""
        pipeline = EntityResolutionPipeline(
            config={
                "project": {"name": "Test", "experiment_id": "test_exp", "seed": 42},
                "paths": {"submissions_dir": "output"},
                "data": {
                    "id_column_s1": "s1_id",
                    "id_column_s2": "s2_id",
                    "id_column_s3": "s3_id",
                },
                "blocking": {"top_k": 2, "blocking_fields": ["name", "address", "phone"]},
                "features": {"string_similarity_metrics": ["jaccard"]},
                "decision": {"threshold": 0.5},
                "evaluation": {"beta": 0.5},
                "submission": {
                    "matching_filename": "matching_results.tsv",
                    "candidates_filename": "candidate_pairs.tsv",
                },
            }
        )

        results = pipeline.run(
            s1_df=self.s1_df,
            s2_df=self.s2_df,
            s3_df=self.s3_df,
            train_matches=self.train_matches,
            output_dir="output",
        )

        self.assertEqual(results["num_s1_entities"], 5)
        self.assertGreater(results["num_candidates_generated"], 0)
        self.assertTrue(results["is_submission_valid"])
        self.assertIn("precision", results["eval_metrics"])
        self.assertIn("f_0.5", results["eval_metrics"])

        # Check that generated files pass official validation
        is_valid, _ = run_official_validator(
            matching_filepath=results["matching_results_path"],
            candidates_filepath=results["candidate_pairs_path"],
        )
        self.assertTrue(is_valid)

    def test_candidate_map_and_resolved_map_integrity(self):
        """Tests that candidate and match maps strictly maintain 100% S1 coverage and subset rule."""
        all_s1 = ["s1_1", "s1_2", "s1_3"]
        cands_df = pd.DataFrame([
            {"s1_id": "s1_1", "s2_id": "s2_101", "s3_id": "s3_201"},
            {"s1_id": "s1_2", "s2_id": "s2_102", "s3_id": None},
        ])
        cand_map = candidates_df_to_map(cands_df, all_s1_ids=all_s1)
        self.assertEqual(cand_map["s1_1"], ["s2_101", "s3_201"])
        self.assertEqual(cand_map["s1_2"], ["s2_102"])
        self.assertEqual(cand_map["s1_3"], [])  # Entity with zero candidates

        resolved_df = pd.DataFrame([
            {"s1_id": "s1_1", "s2_id": "s2_101", "s3_id": "s3_201", "match_score": 0.9},
            {"s1_id": "s1_2", "s2_id": "s2_102", "s3_id": "s3_999", "match_score": 0.8},  # s3_999 not in cands
        ])
        match_map = resolved_df_to_map(resolved_df, all_s1_ids=all_s1, candidate_map=cand_map)
        self.assertEqual(match_map["s1_1"], ["s2_101", "s3_201"])
        self.assertEqual(match_map["s1_2"], ["s2_102"])  # s3_999 safely excluded
        self.assertEqual(match_map["s1_3"], [])

    def test_threshold_sensitivity(self):
        """Tests that varying threshold changes resolution results predictably."""
        pipeline = EntityResolutionPipeline()
        cands_df = pd.DataFrame([
            {"s1_id": "s1_1", "s2_id": "s2_101", "s3_id": "s3_201", "match_score": 0.4},
            {"s1_id": "s1_2", "s2_id": "s2_102", "s3_id": "s3_202", "match_score": 0.7},
        ])

        # High threshold (0.8): neither passes
        _, high_map = pipeline.resolve_decisions(
            cands_df, all_s1_ids=["s1_1", "s1_2"], candidate_map={"s1_1": ["s2_101"], "s1_2": ["s2_102"]}, threshold=0.8
        )
        self.assertEqual(high_map["s1_1"], [])
        self.assertEqual(high_map["s1_2"], [])

        # Low threshold (0.3): both pass
        _, low_map = pipeline.resolve_decisions(
            cands_df, all_s1_ids=["s1_1", "s1_2"], candidate_map={"s1_1": ["s2_101"], "s1_2": ["s2_102"]}, threshold=0.3
        )
        self.assertEqual(len(low_map["s1_1"]), 1)
        self.assertEqual(len(low_map["s1_2"]), 1)

    def test_predict_standalone_inference(self):
        """Tests src.predict.run_inference module standalone."""
        results = run_inference(
            s1_df=self.s1_df,
            s2_df=self.s2_df,
            s3_df=self.s3_df,
            threshold=0.5,
            output_dir="output",
            validate=True,
        )
        self.assertTrue(results["is_submission_valid"])
        self.assertEqual(results["num_s1_entities"], len(self.s1_df))


if __name__ == "__main__":
    unittest.main()
