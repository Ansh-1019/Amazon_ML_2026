"""
End-to-End Comprehensive Pipeline Integration Test for Amazon ML Challenge 2026.

Verifies the unified 12-stage execution flow:
  1. Data loads (Ansh DataLoader)
  2. Normalization works (Maithili Normalization)
  3. Blocking works (Maithili CandidatePipeline + IdAdapter)
  4. Candidate pairs are generated (Blocking Adapter)
  5. Features are generated (Anmol build_pair_features)
  6. Model trains (Anmol CatBoost / LightGBM + Hard Negatives)
  7. Predictions are generated (ModelPredictor / Model Adapter)
  8. Threshold is selected (Anmol choose_threshold)
  9. Entity resolution works (Ansh EntityResolver)
  10. Macro F0.5 is calculated (Ansh EntityEvaluator)
  11. Submission files are generated (Ansh SubmissionGenerator)
  12. Official validator passes (run_official_validator)
"""

import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

from src.pipeline import EntityResolutionPipeline, set_seed
from src.submission import run_official_validator
from scripts.run_integrated_training import generate_synthetic_benchmark


class TestEndToEndPipeline(unittest.TestCase):

    def setUp(self):
        set_seed(42)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

        # Generate synthetic benchmark
        self.s1_df, self.s2_df, self.s3_df, self.ground_truth = generate_synthetic_benchmark(
            num_entities=60, random_seed=42
        )

        # Format ground_truth as DataFrame
        gt_rows = []
        for s1_id, t_ids in self.ground_truth.items():
            if t_ids:
                gt_rows.append({
                    "source1_entity_id": s1_id,
                    "matched_entity_ids": ",".join(t_ids),
                })
        self.train_matches = pd.DataFrame(gt_rows)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_end_to_end_pipeline_with_training_and_threshold_optimization(self):
        """Runs the entire 12-stage pipeline from raw DataFrames to validated submissions."""
        # 1. Initialize pipeline
        pipeline = EntityResolutionPipeline(
            config={
                "project": {"name": "TestEndToEnd", "experiment_id": "test_e2e", "seed": 42},
                "paths": {"submissions_dir": str(self.output_dir)},
                "data": {
                    "id_column_s1": "entity_id",
                    "id_column_s2": "entity_id",
                    "id_column_s3": "entity_id",
                    "name_column": "business_name",
                    "address_column": "business_address",
                    "country_column": "country",
                },
                "blocking": {
                    "top_k": 10,
                    "blocking_fields": ["business_name", "business_address", "country"],
                },
                "features": {"string_similarity_metrics": ["levenshtein", "jaccard"]},
                "modeling": {"model_type": "catboost"},
                "decision": {"threshold": 0.5},
                "evaluation": {"beta": 0.5},
                "submission": {
                    "matching_filename": "matching_results.tsv",
                    "candidates_filename": "candidate_pairs.tsv",
                },
            }
        )

        # 2. Stage 1: Data load & normalization
        s1_norm, s2_norm, s3_norm, train_matches, all_s1_ids, valid_target_ids = pipeline.load_and_normalize_data(
            s1_df=self.s1_df, s2_df=self.s2_df, s3_df=self.s3_df, train_matches=self.train_matches
        )
        self.assertEqual(len(s1_norm), 60)
        self.assertIn("business_name_clean", s1_norm.columns)
        self.assertIn("business_address_normalized", s1_norm.columns)

        # 3. Stage 2: Candidate Generation (Blocking)
        candidates_df, candidate_map = pipeline.generate_candidates(
            s1_norm=s1_norm, s2_norm=s2_norm, s3_norm=s3_norm, all_s1_ids=all_s1_ids, valid_target_ids=valid_target_ids
        )
        self.assertFalse(candidates_df.empty)
        self.assertEqual(len(candidate_map), 60)

        # 4. Stage 3: Feature Extraction
        features_df = pipeline.extract_features(
            candidates_df=candidates_df, s1_norm=s1_norm, s2_norm=s2_norm, s3_norm=s3_norm
        )
        self.assertFalse(features_df.empty)
        self.assertGreaterEqual(features_df.shape[1], 15)

        # 5. Stage 3.5: Model Training & Threshold Optimization
        model, optimal_threshold = pipeline.train_model(
            candidates_df=candidates_df,
            s1_norm=s1_norm,
            s2_norm=s2_norm,
            s3_norm=s3_norm,
            train_matches=self.train_matches,
            model_type="catboost",
            optimize_threshold=True,
        )
        self.assertIsNotNone(model)
        self.assertTrue(0.0 <= optimal_threshold <= 1.0)

        # 6. Stage 4: Model Inference & Probability Scoring
        scored_df = pipeline.score_candidates(
            candidates_df=candidates_df, features_df=features_df
        )
        self.assertIn("match_score", scored_df.columns)
        self.assertIn("s1_id", scored_df.columns)
        self.assertIn("target_id", scored_df.columns)

        # 7. Stage 5: Entity-Level Decision Resolution
        resolved_df, match_map = pipeline.resolve_decisions(
            scored_candidates_df=scored_df,
            all_s1_ids=all_s1_ids,
            candidate_map=candidate_map,
            threshold=optimal_threshold,
            valid_target_ids=valid_target_ids,
        )
        self.assertEqual(len(match_map), 60)
        # Verify candidate subset constraint
        for s1, matched in match_map.items():
            cand_set = set(candidate_map.get(s1, []))
            for m in matched:
                self.assertIn(m, cand_set)

        # 8. Stage 6: Evaluation & Submission Generation
        matching_path, candidate_path, eval_metrics, is_valid = pipeline.evaluate_and_generate_submission(
            candidate_map=candidate_map,
            match_map=match_map,
            all_s1_ids=all_s1_ids,
            train_matches=self.train_matches,
            valid_target_ids=valid_target_ids,
            output_dir=self.output_dir,
        )

        self.assertTrue(matching_path.exists())
        self.assertTrue(candidate_path.exists())
        self.assertTrue(is_valid)
        self.assertIn("macro_f0.5", eval_metrics)
        self.assertGreater(eval_metrics["macro_f0.5"], 0.70)

        # 9. Test full pipeline.run() method
        full_results = pipeline.run(
            s1_df=self.s1_df,
            s2_df=self.s2_df,
            s3_df=self.s3_df,
            train_matches=self.train_matches,
            train_model=False,  # model already trained
            output_dir=self.output_dir,
        )
        self.assertTrue(full_results["is_submission_valid"])
        self.assertGreater(full_results["eval_metrics"]["macro_f0.5"], 0.70)


if __name__ == "__main__":
    unittest.main()
