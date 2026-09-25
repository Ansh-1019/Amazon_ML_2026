import argparse
import sys
from pathlib import Path

# Add src directory to system path
sys.path.append(str(Path(__file__).parent))

from src.utils import setup_logger, load_config, ExperimentTracker
from src.data import DataLoader, DataNormalizer
from src.blocking import CandidateGenerator
from src.features import FeatureExtractor
from src.modeling import ModelTrainer, ModelPredictor
from src.evaluation import EntityEvaluator
from src.decision import EntityResolver
from src.submission import SubmissionGenerator


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 - Entity Resolution Pipeline")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    # 1. Load Configuration
    config = load_config(args.config)
    exp_id = config["project"]["experiment_id"]

    # 2. Setup Logger & Tracker
    tracker = ExperimentTracker(experiment_id=exp_id, experiments_dir=config["paths"]["experiments_dir"])
    logger = setup_logger(name="pipeline", log_file=tracker.get_run_dir() / "run.log")

    logger.info(f"--- Starting Pipeline Run: {exp_id} ---")
    tracker.log_params(config)

    # 3. Load Raw Data
    data_loader = DataLoader(raw_data_dir=config["paths"]["raw_data_dir"])
    s1_df, s2_df, s3_df, train_matches = data_loader.load_sources(config)

    id_s1 = config["data"]["id_column_s1"]
    all_s1_ids = s1_df[id_s1].tolist() if not s1_df.empty and id_s1 in s1_df.columns else []

    # 4. Normalize Data
    normalizer = DataNormalizer()
    s1_norm = normalizer.normalize_dataframe(s1_df, text_columns=config["blocking"]["blocking_fields"])
    s2_norm = normalizer.normalize_dataframe(s2_df, text_columns=config["blocking"]["blocking_fields"])
    s3_norm = normalizer.normalize_dataframe(s3_df, text_columns=config["blocking"]["blocking_fields"])

    # 5. Blocking / Candidate Generation
    blocking_module = CandidateGenerator(
        top_k=config["blocking"]["top_k"],
        blocking_fields=config["blocking"]["blocking_fields"]
    )
    candidates_df = blocking_module.generate_candidates(s1_norm, s2_norm, s3_norm, config)

    # 6. Pairwise Feature Extraction
    feature_extractor = FeatureExtractor()
    features_df = feature_extractor.extract_features(candidates_df, s1_norm, s2_norm, s3_norm, config)

    # 7. Model Scoring
    if not features_df.empty:
        # Dummy probability scores for baseline pipeline verification
        candidates_df["match_score"] = 0.8
    else:
        candidates_df["match_score"] = []

    # 8. Entity-Level Decision Layer
    resolver = EntityResolver(threshold=config["decision"]["threshold"])
    resolved_matches_df = resolver.resolve(candidates_df, config)

    # 9. Evaluation (if ground truth available)
    if train_matches is not None and not train_matches.empty:
        evaluator = EntityEvaluator(beta=config["evaluation"]["beta"])
        # Format true vs pred dictionaries
        y_true = dict(zip(train_matches["s1_id"], zip(train_matches["s2_id"], train_matches["s3_id"])))
        y_pred = dict(zip(resolved_matches_df["s1_id"], zip(resolved_matches_df["s2_id"], resolved_matches_df["s3_id"])))
        eval_metrics = evaluator.evaluate(y_true, y_pred)
        tracker.log_metrics(eval_metrics)

    # 10. Submission Generation & Validation
    sub_generator = SubmissionGenerator(output_dir=config["paths"]["submissions_dir"])
    submission_path = sub_generator.validate_and_generate(
        resolved_matches_df=resolved_matches_df,
        all_s1_ids=all_s1_ids,
        candidates_df=candidates_df,
        output_filename=config["submission"]["output_filename"],
        config=config
    )

    logger.info(f"--- Pipeline Execution Completed Successfully. Output: {submission_path} ---")


if __name__ == "__main__":
    main()
