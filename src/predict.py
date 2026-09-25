"""
Standalone Inference and Prediction Module for Amazon ML Challenge 2026.

Executes test-time prediction using the integrated pipeline:
  1. Ingests test source data (Source 1, Source 2, Source 3).
  2. Generates candidate pairs via blocking.
  3. Computes similarity features.
  4. Predicts match probabilities using trained model (or fallback heuristic).
  5. Applies configurable entity-level decision threshold.
  6. Emits validated matching_results.tsv and candidate_pairs.tsv.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from src.utils import setup_logger, load_config
from src.pipeline import EntityResolutionPipeline
from src.submission import run_official_validator

logger = logging.getLogger("pipeline.predict")


def run_inference(
    config_path: Union[str, Path] = "configs/default_config.yaml",
    threshold: Optional[float] = None,
    output_dir: Optional[Union[str, Path]] = "output",
    model: Optional[Any] = None,
    s1_df: Optional[pd.DataFrame] = None,
    s2_df: Optional[pd.DataFrame] = None,
    s3_df: Optional[pd.DataFrame] = None,
    validate: bool = True,
) -> Dict[str, Any]:
    """
    Executes end-to-end inference and generates validated submission files.

    Args:
        config_path: Path to YAML configuration.
        threshold: Optional decision threshold override.
        output_dir: Target directory for submission files.
        model: Optional pre-loaded or trained model instance.
        s1_df: Optional in-memory Source 1 DataFrame.
        s2_df: Optional in-memory Source 2 DataFrame.
        s3_df: Optional in-memory Source 3 DataFrame.
        validate: Whether to run the official submission validator.

    Returns:
        Dict containing prediction summary and file paths.
    """
    pipeline = EntityResolutionPipeline(config_path=config_path, model=model)
    
    results = pipeline.run(
        s1_df=s1_df,
        s2_df=s2_df,
        s3_df=s3_df,
        train_matches=None,  # Pure inference mode
        threshold=threshold,
        output_dir=output_dir,
    )

    if validate:
        is_valid, log = run_official_validator(
            matching_filepath=results["matching_results_path"],
            candidates_filepath=results["candidate_pairs_path"],
        )
        results["is_submission_valid"] = is_valid
        results["validator_log"] = log

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026 - Standalone Prediction & Submission Generation"
    )
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold for entity matching")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory for output TSV files")
    parser.add_argument("--no-validate", action="store_true", help="Skip official validator execution")
    args = parser.parse_args()

    results = run_inference(
        config_path=args.config,
        threshold=args.threshold,
        output_dir=args.output_dir,
        validate=not args.no_validate,
    )

    print("\n" + "=" * 50)
    print("INFERENCE SUMMARY")
    print("=" * 50)
    print(f"Matching Results: {results['matching_results_path']}")
    print(f"Candidate Pairs:  {results['candidate_pairs_path']}")
    print(f"Entities Processed: {results['num_s1_entities']}")
    print(f"Candidates Generated: {results['num_candidates_generated']}")
    print(f"Matches Resolved: {results['num_resolved_matches']}")
    print(f"Submission Valid: {results['is_submission_valid']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
