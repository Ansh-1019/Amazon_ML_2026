import logging
from pathlib import Path
from typing import List, Set, Union
import pandas as pd


logger = logging.getLogger("pipeline.submission")


class SubmissionGenerator:
    """Validates and exports final competition submission files."""

    def __init__(self, output_dir: Union[str, Path] = "submissions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate_and_generate(
        self,
        resolved_matches_df: pd.DataFrame,
        all_s1_ids: List[str],
        candidates_df: pd.DataFrame,
        output_filename: str = "submission.csv",
        config: dict = None
    ) -> Path:
        """
        Validates submission constraints and saves CSV file:
        1. Final matches must be a subset of candidate pairs.
        2. Every Source 1 entity must appear in final output.

        Args:
            resolved_matches_df: DataFrame of matched pairs ['s1_id', 's2_id', 's3_id']
            all_s1_ids: List of all Source 1 entity IDs that MUST be present
            candidates_df: Candidate pairs DataFrame generated in blocking step
            output_filename: Output file name
            config: Pipeline configuration dictionary

        Returns:
            Path to exported submission file.
        """
        logger.info("Validating submission requirements...")
        
        # 1. Validate Subset of Candidate Pairs
        if not candidates_df.empty and not resolved_matches_df.empty:
            cand_pairs = set(zip(candidates_df["s1_id"], candidates_df["s2_id"].fillna(""), candidates_df["s3_id"].fillna("")))
            resolved_pairs = set(zip(resolved_matches_df["s1_id"], resolved_matches_df["s2_id"].fillna(""), resolved_matches_df["s3_id"].fillna("")))
            invalid_pairs = resolved_pairs - cand_pairs
            if invalid_pairs:
                raise ValueError(f"Submission validation failed: {len(invalid_pairs)} predicted matches are not in candidate set.")
            logger.info("VALIDATED: All predicted matches are a valid subset of generated candidate pairs.")

        # 2. Enforce Mandatory Source 1 Coverage
        submission_df = pd.DataFrame({"s1_id": all_s1_ids})
        if not resolved_matches_df.empty:
            submission_df = submission_df.merge(
                resolved_matches_df[["s1_id", "s2_id", "s3_id"]],
                on="s1_id",
                how="left"
            )
        else:
            submission_df["s2_id"] = None
            submission_df["s3_id"] = None

        missing_s1 = set(all_s1_ids) - set(submission_df["s1_id"])
        if missing_s1:
            raise ValueError(f"Submission validation failed: {len(missing_s1)} Source 1 entities missing from output.")
        logger.info(f"VALIDATED: 100% Source 1 coverage achieved ({len(submission_df)} entities).")

        # Save to output path
        output_path = self.output_dir / output_filename
        submission_df.to_csv(output_path, index=False)
        logger.info(f"Submission file successfully generated and saved to: {output_path}")

        return output_path
