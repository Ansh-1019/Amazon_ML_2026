import logging
import pandas as pd


logger = logging.getLogger("pipeline.decision")


class EntityResolver:
    """Converts match probabilities/scores into entity-level match assignments for Source 1 entities."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def resolve(self, scored_candidates_df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """
        Selects top-scoring match for each Source 1 entity above threshold.

        Args:
            scored_candidates_df: DataFrame with ['s1_id', 's2_id', 's3_id', 'match_score']

        Returns:
            DataFrame with resolved entity matches: ['s1_id', 's2_id', 's3_id', 'match_score']
        """
        logger.info(f"Resolving entity matches with score threshold {self.threshold}...")
        if scored_candidates_df.empty:
            return pd.DataFrame(columns=["s1_id", "s2_id", "s3_id", "match_score"])

        # Filter by threshold
        valid_df = scored_candidates_df[scored_candidates_df["match_score"] >= self.threshold].copy()

        if valid_df.empty:
            logger.warning("No candidate pairs exceeded match threshold.")
            return pd.DataFrame(columns=["s1_id", "s2_id", "s3_id", "match_score"])

        # Group by s1_id and pick candidate triplet with maximum score
        resolved_df = (
            valid_df.sort_values(by=["s1_id", "match_score"], ascending=[True, False])
            .groupby("s1_id", as_index=False)
            .first()
        )

        logger.info(f"Resolved matches for {len(resolved_df)} Source 1 entities.")
        return resolved_df
