import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd

logger = logging.getLogger("pipeline.blocking")


def resolve_id_col(df: pd.DataFrame, configured_col: Optional[str], fallback_aliases: List[str]) -> str:
    """Resolves the ID column name from DataFrame columns given configuration and fallbacks."""
    if configured_col and configured_col in df.columns:
        return configured_col
    for alias in fallback_aliases:
        if alias in df.columns:
            return alias
    if "entity_id" in df.columns:
        return "entity_id"
    return configured_col or "entity_id"


class CandidateGenerator:
    """Generates candidate matches from Source 2 and Source 3 for each entity in Source 1."""

    def __init__(self, top_k: int = 20, blocking_fields: Optional[List[str]] = None):
        self.top_k = top_k
        self.blocking_fields = blocking_fields or ["business_name", "business_address", "country"]

    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        config: dict
    ) -> pd.DataFrame:
        """
        Generates candidate pairs using vector similarity / blocking.
        Returns DataFrame containing candidate pairs: ['s1_id', 's2_id', 's3_id', 'blocking_score'].
        """
        logger.info("Generating candidate pairs...")
        if s1_df.empty:
            logger.warning("Source 1 DataFrame is empty.")
            return pd.DataFrame(columns=["s1_id", "s2_id", "s3_id", "blocking_score"])

        data_cfg = config.get("data", {})
        id_s1 = resolve_id_col(s1_df, data_cfg.get("id_column_s1"), ["entity_id", "s1_id", "id"])
        id_s2 = resolve_id_col(s2_df, data_cfg.get("id_column_s2"), ["entity_id", "s2_id", "id"])
        id_s3 = resolve_id_col(s3_df, data_cfg.get("id_column_s3"), ["entity_id", "s3_id", "id"])

        candidates = []

        # Baseline blocking: candidate generation per source
        for _, s1_row in s1_df.iterrows():
            s1_id_val = str(s1_row[id_s1]).strip()
            s2_matches = s2_df[id_s2].head(self.top_k).tolist() if not s2_df.empty and id_s2 in s2_df.columns else [None]
            s3_matches = s3_df[id_s3].head(self.top_k).tolist() if not s3_df.empty and id_s3 in s3_df.columns else [None]

            for s2_id_val in s2_matches:
                for s3_id_val in s3_matches:
                    candidates.append({
                        "s1_id": s1_id_val,
                        "s2_id": str(s2_id_val).strip() if s2_id_val is not None else None,
                        "s3_id": str(s3_id_val).strip() if s3_id_val is not None else None,
                        "blocking_score": 1.0
                    })

        cand_df = pd.DataFrame(candidates)
        logger.info(f"Generated {len(cand_df)} candidate triplets.")
        return cand_df
