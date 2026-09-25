import logging
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


logger = logging.getLogger("pipeline.blocking")


class CandidateGenerator:
    """Generates candidate matches from Source 2 and Source 3 for each entity in Source 1."""

    def __init__(self, top_k: int = 20, blocking_fields: List[str] = None):
        self.top_k = top_k
        self.blocking_fields = blocking_fields or ["name"]

    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        config: dict
    ) -> pd.DataFrame:
        """
        Generates candidate pairs using TF-IDF vector similarity / nearest neighbors.
        Returns DataFrame containing candidate pairs: ['s1_id', 's2_id', 's3_id', 'blocking_score'].
        """
        logger.info("Generating candidate pairs...")
        if s1_df.empty:
            logger.warning("Source 1 DataFrame is empty.")
            return pd.DataFrame(columns=["s1_id", "s2_id", "s3_id", "blocking_score"])

        id_s1 = config["data"]["id_column_s1"]
        id_s2 = config["data"]["id_column_s2"]
        id_s3 = config["data"]["id_column_s3"]

        candidates = []

        # Baseline blocking: Cartesian or nearest neighbor blocking per source
        # For each S1 entity, find top_k S2 and top_k S3 candidates
        # (Modular placeholder for team blocking algorithm)
        for _, s1_row in s1_df.iterrows():
            s1_id_val = s1_row[id_s1]
            s2_matches = s2_df[id_s2].head(self.top_k).tolist() if not s2_df.empty else [None]
            s3_matches = s3_df[id_s3].head(self.top_k).tolist() if not s3_df.empty else [None]

            for s2_id_val in s2_matches:
                for s3_id_val in s3_matches:
                    candidates.append({
                        "s1_id": s1_id_val,
                        "s2_id": s2_id_val,
                        "s3_id": s3_id_val,
                        "blocking_score": 1.0
                    })

        cand_df = pd.DataFrame(candidates)
        logger.info(f"Generated {len(cand_df)} candidate triplets.")
        return cand_df
