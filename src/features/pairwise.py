import logging
import pandas as pd
import numpy as np


logger = logging.getLogger("pipeline.features")


class FeatureExtractor:
    """Extracts pairwise similarity features between Source 1, Source 2, and Source 3 candidate records."""

    @staticmethod
    def jaccard_similarity(str1: str, str2: str) -> float:
        if not isinstance(str1, str) or not isinstance(str2, str):
            return 0.0
        set1, set2 = set(str1.split()), set(str2.split())
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def extract_features(
        self,
        candidates_df: pd.DataFrame,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        config: dict
    ) -> pd.DataFrame:
        """
        Calculates similarity features for each candidate triplet.
        Returns DataFrame of features X matching candidates_df order.
        """
        logger.info("Extracting pairwise similarity features...")
        if candidates_df.empty:
            return pd.DataFrame()

        # Merge text fields to compute similarities
        id_s1 = config["data"]["id_column_s1"]
        id_s2 = config["data"]["id_column_s2"]
        id_s3 = config["data"]["id_column_s3"]

        features_df = candidates_df.copy()

        # Placeholder similarity feature computations
        features_df["s1_s2_jaccard"] = 0.5
        features_df["s1_s3_jaccard"] = 0.5
        features_df["s2_s3_jaccard"] = 0.5

        feature_cols = [c for c in features_df.columns if c not in ["s1_id", "s2_id", "s3_id"]]
        logger.info(f"Extracted {len(feature_cols)} feature columns.")
        return features_df[feature_cols]
