"""
Text Normalization Module for Amazon ML Challenge 2026.

Cleans and standardizes text fields across Source 1, Source 2, and Source 3 tables.
Handles missing values, unicode characters, case folding, and whitespace normalization.
"""

import re
import logging
from typing import List, Optional
import pandas as pd

logger = logging.getLogger("pipeline.normalizer")


class DataNormalizer:
    """Normalizes string fields, handles missing values, and formats data frames."""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Standard text cleaning:
          - Converts to string and handles nulls/empty values
          - Lowercase transformation
          - Strips non-alphanumeric punctuation while preserving Unicode words (\w)
          - Collapses multiple whitespace characters to a single space
        """
        if text is None or not isinstance(text, str):
            return ""
        text = text.lower()
        # Replace non-word / non-space punctuation with space (preserving unicode letters)
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_dataframe(
        self,
        df: pd.DataFrame,
        text_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Applies normalization to specified text columns in a DataFrame.
        Creates '{col}_clean' for each text column.
        
        Args:
            df: Input source DataFrame.
            text_columns: List of text column names to normalize 
                          (default: ['business_name', 'business_address', 'country']).
        
        Returns:
            Normalized copy of DataFrame with '{col}_clean' columns added.
        """
        if df.empty:
            return df.copy()

        df_norm = df.copy()
        
        cols_to_clean = text_columns or [c for c in ["business_name", "business_address", "country", "name", "address"] if c in df_norm.columns]

        clean_cols = []
        for col in cols_to_clean:
            if col in df_norm.columns:
                clean_col_name = f"{col}_clean"
                df_norm[clean_col_name] = df_norm[col].fillna("").astype(str).apply(self.clean_text)
                clean_cols.append(clean_col_name)

        # Generate combined text representation for multi-field matching/blocking
        if clean_cols:
            df_norm["text_clean_combined"] = df_norm[clean_cols].agg(
                lambda row: " ".join([v for v in row if v]), axis=1
            )

        return df_norm
