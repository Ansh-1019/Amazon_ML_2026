"""
Text Normalization Module for Amazon ML Challenge 2026.

Cleans and standardizes text fields across Source 1, Source 2, and Source 3 tables.
Handles missing values, unicode characters, case folding, and whitespace normalization.

Integration note (2026-09-26):
    clean_text() now delegates to normalize_name() from src.data.normalization
    (Maithili's Unicode-aware implementation) so that the DataNormalizer class
    benefits from NFKC casefold, typographic punctuation normalization, and
    legal-abbreviation expansion.  The public interface is UNCHANGED.
"""

import logging
from typing import List, Optional
import pandas as pd

from .normalization import normalize_name

logger = logging.getLogger("pipeline.normalizer")


class DataNormalizer:
    """Normalizes string fields, handles missing values, and formats data frames."""

    @staticmethod
    def clean_text(text: object) -> str:
        """
        Standard text cleaning, now backed by Maithili's normalize_name():
          - Converts to string and handles nulls/empty values
          - NFKC casefold (Unicode-aware lowercase)
          - Typographic quote/hyphen normalization
          - Whitespace-delimited '&' → 'and'
          - Trailing legal-abbreviation expansion (corp, ltd, inc, pvt, co)
          - Collapses multiple whitespace characters to a single space

        The public behaviour contract (returns str, empty for None/empty input)
        is identical to the previous regex-based implementation.
        """
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        if text.strip() == "":
            return ""
        return normalize_name(text)

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

        cols_to_clean = text_columns or [
            c for c in ["business_name", "business_address", "country", "name", "address"]
            if c in df_norm.columns
        ]

        clean_cols = []
        for col in cols_to_clean:
            if col in df_norm.columns:
                clean_col_name = f"{col}_clean"
                df_norm[clean_col_name] = (
                    df_norm[col].fillna("").astype(str).apply(self.clean_text)
                )
                clean_cols.append(clean_col_name)

        # Generate combined text representation for multi-field matching/blocking
        if clean_cols:
            df_norm["text_clean_combined"] = df_norm[clean_cols].agg(
                lambda row: " ".join([v for v in row if v]), axis=1
            )

        return df_norm
