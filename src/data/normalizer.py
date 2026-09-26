"""
Text Normalization Module for Amazon ML Challenge 2026.

Cleans and standardizes text fields across Source 1, Source 2, and Source 3 tables.
Handles missing values, unicode characters, case folding, and whitespace normalization.

Integration notes:
    2026-09-26 (normalization): clean_text() delegates to normalize_name() from
        src.data.normalization (Maithili's Unicode-aware implementation) for NFKC
        casefold, typographic punctuation normalization, and legal-abbreviation
        expansion. The public interface is UNCHANGED.

    2026-09-26 (address): normalize_dataframe() now applies normalize_address() from
        src.data.address (Maithili's implementation) to the 'business_address' column,
        producing 'business_address_normalized' alongside the existing
        'business_address_clean'. The original 'business_address' column is preserved.
        All other column behaviour is unchanged.
"""

import logging
from typing import List, Optional
import pandas as pd

from .normalization import normalize_name
from .address import normalize_address

logger = logging.getLogger("pipeline.normalizer")

# Address columns that receive specialized address normalization in addition to
# the standard clean_text() pass.
_ADDRESS_COLUMNS = frozenset({"business_address", "address"})


class DataNormalizer:
    """Normalizes string fields, handles missing values, and formats data frames."""

    @staticmethod
    def clean_text(text: object) -> str:
        """
        Standard text cleaning, backed by Maithili's normalize_name():
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

    @staticmethod
    def clean_address(address: object) -> str:
        """
        Address-specific normalization backed by Maithili's normalize_address().

        Differs from clean_text() in that it:
          - Preserves commas, slashes, and separator-adjacent characters as spaces
            (intentionally loses separator distinctions for blocking heuristics)
          - Does NOT expand legal abbreviations (addresses have no such vocabulary)
          - Handles float/Decimal NaN as missing (returns "")
          - Converts non-string types via str()

        Returns "" for None, float NaN, Decimal NaN, and whitespace-only inputs.
        """
        return normalize_address(address)

    def normalize_dataframe(
        self,
        df: pd.DataFrame,
        text_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Applies normalization to specified text columns in a DataFrame.

        For every column in text_columns:
          - Creates '{col}_clean'  via clean_text()  (name-oriented normalization)

        Additionally, for address columns (business_address, address):
          - Creates '{col}_normalized' via clean_address() (address-specific
            normalization, preserves original column unchanged)

        The 'text_clean_combined' column is built from the '{col}_clean' columns only.
        Original columns are never modified.

        Args:
            df: Input source DataFrame.
            text_columns: List of text column names to normalize
                          (default: ['business_name', 'business_address', 'country']).

        Returns:
            Normalized copy of DataFrame with '{col}_clean' (and optionally
            '{col}_normalized' for address columns) added.
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
            if col not in df_norm.columns:
                continue

            # Standard name-oriented clean for every text column
            clean_col_name = f"{col}_clean"
            df_norm[clean_col_name] = (
                df_norm[col].fillna("").astype(str).apply(self.clean_text)
            )
            clean_cols.append(clean_col_name)

            # Address columns additionally get a dedicated address normalization column
            if col in _ADDRESS_COLUMNS:
                norm_col_name = f"{col}_normalized"
                df_norm[norm_col_name] = df_norm[col].apply(self.clean_address)

        # Generate combined text representation for multi-field matching/blocking
        if clean_cols:
            df_norm["text_clean_combined"] = df_norm[clean_cols].agg(
                lambda row: " ".join([v for v in row if v]), axis=1
            )

        return df_norm
