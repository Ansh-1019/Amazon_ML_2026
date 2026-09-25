import re
import logging
import pandas as pd


logger = logging.getLogger("pipeline.normalizer")


class DataNormalizer:
    """Normalizes string fields, handles missing values, and formats data frames."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Standard text cleaning: lowercase, strip punctuation, remove extra whitespaces."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_dataframe(self, df: pd.DataFrame, text_columns: list) -> pd.DataFrame:
        """Applies normalization to text columns in a DataFrame."""
        if df.empty:
            return df
        
        df_norm = df.copy()
        for col in text_columns:
            if col in df_norm.columns:
                df_norm[f"{col}_clean"] = df_norm[col].fillna("").astype(str).apply(self.clean_text)
        return df_norm
