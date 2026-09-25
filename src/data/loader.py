"""
Data Loading and Schema Handling Module for Amazon ML Challenge 2026.

Handles ingestion of Source 1, Source 2, Source 3, and Ground Truth datasets.
Enforces canonical schema validation, TSV parsing, whitespace stripping,
UTF-8 text encoding, duplicate ID handling, and source identity preservation.

Canonical Source Schema:
  - entity_id: str (unique record identifier)
  - business_name: str (name of business entity)
  - business_address: str (street address, city, state, postal code)
  - country: str (country name/code, unconstrained to allow any country e.g. France, US, IN)
  - source: str ('source1', 'source2', 'source3')
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger("pipeline.data_loader")

# Canonical required columns for Source 1, Source 2, Source 3 tables
CANONICAL_SOURCE_COLUMNS: List[str] = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

# Supported ground truth schema column configurations
GROUND_TRUTH_COMPETITION_COLUMNS: List[str] = [
    "source1_entity_id",
    "matched_entity_ids",
]

GROUND_TRUTH_LEGACY_COLUMNS: List[str] = [
    "s1_id",
    "s2_id",
    "s3_id",
]


class SchemaValidationError(ValueError):
    """Raised when an input table fails schema validation (e.g. missing required columns)."""
    pass


class DataLoader:
    """
    Robust data ingestion module for multi-source business entity resolution.
    
    Reads TSV tables by default, enforces strict schema validation without fabricating
    missing columns, preserves source identity, and cleans text data.
    """

    def __init__(self, raw_data_dir: Union[str, Path], strict: bool = True):
        self.raw_data_dir = Path(raw_data_dir)
        self.strict = strict

    def load_sources(
        self, config: dict
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Loads and standardizes Source 1, Source 2, Source 3, and optional training matches.
        
        Returns:
            s1_df: DataFrame for Source 1 (Reference)
            s2_df: DataFrame for Source 2
            s3_df: DataFrame for Source 3
            train_matches: Optional DataFrame for Ground Truth matches
        """
        data_cfg = config.get("data", {})
        s1_filename = data_cfg.get("source1_filename", "source1.tsv")
        s2_filename = data_cfg.get("source2_filename", "source2.tsv")
        s3_filename = data_cfg.get("source3_filename", "source3.tsv")
        train_matches_filename = data_cfg.get("train_matches_filename", "train_matches.tsv")

        s1_file = self.raw_data_dir / s1_filename
        s2_file = self.raw_data_dir / s2_filename
        s3_file = self.raw_data_dir / s3_filename
        train_matches_file = self.raw_data_dir / train_matches_filename

        # Load Source 1
        if not s1_file.exists():
            logger.warning(f"Source 1 file not found at '{s1_file}'. Returning empty canonical DataFrame.")
            s1_df = self._empty_source_df(source_name="source1")
        else:
            s1_df = self.load_source_file(s1_file, source_name="source1")

        # Load Source 2
        if not s2_file.exists():
            logger.warning(f"Source 2 file not found at '{s2_file}'. Returning empty canonical DataFrame.")
            s2_df = self._empty_source_df(source_name="source2")
        else:
            s2_df = self.load_source_file(s2_file, source_name="source2")

        # Load Source 3
        if not s3_file.exists():
            logger.warning(f"Source 3 file not found at '{s3_file}'. Returning empty canonical DataFrame.")
            s3_df = self._empty_source_df(source_name="source3")
        else:
            s3_df = self.load_source_file(s3_file, source_name="source3")

        # Load Training Matches
        train_matches = None
        if train_matches_file.exists():
            train_matches = self.load_train_matches(train_matches_file)
        else:
            # Check legacy csv fallback if tsv not found
            csv_fallback = self.raw_data_dir / "train_matches.csv"
            if csv_fallback.exists():
                train_matches = self.load_train_matches(csv_fallback)

        return s1_df, s2_df, s3_df, train_matches

    def load_source_file(
        self,
        file_path: Union[str, Path],
        source_name: str,
        required_cols: Optional[List[str]] = None,
        deduplicate: bool = True,
    ) -> pd.DataFrame:
        """
        Loads a single source table (Source 1, 2, or 3), validates schema, and standardizes columns.

        Args:
            file_path: Path to the raw source file (TSV/CSV/Parquet).
            source_name: Name of source ('source1', 'source2', 'source3').
            required_cols: Optional override of required columns (default: CANONICAL_SOURCE_COLUMNS).
            deduplicate: If True, warns and deduplicates identical entity_ids keeping the first.

        Returns:
            Standardized DataFrame with ['entity_id', 'business_name', 'business_address', 'country', 'source'].

        Raises:
            SchemaValidationError: If required columns are missing from the file.
        """
        path = Path(file_path)
        logger.info(f"Loading {source_name} from: {path}")
        df = self._read_raw_file(path)

        # Validate Schema
        cols_to_check = required_cols or CANONICAL_SOURCE_COLUMNS
        self._validate_source_schema(df, source_name=source_name, file_path=path, required_cols=cols_to_check)

        # Clean string values: strip whitespace, normalize NaNs to empty strings
        for col in cols_to_check:
            df[col] = df[col].fillna("").astype(str).str.strip()

        # Handle and validate entity_id
        if "entity_id" in df.columns:
            # Check for empty/blank entity IDs
            blank_mask = df["entity_id"] == ""
            if blank_mask.any():
                blank_count = blank_mask.sum()
                logger.warning(f"Found {blank_count} rows with blank entity_id in {source_name} ({path.name}). Dropping blank ID rows.")
                df = df[~blank_mask].copy()

            # Check for duplicate entity_ids
            dups = df[df.duplicated(subset=["entity_id"], keep=False)]
            if not dups.empty:
                unique_dups = dups["entity_id"].unique()
                logger.warning(
                    f"Found {len(dups)} duplicate entity_id rows ({len(unique_dups)} unique IDs) in {source_name} ({path.name}). "
                    f"Example duplicate IDs: {list(unique_dups[:3])}"
                )
                if deduplicate:
                    logger.info(f"Deduplicating {source_name} by entity_id (keeping first occurrence).")
                    df = df.drop_duplicates(subset=["entity_id"], keep="first").copy()

        # Preserve Source Identity
        df["source"] = source_name

        # Ensure canonical column order first, followed by any extra columns
        ordered_cols = [c for c in CANONICAL_SOURCE_COLUMNS if c in df.columns]
        extra_cols = [c for c in df.columns if c not in ordered_cols and c != "source"]
        final_cols = ordered_cols + ["source"] + extra_cols
        df = df[final_cols].reset_index(drop=True)

        logger.info(f"Successfully loaded {len(df)} records for {source_name}.")
        return df

    def load_train_matches(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Loads ground truth match labels file.
        Supports competition schema ('source1_entity_id', 'matched_entity_ids')
        and legacy schema ('s1_id', 's2_id', 's3_id').
        """
        path = Path(file_path)
        logger.info(f"Loading ground truth matches from: {path}")
        df = self._read_raw_file(path)

        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]

        # Determine schema format
        if "source1_entity_id" in df.columns and "matched_entity_ids" in df.columns:
            df["source1_entity_id"] = df["source1_entity_id"].fillna("").astype(str).str.strip()
            df["matched_entity_ids"] = df["matched_entity_ids"].fillna("").astype(str).str.strip()
            # Clean out blank s1 IDs
            df = df[df["source1_entity_id"] != ""].reset_index(drop=True)
            logger.info(f"Loaded {len(df)} ground truth matches (competition schema).")
            return df
        elif "s1_id" in df.columns:
            for col in ["s1_id", "s2_id", "s3_id"]:
                if col in df.columns:
                    df[col] = df[col].fillna("").astype(str).str.strip()
            df = df[df["s1_id"] != ""].reset_index(drop=True)
            logger.info(f"Loaded {len(df)} ground truth matches (triplet schema).")
            return df
        else:
            raise SchemaValidationError(
                f"Invalid ground truth schema in '{path.name}'. "
                f"Expected either {GROUND_TRUTH_COMPETITION_COLUMNS} or {GROUND_TRUTH_LEGACY_COLUMNS}, "
                f"but found columns: {list(df.columns)}"
            )

    def _validate_source_schema(
        self,
        df: pd.DataFrame,
        source_name: str,
        file_path: Path,
        required_cols: List[str],
    ) -> None:
        """
        Validates that all required columns are present in DataFrame headers.
        Does NOT fabricate missing columns; raises SchemaValidationError with a clear message.
        """
        present_cols = set(df.columns)
        missing_cols = [c for c in required_cols if c not in present_cols]

        if missing_cols:
            raise SchemaValidationError(
                f"Schema validation failed for {source_name} ({file_path.name})! "
                f"Missing required column(s): {missing_cols}. "
                f"Found columns: {list(df.columns)}. "
                f"Expected schema: {required_cols}"
            )

    def _read_raw_file(self, file_path: Path) -> pd.DataFrame:
        """
        Reads a data file using TSV as the default delimiter for text files.
        Robust to UTF-8 text, whitespace in headers, and empty files.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Data file does not exist: {file_path}")

        ext = file_path.suffix.lower()

        if ext == ".parquet":
            df = pd.read_parquet(file_path)
        elif ext in [".json", ".jsonl"]:
            df = pd.read_json(file_path, lines=True if ext == ".jsonl" else False, dtype=False)
        else:
            # Default text reader: TSV format
            # Auto-detect delimiter if file is explicitly .csv and not tab-separated
            sep = "\t" if ext != ".csv" else ","
            try:
                # First attempt with determined separator
                df = pd.read_csv(
                    file_path,
                    sep=sep,
                    dtype=str,
                    keep_default_na=False,
                    encoding="utf-8",
                    on_bad_lines="error",
                )
                # If only 1 column was parsed but tabs exist in header, retry with tab
                if len(df.columns) == 1 and "\t" in df.columns[0]:
                    df = pd.read_csv(
                        file_path,
                        sep="\t",
                        dtype=str,
                        keep_default_na=False,
                        encoding="utf-8",
                    )
            except Exception as e:
                # Fallback attempt with tab separator
                try:
                    df = pd.read_csv(
                        file_path,
                        sep="\t",
                        dtype=str,
                        keep_default_na=False,
                        encoding="utf-8",
                    )
                except Exception:
                    raise IOError(f"Failed to read data file '{file_path.name}': {str(e)}")

        # Strip whitespace from column headers
        df.columns = [str(c).strip() for c in df.columns]
        return df

    def _empty_source_df(self, source_name: str) -> pd.DataFrame:
        """Returns an empty DataFrame conforming to the canonical source schema."""
        df = pd.DataFrame(columns=CANONICAL_SOURCE_COLUMNS)
        df["source"] = pd.Series(dtype=str)
        return df
