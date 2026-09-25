import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd


logger = logging.getLogger("pipeline.data_loader")


class DataLoader:
    """Handles data ingestion for Source 1, Source 2, Source 3, and training matches."""

    def __init__(self, raw_data_dir: Union[str, Path]):
        self.raw_data_dir = Path(raw_data_dir)

    def load_sources(self, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Loads raw data tables.
        Returns:
            s1_df: DataFrame for Source 1 (Reference)
            s2_df: DataFrame for Source 2
            s3_df: DataFrame for Source 3
            train_matches: Optional DataFrame for Ground Truth training matches
        """
        s1_file = self.raw_data_dir / config["data"]["source1_filename"]
        s2_file = self.raw_data_dir / config["data"]["source2_filename"]
        s3_file = self.raw_data_dir / config["data"]["source3_filename"]
        train_matches_file = self.raw_data_dir / config["data"].get("train_matches_filename", "train_matches.csv")

        if not s1_file.exists():
            logger.warning(f"Source 1 file not found at {s1_file}. Returning empty DataFrame.")
            s1_df = pd.DataFrame()
        else:
            s1_df = self._read_file(s1_file)

        if not s2_file.exists():
            logger.warning(f"Source 2 file not found at {s2_file}. Returning empty DataFrame.")
            s2_df = pd.DataFrame()
        else:
            s2_df = self._read_file(s2_file)

        if not s3_file.exists():
            logger.warning(f"Source 3 file not found at {s3_file}. Returning empty DataFrame.")
            s3_df = pd.DataFrame()
        else:
            s3_df = self._read_file(s3_file)

        train_matches = None
        if train_matches_file.exists():
            train_matches = self._read_file(train_matches_file)

        return s1_df, s2_df, s3_df, train_matches

    def _read_file(self, file_path: Path) -> pd.DataFrame:
        logger.info(f"Reading file: {file_path}")
        if file_path.suffix.lower() == ".parquet":
            return pd.read_parquet(file_path)
        elif file_path.suffix.lower() in [".json", ".jsonl"]:
            return pd.read_json(file_path, lines=True if file_path.suffix.lower() == ".jsonl" else False)
        else:
            return pd.read_csv(file_path)
