"""Pairwise Feature Extraction Bridge for Amazon ML Challenge 2026.

Integrates Anmol's authoritative feature extraction implementation (src.features.features)
with the canonical candidates DataFrame and source entity DataFrames.

Flow:
    candidates_df (s1_id, s2_id / s3_id / target_id) + s1_df, s2_df, s3_df
                   ↓
    Candidate/Entity join & wide pair construction (left_*, right_*)
                   ↓
    Anmol build_pair_features()
                   ↓
    Feature matrix with metadata identifiers (source1_entity_id, candidate_entity_id, target_source)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.features.features import build_pair_features, _clean_text, _tokenize, _jaccard_similarity

logger = logging.getLogger("pipeline.features")


def _resolve_col(df: pd.DataFrame, candidates: List[str], default: Optional[str] = None) -> Optional[str]:
    """Finds the first matching column name in a DataFrame."""
    if df is None:
        return default
    for c in candidates:
        if c in df.columns:
            return c
    return default


def _extract_entity_record_map(df: pd.DataFrame, id_col: str, name_col: Optional[str], addr_col: Optional[str], country_col: Optional[str]) -> Dict[str, Dict[str, str]]:
    """Builds a fast lookup map: entity_id -> {name, address, country}."""
    if df is None or df.empty or id_col not in df.columns:
        return {}

    record_map: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        eid = str(row[id_col]).strip()
        if not eid or eid in ("nan", "None"):
            continue
        record_map[eid] = {
            "name": str(row[name_col]).strip() if name_col and pd.notna(row.get(name_col)) else "",
            "address": str(row[addr_col]).strip() if addr_col and pd.notna(row.get(addr_col)) else "",
            "country": str(row[country_col]).strip() if country_col and pd.notna(row.get(country_col)) else "",
        }
    return record_map


class FeatureExtractor:
    """Extracts pairwise similarity features between Source 1, Source 2, and Source 3 candidate records."""

    @staticmethod
    def jaccard_similarity(str1: str, str2: str) -> float:
        """Computes Jaccard token similarity between two strings."""
        if not isinstance(str1, str) or not isinstance(str2, str):
            return 0.0
        s1 = _tokenize(str1)
        s2 = _tokenize(str2)
        if not s1 or not s2:
            return 0.0
        return _jaccard_similarity(s1, s2)

    def extract_features(
        self,
        candidates_df: pd.DataFrame,
        s1_df: Optional[pd.DataFrame] = None,
        s2_df: Optional[pd.DataFrame] = None,
        s3_df: Optional[pd.DataFrame] = None,
        config: Optional[dict] = None,
        target_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Extracts pairwise similarity features for all candidate records.
        
        Supports:
          1. Candidate DataFrame + Source DataFrames: Joins records into wide format and extracts features.
          2. Pre-joined wide pairs DataFrame (e.g. left_name, right_name): Directly computes features.
        
        Returns:
            DataFrame of features with attached metadata columns:
            [source1_entity_id, candidate_entity_id, target_source, s1_id, s2_id, s3_id, ...]
        """
        logger.info("Extracting pairwise similarity features via Anmol build_pair_features...")
        if candidates_df is None or candidates_df.empty:
            logger.warning("Empty candidates DataFrame received.")
            return pd.DataFrame()

        # Case 1: If candidates_df is already in wide format with left_* / right_* columns
        has_left = any(str(c).lower().startswith(("left_", "source_", "entity1_", "a_")) for c in candidates_df.columns)
        has_right = any(str(c).lower().startswith(("right_", "target_", "entity2_", "b_")) for c in candidates_df.columns)
        if has_left and has_right:
            return build_pair_features(candidates_df, target_col=target_col)

        # Case 2: Join candidate pairs with source DataFrames
        cfg = config or {}
        data_cfg = cfg.get("data", {})

        id_s1 = _resolve_col(s1_df, [data_cfg.get("id_column_s1"), "entity_id", "s1_id", "id", "source1_entity_id"], "entity_id")
        id_s2 = _resolve_col(s2_df, [data_cfg.get("id_column_s2"), "entity_id", "s2_id", "id", "candidate_entity_id"], "entity_id")
        id_s3 = _resolve_col(s3_df, [data_cfg.get("id_column_s3"), "entity_id", "s3_id", "id", "candidate_entity_id"], "entity_id")

        name_s1 = _resolve_col(s1_df, ["business_name_clean", "business_name", "name"])
        addr_s1 = _resolve_col(s1_df, ["business_address_clean", "business_address", "address"])
        country_s1 = _resolve_col(s1_df, ["country_clean", "country", "nation"])

        name_s2 = _resolve_col(s2_df, ["business_name_clean", "business_name", "name"])
        addr_s2 = _resolve_col(s2_df, ["business_address_clean", "business_address", "address"])
        country_s2 = _resolve_col(s2_df, ["country_clean", "country", "nation"])

        name_s3 = _resolve_col(s3_df, ["business_name_clean", "business_name", "name"])
        addr_s3 = _resolve_col(s3_df, ["business_address_clean", "business_address", "address"])
        country_s3 = _resolve_col(s3_df, ["country_clean", "country", "nation"])

        s1_map = _extract_entity_record_map(s1_df, id_s1, name_s1, addr_s1, country_s1)
        s2_map = _extract_entity_record_map(s2_df, id_s2, name_s2, addr_s2, country_s2)
        s3_map = _extract_entity_record_map(s3_df, id_s3, name_s3, addr_s3, country_s3)

        s2_id_set = set(s2_map.keys())
        s3_id_set = set(s3_map.keys())

        wide_rows: List[Dict[str, Any]] = []
        meta_rows: List[Dict[str, Any]] = []

        # Find s1 ID column in candidates_df
        cand_s1_col = _resolve_col(candidates_df, ["s1_id", "source1_entity_id", "entity_id", "id_s1"], "s1_id")

        for _, row in candidates_df.iterrows():
            s1_id_val = str(row.get(cand_s1_col, "")).strip().strip("\"'")
            if not s1_id_val or s1_id_val in ("nan", "None"):
                continue

            # Determine candidate ID and target source
            cand_id_val: Optional[str] = None
            target_source: str = "unknown"

            # Check explicit source column
            if "source" in row and pd.notna(row["source"]):
                src_str = str(row["source"]).strip().lower()
                if "2" in src_str or src_str == "s2":
                    target_source = "s2"
                elif "3" in src_str or src_str == "s3":
                    target_source = "s3"

            # Check explicit s2_id / s3_id columns
            s2_id_val = str(row.get("s2_id", "")).strip().strip("\"'") if "s2_id" in row and pd.notna(row["s2_id"]) else ""
            s3_id_val = str(row.get("s3_id", "")).strip().strip("\"'") if "s3_id" in row and pd.notna(row["s3_id"]) else ""
            target_id_val = str(row.get("target_id", "")).strip().strip("\"'") if "target_id" in row and pd.notna(row["target_id"]) else ""
            candidate_entity_id_val = str(row.get("candidate_entity_id", "")).strip().strip("\"'") if "candidate_entity_id" in row and pd.notna(row["candidate_entity_id"]) else ""

            if s2_id_val and s2_id_val not in ("nan", "None", ""):
                cand_id_val = s2_id_val
                target_source = "s2"
            elif s3_id_val and s3_id_val not in ("nan", "None", ""):
                cand_id_val = s3_id_val
                target_source = "s3"
            elif target_id_val and target_id_val not in ("nan", "None", ""):
                cand_id_val = target_id_val
                if target_source == "unknown":
                    if cand_id_val in s2_id_set:
                        target_source = "s2"
                    elif cand_id_val in s3_id_set:
                        target_source = "s3"
                    elif cand_id_val.startswith("S2-") or cand_id_val.startswith("s2_"):
                        target_source = "s2"
                    elif cand_id_val.startswith("S3-") or cand_id_val.startswith("s3_"):
                        target_source = "s3"
            elif candidate_entity_id_val and candidate_entity_id_val not in ("nan", "None", ""):
                cand_id_val = candidate_entity_id_val
                if target_source == "unknown":
                    if cand_id_val in s2_id_set:
                        target_source = "s2"
                    elif cand_id_val in s3_id_set:
                        target_source = "s3"
                    elif cand_id_val.startswith("S2-") or cand_id_val.startswith("s2_"):
                        target_source = "s2"
                    elif cand_id_val.startswith("S3-") or cand_id_val.startswith("s3_"):
                        target_source = "s3"

            if not cand_id_val:
                continue

            left_info = s1_map.get(s1_id_val, {})
            right_info = s2_map.get(cand_id_val, {}) if target_source == "s2" else s3_map.get(cand_id_val, {})
            if not right_info and target_source == "unknown":
                right_info = s2_map.get(cand_id_val) or s3_map.get(cand_id_val, {})

            wide_rows.append({
                "left_name": left_info.get("name", ""),
                "right_name": right_info.get("name", ""),
                "left_address": left_info.get("address", ""),
                "right_address": right_info.get("address", ""),
                "left_country": left_info.get("country", ""),
                "right_country": right_info.get("country", ""),
            })

            meta_rows.append({
                "source1_entity_id": s1_id_val,
                "candidate_entity_id": cand_id_val,
                "target_source": target_source,
                "s1_id": s1_id_val,
                "s2_id": cand_id_val if target_source == "s2" else None,
                "s3_id": cand_id_val if target_source == "s3" else None,
                "target_id": cand_id_val,
            })

        if not wide_rows:
            logger.warning("No valid candidate pair rows constructed.")
            return pd.DataFrame()

        wide_df = pd.DataFrame(wide_rows)
        meta_df = pd.DataFrame(meta_rows)

        # Build similarity features using Anmol's canonical build_pair_features
        features_df = build_pair_features(wide_df, target_col=None)

        # Prepend metadata identifier columns
        for col in reversed(["source1_entity_id", "candidate_entity_id", "target_source", "s1_id", "s2_id", "s3_id", "target_id"]):
            if col in meta_df.columns:
                features_df.insert(0, col, meta_df[col].values)

        logger.info(
            f"Extracted {features_df.select_dtypes(include=[np.number]).shape[1]} numeric features "
            f"for {len(features_df)} candidate pairs."
        )
        return features_df
