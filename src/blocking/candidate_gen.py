"""Candidate generation adapter integrating Maithili's multi-block pipeline into the main project.

Preserves Maithili's authoritative blocking algorithms and components:
  - Exact name and address fingerprint blocking (exact_blocking.py)
  - Token-based inverted index blocking (blocking.py)
  - Character n-gram TF-IDF similarity blocking (char_ngram_blocking.py)
  - Rare token blocking (rare_blocking.py)
  - Deterministic candidate union (candidate_union.py)
  - Candidate pipeline orchestration (candidate_pipeline.py)
  - Candidate retrieval diagnostics (candidate_diagnostics.py)

Provides a clean, prefix-agnostic, reversible ID adapter at the boundary so that
arbitrary entity ID formats are supported while preserving the original IDs.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple
import pandas as pd

from src.data.normalization import normalize_name, tokenize_name
from src.data.address import address_fingerprint
from src.blocking.blocking import DEFAULT_STOPWORDS, build_token_index, generate_token_candidates
from src.blocking.exact_blocking import build_exact_index, generate_exact_candidates
from src.blocking.char_ngram_blocking import build_char_ngram_index, generate_char_ngram_candidates
from src.blocking.rare_blocking import build_rare_token_index, generate_rare_token_candidates
from src.blocking.candidate_union import union_candidates
from src.blocking.candidate_pipeline import CandidatePipeline, CandidateResult

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


def resolve_text_col(df: pd.DataFrame, preferred: List[str]) -> Optional[str]:
    """Finds the first matching column name from a list of preferred column names."""
    for col in preferred:
        if col in df.columns:
            return col
    return None


class IdAdapter:
    """Reversible ID adapter that maps source entity IDs to source-tagged internal IDs and back.
    
    Ensures Maithili's prefix-sensitive diagnostic and evaluation utilities can reliably
    differentiate Source 2 from Source 3, while preserving original entity IDs verbatim.
    """

    def __init__(self) -> None:
        self.internal_to_original: dict[str, tuple[str, str]] = {}

    def register_s2(self, original_ids: Sequence[object]) -> list[str]:
        internal_ids: list[str] = []
        for i, raw_id in enumerate(original_ids):
            orig = str(raw_id).strip()
            if orig.startswith("S2-"):
                internal = orig
            else:
                internal = f"S2-@{i}@{orig}"
            self.internal_to_original[internal] = (orig, "s2")
            internal_ids.append(internal)
        return internal_ids

    def register_s3(self, original_ids: Sequence[object]) -> list[str]:
        internal_ids: list[str] = []
        for i, raw_id in enumerate(original_ids):
            orig = str(raw_id).strip()
            if orig.startswith("S3-"):
                internal = orig
            else:
                internal = f"S3-@{i}@{orig}"
            self.internal_to_original[internal] = (orig, "s3")
            internal_ids.append(internal)
        return internal_ids

    def decode(self, internal_id: str) -> tuple[str, str]:
        """Decodes an internal ID back to (original_id, source)."""
        if internal_id in self.internal_to_original:
            return self.internal_to_original[internal_id]
        if internal_id.startswith("S2-"):
            return internal_id, "s2"
        if internal_id.startswith("S3-"):
            return internal_id, "s3"
        return internal_id, "unknown"


class CandidateGenerator:
    """Generates candidate matches from Source 2 and Source 3 for each entity in Source 1.
    
    Acts as a thin adapter around Maithili's multi-block CandidatePipeline.
    """

    def __init__(
        self,
        top_k: int = 20,
        blocking_fields: Optional[List[str]] = None,
        min_token_length: int = 2,
        max_token_frequency: Optional[int] = None,
        ngram_range: Tuple[int, int] = (2, 5),
        enabled_blocks: Optional[Sequence[str]] = None,
    ) -> None:
        self.top_k = top_k
        self.blocking_fields = blocking_fields or ["business_name", "business_address", "country"]
        self.min_token_length = min_token_length
        self.max_token_frequency = max_token_frequency
        self.ngram_range = ngram_range
        self.enabled_blocks = tuple(
            enabled_blocks or ("exact_name", "exact_address", "token", "char_ngram")
        )

    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        config: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Generates candidate matches for all Source 1 records using Maithili's CandidatePipeline.
        
        Returns a canonical candidate DataFrame with columns:
            ['s1_id', 's2_id', 's3_id', 'target_id', 'source', 'block_reasons', 'blocking_score']
        """
        logger.info("Generating candidate pairs via Maithili CandidatePipeline...")
        columns = ["s1_id", "s2_id", "s3_id", "target_id", "source", "block_reasons", "blocking_score"]
        if s1_df is None or s1_df.empty:
            logger.warning("Source 1 DataFrame is empty.")
            return pd.DataFrame(columns=columns)

        cfg = config or {}
        data_cfg = cfg.get("data", {})
        blocking_cfg = cfg.get("blocking", {})

        id_s1 = resolve_id_col(s1_df, data_cfg.get("id_column_s1"), ["entity_id", "s1_id", "id"])
        id_s2 = resolve_id_col(s2_df, data_cfg.get("id_column_s2"), ["entity_id", "s2_id", "id"]) if s2_df is not None and not s2_df.empty else "entity_id"
        id_s3 = resolve_id_col(s3_df, data_cfg.get("id_column_s3"), ["entity_id", "s3_id", "id"]) if s3_df is not None and not s3_df.empty else "entity_id"

        top_k = blocking_cfg.get("top_k", self.top_k)
        min_token_length = blocking_cfg.get("min_token_length", self.min_token_length)
        max_token_frequency = blocking_cfg.get("max_token_frequency", self.max_token_frequency)
        max_candidates = blocking_cfg.get("max_candidates", None)
        ngram_range = tuple(blocking_cfg.get("ngram_range", self.ngram_range))

        # Identify name and address columns across sources
        name_s1_col = resolve_text_col(s1_df, ["business_name_clean", "business_name", "name"])
        addr_s1_col = resolve_text_col(s1_df, ["business_address_clean", "business_address", "address"])

        name_s2_col = resolve_text_col(s2_df, ["business_name_clean", "business_name", "name"]) if s2_df is not None and not s2_df.empty else None
        addr_s2_col = resolve_text_col(s2_df, ["business_address_clean", "business_address", "address"]) if s2_df is not None and not s2_df.empty else None

        name_s3_col = resolve_text_col(s3_df, ["business_name_clean", "business_name", "name"]) if s3_df is not None and not s3_df.empty else None
        addr_s3_col = resolve_text_col(s3_df, ["business_address_clean", "business_address", "address"]) if s3_df is not None and not s3_df.empty else None

        # Build reversible ID mappings
        id_adapter = IdAdapter()
        target_ids: list[str] = []
        target_names: list[object] = []
        target_addresses: list[object] = []

        if s2_df is not None and not s2_df.empty and id_s2 in s2_df.columns:
            s2_orig_ids = s2_df[id_s2].tolist()
            s2_internal_ids = id_adapter.register_s2(s2_orig_ids)
            target_ids.extend(s2_internal_ids)
            target_names.extend(s2_df[name_s2_col].tolist() if name_s2_col else [""] * len(s2_df))
            target_addresses.extend(s2_df[addr_s2_col].tolist() if addr_s2_col else [""] * len(s2_df))

        if s3_df is not None and not s3_df.empty and id_s3 in s3_df.columns:
            s3_orig_ids = s3_df[id_s3].tolist()
            s3_internal_ids = id_adapter.register_s3(s3_orig_ids)
            target_ids.extend(s3_internal_ids)
            target_names.extend(s3_df[name_s3_col].tolist() if name_s3_col else [""] * len(s3_df))
            target_addresses.extend(s3_df[addr_s3_col].tolist() if addr_s3_col else [""] * len(s3_df))

        if not target_ids:
            logger.warning("No targets available in Source 2 or Source 3.")
            return pd.DataFrame(columns=columns)

        # Build Maithili's indexes over aligned targets
        exact_name_index = build_exact_index(target_ids, (normalize_name(n) if n is not None else None for n in target_names))
        exact_addr_index = build_exact_index(target_ids, (address_fingerprint(a) if a is not None else None for a in target_addresses))
        token_index = build_token_index(
            target_ids, target_names,
            stopwords=DEFAULT_STOPWORDS,
            min_token_length=min_token_length,
            max_token_frequency=max_token_frequency,
        )
        char_ngram_index = build_char_ngram_index(
            target_ids, target_names,
            ngram_range=ngram_range,
            min_df=1,
            max_features=None,
        )

        # Configure Maithili's CandidatePipeline
        blocks: dict[str, Any] = {}
        if "exact_name" in self.enabled_blocks:
            blocks["exact_name"] = lambda rec: generate_exact_candidates(
                normalize_name(rec.get("name")), exact_name_index
            )
        if "exact_address" in self.enabled_blocks:
            blocks["exact_address"] = lambda rec: generate_exact_candidates(
                address_fingerprint(rec.get("address")), exact_addr_index
            )
        if "token" in self.enabled_blocks:
            blocks["token"] = lambda rec: generate_token_candidates(
                rec.get("name"), token_index,
                stopwords=DEFAULT_STOPWORDS,
                min_token_length=min_token_length,
                max_candidates=max_candidates,
            )
        if "char_ngram" in self.enabled_blocks:
            blocks["char_ngram"] = lambda rec: generate_char_ngram_candidates(
                rec.get("name"), char_ngram_index, top_k=top_k
            )

        pipeline = CandidatePipeline(blocks)

        # Generate candidates per S1 entity
        candidate_rows: list[dict[str, Any]] = []
        for _, s1_row in s1_df.iterrows():
            s1_id_val = str(s1_row[id_s1]).strip()
            s1_name_val = s1_row[name_s1_col] if name_s1_col and pd.notna(s1_row[name_s1_col]) else ""
            s1_addr_val = s1_row[addr_s1_col] if addr_s1_col and pd.notna(s1_row[addr_s1_col]) else ""

            rec = {"id": s1_id_val, "name": s1_name_val, "address": s1_addr_val}
            result: CandidateResult = pipeline.generate(rec)

            for internal_cand_id in result.all_candidates:
                orig_id, source = id_adapter.decode(internal_cand_id)
                reasons = result.provenance.get(internal_cand_id, ())
                reason_str = ",".join(reasons)

                if source == "s2":
                    candidate_rows.append({
                        "s1_id": s1_id_val,
                        "s2_id": orig_id,
                        "s3_id": None,
                        "target_id": orig_id,
                        "source": "s2",
                        "block_reasons": reason_str,
                        "blocking_score": 1.0,
                    })
                else:
                    candidate_rows.append({
                        "s1_id": s1_id_val,
                        "s2_id": None,
                        "s3_id": orig_id,
                        "target_id": orig_id,
                        "source": "s3",
                        "block_reasons": reason_str,
                        "blocking_score": 1.0,
                    })

        cand_df = pd.DataFrame(candidate_rows, columns=columns)
        logger.info(f"Generated {len(cand_df)} candidate pairs across {s1_df[id_s1].nunique()} Source 1 entities.")
        return cand_df
