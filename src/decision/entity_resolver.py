"""
Entity-Level Decision Layer for Amazon ML Challenge 2026.

Converts per-candidate match scores into entity-level match SETS.

Design Principles
-----------------
- Every Source 1 entity MUST appear in the output (zero, one, or many matches).
- A match set may be empty (no candidates above threshold).
- A match set may contain multiple IDs (several high-scoring candidates).
- The output match set is ALWAYS a strict subset of the entity's candidate set.
- Duplicate target IDs are never emitted.
- No assumption of one-to-one matching.

Decision Flow
-------------
scored candidate pairs
        ↓
optional per-source threshold filtering
        ↓
global threshold filtering
        ↓
optional top-margin pruning
        ↓
{s1_id → [matched_id, ...]}

Thresholds
----------
threshold         : Global minimum score to accept any match (default 0.5).
threshold_s2      : Optional separate minimum for Source 2 IDs.
threshold_s3      : Optional separate minimum for Source 3 IDs.
top_margin        : If set, only keep IDs within `top_margin` of the top score
                    for that entity (e.g. margin=0.15 keeps all IDs within 0.15
                    of the best score). Applied AFTER threshold filtering.
                    Set to None (default) to disable.

Example
-------
  S1_A candidates after threshold filter:
      s2_001  score=0.97
      s2_002  score=0.95
      s3_101  score=0.22  <- below threshold=0.5, already removed

  With top_margin=None  → matches = [s2_001, s2_002]  (both kept)
  With top_margin=0.10  → best=0.97, drop anything < 0.87
                        → matches = [s2_001, s2_002]  (0.95 >= 0.87, kept)
  With top_margin=0.01  → best=0.97, drop anything < 0.96
                        → matches = [s2_001]           (0.95 < 0.96, dropped)
"""

import logging
from typing import Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger("pipeline.decision")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: detect which source a target ID belongs to
# ─────────────────────────────────────────────────────────────────────────────

def _infer_source(target_id: str, s2_ids: Optional[Set[str]], s3_ids: Optional[Set[str]]) -> str:
    """Returns 's2', 's3', or 'unknown' for a target ID."""
    if s2_ids is not None and target_id in s2_ids:
        return "s2"
    if s3_ids is not None and target_id in s3_ids:
        return "s3"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class EntityResolver:
    """
    Converts candidate-level match scores into entity-level match SETS.

    Supports:
      - Zero matches per Source 1 entity
      - One match per Source 1 entity
      - Multiple matches per Source 1 entity

    Never forces exactly one match (old behaviour was: groupby/first → always 1).
    """

    def __init__(
        self,
        threshold: float = 0.5,
        threshold_s2: Optional[float] = None,
        threshold_s3: Optional[float] = None,
        top_margin: Optional[float] = None,
    ):
        """
        Args:
            threshold:    Global minimum match score (applied to all candidates).
            threshold_s2: Optional override minimum score for Source 2 IDs.
                          If None, `threshold` is used for Source 2 IDs.
            threshold_s3: Optional override minimum score for Source 3 IDs.
                          If None, `threshold` is used for Source 3 IDs.
            top_margin:   Optional margin below the entity's top score.
                          After threshold filtering, drop any candidate whose
                          score < (top_score - top_margin).  Set None to disable.
        """
        self.threshold = threshold
        self.threshold_s2 = threshold_s2
        self.threshold_s3 = threshold_s3
        self.top_margin = top_margin

    # ------------------------------------------------------------------
    # Public API (preserves the pipeline's existing .resolve(df, config) call)
    # ------------------------------------------------------------------

    def resolve(
        self,
        scored_candidates_df: pd.DataFrame,
        config: dict,
        s2_ids: Optional[Set[str]] = None,
        s3_ids: Optional[Set[str]] = None,
    ) -> pd.DataFrame:
        """
        Resolves entity-level match sets from scored candidate pairs.

        Args:
            scored_candidates_df: DataFrame with columns including an s1 ID column,
                                  one or more target ID columns (s2_id / s3_id /
                                  target_id), and 'match_score'.
            config: Pipeline configuration dict (reads decision sub-keys if present).
            s2_ids: Optional set of all valid Source 2 IDs (for per-source thresholds).
            s3_ids: Optional set of all valid Source 3 IDs.

        Returns:
            DataFrame with one row per (s1_entity, matched_target) pair.
            Columns: [s1_id_col, target_id, match_score, source]
            If an S1 entity has zero matches above threshold, it produces NO rows
            (the pipeline's resolved_df_to_map guarantees all S1 entities still
            appear in the final map, defaulting to []).
        """
        # ── Read thresholds from config (fall back to constructor values) ──
        dec_cfg = config.get("decision", {})
        threshold    = dec_cfg.get("threshold",    self.threshold)
        threshold_s2 = dec_cfg.get("threshold_s2", self.threshold_s2)
        threshold_s3 = dec_cfg.get("threshold_s3", self.threshold_s3)
        top_margin   = dec_cfg.get("top_margin",   self.top_margin)

        # Effective per-source thresholds (fall back to global)
        thr_s2 = threshold_s2 if threshold_s2 is not None else threshold
        thr_s3 = threshold_s3 if threshold_s3 is not None else threshold

        logger.info(
            f"EntityResolver: global_threshold={threshold}, "
            f"thr_s2={thr_s2}, thr_s3={thr_s3}, top_margin={top_margin}"
        )

        if scored_candidates_df is None or scored_candidates_df.empty:
            logger.warning("EntityResolver received empty candidates DataFrame.")
            return pd.DataFrame(columns=["s1_id", "target_id", "match_score", "source"])

        df = scored_candidates_df.copy()

        # ── Detect s1 ID column ──
        s1_col = self._detect_s1_col(df)
        if s1_col is None:
            logger.error("EntityResolver: cannot find an s1 ID column in candidates DataFrame.")
            return pd.DataFrame(columns=["s1_id", "target_id", "match_score", "source"])

        # ── Detect target ID columns and melt into long form ──
        df_long = self._melt_to_long_form(df, s1_col)
        if df_long.empty:
            logger.warning("EntityResolver: no target IDs found after melting candidates.")
            return pd.DataFrame(columns=["s1_id", "target_id", "match_score", "source"])

        # ── Ensure match_score column exists ──
        if "match_score" not in df_long.columns:
            logger.warning("EntityResolver: 'match_score' column missing; defaulting all scores to 0.0.")
            df_long["match_score"] = 0.0

        df_long["match_score"] = pd.to_numeric(df_long["match_score"], errors="coerce").fillna(0.0)

        # ── Tag each row with its source ──
        df_long["source"] = df_long["target_id"].apply(
            lambda tid: _infer_source(tid, s2_ids, s3_ids)
        )

        # ── Apply per-source threshold filter ──
        mask_s2 = (df_long["source"] == "s2") & (df_long["match_score"] >= thr_s2)
        mask_s3 = (df_long["source"] == "s3") & (df_long["match_score"] >= thr_s3)
        mask_uk = (df_long["source"] == "unknown") & (df_long["match_score"] >= threshold)
        passed_df = df_long[mask_s2 | mask_s3 | mask_uk].copy()

        if passed_df.empty:
            logger.info("EntityResolver: no candidate passed the score threshold(s).")
            return pd.DataFrame(columns=["s1_id", "target_id", "match_score", "source"])

        # ── Remove duplicate (s1_id, target_id) pairs, keeping highest score ──
        passed_df = (
            passed_df
            .sort_values("match_score", ascending=False)
            .drop_duplicates(subset=["s1_id", "target_id"], keep="first")
        )

        # ── Optional top-margin pruning (entity-local) ──
        if top_margin is not None and top_margin >= 0.0:
            passed_df = self._apply_top_margin(passed_df, top_margin)

        # ── Rename s1 column to a canonical name for downstream ──
        if s1_col != "s1_id":
            passed_df = passed_df.rename(columns={s1_col: "s1_id"})

        result = passed_df[["s1_id", "target_id", "match_score", "source"]].reset_index(drop=True)

        # Diagnostics
        n_entities_matched = result["s1_id"].nunique()
        n_total_matches = len(result)
        multi_match = (result.groupby("s1_id").size() > 1).sum()
        logger.info(
            f"EntityResolver: {n_entities_matched} S1 entities matched, "
            f"{n_total_matches} total match pairs, "
            f"{multi_match} entities with multiple matches."
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_s1_col(self, df: pd.DataFrame) -> Optional[str]:
        """Returns the first s1 ID column found in the DataFrame."""
        for col in ("s1_id", "source1_entity_id", "entity_id_s1"):
            if col in df.columns:
                return col
        # Heuristic: any column whose name contains 's1'
        for col in df.columns:
            if "s1" in col.lower():
                return col
        return None

    def _melt_to_long_form(self, df: pd.DataFrame, s1_col: str) -> pd.DataFrame:
        """
        Converts a wide-form candidates DataFrame (one row per s1↔s2/s3 pair/triplet)
        into a long-form DataFrame with columns [s1_col, target_id, match_score].

        Handles:
          - Triplet format:  s1_id | s2_id | s3_id | match_score
          - Pair format:     s1_id | target_id | match_score
          - Mixed:           s1_id | s2_id | match_score   (no s3_id column)
        """
        target_cols = [
            c for c in df.columns
            if c in ("s2_id", "s3_id", "target_id", "candidate_entity_id", "candidate_id", "matched_id")
        ]

        if not target_cols:
            return pd.DataFrame()

        score_col = None
        for sc in ("match_score", "probability", "score"):
            if sc in df.columns:
                score_col = sc
                break

        rows = []
        for _, row in df.iterrows():
            s1_val = str(row[s1_col]).strip().strip("\"'")
            if not s1_val or s1_val in ("nan", "None"):
                continue
            score = float(row[score_col]) if score_col and pd.notna(row.get(score_col)) else 0.0
            src_val = str(row.get("source", row.get("target_source", ""))).strip() if "source" in row or "target_source" in row else None
            for tc in target_cols:
                if tc not in row or pd.isna(row[tc]):
                    continue
                tid = str(row[tc]).strip().strip("\"'")
                if not tid or tid in ("nan", "None"):
                    continue
                entry = {"s1_id": s1_val, "target_id": tid, "match_score": score}
                if src_val:
                    entry["source"] = src_val
                rows.append(entry)

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _apply_top_margin(self, df: pd.DataFrame, margin: float) -> pd.DataFrame:
        """
        For each S1 entity, drops any candidate whose score is more than
        `margin` below the entity's best score.

        Example: best=0.95, margin=0.10 → keep only scores >= 0.85.
        """
        if df.empty:
            return df

        top_scores = df.groupby("s1_id")["match_score"].transform("max")
        cutoff = top_scores - margin
        kept = df[df["match_score"] >= cutoff].copy()

        dropped = len(df) - len(kept)
        if dropped > 0:
            logger.info(f"EntityResolver top_margin={margin}: pruned {dropped} low-confidence candidates.")
        return kept
