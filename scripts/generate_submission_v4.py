"""V4 Submission Generation — Multi-Signal Blocker + 24 Features + Macro F0.5 Decision Layer.

Key V4 improvements over generate_submission_v2.py:
  1. Multi-signal blocker (Name + Address + Address-Numbers) fixes 14.9% blindspot.
  2. 24-feature set (V3 core + 6 new targeted signals).
  3. Macro F0.5 Calibrated Decision Layer:
       - Empty-prediction guard (entities predicted 0 matches → F0.5 = 1.0)
       - Top-N cap (max 6 matches per entity to avoid precision collapse)
       - Relative-gap pruning (reject candidates more than delta below top score)

Memory budget: peak < 3.2 GB on Apple Silicon M2 (8 GB unified RAM).

Usage:
    python3 scripts/generate_submission_v4.py \
        --test-dir  data/raw/dataset/test \
        --model-dir models/v4 \
        --output    output/matching_results_v4.tsv
"""
from __future__ import annotations

import argparse
import array
import gc
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.features_v4 import build_pair_features_v4, get_feature_columns_v4

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("submission_v4")


# ══════════════════════════════════════════════════════════════════════════════
# Text helpers
# ══════════════════════════════════════════════════════════════════════════════

_RE_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_RE_NUMBERS   = re.compile(r"\b(\d{3,})\b")

STOP_TOKENS = frozenset({
    "and", "co", "company", "corp", "corporation", "inc", "incorporated",
    "ltd", "limited", "llc", "llp", "pvt", "private", "services",
    "solutions", "systems", "the", "of", "for", "in", "a",
})
ADDR_STOP = frozenset({
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln",
    "drive", "dr", "blvd", "boulevard", "court", "ct", "way",
    "near", "opp", "floor", "fl", "building", "bldg", "suite",
    "ste", "unit", "no", "number", "main", "sector", "phase",
    "block", "plot", "flat", "and", "the", "of",
})


def _clean(value) -> str:
    if pd.isna(value) or value is None:
        return ""
    text = str(value).strip().lower()
    return " ".join(_RE_NON_ALNUM.sub(" ", text).split())


def _name_tokens(value) -> Set[str]:
    return {t for t in _clean(value).split() if t not in STOP_TOKENS and len(t) >= 2}


def _addr_tokens(value) -> Set[str]:
    return {t for t in _clean(value).split() if t not in ADDR_STOP and len(t) >= 4}


def _addr_numbers(value) -> Set[str]:
    return set(_RE_NUMBERS.findall(str(value)))


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Signal Blocker (same as train_v4.py – copied for standalone use)
# ══════════════════════════════════════════════════════════════════════════════

class MultiSignalBlocker:
    def __init__(
        self,
        name_max_freq: int = 5_000,
        addr_max_freq: int = 3_000,
        num_max_freq: int  = 10_000,
    ):
        self.name_max_freq = name_max_freq
        self.addr_max_freq = addr_max_freq
        self.num_max_freq  = num_max_freq
        self._id_to_int: Dict[str, int] = {}
        self._int_to_id: List[str]      = []
        self._name_idx: Dict[str, array.array] = {}
        self._addr_idx: Dict[str, array.array] = {}
        self._num_idx:  Dict[str, array.array] = {}
        self._country_idx: Dict[str, Set[str]] = defaultdict(set)

    def _register(self, eid: str) -> int:
        if eid not in self._id_to_int:
            idx = len(self._int_to_id)
            self._id_to_int[eid] = idx
            self._int_to_id.append(eid)
            return idx
        return self._id_to_int[eid]

    def build(self, entities: Dict[str, dict]) -> None:
        logger.info(f"Building multi-signal index over {len(entities):,} entities...")
        t0 = time.time()
        name_raw: Dict[str, List[int]] = defaultdict(list)
        addr_raw: Dict[str, List[int]] = defaultdict(list)
        num_raw:  Dict[str, List[int]] = defaultdict(list)

        for eid, rec in entities.items():
            uid = self._register(eid)
            c = _clean(rec.get("country", ""))
            if c:
                self._country_idx[c].add(eid)
            for t in _name_tokens(rec.get("name", "")):
                name_raw[t].append(uid)
            for t in _addr_tokens(rec.get("address", "")):
                addr_raw[t].append(uid)
            for t in _addr_numbers(rec.get("address", "")):
                num_raw[t].append(uid)

        self._name_idx = {t: array.array("I", ids) for t, ids in name_raw.items() if len(ids) <= self.name_max_freq}
        self._addr_idx = {t: array.array("I", ids) for t, ids in addr_raw.items() if len(ids) <= self.addr_max_freq}
        self._num_idx  = {t: array.array("I", ids) for t, ids in num_raw.items()  if len(ids) <= self.num_max_freq}

        logger.info(
            f"Index built in {time.time()-t0:.1f}s — "
            f"name: {len(self._name_idx):,} tokens, "
            f"addr: {len(self._addr_idx):,} tokens, "
            f"num: {len(self._num_idx):,} tokens"
        )

    def retrieve(self, s1_rec: dict, top_k: int = 35) -> List[Tuple[str, float]]:
        s1_name_toks = _name_tokens(s1_rec.get("name", ""))
        s1_addr_toks = _addr_tokens(s1_rec.get("address", ""))
        s1_addr_nums = _addr_numbers(s1_rec.get("address", ""))
        s1_country   = _clean(s1_rec.get("country", ""))

        scores: Dict[int, float] = defaultdict(float)
        for t in s1_name_toks:
            if t in self._name_idx:
                for uid in self._name_idx[t]:
                    scores[uid] += 2.0
        for t in s1_addr_toks:
            if t in self._addr_idx:
                for uid in self._addr_idx[t]:
                    scores[uid] += 1.2
        for t in s1_addr_nums:
            if t in self._num_idx:
                for uid in self._num_idx[t]:
                    scores[uid] += 1.5
        if not scores:
            return []
        if s1_country and s1_country in self._country_idx:
            same_ctry = self._country_idx[s1_country]
            for uid, eid in enumerate(self._int_to_id):
                if uid in scores and eid in same_ctry:
                    scores[uid] += 1.0

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self._int_to_id[uid], score) for uid, score in top]


# ══════════════════════════════════════════════════════════════════════════════
# Macro F0.5 Calibrated Decision Layer
# ══════════════════════════════════════════════════════════════════════════════

def apply_decision_layer(
    candidates_with_probs: List[Tuple[str, float]],
    theta_min: float = 0.76,
    max_k: int = 6,
    delta_p: float = 0.15,
) -> List[str]:
    """
    Convert ranked (entity_id, probability) pairs to a final match list.

    Rules:
    1. Empty-prediction guard: if max(prob) < theta_min → predict [] (no matches).
       Entities with 0 true matches need to predict 0; this yields F0.5 = 1.0.
    2. Rank by probability descending.
    3. Relative-gap filter: only include candidates within delta_p of the top score.
    4. Hard cap at max_k candidates to prevent precision collapse.
    """
    if not candidates_with_probs:
        return []

    # Sort by probability
    sorted_cands = sorted(candidates_with_probs, key=lambda x: x[1], reverse=True)
    max_prob = sorted_cands[0][1]

    # Empty-prediction guard
    if max_prob < theta_min:
        return []

    # Relative-gap pruning + cap
    matched = []
    for eid, prob in sorted_cands:
        if prob < theta_min:
            break
        if prob < max_prob - delta_p:
            break
        matched.append(eid)
        if len(matched) >= max_k:
            break

    return matched


# ══════════════════════════════════════════════════════════════════════════════
# Entity Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_all_entities(tsv_path: Path, prefix: str) -> Dict[str, dict]:
    logger.info(f"Loading {prefix} from {tsv_path.name}...")
    t0 = time.time()
    entities: Dict[str, dict] = {}
    for chunk in pd.read_csv(tsv_path, sep="\t", chunksize=200_000, dtype=str, keep_default_na=False):
        for _, row in chunk.iterrows():
            eid = str(row.get("entity_id", "")).strip()
            if eid:
                entities[eid] = {
                    "name":    str(row.get("business_name", "")).strip(),
                    "address": str(row.get("business_address", "")).strip(),
                    "country": str(row.get("country", "")).strip(),
                }
    logger.info(f"Loaded {len(entities):,} {prefix} entities in {time.time()-t0:.1f}s")
    return entities


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="V4 Submission Generation")
    parser.add_argument("--test-dir",    default="data/raw/dataset/test")
    parser.add_argument("--model-dir",   default="models/v4")
    parser.add_argument("--top-k",       type=int,   default=35)
    parser.add_argument("--theta-min",   type=float, default=0.76,  help="Minimum probability to predict a match")
    parser.add_argument("--delta-p",     type=float, default=0.15,  help="Max prob drop below top candidate")
    parser.add_argument("--max-k",       type=int,   default=6,     help="Max matches per S1 entity")
    parser.add_argument("--chunk-size",  type=int,   default=50_000)
    parser.add_argument("--output",      default="output/matching_results_v4.tsv")
    args = parser.parse_args()

    test_dir    = ROOT / args.test_dir
    model_dir   = ROOT / args.model_dir
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    # ── 1. Load Ensemble Config & Models ─────────────────────────────────
    with open(model_dir / "training_summary_v4.json") as f:
        summary = json.load(f)
    weights   = summary["ensemble_weights"]
    threshold = summary["ensemble_threshold"]
    feature_names = get_feature_columns_v4()
    logger.info(f"Loaded config: weights={weights}, pair_threshold={threshold:.2f}")

    import lightgbm as lgb
    from catboost import CatBoostClassifier
    import xgboost as xgb

    lgb_model = lgb.Booster(model_file=str(model_dir / "lightgbm_v4.txt"))
    cb_model  = CatBoostClassifier()
    cb_model.load_model(str(model_dir / "catboost_v4.cbm"))
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(model_dir / "xgboost_v4.json"))
    logger.info("All three models loaded.")

    # ── 2. Load Target Entities ───────────────────────────────────────────
    target_entities: Dict[str, dict] = {}
    target_entities.update(load_all_entities(test_dir / "test_source2.tsv", "S2"))
    target_entities.update(load_all_entities(test_dir / "test_source3.tsv", "S3"))
    logger.info(f"Total target entities: {len(target_entities):,}")

    # ── 3. Build Multi-Signal Blocking Index ─────────────────────────────
    blocker = MultiSignalBlocker()
    blocker.build(target_entities)
    gc.collect()

    # ── 4. Stream S1 and Predict ─────────────────────────────────────────
    s1_path  = test_dir / "test_source1.tsv"
    total_s1 = sum(len(c) for c in pd.read_csv(
        s1_path, sep="\t", chunksize=200_000, dtype=str, keep_default_na=False
    ))
    logger.info(f"Total S1 entities: {total_s1:,}")

    processed     = 0
    matched_count = 0

    with open(output_path, "w") as out_f:
        out_f.write("source1_entity_id\tmatched_entity_ids\n")

        for chunk in pd.read_csv(
            s1_path, sep="\t", chunksize=args.chunk_size,
            dtype=str, keep_default_na=False
        ):
            chunk_s1:   Dict[str, dict] = {}
            s1_ids_order: List[str] = []
            for _, row in chunk.iterrows():
                eid = str(row.get("entity_id", "")).strip()
                if eid:
                    chunk_s1[eid] = {
                        "name":    str(row.get("business_name", "")).strip(),
                        "address": str(row.get("business_address", "")).strip(),
                        "country": str(row.get("country", "")).strip(),
                    }
                    s1_ids_order.append(eid)

            # Retrieve candidates using multi-signal blocker
            all_pairs:           List[Tuple[str, str]] = []
            pair_blocker_scores: List[float]           = []
            pair_is_source2:     List[int]             = []

            for s1_id in s1_ids_order:
                cands = blocker.retrieve(chunk_s1[s1_id], top_k=args.top_k)
                max_score = cands[0][1] if cands else 1.0
                for t_id, score in cands:
                    all_pairs.append((s1_id, t_id))
                    pair_blocker_scores.append(score / max_score if max_score > 0 else 0.0)
                    pair_is_source2.append(int(t_id.startswith("S2")))

            # Feature extraction + inference
            s1_candidate_probs: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

            if all_pairs:
                rows = []
                for (s1_id, t_id), bscore, is_s2 in zip(all_pairs, pair_blocker_scores, pair_is_source2):
                    s1_r = chunk_s1[s1_id]
                    t_r  = target_entities.get(t_id, {})
                    rows.append({
                        "source_name":    s1_r.get("name", ""),
                        "target_name":    t_r.get("name", ""),
                        "source_address": s1_r.get("address", ""),
                        "target_address": t_r.get("address", ""),
                        "source_country": s1_r.get("country", ""),
                        "target_country": t_r.get("country", ""),
                        "blocker_score":  bscore,
                        "is_source2":     is_s2,
                    })
                feat_df = build_pair_features_v4(pd.DataFrame(rows))
                if not feat_df.empty:
                    X = feat_df[feature_names].values
                    p_lgb = lgb_model.predict(X)
                    p_cb  = cb_model.predict_proba(X)[:, 1]
                    p_xgb = xgb_model.predict_proba(X)[:, 1]
                    probs = (
                        weights["lightgbm"]  * p_lgb +
                        weights["catboost"]  * p_cb  +
                        weights["xgboost"]   * p_xgb
                    )
                    for (s1_id, t_id), prob in zip(all_pairs, probs):
                        s1_candidate_probs[s1_id].append((t_id, float(prob)))

            # Apply calibrated Macro F0.5 decision layer
            for s1_id in s1_ids_order:
                cand_probs = s1_candidate_probs.get(s1_id, [])
                matched = apply_decision_layer(
                    cand_probs,
                    theta_min=args.theta_min,
                    max_k=args.max_k,
                    delta_p=args.delta_p,
                )
                out_f.write(f"{s1_id}\t{','.join(matched)}\n")
                if matched:
                    matched_count += 1

            processed += len(s1_ids_order)
            logger.info(
                "  Processed %d/%d (%.1f%%) — matched: %d",
                processed, total_s1, 100 * processed / total_s1, matched_count,
            )

    elapsed = time.time() - total_start
    logger.info(
        "DONE: Wrote %d rows (%d with matches, %.1f%% coverage) in %.1f min",
        processed, matched_count, 100 * matched_count / max(1, processed), elapsed / 60,
    )
    logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
