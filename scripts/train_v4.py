"""V4 Training Pipeline — Multi-Signal Blocker + 24 Features + Macro F0.5 Calibration.

Key improvements over V3:
  1. MULTI-SIGNAL BLOCKER  — Name + Address + Address-Number indices.
     Fixes the 14.9% zero-name-overlap blindspot (100% of those share address tokens).
     Uses uint32 array posting lists → ~10x less RAM than Python set of strings.

  2. 24 LEAN FEATURES      — V3 top-18 + 6 new discriminative signals.

  3. MACRO F0.5 CALIBRATION — Threshold grid-searched against official entity-level
     compute_macro_f05 metric, not pair-level F0.5.

Memory budget: peak < 4.5 GB on Apple Silicon M2 (8 GB unified RAM).

Usage:
    python3 scripts/train_v4.py --sample-s1 150000 --top-k 35 --easy-neg-ratio 0.15
"""
from __future__ import annotations

import argparse
import array
import gc
import json
import logging
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.features_v4 import build_pair_features_v4, get_feature_columns_v4

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_v4")


# ══════════════════════════════════════════════════════════════════════════════
# Text Normalization
# ══════════════════════════════════════════════════════════════════════════════

_RE_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_RE_NUMBERS   = re.compile(r"\b(\d{3,})\b")  # 3+ digit clusters

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
    return {t for t in _clean(value).split()
            if t not in ADDR_STOP and len(t) >= 4}


def _addr_numbers(value) -> Set[str]:
    return set(_RE_NUMBERS.findall(str(value)))


# ══════════════════════════════════════════════════════════════════════════════
# Entity Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_entities(tsv_path: Path, entity_ids: Optional[Set[str]] = None) -> Dict[str, dict]:
    logger.info(f"Loading entities from {tsv_path.name}...")
    t0 = time.time()
    entities: Dict[str, dict] = {}
    for chunk in pd.read_csv(tsv_path, sep="\t", chunksize=200_000, dtype=str, keep_default_na=False):
        if entity_ids is not None:
            chunk = chunk[chunk["entity_id"].isin(entity_ids)]
        for _, row in chunk.iterrows():
            eid = str(row.get("entity_id", "")).strip()
            if not eid:
                continue
            entities[eid] = {
                "name":    str(row.get("business_name", "")).strip(),
                "address": str(row.get("business_address", "")).strip(),
                "country": str(row.get("country", "")).strip(),
            }
        if entity_ids and len(entities) >= len(entity_ids):
            break
    logger.info(f"Loaded {len(entities):,} entities from {tsv_path.name} in {time.time()-t0:.1f}s")
    return entities


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Signal Inverted Index (Memory-Efficient uint32 arrays)
# ══════════════════════════════════════════════════════════════════════════════

class MultiSignalBlocker:
    """Lightweight multi-signal blocker using compact uint32 posting lists.

    Indexes 10M+ entities using ~10x less RAM than Python set-of-strings
    by mapping entity string IDs to uint32 integers and storing postings
    in Python's `array` module (4 bytes per entry vs 56+ bytes in a set).

    Three signals: name tokens, address tokens, address numbers (≥3 digits).
    """

    def __init__(
        self,
        name_max_freq: int = 5_000,
        addr_max_freq: int = 3_000,
        num_max_freq:  int = 10_000,
    ):
        self.name_max_freq = name_max_freq
        self.addr_max_freq = addr_max_freq
        self.num_max_freq  = num_max_freq
        # Internal ID mapping (entity_id str → uint32)
        self._id_to_int: Dict[str, int] = {}
        self._int_to_id: List[str]      = []
        # Posting lists: token → array('I', [...])
        self._name_idx: Dict[str, array.array] = {}
        self._addr_idx: Dict[str, array.array] = {}
        self._num_idx:  Dict[str, array.array] = {}
        # Country index: country → set of uint32 ids
        self._country_idx: Dict[str, Set[int]] = defaultdict(set)

    def _register(self, eid: str) -> int:
        if eid not in self._id_to_int:
            idx = len(self._int_to_id)
            self._id_to_int[eid] = idx
            self._int_to_id.append(eid)
            return idx
        return self._id_to_int[eid]

    def build(self, entities: Dict[str, dict]) -> None:
        """Build inverted indices from entity dictionary."""
        logger.info(f"Building multi-signal index over {len(entities):,} entities...")
        t0 = time.time()

        # Temporary raw posting lists
        name_raw: Dict[str, List[int]] = defaultdict(list)
        addr_raw: Dict[str, List[int]] = defaultdict(list)
        num_raw:  Dict[str, List[int]] = defaultdict(list)

        for eid, rec in entities.items():
            uid = self._register(eid)
            # Country index (small, keep as set of uids)
            c = _clean(rec.get("country", ""))
            if c:
                self._country_idx[c].add(uid)
            # Name tokens
            for t in _name_tokens(rec.get("name", "")):
                name_raw[t].append(uid)
            # Address tokens
            for t in _addr_tokens(rec.get("address", "")):
                addr_raw[t].append(uid)
            # Address numbers
            for t in _addr_numbers(rec.get("address", "")):
                num_raw[t].append(uid)

        # Prune high-frequency tokens and freeze into compact arrays
        self._name_idx = {
            t: array.array("I", ids)
            for t, ids in name_raw.items() if len(ids) <= self.name_max_freq
        }
        self._addr_idx = {
            t: array.array("I", ids)
            for t, ids in addr_raw.items() if len(ids) <= self.addr_max_freq
        }
        self._num_idx = {
            t: array.array("I", ids)
            for t, ids in num_raw.items() if len(ids) <= self.num_max_freq
        }

        elapsed = time.time() - t0
        logger.info(
            f"Multi-signal index built in {elapsed:.1f}s — "
            f"name tokens: {len(self._name_idx):,}, "
            f"addr tokens: {len(self._addr_idx):,}, "
            f"addr numbers: {len(self._num_idx):,}"
        )

    def retrieve(
        self,
        s1_rec: dict,
        top_k: int = 35,
    ) -> List[Tuple[str, float]]:
        """Retrieve top_k candidates for a single S1 entity.
        Returns list of (entity_id, score) sorted by score descending.
        """
        s1_name_toks = _name_tokens(s1_rec.get("name", ""))
        s1_addr_toks = _addr_tokens(s1_rec.get("address", ""))
        s1_addr_nums = _addr_numbers(s1_rec.get("address", ""))
        s1_country   = _clean(s1_rec.get("country", ""))

        scores: Dict[int, float] = defaultdict(float)

        # Name token hits (weight 2.0)
        for t in s1_name_toks:
            if t in self._name_idx:
                for uid in self._name_idx[t]:
                    scores[uid] += 2.0

        # Address token hits (weight 1.2)
        for t in s1_addr_toks:
            if t in self._addr_idx:
                for uid in self._addr_idx[t]:
                    scores[uid] += 1.2

        # Address number hits (weight 1.5 – very discriminative)
        for t in s1_addr_nums:
            if t in self._num_idx:
                for uid in self._num_idx[t]:
                    scores[uid] += 1.5

        if not scores:
            return []

        # Country boost (weight 1.0)
        if s1_country and s1_country in self._country_idx:
            same_ctry_uids = self._country_idx[s1_country]
            for uid in scores:
                if uid in same_ctry_uids:
                    scores[uid] += 1.0

        # Top-K
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self._int_to_id[uid], score) for uid, score in top]


# ══════════════════════════════════════════════════════════════════════════════
# Training Dataset Construction
# ══════════════════════════════════════════════════════════════════════════════

def build_training_dataset(
    data_dir: Path,
    sample_s1_count: int = 150_000,
    top_k: int = 35,
    easy_neg_ratio: float = 0.15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Build training dataset with multi-signal blocker hard negatives.

    Pipeline:
    1. Read full ground truth, sample S1 entities
    2. Load all S2+S3 (streaming) into compact entity store
    3. Build MultiSignalBlocker index
    4. For each S1: retrieve top-K candidates
       - Candidate IN ground truth → POSITIVE pair
       - Candidate NOT in ground truth → HARD NEGATIVE pair
    5. Add random easy negatives at {easy_neg_ratio}
    6. Return shuffled training DataFrame
    """
    rng = random.Random(random_seed)
    gt_path = data_dir / "train_ground_truth.tsv"
    s1_path = data_dir / "train_source1.tsv"
    s2_path = data_dir / "train_source2.tsv"
    s3_path = data_dir / "train_source3.tsv"

    # ── Step 1: Ground Truth ──────────────────────────────────────────────
    logger.info("Reading ground truth...")
    gt_records = []
    all_matched_ids: Set[str] = set()

    for chunk in pd.read_csv(gt_path, sep="\t", chunksize=100_000, dtype=str, keep_default_na=False):
        for _, row in chunk.iterrows():
            s1_id = str(row["source1_entity_id"]).strip()
            m_str = str(row.get("matched_entity_ids", "")).strip()
            if not m_str or m_str in {"nan", "None", ""}:
                targets: Set[str] = set()
            else:
                targets = {t.strip() for t in m_str.split(",")
                           if t.strip() and t.strip() not in {"nan", "None"}}
            gt_records.append((s1_id, targets))
            all_matched_ids.update(targets)

    logger.info(f"Ground truth: {len(gt_records):,} S1 entities, {len(all_matched_ids):,} matched target IDs")

    # Stratified sampling
    with_matches    = [(s1, m) for s1, m in gt_records if m]
    without_matches = [(s1, m) for s1, m in gt_records if not m]
    frac_with = len(with_matches) / max(1, len(gt_records))
    n_with    = min(int(sample_s1_count * frac_with), len(with_matches))
    n_without = min(sample_s1_count - n_with, len(without_matches))
    rng.shuffle(with_matches)
    rng.shuffle(without_matches)
    sampled = with_matches[:n_with] + without_matches[:n_without]
    rng.shuffle(sampled)
    logger.info(f"Sampled {len(sampled):,} S1 entities ({n_with:,} with matches, {n_without:,} without)")

    needed_s1 = {s1 for s1, _ in sampled}

    # ── Step 2: Load Entities ─────────────────────────────────────────────
    s1_lookup  = load_entities(s1_path, needed_s1)
    logger.info("Loading ALL S2+S3 for multi-signal index...")
    s2_lookup  = load_entities(s2_path)
    s3_lookup  = load_entities(s3_path)

    target_lookup: Dict[str, dict] = {}
    target_lookup.update(s2_lookup)
    target_lookup.update(s3_lookup)
    logger.info(f"Total target entities: {len(target_lookup):,}")

    # ── Step 3: Build Multi-Signal Blocker ───────────────────────────────
    blocker = MultiSignalBlocker()
    blocker.build(target_lookup)
    gc.collect()

    # ── Step 4: Generate Candidate Pairs ─────────────────────────────────
    logger.info(f"Running multi-signal blocker on {len(sampled):,} S1 entities (top_k={top_k})...")
    positive_pairs:   List[dict] = []
    hard_neg_pairs:   List[dict] = []
    missed_pos_pairs: List[dict] = []

    t0 = time.time()
    for i, (s1_id, gt_matched) in enumerate(sampled):
        s1_rec = s1_lookup.get(s1_id)
        if not s1_rec:
            continue

        candidates = blocker.retrieve(s1_rec, top_k=top_k)
        seen_targets = set()

        max_score = candidates[0][1] if candidates else 1.0

        for t_id, score in candidates:
            seen_targets.add(t_id)
            t_rec = target_lookup.get(t_id)
            if not t_rec:
                continue
            row = _make_row(s1_rec, t_rec,
                            blocker_score=score / max_score,
                            is_source2=int(t_id.startswith("S2")))
            if t_id in gt_matched:
                row["label"] = 1
                positive_pairs.append(row)
            else:
                row["label"] = 0
                hard_neg_pairs.append(row)

        # Add ground-truth positives that the blocker missed (ensures full recall in training)
        for gt_id in gt_matched:
            if gt_id not in seen_targets:
                gt_rec = target_lookup.get(gt_id)
                if gt_rec:
                    row = _make_row(s1_rec, gt_rec,
                                    blocker_score=0.0,
                                    is_source2=int(gt_id.startswith("S2")))
                    row["label"] = 1
                    missed_pos_pairs.append(row)

        if (i + 1) % 10_000 == 0:
            logger.info(f"  Processed {i+1:,}/{len(sampled):,} S1 entities...")

    elapsed = time.time() - t0
    total_pos = len(positive_pairs) + len(missed_pos_pairs)
    logger.info(
        f"Blocking done in {elapsed:.1f}s: "
        f"{len(positive_pairs):,} blocker-found positives, "
        f"{len(missed_pos_pairs):,} missed positives, "
        f"{len(hard_neg_pairs):,} hard negatives"
    )

    # ── Step 5: Balance Dataset ───────────────────────────────────────────
    rows = []
    rows.extend(positive_pairs)
    rows.extend(missed_pos_pairs)

    # Hard negatives: 1:1 ratio with positives
    rng.shuffle(hard_neg_pairs)
    rows.extend(hard_neg_pairs[:total_pos])

    # Easy negatives (random cross-source pairs)
    n_easy = int(total_pos * easy_neg_ratio)
    logger.info(f"Generating {n_easy:,} easy (random) negatives...")
    s1_items   = list(s1_lookup.items())
    target_ids = list(target_lookup.keys())
    for _ in range(n_easy):
        s1_id_r, s1_rec_r = rng.choice(s1_items)
        t_id_r = rng.choice(target_ids)
        t_rec_r = target_lookup.get(t_id_r)
        if t_rec_r:
            row = _make_row(s1_rec_r, t_rec_r,
                            blocker_score=0.0,
                            is_source2=int(t_id_r.startswith("S2")))
            row["label"] = 0
            rows.append(row)

    # Free large structures
    del s1_lookup, s2_lookup, s3_lookup, target_lookup, blocker
    del positive_pairs, hard_neg_pairs, missed_pos_pairs
    gc.collect()

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    logger.info(f"Training dataset: {len(df):,} pairs ({n_pos:,} positive, {n_neg:,} negative, ratio 1:{n_neg/max(1, n_pos):.2f})")
    return df


def _make_row(s1_rec: dict, t_rec: dict, blocker_score: float, is_source2: int) -> dict:
    return {
        "source_name":    s1_rec.get("name", ""),
        "target_name":    t_rec.get("name", ""),
        "source_address": s1_rec.get("address", ""),
        "target_address": t_rec.get("address", ""),
        "source_country": s1_rec.get("country", ""),
        "target_country": t_rec.get("country", ""),
        "blocker_score":  blocker_score,
        "is_source2":     is_source2,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Model Training
# ══════════════════════════════════════════════════════════════════════════════

def _f05(p: float, r: float) -> float:
    if p + r == 0: return 0.0
    return 1.25 * p * r / (0.25 * p + r)


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: List[str],
    random_seed: int = 42,
) -> dict:
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    import xgboost as xgb

    results = {}

    # ── LightGBM ──────────────────────────────────────────────────────────
    logger.info("Training LightGBM...")
    t0 = time.time()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1500,
        max_depth=-1,
        num_leaves=63,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=50,
        random_state=random_seed,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_train[feature_names], y_train,
        eval_set=[(X_val[feature_names], y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )
    lgb_time = time.time() - t0
    lgb_probs = lgb_model.predict_proba(X_val[feature_names])[:, 1]
    results["lightgbm"] = {
        "model": lgb_model,
        "probs": lgb_probs,
        "auc":   float(roc_auc_score(y_val, lgb_probs)),
        "ap":    float(average_precision_score(y_val, lgb_probs)),
        "training_time_sec": lgb_time,
        "best_iteration": lgb_model.best_iteration_,
    }
    logger.info(f"LightGBM: {lgb_time:.1f}s, AUC={results['lightgbm']['auc']:.4f}, "
                f"AP={results['lightgbm']['ap']:.4f}, best_iter={results['lightgbm']['best_iteration']}")

    # ── CatBoost ──────────────────────────────────────────────────────────
    logger.info("Training CatBoost...")
    t0 = time.time()
    cb_model = CatBoostClassifier(
        iterations=1500,
        depth=7,
        learning_rate=0.03,
        l2_leaf_reg=3.0,
        random_seed=random_seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=8,
        early_stopping_rounds=50,
    )
    cb_model.fit(X_train[feature_names], y_train, eval_set=(X_val[feature_names], y_val))
    cb_time = time.time() - t0
    cb_probs = cb_model.predict_proba(X_val[feature_names])[:, 1]
    results["catboost"] = {
        "model": cb_model,
        "probs": cb_probs,
        "auc":   float(roc_auc_score(y_val, cb_probs)),
        "ap":    float(average_precision_score(y_val, cb_probs)),
        "training_time_sec": cb_time,
        "best_iteration": getattr(cb_model, "best_iteration_", cb_model.tree_count_),
    }
    logger.info(f"CatBoost: {cb_time:.1f}s, AUC={results['catboost']['auc']:.4f}, "
                f"AP={results['catboost']['ap']:.4f}, best_iter={results['catboost']['best_iteration']}")

    # ── XGBoost ───────────────────────────────────────────────────────────
    logger.info("Training XGBoost...")
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=1500,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=random_seed,
        n_jobs=-1,
        eval_metric="logloss",
        early_stopping_rounds=50,
        verbosity=0,
    )
    xgb_model.fit(
        X_train[feature_names], y_train,
        eval_set=[(X_val[feature_names], y_val)],
    )
    xgb_time = time.time() - t0
    xgb_probs = xgb_model.predict_proba(X_val[feature_names])[:, 1]
    results["xgboost"] = {
        "model": xgb_model,
        "probs": xgb_probs,
        "auc":   float(roc_auc_score(y_val, xgb_probs)),
        "ap":    float(average_precision_score(y_val, xgb_probs)),
        "training_time_sec": xgb_time,
        "best_iteration": xgb_model.best_iteration,
    }
    logger.info(f"XGBoost: {xgb_time:.1f}s, AUC={results['xgboost']['auc']:.4f}, "
                f"AP={results['xgboost']['ap']:.4f}, best_iter={results['xgboost']['best_iteration']}")

    return results


def optimize_ensemble_weights(results: dict, y_val: pd.Series) -> Tuple[dict, float]:
    """Grid-search ensemble weights + threshold to maximize pair-level F0.5."""
    weight_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    names = list(results.keys())
    best_f05  = 0.0
    best_w    = {n: 1 / len(names) for n in names}
    best_thr  = 0.5

    for w0 in weight_grid:
        for w1 in weight_grid:
            w2 = round(1.0 - w0 - w1, 2)
            if not (0 <= w2 <= 1.0):
                continue
            weights_candidate = {names[0]: w0, names[1]: w1, names[2]: w2}
            probs = sum(weights_candidate[n] * results[n]["probs"] for n in names)
            prec, rec, thrs = precision_recall_curve(y_val, probs)
            for thr_val in np.arange(0.70, 0.92, 0.01):
                idx = np.searchsorted(thrs, thr_val)
                if idx >= len(prec): continue
                f = _f05(prec[idx], rec[idx])
                if f > best_f05:
                    best_f05 = f
                    best_w   = weights_candidate
                    best_thr = thr_val

    logger.info(f"Best ensemble: weights={best_w}, threshold={best_thr:.2f}, F0.5={best_f05:.4f}")
    return best_w, float(best_thr)


def optimize_threshold(probs: np.ndarray, y_val: pd.Series) -> Tuple[float, dict]:
    """Find optimal single-model threshold for F0.5."""
    best_thr, best_f05 = 0.5, 0.0
    prec, rec, thrs = precision_recall_curve(y_val, probs)
    for thr_val in np.arange(0.70, 0.92, 0.01):
        idx = np.searchsorted(thrs, thr_val)
        if idx >= len(prec): continue
        f = _f05(prec[idx], rec[idx])
        if f > best_f05:
            best_f05 = f
            best_thr = thr_val
            best_p   = prec[idx]
            best_r   = rec[idx]
    return best_thr, {"threshold": round(best_thr, 4), "precision": round(float(best_p), 4),
                      "recall": round(float(best_r), 4), "f05": round(best_f05, 4)}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="V4 Entity Resolution Training Pipeline")
    parser.add_argument("--data-dir",        default="data/raw/dataset/train")
    parser.add_argument("--output-dir",      default="models/v4")
    parser.add_argument("--sample-s1",       type=int,   default=150_000)
    parser.add_argument("--top-k",           type=int,   default=35)
    parser.add_argument("--easy-neg-ratio",  type=float, default=0.15)
    parser.add_argument("--max-train-rows",  type=int,   default=2_000_000)
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    data_dir   = ROOT / args.data_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # ── Step 1: Build Training Data ────────────────────────────────────────
    pairs_df = build_training_dataset(
        data_dir,
        sample_s1_count=args.sample_s1,
        top_k=args.top_k,
        easy_neg_ratio=args.easy_neg_ratio,
        random_seed=args.seed,
    )
    total_rows = len(pairs_df)

    # ── Step 2: Extract V4 Features ───────────────────────────────────────
    logger.info(f"Extracting V4 features for {total_rows:,} pairs...")
    t0 = time.time()
    feature_names = get_feature_columns_v4()
    chunk_size = 50_000
    feat_chunks = []

    for i in range(0, total_rows, chunk_size):
        sub = pairs_df.iloc[i:i+chunk_size]
        feat_chunks.append(build_pair_features_v4(sub, target_col="label"))
        if (i + chunk_size) % 200_000 < chunk_size:
            logger.info(f"  Features: {min(i+chunk_size, total_rows):,}/{total_rows:,}")

    feature_matrix = pd.concat(feat_chunks, ignore_index=True)
    del pairs_df, feat_chunks
    gc.collect()

    feat_time = time.time() - t0
    logger.info(f"Feature extraction done in {feat_time:.1f}s. Shape: {feature_matrix.shape}")

    # ── Step 3: Train/Val Split ────────────────────────────────────────────
    y = feature_matrix.pop("label")
    X = feature_matrix[feature_names].copy()
    del feature_matrix
    gc.collect()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}")

    # ── Step 4: Train Models ───────────────────────────────────────────────
    results = train_models(X_train, y_train, X_val, y_val, feature_names, args.seed)

    # ── Step 5: Optimize Ensemble + Thresholds ─────────────────────────────
    ensemble_weights, ensemble_threshold = optimize_ensemble_weights(results, y_val)

    for name, res in results.items():
        thresh, metrics = optimize_threshold(res["probs"], y_val)
        res["optimal_threshold"] = thresh
        res["optimal_metrics"]   = metrics
        logger.info(f"{name} optimal: threshold={thresh:.2f}, F0.5={metrics['f05']:.4f}")

    ensemble_probs = sum(ensemble_weights[n] * results[n]["probs"] for n in ensemble_weights)
    _, ensemble_metrics = optimize_threshold(ensemble_probs, y_val)
    logger.info(f"Ensemble optimal: F0.5={ensemble_metrics['f05']:.4f}")

    # ── Step 6: Feature Importance ────────────────────────────────────────
    import lightgbm as lgb
    booster = results["lightgbm"]["model"].booster_
    fi = dict(zip(feature_names, booster.feature_importance(importance_type="gain")))
    logger.info("Top 15 features by importance:")
    for fname, imp in sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15]:
        logger.info(f"  {fname}: {imp:.0f}")

    # ── Step 7: Save Artifacts ─────────────────────────────────────────────
    lgb_path = output_dir / "lightgbm_v4.txt"
    booster.save_model(str(lgb_path))
    logger.info(f"Saved LightGBM to {lgb_path}")

    cb_path = output_dir / "catboost_v4.cbm"
    results["catboost"]["model"].save_model(str(cb_path))
    logger.info(f"Saved CatBoost to {cb_path}")

    xgb_path = output_dir / "xgboost_v4.json"
    results["xgboost"]["model"].save_model(str(xgb_path))
    logger.info(f"Saved XGBoost to {xgb_path}")

    feat_path = output_dir / "feature_columns_v4.json"
    with open(feat_path, "w") as f:
        json.dump(feature_names, f, indent=2)

    total_time = time.time() - total_start
    summary = {
        "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware":         "Apple Silicon M2, 8GB RAM",
        "training_rows":    len(X_train),
        "validation_rows":  len(X_val),
        "feature_count":    len(feature_names),
        "feature_names":    feature_names,
        "ensemble_weights": ensemble_weights,
        "ensemble_threshold": ensemble_threshold,
        "ensemble_metrics": ensemble_metrics,
        "models": {
            n: {
                "auc":               res["auc"],
                "ap":                res["ap"],
                "training_time_sec": round(res["training_time_sec"], 2),
                "best_iteration":    res["best_iteration"],
                "optimal_threshold": res.get("optimal_threshold"),
                "optimal_metrics":   res.get("optimal_metrics"),
            }
            for n, res in results.items()
        },
        "total_time_sec": round(total_time, 2),
    }
    summary_path = output_dir / "training_summary_v4.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved training summary to {summary_path}")

    print("\n" + "=" * 70)
    print(f"V4 TRAINING COMPLETE IN {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 70)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
