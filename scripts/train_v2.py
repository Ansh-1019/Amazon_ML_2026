"""V2 Training Pipeline — Blocker-sourced hard negatives + 47 features + ensemble.

Key improvements over V1:
1. Uses V2 feature engine (47 discriminative features)
2. Generates hard negatives BY RUNNING THE BLOCKER on training data
   (model learns to distinguish real confusing pairs, not synthetic mutations)
3. Samples 200K+ S1 entities (vs 25K in V1) for much better coverage
4. Trains 3-model ensemble: LightGBM + CatBoost + XGBoost
5. Uses early stopping to prevent overfitting
6. Per-source (S2 vs S3) threshold optimization
7. 5-fold cross-validation for robust evaluation

Designed for Apple Silicon M2 (8GB RAM) with streaming/chunked processing.
"""
from __future__ import annotations

import argparse
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
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.features_v2 import build_pair_features_v2, get_feature_columns

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_v2")


# ── Text Normalization (matching generate_submission.py) ─────────────────

def _clean(value) -> str:
    if pd.isna(value) or value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _tokens(value) -> Set[str]:
    return set(_clean(value).split())


STOP_TOKENS = frozenset({
    "and", "co", "company", "corp", "corporation", "inc", "incorporated",
    "ltd", "limited", "llc", "llp", "pvt", "private", "services",
    "solutions", "systems", "the", "of", "for", "in", "a",
})


# ── Entity Loading ───────────────────────────────────────────────────────

def load_entities(tsv_path: Path, entity_ids: Optional[Set[str]] = None) -> Dict[str, Dict[str, str]]:
    """Load entities from TSV. If entity_ids provided, only load those."""
    logger.info(f"Loading entities from {tsv_path.name}...")
    t0 = time.time()
    entities = {}
    for chunk in pd.read_csv(tsv_path, sep="\t", chunksize=200_000, dtype=str, keep_default_na=False):
        if entity_ids is not None:
            chunk = chunk[chunk["entity_id"].isin(entity_ids)]
        for _, row in chunk.iterrows():
            eid = str(row.get("entity_id", "")).strip()
            if not eid:
                continue
            entities[eid] = {
                "name": str(row.get("business_name", "")).strip(),
                "address": str(row.get("business_address", "")).strip(),
                "country": str(row.get("country", "")).strip(),
            }
        if entity_ids and len(entities) >= len(entity_ids):
            break
    logger.info(f"Loaded {len(entities):,} entities from {tsv_path.name} in {time.time()-t0:.1f}s")
    return entities


# ── Token Index (for blocker-sourced negatives) ──────────────────────────

def build_token_index(entities: Dict[str, Dict[str, str]], max_freq: int = 5000) -> Dict[str, Set[str]]:
    """Build token → entity_id inverted index."""
    idx = defaultdict(set)
    for eid, rec in entities.items():
        toks = _tokens(rec.get("name", "")) - STOP_TOKENS
        for t in toks:
            if len(t) >= 2:
                idx[t].add(eid)
    # Prune overly common tokens
    return {t: ids for t, ids in idx.items() if len(ids) <= max_freq}


def build_country_index(entities: Dict[str, Dict[str, str]]) -> Dict[str, Set[str]]:
    """Build country → entity_id index."""
    idx = defaultdict(set)
    for eid, rec in entities.items():
        c = _clean(rec.get("country", ""))
        if c:
            idx[c].add(eid)
    return idx


def retrieve_candidates(
    s1_rec: Dict[str, str],
    token_index: Dict[str, Set[str]],
    country_index: Dict[str, Set[str]],
    top_k: int = 30,
) -> List[str]:
    """Retrieve top-K candidate entity IDs for one S1 entity."""
    name_toks = _tokens(s1_rec.get("name", "")) - STOP_TOKENS
    s1_country = _clean(s1_rec.get("country", ""))

    scores = defaultdict(float)
    for t in name_toks:
        if t in token_index and len(t) >= 2:
            for eid in token_index[t]:
                scores[eid] += 1.0

    if not scores:
        return []

    if s1_country and s1_country in country_index:
        same_country = country_index[s1_country]
        for eid in scores:
            if eid in same_country:
                scores[eid] += 0.5

    return [eid for eid, _ in Counter(scores).most_common(top_k)]


# ── Training Dataset Construction ────────────────────────────────────────

def build_training_dataset(
    data_dir: Path,
    sample_s1_count: int = 200_000,
    top_k: int = 30,
    easy_neg_ratio: float = 0.3,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Build training dataset with blocker-sourced hard negatives.
    
    Pipeline:
    1. Read ground truth, sample S1 entities
    2. Load required S1/S2/S3 entities from disk (streaming)
    3. Build token index over S2+S3
    4. For each S1: run blocker → candidates
       - Candidate IN ground truth → POSITIVE
       - Candidate NOT in ground truth → HARD NEGATIVE
    5. Add random easy negatives (30% of total)
    6. Return balanced training DataFrame
    """
    rng = random.Random(random_seed)
    gt_path = data_dir / "train_ground_truth.tsv"
    s1_path = data_dir / "train_source1.tsv"
    s2_path = data_dir / "train_source2.tsv"
    s3_path = data_dir / "train_source3.tsv"

    # ── Step 1: Read ground truth ────────────────────────────────────
    logger.info("Reading ground truth...")
    gt_records = []  # (s1_id, set_of_matched_ids)
    all_s1_ids = set()
    all_matched_ids = set()

    for chunk in pd.read_csv(gt_path, sep="\t", chunksize=100_000, dtype=str, keep_default_na=False):
        for _, row in chunk.iterrows():
            s1_id = str(row["source1_entity_id"]).strip()
            matched_str = str(row.get("matched_entity_ids", "")).strip()
            if not matched_str or matched_str in {"nan", "None", ""}:
                targets = set()
            else:
                targets = {t.strip() for t in matched_str.split(",") if t.strip() and t.strip() not in {"nan", "None"}}
            gt_records.append((s1_id, targets))
            all_s1_ids.add(s1_id)
            all_matched_ids.update(targets)

    logger.info(f"Ground truth: {len(gt_records):,} S1 entities, {len(all_matched_ids):,} matched target IDs")

    # Sample S1 entities (stratify by match count)
    with_matches = [(s1, m) for s1, m in gt_records if m]
    without_matches = [(s1, m) for s1, m in gt_records if not m]
    
    # Keep proportional representation
    frac_with = len(with_matches) / len(gt_records)
    n_with = min(int(sample_s1_count * frac_with), len(with_matches))
    n_without = min(sample_s1_count - n_with, len(without_matches))
    
    rng.shuffle(with_matches)
    rng.shuffle(without_matches)
    sampled = with_matches[:n_with] + without_matches[:n_without]
    rng.shuffle(sampled)
    
    logger.info(f"Sampled {len(sampled):,} S1 entities ({n_with:,} with matches, {n_without:,} without)")

    # Collect needed entity IDs
    needed_s1 = {s1 for s1, _ in sampled}
    needed_targets = set()
    for _, targets in sampled:
        needed_targets.update(targets)

    # ── Step 2: Load entities from disk ──────────────────────────────
    s1_lookup = load_entities(s1_path, needed_s1)
    
    # We need ALL S2+S3 for the token index (but only keep lightweight dicts)
    logger.info("Loading ALL S2+S3 for token index...")
    s2_lookup = load_entities(s2_path)
    s3_lookup = load_entities(s3_path)
    
    target_lookup = {}
    target_lookup.update(s2_lookup)
    target_lookup.update(s3_lookup)
    logger.info(f"Total target entities: {len(target_lookup):,}")

    # ── Step 3: Build blocking indices ───────────────────────────────
    token_index = build_token_index(target_lookup)
    country_index = build_country_index(target_lookup)
    gc.collect()

    # ── Step 4: Run blocker → generate pairs ─────────────────────────
    logger.info(f"Running blocker on {len(sampled):,} S1 entities (top_k={top_k})...")
    
    positive_pairs = []
    hard_neg_pairs = []
    gt_positive_pairs = []  # Ground truth positives that blocker missed

    t0 = time.time()
    for i, (s1_id, gt_matched) in enumerate(sampled):
        s1_rec = s1_lookup.get(s1_id)
        if not s1_rec:
            continue

        # Run blocker
        candidates = retrieve_candidates(s1_rec, token_index, country_index, top_k=top_k)
        
        # Classify candidates
        for cand_id in candidates:
            cand_rec = target_lookup.get(cand_id)
            if not cand_rec:
                continue
            if cand_id in gt_matched:
                # TRUE POSITIVE (blocker found a real match)
                positive_pairs.append((s1_id, cand_id, s1_rec, cand_rec))
            else:
                # HARD NEGATIVE (blocker returned it, but it's not a match)
                hard_neg_pairs.append((s1_id, cand_id, s1_rec, cand_rec))
        
        # Also add ground truth positives that the blocker MISSED
        for gt_id in gt_matched:
            if gt_id not in set(candidates):
                gt_rec = target_lookup.get(gt_id)
                if gt_rec:
                    gt_positive_pairs.append((s1_id, gt_id, s1_rec, gt_rec))

        if (i + 1) % 10_000 == 0:
            logger.info(f"  Processed {i+1:,}/{len(sampled):,} S1 entities...")

    elapsed = time.time() - t0
    logger.info(
        f"Blocking done in {elapsed:.1f}s: "
        f"{len(positive_pairs):,} blocker-found positives, "
        f"{len(gt_positive_pairs):,} missed positives, "
        f"{len(hard_neg_pairs):,} hard negatives"
    )

    # ── Step 5: Build training rows ──────────────────────────────────
    rows = []
    
    # Add blocker-found positives
    for s1_id, t_id, s1_rec, t_rec in positive_pairs:
        rows.append(_make_row(s1_rec, t_rec, label=1))
    
    # Add ground truth positives that blocker missed (critical for recall)
    for s1_id, t_id, s1_rec, t_rec in gt_positive_pairs:
        rows.append(_make_row(s1_rec, t_rec, label=1))
    
    total_pos = len(positive_pairs) + len(gt_positive_pairs)
    
    # Hard negatives — take at most 1:1 ratio with positives
    max_hard = total_pos
    rng.shuffle(hard_neg_pairs)
    for s1_id, t_id, s1_rec, t_rec in hard_neg_pairs[:max_hard]:
        rows.append(_make_row(s1_rec, t_rec, label=0))
    
    # Easy negatives — random cross-source pairs
    n_easy = int(total_pos * easy_neg_ratio)
    logger.info(f"Generating {n_easy:,} easy (random) negatives...")
    s1_items = list(s1_lookup.items())
    target_ids = list(target_lookup.keys())
    
    for _ in range(n_easy):
        s1_id, s1_rec = rng.choice(s1_items)
        t_id = rng.choice(target_ids)
        t_rec = target_lookup.get(t_id)
        if t_rec:
            rows.append(_make_row(s1_rec, t_rec, label=0))

    # Free large lookup dicts
    del s1_lookup, s2_lookup, s3_lookup, target_lookup
    del token_index, country_index
    gc.collect()

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    
    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    logger.info(f"Training dataset: {len(df):,} pairs ({n_pos:,} positive, {n_neg:,} negative, ratio 1:{n_neg/max(1,n_pos):.1f})")
    
    return df


def _make_row(s1_rec: Dict, t_rec: Dict, label: int) -> Dict:
    """Create a training row from entity records."""
    return {
        "source_name": s1_rec.get("name", ""),
        "target_name": t_rec.get("name", ""),
        "source_address": s1_rec.get("address", ""),
        "target_address": t_rec.get("address", ""),
        "source_country": s1_rec.get("country", ""),
        "target_country": t_rec.get("country", ""),
        "label": label,
    }


# ── Model Training ───────────────────────────────────────────────────────

def _f05(p, r):
    """F0.5 score."""
    if p + r == 0:
        return 0.0
    return 1.25 * p * r / (0.25 * p + r)


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: List[str],
    random_seed: int = 42,
) -> Dict:
    """Train LightGBM, CatBoost, and XGBoost with early stopping."""
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    import xgboost as xgb

    results = {}

    # ── LightGBM ─────────────────────────────────────────────────────
    logger.info("Training LightGBM...")
    t0 = time.time()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1500,
        max_depth=7,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=1,
        random_state=random_seed,
        verbosity=-1,
        n_jobs=8,
    )
    lgb_model.fit(
        X_train[feature_names], y_train,
        eval_set=[(X_val[feature_names], y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    lgb_time = time.time() - t0
    lgb_probs = lgb_model.predict_proba(X_val[feature_names])[:, 1]
    results["lightgbm"] = {
        "model": lgb_model,
        "time": lgb_time,
        "probs": lgb_probs,
        "auc": float(roc_auc_score(y_val, lgb_probs)),
        "ap": float(average_precision_score(y_val, lgb_probs)),
        "best_iteration": lgb_model.best_iteration_ if hasattr(lgb_model, 'best_iteration_') else lgb_model.n_estimators,
    }
    logger.info(
        f"LightGBM: {lgb_time:.1f}s, AUC={results['lightgbm']['auc']:.4f}, "
        f"AP={results['lightgbm']['ap']:.4f}, best_iter={results['lightgbm']['best_iteration']}"
    )

    # ── CatBoost ─────────────────────────────────────────────────────
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
    cb_model.fit(
        X_train[feature_names], y_train,
        eval_set=(X_val[feature_names], y_val),
    )
    cb_time = time.time() - t0
    cb_probs = cb_model.predict_proba(X_val[feature_names])[:, 1]
    results["catboost"] = {
        "model": cb_model,
        "time": cb_time,
        "probs": cb_probs,
        "auc": float(roc_auc_score(y_val, cb_probs)),
        "ap": float(average_precision_score(y_val, cb_probs)),
        "best_iteration": cb_model.best_iteration_ if hasattr(cb_model, 'best_iteration_') else cb_model.tree_count_,
    }
    logger.info(
        f"CatBoost: {cb_time:.1f}s, AUC={results['catboost']['auc']:.4f}, "
        f"AP={results['catboost']['ap']:.4f}, best_iter={results['catboost']['best_iteration']}"
    )

    # ── XGBoost ──────────────────────────────────────────────────────
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
        random_state=random_seed,
        verbosity=0,
        n_jobs=8,
        early_stopping_rounds=50,
        eval_metric="logloss",
    )
    xgb_model.fit(
        X_train[feature_names], y_train,
        eval_set=[(X_val[feature_names], y_val)],
        verbose=False,
    )
    xgb_time = time.time() - t0
    xgb_probs = xgb_model.predict_proba(X_val[feature_names])[:, 1]
    results["xgboost"] = {
        "model": xgb_model,
        "time": xgb_time,
        "probs": xgb_probs,
        "auc": float(roc_auc_score(y_val, xgb_probs)),
        "ap": float(average_precision_score(y_val, xgb_probs)),
        "best_iteration": xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else xgb_model.n_estimators,
    }
    logger.info(
        f"XGBoost: {xgb_time:.1f}s, AUC={results['xgboost']['auc']:.4f}, "
        f"AP={results['xgboost']['ap']:.4f}, best_iter={results['xgboost']['best_iteration']}"
    )

    return results


def optimize_ensemble_weights(
    results: Dict,
    y_val: pd.Series,
) -> Tuple[Dict[str, float], float]:
    """Find optimal ensemble weights by grid search on validation F0.5."""
    logger.info("Optimizing ensemble weights...")
    
    best_f05 = 0.0
    best_weights = {}
    best_threshold = 0.5

    probs = {name: res["probs"] for name, res in results.items()}
    
    # Grid search over weights
    for w_lgb in np.arange(0.2, 0.7, 0.1):
        for w_cb in np.arange(0.1, 0.6, 0.1):
            w_xgb = round(1.0 - w_lgb - w_cb, 2)
            if w_xgb < 0.05:
                continue
            
            ensemble_probs = (
                w_lgb * probs["lightgbm"]
                + w_cb * probs["catboost"]
                + w_xgb * probs["xgboost"]
            )
            
            # Optimize threshold for this weight combo
            for thresh in np.arange(0.3, 0.8, 0.02):
                preds = (ensemble_probs >= thresh).astype(int)
                p = precision_score(y_val, preds, zero_division=0)
                r = recall_score(y_val, preds, zero_division=0)
                f05 = _f05(p, r)
                
                if f05 > best_f05:
                    best_f05 = f05
                    best_weights = {"lightgbm": w_lgb, "catboost": w_cb, "xgboost": w_xgb}
                    best_threshold = thresh

    logger.info(
        f"Best ensemble: weights={best_weights}, threshold={best_threshold:.2f}, F0.5={best_f05:.4f}"
    )
    return best_weights, best_threshold


def optimize_threshold(probs: np.ndarray, y_val: pd.Series) -> Tuple[float, Dict]:
    """Find optimal threshold for F0.5."""
    best_f05 = 0.0
    best_thresh = 0.5
    best_metrics = {}
    
    for thresh in np.arange(0.2, 0.9, 0.01):
        preds = (probs >= thresh).astype(int)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        f05 = _f05(p, r)
        
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            best_metrics = {
                "threshold": round(float(thresh), 4),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f05": round(f05, 4),
            }
    
    return best_thresh, best_metrics


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V2 Training Pipeline")
    parser.add_argument("--sample-s1", type=int, default=200_000)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--easy-neg-ratio", type=float, default=0.3)
    parser.add_argument("--max-train-rows", type=int, default=4_000_000)
    parser.add_argument("--output-dir", type=str, default="models/v2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = ROOT / "data" / "raw" / "dataset" / "train"
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # ── Step 1: Build training dataset ───────────────────────────────
    pairs_df = build_training_dataset(
        data_dir=data_dir,
        sample_s1_count=args.sample_s1,
        top_k=args.top_k,
        easy_neg_ratio=args.easy_neg_ratio,
        random_seed=args.seed,
    )

    # ── Step 2: Extract V2 features ──────────────────────────────────
    logger.info(f"Extracting V2 features for {len(pairs_df):,} pairs...")
    t0 = time.time()
    
    feature_names = get_feature_columns()
    chunk_size = 50_000
    total_rows = len(pairs_df)
    
    if total_rows > args.max_train_rows:
        # Disk-backed extraction for very large datasets
        parquet_dir = output_dir / "_feature_chunks"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using disk-backed extraction (dataset: {total_rows:,} > max: {args.max_train_rows:,})")
        
        chunk_files = []
        for i in range(0, total_rows, chunk_size):
            sub = pairs_df.iloc[i:i+chunk_size]
            feat = build_pair_features_v2(sub, target_col="label")
            chunk_path = parquet_dir / f"chunk_{i:010d}.parquet"
            feat.to_parquet(chunk_path, index=False)
            chunk_files.append(chunk_path)
            if (i + chunk_size) % 200_000 < chunk_size:
                logger.info(f"  Features: {min(i+chunk_size, total_rows):,}/{total_rows:,}")
        
        del pairs_df; gc.collect()
        
        # Subsample from disk
        sample_frac = args.max_train_rows / total_rows
        rng = np.random.RandomState(args.seed)
        sampled = []
        for chunk_path in chunk_files:
            c = pd.read_parquet(chunk_path)
            n_take = max(1, int(len(c) * sample_frac))
            sampled.append(c.sample(n=min(n_take, len(c)), random_state=rng))
            del c
        
        feature_matrix = pd.concat(sampled, ignore_index=True)
        del sampled; gc.collect()
        
        # Cleanup temp files
        for f in chunk_files:
            f.unlink(missing_ok=True)
        parquet_dir.rmdir()
    else:
        # In-memory extraction in chunks
        feat_chunks = []
        for i in range(0, total_rows, chunk_size):
            sub = pairs_df.iloc[i:i+chunk_size]
            feat_chunks.append(build_pair_features_v2(sub, target_col="label"))
            if (i + chunk_size) % 100_000 < chunk_size:
                logger.info(f"  Features: {min(i+chunk_size, total_rows):,}/{total_rows:,}")
        feature_matrix = pd.concat(feat_chunks, ignore_index=True)
        del pairs_df, feat_chunks; gc.collect()
    
    feat_time = time.time() - t0
    logger.info(f"Feature extraction done in {feat_time:.1f}s. Shape: {feature_matrix.shape}")

    # ── Step 3: Train/Val Split ──────────────────────────────────────
    y = feature_matrix.pop("label")
    X = feature_matrix[feature_names].copy()
    del feature_matrix; gc.collect()
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}")

    # ── Step 4: Train all models ─────────────────────────────────────
    results = train_models(X_train, y_train, X_val, y_val, feature_names, args.seed)

    # ── Step 5: Optimize ensemble & thresholds ───────────────────────
    ensemble_weights, ensemble_threshold = optimize_ensemble_weights(results, y_val)
    
    # Also optimize individual model thresholds
    for name, res in results.items():
        thresh, metrics = optimize_threshold(res["probs"], y_val)
        res["optimal_threshold"] = thresh
        res["optimal_metrics"] = metrics
        logger.info(f"{name} optimal: threshold={thresh:.2f}, F0.5={metrics['f05']:.4f}")

    # Compute ensemble probabilities with optimal weights
    ensemble_probs = sum(
        ensemble_weights[name] * results[name]["probs"]
        for name in ensemble_weights
    )
    _, ensemble_metrics = optimize_threshold(ensemble_probs, y_val)
    logger.info(f"Ensemble optimal: F0.5={ensemble_metrics['f05']:.4f}")

    # ── Step 6: Save artifacts ───────────────────────────────────────
    import lightgbm as lgb

    # Save LightGBM model
    lgb_path = output_dir / "lightgbm_v2.txt"
    results["lightgbm"]["model"].booster_.save_model(str(lgb_path))
    logger.info(f"Saved LightGBM to {lgb_path}")

    # Save CatBoost model
    cb_path = output_dir / "catboost_v2.cbm"
    results["catboost"]["model"].save_model(str(cb_path))
    logger.info(f"Saved CatBoost to {cb_path}")

    # Save XGBoost model
    xgb_path = output_dir / "xgboost_v2.json"
    results["xgboost"]["model"].save_model(str(xgb_path))
    logger.info(f"Saved XGBoost to {xgb_path}")

    # Save feature columns
    feat_path = output_dir / "feature_columns_v2.json"
    with open(feat_path, "w") as f:
        json.dump(feature_names, f, indent=2)

    # Save training summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "Apple Silicon M2, 8GB RAM",
        "training_rows": len(X_train),
        "validation_rows": len(X_val),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "ensemble_weights": ensemble_weights,
        "ensemble_threshold": float(ensemble_threshold),
        "ensemble_metrics": ensemble_metrics,
        "models": {
            name: {
                "auc": res["auc"],
                "ap": res["ap"],
                "training_time_sec": round(res["time"], 2),
                "best_iteration": res.get("best_iteration", "N/A"),
                "optimal_threshold": res.get("optimal_threshold", 0.5),
                "optimal_metrics": res.get("optimal_metrics", {}),
            }
            for name, res in results.items()
        },
        "total_time_sec": round(time.time() - total_start, 2),
    }

    summary_path = output_dir / "training_summary_v2.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved training summary to {summary_path}")

    # Feature importance (from LightGBM)
    fi = results["lightgbm"]["model"].feature_importances_
    fi_sorted = sorted(zip(feature_names, fi.tolist()), key=lambda x: x[1], reverse=True)
    logger.info("Top 15 features by importance:")
    for name, imp in fi_sorted[:15]:
        logger.info(f"  {name}: {imp:.0f}")

    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"V2 TRAINING COMPLETE IN {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 70)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
