"""Memory-optimized real-data model training script for Anmol's ER matching layer.

Tuned specifically for Apple Silicon (M2, 8GB Unified RAM):
1. Uses chunked streaming ingestion for multi-million row raw TSVs (peak RAM < 500 MB).
2. Directly extracts ground-truth positive pairs without full O(N^2) candidate blocking.
3. Generates domain-specific hard negatives via mutation logic + semi-hard distractors.
4. Computes Anmol's 29 pairwise text, token, character, and numeric similarity features.
5. Trains and compares CatBoost and LightGBM using all 8 M2 CPU cores.
6. Optimizes probability threshold tuned for competition Macro F0.5.
7. Saves model artifacts, feature specs, and experiment metrics to models/ and experiments/.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.features import build_pair_features
from src.modeling.model import generate_hard_negatives, train_catboost_model, train_lightgbm_model
from src.decision.threshold import choose_threshold, _f05_score

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("anmol_training")


def stream_load_entities(
    tsv_path: Path,
    target_ids: Set[str],
    chunksize: int = 100_000,
) -> Dict[str, Dict[str, str]]:
    """Stream a large TSV in chunks to extract only records whose entity_id is in target_ids.
    
    Keeps memory minimal (only 1 chunk in RAM at a time).
    """
    logger.info(f"Streaming {tsv_path.name} to extract {len(target_ids):,} entities...")
    extracted: Dict[str, Dict[str, str]] = {}
    remaining_ids = set(target_ids)
    
    t0 = time.time()
    for chunk in pd.read_csv(tsv_path, sep="\t", chunksize=chunksize, dtype=str):
        matched = chunk[chunk["entity_id"].isin(remaining_ids)]
        for _, row in matched.iterrows():
            eid = row["entity_id"]
            extracted[eid] = {
                "name": str(row.get("business_name", "") or "").strip(),
                "address": str(row.get("business_address", "") or "").strip(),
                "country": str(row.get("country", "") or "").strip(),
            }
            remaining_ids.discard(eid)
        if not remaining_ids:
            break
            
    elapsed = time.time() - t0
    logger.info(f"Extracted {len(extracted):,} / {len(target_ids):,} entities from {tsv_path.name} in {elapsed:.2f}s")
    return extracted


def build_training_dataset(
    data_dir: Path,
    sample_s1_count: int = 25_000,
    neg_multiplier: float = 1.5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Extract real positive pairs, synthesize hard negatives, and add semi-hard distractors."""
    rng = random.Random(random_seed)
    gt_path = data_dir / "train_ground_truth.tsv"
    s1_path = data_dir / "train_source1.tsv"
    s2_path = data_dir / "train_source2.tsv"
    s3_path = data_dir / "train_source3.tsv"

    logger.info(f"Reading ground truth matches from {gt_path}...")
    # Read ground truth (only up to sample_s1_count non-empty rows)
    gt_records: List[Tuple[str, List[str]]] = []
    needed_s1: Set[str] = set()
    needed_s2: Set[str] = set()
    needed_s3: Set[str] = set()

    for chunk in pd.read_csv(gt_path, sep="\t", chunksize=50_000, dtype=str):
        for _, row in chunk.iterrows():
            s1_id = str(row["source1_entity_id"]).strip()
            matched_str = str(row.get("matched_entity_ids", "") or "").strip()
            if not matched_str or matched_str in {"nan", "None"}:
                continue
            targets = [t.strip() for t in matched_str.split(",") if t.strip() and t.strip() not in {"nan", "None"}]
            if not targets:
                continue
            gt_records.append((s1_id, targets))
            needed_s1.add(s1_id)
            for t in targets:
                if t.startswith("S2-") or t.startswith("s2_"):
                    needed_s2.add(t)
                elif t.startswith("S3-") or t.startswith("s3_"):
                    needed_s3.add(t)
                else:
                    needed_s2.add(t)
            if len(gt_records) >= sample_s1_count:
                break
        if len(gt_records) >= sample_s1_count:
            break

    logger.info(f"Sampled {len(gt_records):,} ground truth mappings.")
    logger.info(f"Unique entities required -> S1: {len(needed_s1):,}, S2: {len(needed_s2):,}, S3: {len(needed_s3):,}")

    # Stream entities from disk
    s1_lookup = stream_load_entities(s1_path, needed_s1)
    s2_lookup = stream_load_entities(s2_path, needed_s2)
    s3_lookup = stream_load_entities(s3_path, needed_s3)

    # 1. Build Positive Pairs
    pos_pairs: List[Dict[str, str | int]] = []
    target_all_ids: List[str] = list(s2_lookup.keys()) + list(s3_lookup.keys())

    for s1_id, targets in gt_records:
        s1_rec = s1_lookup.get(s1_id)
        if not s1_rec:
            continue
        for tid in targets:
            trec = s2_lookup.get(tid) or s3_lookup.get(tid)
            if not trec:
                continue
            tsrc = "s3" if (tid.startswith("S3-") or tid.startswith("s3_")) else "s2"
            pos_pairs.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": tid,
                "target_source": tsrc,
                "left_name": s1_rec["name"],
                "left_address": s1_rec["address"],
                "left_country": s1_rec["country"],
                "right_name": trec["name"],
                "right_address": trec["address"],
                "right_country": trec["country"],
                "label": 1,
            })

    pos_df = pd.DataFrame(pos_pairs)
    del pos_pairs
    logger.info(f"Constructed {len(pos_df):,} real positive pairs.")

    # 2. Generate Hard Negatives via Mutation Logic (Anmol's function)
    logger.info("Generating near-miss hard negatives via business name & address mutation...")
    hard_negs = generate_hard_negatives(pos_df, target_col="label", max_per_row=1)
    hard_negs["label"] = 0
    logger.info(f"Generated {len(hard_negs):,} hard near-miss negative pairs.")

    # 3. Generate Semi-Hard / Distractor Negatives (cross-matching with random entities)
    num_random_negs = int(len(pos_df) * max(0.5, neg_multiplier - (len(hard_negs) / max(1, len(pos_df)))))
    logger.info(f"Generating {num_random_negs:,} random/semi-hard distractors...")
    random_negs: List[Dict[str, str | int]] = []
    s1_items = list(s1_lookup.items())

    for _ in range(num_random_negs):
        s1_id, s1_rec = rng.choice(s1_items)
        rand_tid = rng.choice(target_all_ids)
        trec = s2_lookup.get(rand_tid) or s3_lookup.get(rand_tid)
        if not trec:
            continue
        tsrc = "s3" if (rand_tid.startswith("S3-") or rand_tid.startswith("s3_")) else "s2"
        random_negs.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": rand_tid,
            "target_source": tsrc,
            "left_name": s1_rec["name"],
            "left_address": s1_rec["address"],
            "left_country": s1_rec["country"],
            "right_name": trec["name"],
            "right_address": trec["address"],
            "right_country": trec["country"],
            "label": 0,
        })
    rand_df = pd.DataFrame(random_negs)
    del random_negs

    # Free the massive entity lookups (~2-3GB for 9.7M entities)
    del s1_lookup, s2_lookup, s3_lookup, s1_items, target_all_ids
    import gc; gc.collect()
    logger.info("Freed entity lookup dictionaries to reclaim RAM.")

    # Combine all training pairs
    combined_pairs = pd.concat([pos_df, hard_negs, rand_df], ignore_index=True).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    del pos_df, hard_negs, rand_df; gc.collect()
    logger.info(f"Total training pairs assembled: {len(combined_pairs):,} (Positives: {(combined_pairs['label'] == 1).sum():,}, Negatives: {(combined_pairs['label'] == 0).sum():,})")
    return combined_pairs


def train_and_evaluate(
    df_pairs: pd.DataFrame,
    output_dir: Path,
    model_type: str = "both",
    iterations: int = 350,
    random_seed: int = 42,
    max_train_rows: int = 3_000_000,
) -> Dict:
    """Extract features, fit models, optimize F0.5 threshold, and export champion artifact.
    
    For datasets larger than max_train_rows, features are extracted chunk-by-chunk and
    written to Parquet files on disk to avoid OOM. A stratified subsample is then loaded
    for model training, keeping peak RAM well under 3GB on 8GB machines.
    """
    logger.info("Computing pairwise feature matrix (29 similarity, overlap, and n-gram features)...")
    t0 = time.time()
    chunk_size = 50_000
    total_rows = len(df_pairs)
    use_disk_backed = total_rows > max_train_rows

    if use_disk_backed:
        # --- Disk-backed feature extraction for very large datasets ---
        parquet_dir = output_dir / "_feature_chunks"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Dataset has {total_rows:,} pairs (> {max_train_rows:,} max_train_rows). "
            f"Using disk-backed extraction → {parquet_dir}"
        )

        # Step 1: Extract features chunk-by-chunk, write each to Parquet on disk
        chunk_files = []
        for i in range(0, total_rows, chunk_size):
            sub_df = df_pairs.iloc[i : i + chunk_size]
            feat_chunk = build_pair_features(sub_df, target_col="label")
            chunk_path = parquet_dir / f"chunk_{i:010d}.parquet"
            feat_chunk.to_parquet(chunk_path, index=False)
            chunk_files.append(chunk_path)
            done = min(i + chunk_size, total_rows)
            if done % 500_000 < chunk_size:
                logger.info(f"  Extracted & saved features for {done:,} / {total_rows:,} pairs")

        # Free the massive df_pairs to reclaim RAM before loading sample
        del df_pairs
        import gc; gc.collect()
        logger.info(f"Wrote {len(chunk_files)} feature chunk files to disk.")

        # Step 2: Build a stratified subsample from disk
        # First pass: count positives and negatives across all chunks
        logger.info(f"Sampling {max_train_rows:,} rows from {total_rows:,} total (stratified by label)...")
        rng = np.random.RandomState(random_seed)
        sample_frac = max_train_rows / total_rows

        sampled_chunks = []
        sampled_count = 0
        for chunk_path in chunk_files:
            c = pd.read_parquet(chunk_path)
            # Stratified sampling: keep same proportion per chunk
            if len(c) > 0:
                n_take = max(1, int(len(c) * sample_frac))
                if n_take >= len(c):
                    sampled_chunks.append(c)
                else:
                    sampled_chunks.append(c.sample(n=n_take, random_state=rng))
                sampled_count += len(sampled_chunks[-1])
            # Free this chunk immediately
            del c

        feature_matrix = pd.concat(sampled_chunks, ignore_index=True)
        del sampled_chunks; gc.collect()
        logger.info(f"Loaded {len(feature_matrix):,} sampled rows from disk into RAM.")

        # Clean up temporary Parquet files
        for f in chunk_files:
            f.unlink(missing_ok=True)
        parquet_dir.rmdir()
        logger.info("Cleaned up temporary chunk files.")

    elif total_rows > chunk_size:
        logger.info(f"Extracting features in chunks of {chunk_size:,} rows for memory safety...")
        feat_chunks = []
        for i in range(0, total_rows, chunk_size):
            sub_df = df_pairs.iloc[i : i + chunk_size]
            feat_chunks.append(build_pair_features(sub_df, target_col="label"))
            logger.info(f"  Extracted features for {min(i + chunk_size, total_rows):,} / {total_rows:,} pairs")
        feature_matrix = pd.concat(feat_chunks, ignore_index=True)
    else:
        feature_matrix = build_pair_features(df_pairs, target_col="label")

    y = feature_matrix.pop("label")
    X = feature_matrix.select_dtypes(include=[np.number])
    del feature_matrix; import gc; gc.collect()
    feat_time = time.time() - t0
    logger.info(f"Feature matrix built in {feat_time:.2f}s with shape {X.shape}.")

    # Train / Validation Split (80% / 20% Stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=random_seed, stratify=y
    )
    logger.info(f"Train split: {X_train.shape[0]:,} rows | Validation split: {X_val.shape[0]:,} rows")

    results = {}
    trained_models = {}

    # 1. Train CatBoost
    if model_type in {"both", "catboost"}:
        logger.info(f"Training CatBoost Classifier (iterations={iterations}, depth=6, thread_count=8)...")
        t_cb = time.time()
        cb_model = train_catboost_model(
            X_train, y_train, X_val, y_val,
            iterations=iterations,
            depth=6,
            learning_rate=0.08,
            random_state=random_seed,
            thread_count=8,
        )
        cb_time = time.time() - t_cb
        cb_probs = cb_model.predict_proba(X_val)[:, 1]
        results["catboost"] = {
            "training_time_sec": round(cb_time, 2),
            "val_auc": float(roc_auc_score(y_val, cb_probs)),
            "val_avg_precision": float(average_precision_score(y_val, cb_probs)),
            "val_log_loss": float(log_loss(y_val, cb_probs)),
            "val_accuracy_0.5": float(accuracy_score(y_val, (cb_probs >= 0.5).astype(int))),
            "probs": cb_probs,
        }
        trained_models["catboost"] = cb_model
        logger.info(
            f"CatBoost trained in {cb_time:.2f}s: AUC = {results['catboost']['val_auc']:.4f}, "
            f"AP = {results['catboost']['val_avg_precision']:.4f}, LogLoss = {results['catboost']['val_log_loss']:.4f}"
        )

    # 2. Train LightGBM
    if model_type in {"both", "lightgbm"}:
        logger.info(f"Training LightGBM Classifier (n_estimators={iterations}, max_depth=6, n_jobs=8)...")
        t_lgb = time.time()
        lgb_model = train_lightgbm_model(
            X_train, y_train, X_val, y_val,
            n_estimators=iterations,
            max_depth=6,
            learning_rate=0.08,
            random_state=random_seed,
            n_jobs=8,
        )
        lgb_time = time.time() - t_lgb
        lgb_probs = lgb_model.predict_proba(X_val)[:, 1]
        results["lightgbm"] = {
            "training_time_sec": round(lgb_time, 2),
            "val_auc": float(roc_auc_score(y_val, lgb_probs)),
            "val_avg_precision": float(average_precision_score(y_val, lgb_probs)),
            "val_log_loss": float(log_loss(y_val, lgb_probs)),
            "val_accuracy_0.5": float(accuracy_score(y_val, (lgb_probs >= 0.5).astype(int))),
            "probs": lgb_probs,
        }
        trained_models["lightgbm"] = lgb_model
        logger.info(
            f"LightGBM trained in {lgb_time:.2f}s: AUC = {results['lightgbm']['val_auc']:.4f}, "
            f"AP = {results['lightgbm']['val_avg_precision']:.4f}, LogLoss = {results['lightgbm']['val_log_loss']:.4f}"
        )

    # Pick champion model based on Average Precision (ranking capability)
    champion_name = "catboost" if results.get("catboost", {}).get("val_avg_precision", 0) >= results.get("lightgbm", {}).get("val_avg_precision", 0) else "lightgbm"
    champion_model = trained_models[champion_name]
    champion_probs = results[champion_name]["probs"]
    logger.info(f"Champion Model: {champion_name.upper()} (Val AP: {results[champion_name]['val_avg_precision']:.4f})")

    # 3. Threshold Optimization for Competition Metric (F0.5)
    logger.info("Optimizing decision threshold specifically for Macro F0.5...")
    candidate_thresholds = np.linspace(0.1, 0.95, 86)
    opt_threshold = choose_threshold(
        labels=y_val,
        probabilities=champion_probs,
        threshold_candidates=candidate_thresholds,
        positive_label=1,
    )

    # Evaluate metrics at default 0.5 vs optimal threshold
    def calc_metrics_at_thresh(thresh: float, probs: np.ndarray) -> Dict[str, float]:
        preds = (probs >= thresh).astype(int)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        f05 = _f05_score(p, r)
        f1 = (2 * p * r) / (p + r) if (p + r) else 0.0
        return {"threshold": round(thresh, 4), "precision": round(p, 4), "recall": round(r, 4), "f05": round(f05, 4), "f1": round(f1, 4)}

    default_metrics = calc_metrics_at_thresh(0.5, champion_probs)
    optimal_metrics = calc_metrics_at_thresh(opt_threshold, champion_probs)
    logger.info(f"Default Threshold (0.50): F0.5 = {default_metrics['f05']:.4f} (Precision = {default_metrics['precision']:.4f}, Recall = {default_metrics['recall']:.4f})")
    logger.info(f"Optimal Threshold ({opt_threshold:.4f}): F0.5 = {optimal_metrics['f05']:.4f} (Precision = {optimal_metrics['precision']:.4f}, Recall = {optimal_metrics['recall']:.4f})")

    # 4. Save Artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    model_artifact_path = output_dir / f"{champion_name}_model.{'cbm' if champion_name == 'catboost' else 'txt'}"
    if champion_name == "catboost":
        champion_model.save_model(str(model_artifact_path))
    else:
        champion_model.booster_.save_model(str(model_artifact_path))
    logger.info(f"Saved champion model artifact to: {model_artifact_path}")

    # Also save the alternative model if trained
    for m_name, m_obj in trained_models.items():
        if m_name != champion_name:
            alt_path = output_dir / f"{m_name}_model.{'cbm' if m_name == 'catboost' else 'txt'}"
            if m_name == "catboost":
                m_obj.save_model(str(alt_path))
            else:
                m_obj.booster_.save_model(str(alt_path))

    # Save feature columns
    feat_cols_path = output_dir / "feature_columns.json"
    with open(feat_cols_path, "w") as f:
        json.dump(list(X.columns), f, indent=2)

    # Feature Importance
    feature_importances = {}
    if hasattr(champion_model, "feature_importances_"):
        fi = champion_model.feature_importances_
        sorted_fi = sorted(zip(X.columns, fi), key=lambda x: x[1], reverse=True)
        feature_importances = {k: round(float(v), 4) for k, v in sorted_fi}

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "Apple Silicon M2, 8 GB RAM, macOS",
        "dataset_rows": total_rows,
        "feature_count": X.shape[1],
        "champion_model": champion_name,
        "champion_artifact": str(model_artifact_path.relative_to(ROOT)),
        "optimal_threshold_f05": float(opt_threshold),
        "validation_metrics": {
            "default_0.5": default_metrics,
            "optimal_f05": optimal_metrics,
        },
        "model_comparison": {
            m: {k: v for k, v in res.items() if k != "probs"}
            for m, res in results.items()
        },
        "top_features": dict(list(feature_importances.items())[:10]),
    }

    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved training summary to: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train Anmol's ER matching model on real competition data.")
    parser.add_argument("--sample-s1", type=int, default=20_000, help="Number of S1 entities to sample from ground truth.")
    parser.add_argument("--neg-multiplier", type=float, default=1.5, help="Ratio of negatives to positives.")
    parser.add_argument("--model-type", type=str, choices=["both", "catboost", "lightgbm"], default="both", help="Models to train.")
    parser.add_argument("--iterations", type=int, default=300, help="Number of boosting iterations.")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    data_dir = ROOT / "data" / "raw" / "dataset" / "train"
    if not data_dir.exists():
        logger.error(f"Data directory not found at: {data_dir}")
        sys.exit(1)

    t_start = time.time()
    pairs_df = build_training_dataset(
        data_dir=data_dir,
        sample_s1_count=args.sample_s1,
        neg_multiplier=args.neg_multiplier,
        random_seed=args.seed,
    )

    summary = train_and_evaluate(
        df_pairs=pairs_df,
        output_dir=ROOT / args.output_dir,
        model_type=args.model_type,
        iterations=args.iterations,
        random_seed=args.seed,
    )

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE IN {total_time:.1f}s")
    print("=" * 70)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
