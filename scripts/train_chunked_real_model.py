"""Train a CatBoost model in small real-data batches for this machine.

This workflow is tuned for a typical 8GB MacBook Air / local workstation where a
full real-dataset training run can exceed memory and time limits. Instead of
training on the whole Source 1 dataset in one shot, we:

1. Load the real train data.
2. Split Source 1 into small deterministic chunks.
3. For each chunk, build candidate pairs and labels from the real ground truth.
4. Train/continue the CatBoost model incrementally on each batch.
5. Keep the best model based on a small validation split.
6. Save the final model artifact.

This gives a practical compromise between accuracy and system constraints.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import DataLoader
from src.data.normalizer import DataNormalizer
from src.blocking.candidate_gen import CandidateGenerator
from src.features.features import build_pair_features
from src.modeling.model import generate_hard_negatives


def _make_gt_lookup(gt_df: pd.DataFrame) -> dict[str, set[str]]:
    lookup: dict[str, set[str]] = {}
    if gt_df is None or gt_df.empty:
        return lookup
    for _, row in gt_df.iterrows():
        s1_id = str(row.get("source1_entity_id", "")).strip()
        if not s1_id or s1_id in {"nan", "None"}:
            continue
        matched = str(row.get("matched_entity_ids", "")).strip()
        targets = {x.strip() for x in matched.split(",") if x.strip() and x.strip() not in {"nan", "None"}}
        lookup.setdefault(s1_id, set()).update(targets)
    return lookup


def build_batch_features(
    s1_chunk: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    block_cfg: dict,
    batch_label: str,
):
    norm = DataNormalizer()
    s1_n = norm.normalize_dataframe(s1_chunk)
    s2_n = norm.normalize_dataframe(s2_df)
    s3_n = norm.normalize_dataframe(s3_df)

    candidate_generator = CandidateGenerator(
        top_k=block_cfg.get("top_k", 10),
        blocking_fields=block_cfg.get("blocking_fields", ["business_name", "business_address", "country"]),
    )
    candidates = candidate_generator.generate_candidates(s1_n, s2_n, s3_n, block_cfg)

    gt_lookup = _make_gt_lookup(gt_df)
    rows: list[dict] = []
    seen = set()

    for _, row in candidates.iterrows():
        s1_id = str(row.get("s1_id") or row.get("source1_entity_id") or row.get("entity_id", "")).strip()
        target_id = str(row.get("target_id") or row.get("candidate_entity_id") or row.get("s2_id") or row.get("s3_id") or "").strip()
        if not s1_id or not target_id:
            continue
        gt_targets = gt_lookup.get(s1_id, set())
        label = 1 if target_id in gt_targets else 0
        rows.append(
            {
                "left_name": row.get("left_name", row.get("business_name", "")),
                "left_address": row.get("left_address", row.get("business_address", "")),
                "left_country": row.get("left_country", row.get("country", "")),
                "right_name": row.get("right_name", row.get("business_name", "")),
                "right_address": row.get("right_address", row.get("business_address", "")),
                "right_country": row.get("right_country", row.get("country", "")),
                "label": int(label),
            }
        )
        seen.add((s1_id, target_id))

    # Include any missed positive matches that were filtered out by blocking.
    for s1_id, gt_targets in gt_lookup.items():
        if s1_id not in set(s1_chunk["entity_id"].astype(str)):
            continue
        for target_id in sorted(gt_targets):
            if (s1_id, target_id) in seen:
                continue
            rows.append(
                {
                    "left_name": "",
                    "left_address": "",
                    "left_country": "",
                    "right_name": "",
                    "right_address": "",
                    "right_country": "",
                    "label": 1,
                }
            )
            seen.add((s1_id, target_id))

    if not rows:
        raise ValueError(f"No training rows created for chunk {batch_label}")

    batch_df = pd.DataFrame(rows)
    hard_neg = generate_hard_negatives(batch_df, target_col="label")
    final_df = pd.concat([batch_df, hard_neg], ignore_index=True) if not hard_neg.empty else batch_df
    X = build_pair_features(final_df, target_col="label")
    y = X.pop("label")
    return X, y


def train_chunked_model(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    batch_size: int,
    max_batches: int,
    random_seed: int,
    verbose: bool = False,
):
    rng = random.Random(random_seed)
    s1_ids = s1_df["entity_id"].astype(str).tolist()
    rng.shuffle(s1_ids)

    model: Optional[CatBoostClassifier] = None
    best_model: Optional[CatBoostClassifier] = None
    best_auc = -1.0
    batch_count = 0

    for i in range(0, len(s1_ids), batch_size):
        if max_batches and batch_count >= max_batches:
            break
        batch_ids = s1_ids[i : i + batch_size]
        batch_s1 = s1_df[s1_df["entity_id"].astype(str).isin(batch_ids)].copy()
        if batch_s1.empty:
            continue

        batch_gt = gt_df[gt_df["source1_entity_id"].astype(str).isin(batch_ids)].copy()
        X, y = build_batch_features(
            batch_s1,
            s2_df,
            s3_df,
            batch_gt,
            block_cfg={
                "top_k": 10,
                "blocking_fields": ["business_name", "business_address", "country"],
            },
            batch_label=f"batch_{batch_count + 1}",
        )

        if y.nunique() < 2:
            if verbose:
                print(f"Skipping batch {batch_count + 1}: insufficient class balance.")
            continue

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=random_seed + batch_count,
            stratify=y,
        )

        init_model = model
        clf = CatBoostClassifier(
            iterations=150,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            random_seed=random_seed,
            verbose=False,
            allow_writing_files=False,
        )

        if init_model is not None:
            clf.fit(X_train, y_train, init_model=init_model)
        else:
            clf.fit(X_train, y_train)

        val_proba = clf.predict_proba(X_val)[:, 1]
        batch_auc = roc_auc_score(y_val, val_proba)
        if batch_auc > best_auc:
            best_auc = batch_auc
            best_model = clf
        model = clf
        batch_count += 1

        if verbose:
            print(f"Batch {batch_count}: AUC={batch_auc:.4f}, rows={len(X)}")

    if best_model is None:
        raise RuntimeError("No valid model was trained from the chunked batches.")

    return best_model, best_auc, batch_count


def main():
    parser = argparse.ArgumentParser(description="Train a CatBoost model in small real-data batches.")
    parser.add_argument("--batch-size", type=int, default=50000, help="Number of Source 1 rows per batch.")
    parser.add_argument("--max-batches", type=int, default=3, help="Maximum number of training batches to process.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic batch order.")
    parser.add_argument("--output", type=str, default="models/chunked_catboost.cbm", help="Path for the trained model artifact.")
    args = parser.parse_args()

    loader = DataLoader(raw_data_dir="data/raw/dataset/train")
    s1_df, s2_df, s3_df, gt_df = loader.load_sources({
        "data": {
            "source1_filename": "train_source1.tsv",
            "source2_filename": "train_source2.tsv",
            "source3_filename": "train_source3.tsv",
            "train_matches_filename": "train_ground_truth.tsv",
        }
    })

    if s1_df.empty:
        raise RuntimeError("Source 1 is empty. Check the raw data path.")
    if gt_df is None or gt_df.empty:
        raise RuntimeError("Ground truth matches are empty. Check the training labels file.")

    model, best_auc, batches_used = train_chunked_model(
        s1_df=s1_df,
        s2_df=s2_df,
        s3_df=s3_df,
        gt_df=gt_df,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        random_seed=args.seed,
        verbose=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))

    summary = {
        "output_model": str(output_path),
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "batches_used": batches_used,
        "best_auc": float(best_auc),
        "source1_rows": int(len(s1_df)),
        "ground_truth_rows": int(len(gt_df)),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
