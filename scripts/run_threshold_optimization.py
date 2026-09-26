"""Threshold optimization experiment using Anmol's choose_threshold() on validation predictions."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data.normalizer import DataNormalizer
from src.blocking.candidate_gen import CandidateGenerator
from src.features.features import build_pair_features
from src.modeling.model import (
    generate_hard_negatives,
    train_catboost_model,
    predict_pair_probabilities,
)
from src.decision.threshold import choose_threshold, _f05_score
from src.decision.entity_resolver import EntityResolver
from scripts.run_integrated_training import generate_synthetic_benchmark, create_wide_pairs


def evaluate_threshold_point(labels: np.ndarray, probs: np.ndarray, thresh: float):
    pred = (probs >= thresh).astype(int)
    tp = float(np.sum((pred == 1) & (labels == 1)))
    fp = float(np.sum((pred == 1) & (labels == 0)))
    fn = float(np.sum((pred == 0) & (labels == 1)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f05 = _f05_score(precision, recall)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "threshold": round(thresh, 4),
        "f05": round(f05, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def main():
    print("=" * 75)
    print("ANMOL THRESHOLD OPTIMIZATION ON VALIDATION SET")
    print("=" * 75)

    # 1. Generate multi-source dataset
    s1_df, s2_df, s3_df, ground_truth = generate_synthetic_benchmark(num_entities=120, random_seed=42)

    # 2. Entity-level Train / Validation split (80% / 20%)
    all_s1_ids = s1_df["entity_id"].tolist()
    rng = np.random.RandomState(42)
    shuffled_s1 = rng.permutation(all_s1_ids)
    split_idx = int(len(shuffled_s1) * 0.8)
    train_s1_ids = set(shuffled_s1[:split_idx])
    valid_s1_ids = set(shuffled_s1[split_idx:])

    s1_train = s1_df[s1_df["entity_id"].isin(train_s1_ids)].copy()
    s1_valid = s1_df[s1_df["entity_id"].isin(valid_s1_ids)].copy()

    norm = DataNormalizer()
    s1_train_n = norm.normalize_dataframe(s1_train)
    s1_valid_n = norm.normalize_dataframe(s1_valid)
    s2_n = norm.normalize_dataframe(s2_df)
    s3_n = norm.normalize_dataframe(s3_df)

    # 3. Generate candidates
    cand_gen = CandidateGenerator(top_k=20)
    train_cands = cand_gen.generate_candidates(s1_train_n, s2_n, s3_n)
    valid_cands = cand_gen.generate_candidates(s1_valid_n, s2_n, s3_n)

    # 4. Attach ground truth labels
    def attach_labels(c_df):
        labels = []
        for _, r in c_df.iterrows():
            s1_id = r["s1_id"]
            target_id = r["target_id"]
            gt_matches = ground_truth.get(s1_id, [])
            labels.append(1 if target_id in gt_matches else 0)
        c_df_labeled = c_df.copy()
        c_df_labeled["label"] = labels
        return c_df_labeled

    train_cands_labeled = attach_labels(train_cands)
    valid_cands_labeled = attach_labels(valid_cands)

    train_wide = create_wide_pairs(train_cands_labeled, s1_train_n, s2_n, s3_n)
    valid_wide = create_wide_pairs(valid_cands_labeled, s1_valid_n, s2_n, s3_n)

    # 5. Training with Hard Negatives
    hard_negs = generate_hard_negatives(train_wide[train_wide["label"] == 1], target_col="label", max_per_row=2)
    train_augmented = pd.concat([train_wide, hard_negs], ignore_index=True) if not hard_negs.empty else train_wide

    X_train_df = build_pair_features(train_augmented, target_col="label")
    y_train = X_train_df.pop("label")

    X_valid_df = build_pair_features(valid_wide, target_col="label")
    y_valid = X_valid_df.pop("label")

    meta_cols = {"source1_entity_id", "candidate_entity_id", "target_source", "s1_id", "s2_id", "s3_id", "target_id", "source"}
    feature_cols = [c for c in X_train_df.select_dtypes(include=[np.number]).columns if c not in meta_cols]

    X_train = X_train_df[feature_cols]
    X_valid = X_valid_df[feature_cols]

    catboost_model = train_catboost_model(
        X_train, y_train, X_valid, y_valid,
        iterations=200, learning_rate=0.05, depth=5, random_state=42
    )

    # 6. Validation Predictions (zero test data leakage)
    val_probs = catboost_model.predict_proba(X_valid)[:, 1]
    val_labels = y_valid.to_numpy().astype(int)

    # 7. Execute Anmol's choose_threshold()
    best_threshold = choose_threshold(val_labels, val_probs)
    opt_metrics = evaluate_threshold_point(val_labels, val_probs, best_threshold)

    print(f"\nOptimization Result (Validation F0.5):")
    print(f"  Best Threshold:   {opt_metrics['threshold']:.4f}")
    print(f"  Validation F0.5:  {opt_metrics['f05']:.4f}")
    print(f"  Precision:        {opt_metrics['precision']:.4f}")
    print(f"  Recall:           {opt_metrics['recall']:.4f}")
    print(f"  True Positives:   {opt_metrics['tp']}")
    print(f"  False Positives:  {opt_metrics['fp']}")
    print(f"  False Negatives:  {opt_metrics['fn']}")

    # 8. Evaluate Range Around Optimum (Stability Check)
    print("\n" + "-" * 75)
    print(f"{'Threshold':>10} | {'F0.5':>8} | {'Precision':>10} | {'Recall':>8} | {'F1':>8} | {'TP':>4} | {'FP':>4} | {'FN':>4}")
    print("-" * 75)

    test_thresholds = sorted(list(set(
        [0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        + [round(best_threshold + delta, 2) for delta in [-0.10, -0.05, 0.0, 0.05, 0.10] if 0.0 <= best_threshold + delta <= 1.0]
    )))

    range_results = []
    for t in test_thresholds:
        m = evaluate_threshold_point(val_labels, val_probs, t)
        range_results.append(m)
        is_opt = " *" if np.isclose(t, best_threshold, atol=1e-3) else ""
        print(f"{m['threshold']:>10.2f}{is_opt:<2} | {m['f05']:>8.4f} | {m['precision']:>10.4f} | {m['recall']:>8.4f} | {m['f1']:>8.4f} | {m['tp']:>4} | {m['fp']:>4} | {m['fn']:>4}")
    print("-" * 75)

    # 9. Pass into Ansh EntityResolver
    valid_features_full = valid_wide.copy()
    for col in feature_cols:
        valid_features_full[col] = X_valid[col]

    preds_df = predict_pair_probabilities(catboost_model, valid_features_full, feature_cols=feature_cols)
    # Ensure columns match EntityResolver expected format
    preds_for_resolver = pd.DataFrame({
        "s1_id": preds_df["source1_entity_id"],
        "source1_entity_id": preds_df["source1_entity_id"],
        "target_id": preds_df["candidate_entity_id"],
        "match_score": preds_df["probability"],
        "source": preds_df["target_source"],
    })

    resolver = EntityResolver(threshold=best_threshold)
    config = {"decision": {"threshold": best_threshold}}
    resolved_df = resolver.resolve(preds_for_resolver, config=config)

    print(f"\nDownstream EntityResolver Resolution:")
    print(f"  Initialized EntityResolver with threshold={best_threshold}")
    print(f"  Total Resolved Pair Matches: {len(resolved_df)}")

    # 10. Store in experiment output
    exp_output = {
        "experiment_name": "anmol_threshold_optimization",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_threshold": best_threshold,
        "metrics_at_optimum": opt_metrics,
        "threshold_range_evaluation": range_results,
        "entity_resolver_config": {
            "threshold": best_threshold,
            "threshold_s2": None,
            "threshold_s3": None,
            "top_margin": None,
        }
    }

    exp_dir = ROOT / "experiments" / "threshold_optimization"
    exp_dir.mkdir(parents=True, exist_ok=True)
    out_file = exp_dir / "threshold_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(exp_output, f, indent=2)

    print(f"\nSaved threshold experiment report to: {out_file}")
    print("=" * 75)


if __name__ == "__main__":
    main()
