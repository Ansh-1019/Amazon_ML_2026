"""Small integrated training experiment combining:
Integrated candidates -> Anmol pairwise features -> Ground-truth labels ->
Hard negatives -> CatBoost / LightGBM -> Validation predictions.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score

from src.data.normalizer import DataNormalizer
from src.blocking.candidate_gen import CandidateGenerator
from src.features.features import build_pair_features
from src.features.pairwise import FeatureExtractor
from src.modeling.model import (
    generate_hard_negatives,
    train_catboost_model,
    train_lightgbm_model,
    predict_pair_probabilities,
)


def generate_synthetic_benchmark(num_entities=80, random_seed=42):
    """Generates synthetic multi-source data with ground truth mappings."""
    rng = np.random.RandomState(random_seed)

    company_bases = [
        ("Acme", "Industrial Solutions", "123 Main Street", "New York", "NY 10001", "US"),
        ("Beta", "Logistics Group", "456 Oak Avenue", "Los Angeles", "CA 90001", "US"),
        ("Gamma", "Healthcare Corp", "789 Pine Road", "Chicago", "IL 60601", "US"),
        ("Delta", "Aerospace Systems", "101 Elm Boulevard", "Houston", "TX 77001", "US"),
        ("Epsilon", "Financial Holdings", "202 Maple Drive", "Phoenix", "AZ 85001", "US"),
        ("Zeta", "Energy Partners", "303 Cedar Lane", "Dallas", "TX 75201", "US"),
        ("Eta", "BioTech Research", "404 Birch Way", "San Jose", "CA 95101", "US"),
        ("Theta", "Retail Network", "505 Walnut Street", "Miami", "FL 33101", "US"),
        ("Iota", "Software Labs", "606 Spruce Court", "Seattle", "WA 98101", "US"),
        ("Kappa", "Consulting Global", "707 Chestnut Ave", "Denver", "CO 80201", "US"),
        ("Lambda", "Media Works", "10 Downing Road", "London", "SW1A 2AA", "UK"),
        ("Mu", "Technologies SARL", "15 Rue de Rivoli", "Paris", "75001", "France"),
        ("Nu", "Automotive AG", "42 Berliner Strasse", "Munich", "80331", "Germany"),
        ("Xi", "Maritime Logistics", "88 Harbour View Rd", "Sydney", "NSW 2000", "Australia"),
        ("Omicron", "Robotics Inc", "12 Science Park Drive", "Singapore", "118257", "Singapore"),
        ("Pi", "Security Systems", "50 Queen Street West", "Toronto", "ON M5H 2N2", "Canada"),
    ]

    s1_rows = []
    s2_rows = []
    s3_rows = []
    ground_truth = {}  # s1_id -> list of target_ids

    for i in range(num_entities):
        base = company_bases[i % len(company_bases)]
        prefix, industry, street, city, zip_code, country = base
        idx = f"{i:03d}"

        s1_id = f"s1_{idx}"
        s1_name = f"{prefix} {industry} {i+1}"
        s1_addr = f"{street}, {city}, {zip_code}"
        s1_rows.append({"entity_id": s1_id, "business_name": s1_name, "business_address": s1_addr, "country": country})

        gt_targets = []

        # 85% chance entity has an S2 match
        if rng.rand() < 0.85:
            s2_id = f"s2_{idx}"
            name_var = s1_name.replace("Solutions", "Solns").replace("Corporation", "Corp").replace("Group", "Grp").replace("Holdings", "Hldgs")
            if rng.rand() < 0.3:
                name_var = f"{prefix} {industry[:4]} {i+1} Inc"
            addr_var = s1_addr.replace("Street", "St").replace("Avenue", "Ave").replace("Road", "Rd").replace("Boulevard", "Blvd")
            if rng.rand() < 0.3:
                addr_var = f"{addr_var} Ste 100"
            s2_rows.append({"entity_id": s2_id, "business_name": name_var, "business_address": addr_var, "country": country})
            gt_targets.append(s2_id)

        # 60% chance entity has an S3 match
        if rng.rand() < 0.60:
            s3_id = f"s3_{idx}"
            name_var = f"{prefix} {industry} #{i+1}".replace("Industries", "Ind").replace("International", "Intl")
            addr_var = s1_addr.replace("Drive", "Dr").replace("Lane", "Ln").replace("Court", "Ct")
            s3_rows.append({"entity_id": s3_id, "business_name": name_var, "business_address": addr_var, "country": country})
            gt_targets.append(s3_id)

        # Add distractor entities in S2 and S3 (not matching any S1)
        if rng.rand() < 0.3:
            d_id = f"s2_distractor_{idx}"
            s2_rows.append({"entity_id": d_id, "business_name": f"{prefix} Unrelated Venture {i}", "business_address": f"999 Random Way, {city}", "country": country})

        ground_truth[s1_id] = gt_targets

    return pd.DataFrame(s1_rows), pd.DataFrame(s2_rows), pd.DataFrame(s3_rows), ground_truth


def create_wide_pairs(cand_df: pd.DataFrame, s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> pd.DataFrame:
    """Constructs Anmol wide pair DataFrame from candidate pairs and entity records."""
    s1_lookup = s1_df.set_index("entity_id")
    s2_lookup = s2_df.set_index("entity_id")
    s3_lookup = s3_df.set_index("entity_id")

    rows = []
    for _, row in cand_df.iterrows():
        s1_id = row.get("s1_id") or row.get("source1_entity_id")
        target_id = row.get("target_id") or row.get("candidate_entity_id") or row.get("s2_id") or row.get("s3_id")
        source = row.get("source") or row.get("target_source") or ("s3" if str(target_id).startswith("s3_") or str(target_id).startswith("S3-") else "s2")

        if s1_id not in s1_lookup.index:
            continue
        s1_rec = s1_lookup.loc[s1_id]

        target_rec = None
        if source == "s2" and target_id in s2_lookup.index:
            target_rec = s2_lookup.loc[target_id]
        elif source == "s3" and target_id in s3_lookup.index:
            target_rec = s3_lookup.loc[target_id]
        elif target_id in s2_lookup.index:
            target_rec = s2_lookup.loc[target_id]
            source = "s2"
        elif target_id in s3_lookup.index:
            target_rec = s3_lookup.loc[target_id]
            source = "s3"

        if target_rec is None:
            continue

        pair = {
            "source1_entity_id": s1_id,
            "candidate_entity_id": target_id,
            "target_source": source,
            "left_name": s1_rec.get("business_name", ""),
            "left_address": s1_rec.get("business_address", ""),
            "left_country": s1_rec.get("country", ""),
            "right_name": target_rec.get("business_name", ""),
            "right_address": target_rec.get("business_address", ""),
            "right_country": target_rec.get("country", ""),
            "label": row.get("label", 0),
        }
        rows.append(pair)

    return pd.DataFrame(rows)


def run_experiment():
    print("=" * 75)
    print("INTEGRATED TRAINING EXPERIMENT")
    print("=" * 75)

    # 1. Generate multi-source dataset
    s1_df, s2_df, s3_df, ground_truth = generate_synthetic_benchmark(num_entities=100, random_seed=42)

    # 2. Entity-level Train / Validation split (80% train / 20% validation)
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

    # 4. Attach ground truth labels (strictly within appropriate split)
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

    pos_rows_train = int((train_cands_labeled["label"] == 1).sum())
    neg_rows_train = int((train_cands_labeled["label"] == 0).sum())

    # 5. Form wide pairs
    train_wide = create_wide_pairs(train_cands_labeled, s1_train_n, s2_n, s3_n)
    valid_wide = create_wide_pairs(valid_cands_labeled, s1_valid_n, s2_n, s3_n)

    # 6. Generate hard negatives ONLY on training split
    hard_negs = generate_hard_negatives(train_wide[train_wide["label"] == 1], target_col="label", max_per_row=2)
    num_hard_negs = len(hard_negs)

    train_augmented = pd.concat([train_wide, hard_negs], ignore_index=True) if not hard_negs.empty else train_wide
    total_train_rows = len(train_augmented)

    # 7. Extract Anmol pairwise features
    X_train_df = build_pair_features(train_augmented, target_col="label")
    y_train = X_train_df.pop("label")

    X_valid_df = build_pair_features(valid_wide, target_col="label")
    y_valid = X_valid_df.pop("label")

    # Keep exactly the intersection of numeric feature columns
    meta_cols = {"source1_entity_id", "candidate_entity_id", "target_source", "s1_id", "s2_id", "s3_id", "target_id", "source"}
    feature_cols = [c for c in X_train_df.select_dtypes(include=[np.number]).columns if c not in meta_cols]
    num_features = len(feature_cols)

    X_train = X_train_df[feature_cols]
    X_valid = X_valid_df[feature_cols]

    # 8. Train CatBoost model
    cb_start = time.time()
    catboost_model = train_catboost_model(
        X_train, y_train, X_valid, y_valid,
        iterations=200, learning_rate=0.05, depth=5, random_state=42
    )
    cb_time = time.time() - cb_start

    # 9. Train LightGBM model
    lgb_start = time.time()
    lightgbm_model = train_lightgbm_model(
        X_train, y_train, X_valid, y_valid,
        n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
    )
    lgb_time = time.time() - lgb_start
    total_training_time = cb_time + lgb_time

    # 10. Evaluate validation metrics
    def evaluate(model, X, y):
        prob = model.predict_proba(X)[:, 1]
        pred_bin = (prob >= 0.5).astype(int)
        return {
            "val_auc": float(roc_auc_score(y, prob)),
            "val_avg_precision": float(average_precision_score(y, prob)),
            "val_accuracy": float(accuracy_score(y, pred_bin)),
            "val_log_loss": float(log_loss(y, prob, labels=[0, 1])),
            "predictions": prob,
        }

    cb_metrics = evaluate(catboost_model, X_valid, y_valid)
    lgb_metrics = evaluate(lightgbm_model, X_valid, y_valid)

    # 11. Run standardized prediction adapter on validation set
    valid_features_full = valid_wide.copy()
    for col in feature_cols:
        valid_features_full[col] = X_valid[col]

    preds_output = predict_pair_probabilities(catboost_model, valid_features_full, feature_cols=feature_cols)

    # Print results
    print(f"Training Rows (Total Augmented): {total_train_rows}")
    print(f"Positive Rows (Initial):         {pos_rows_train}")
    print(f"Negative Rows (Initial):         {neg_rows_train}")
    print(f"Hard Negatives (Generated):      {num_hard_negs}")
    print(f"Feature Count:                   {num_features}")
    print(f"Validation Rows:                 {len(valid_wide)}")
    print(f"Validation Positive Rows:        {int((y_valid == 1).sum())}")
    print(f"Validation Negative Rows:        {int((y_valid == 0).sum())}")
    print("-" * 75)
    print("CatBoost Validation Metrics:")
    print(f"  ROC-AUC:           {cb_metrics['val_auc']:.4f}")
    print(f"  Average Precision: {cb_metrics['val_avg_precision']:.4f}")
    print(f"  Accuracy (@0.5):   {cb_metrics['val_accuracy']:.4f}")
    print(f"  Log Loss:          {cb_metrics['val_log_loss']:.4f}")
    print(f"  Training Time:     {cb_time:.3f}s")
    print("-" * 75)
    print("LightGBM Validation Metrics:")
    print(f"  ROC-AUC:           {lgb_metrics['val_auc']:.4f}")
    print(f"  Average Precision: {lgb_metrics['val_avg_precision']:.4f}")
    print(f"  Accuracy (@0.5):   {lgb_metrics['val_accuracy']:.4f}")
    print(f"  Log Loss:          {lgb_metrics['val_log_loss']:.4f}")
    print(f"  Training Time:     {lgb_time:.3f}s")
    print("-" * 75)
    print(f"Total Training Time (both):      {total_training_time:.3f}s")
    print(f"Prediction Output Schema:        {list(preds_output.columns)}")
    print(f"Sample Prediction Rows (Top 5):\n{preds_output.head(5)}")
    print("=" * 75)


if __name__ == "__main__":
    run_experiment()
