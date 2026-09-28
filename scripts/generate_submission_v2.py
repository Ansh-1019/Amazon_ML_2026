"""V2 Submission Generation Script — Uses V3 Top-18 features and ensemble models.

Designed for Apple Silicon M2, 8GB RAM.
1. Loads S2/S3 into lightweight blocker index.
2. Streams S1 in chunks.
3. Retrieves Top-K candidates.
4. Extracts 18 V3 features.
5. Runs LightGBM, CatBoost, XGBoost and blends with optimal weights.
6. Writes to output/matching_results_v2.tsv
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.features_v3 import build_pair_features_v3, get_feature_columns_v3

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("submission_v2")

# ── text normalisation helpers ──────────────────────────────────────────
def _clean(value) -> str:
    if pd.isna(value): return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())

def _tokens(value) -> Set[str]:
    return set(_clean(value).split())

STOP_TOKENS = {
    "and", "co", "company", "corp", "corporation", "inc", "incorporated",
    "ltd", "limited", "llc", "llp", "pvt", "private", "services",
    "solutions", "systems", "the", "of", "for", "in", "a",
}

# ── build inverted index over S2+S3 ────────────────────────────────────
def build_token_index(entities: Dict[str, Dict[str, str]]) -> Dict[str, Set[str]]:
    logger.info("Building token index...")
    idx: Dict[str, Set[str]] = defaultdict(set)
    for eid, rec in entities.items():
        toks = _tokens(rec.get("name", "")) - STOP_TOKENS
        for t in toks:
            if len(t) >= 2:
                idx[t].add(eid)
    pruned = {t: ids for t, ids in idx.items() if len(ids) <= 5000}
    return pruned

def build_country_index(entities: Dict[str, Dict[str, str]]) -> Dict[str, Set[str]]:
    logger.info("Building country index...")
    idx: Dict[str, Set[str]] = defaultdict(set)
    for eid, rec in entities.items():
        c = _clean(rec.get("country", ""))
        if c: idx[c].add(eid)
    return idx

# ── candidate retrieval ─────────────────────────────────────────────────
def retrieve_candidates(
    s1_rec: Dict[str, str],
    token_index: Dict[str, Set[str]],
    country_index: Dict[str, Set[str]],
    top_k: int = 30,
) -> List[str]:
    name_toks = _tokens(s1_rec.get("name", "")) - STOP_TOKENS
    s1_country = _clean(s1_rec.get("country", ""))

    candidate_scores: Dict[str, float] = defaultdict(float)
    for t in name_toks:
        if t in token_index and len(t) >= 2:
            for eid in token_index[t]:
                candidate_scores[eid] += 1.0
    if not candidate_scores: return []

    if s1_country and s1_country in country_index:
        same_country_ids = country_index[s1_country]
        for eid in candidate_scores:
            if eid in same_country_ids:
                candidate_scores[eid] += 0.5

    c = Counter(candidate_scores)
    return [eid for eid, _ in c.most_common(top_k)]

# ── feature extraction ──────────────────────────────────────────────────
def extract_features_for_pairs(
    pairs: List[Tuple[str, str]],
    s1_entities: Dict[str, Dict[str, str]],
    target_entities: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    rows = []
    for s1_id, t_id in pairs:
        s1_rec = s1_entities.get(s1_id, {})
        t_rec = target_entities.get(t_id, {})
        rows.append({
            "source_name": s1_rec.get("name", ""),
            "target_name": t_rec.get("name", ""),
            "source_address": s1_rec.get("address", ""),
            "target_address": t_rec.get("address", ""),
            "source_country": s1_rec.get("country", ""),
            "target_country": t_rec.get("country", ""),
        })

    pair_df = pd.DataFrame(rows)
    if pair_df.empty: return pd.DataFrame()
    return build_pair_features_v3(pair_df)

# ── main ─────────────────────────────────────────────────────────────────
def load_all_entities(tsv_path: Path, prefix: str) -> Dict[str, Dict[str, str]]:
    logger.info("Loading %s from %s ...", prefix, tsv_path.name)
    entities: Dict[str, Dict[str, str]] = {}
    for chunk in pd.read_csv(tsv_path, sep="\t", chunksize=200_000, dtype=str, keep_default_na=False):
        for _, row in chunk.iterrows():
            eid = str(row.get("entity_id", "")).strip()
            if eid:
                entities[eid] = {
                    "name": str(row.get("business_name", "")).strip(),
                    "address": str(row.get("business_address", "")).strip(),
                    "country": str(row.get("country", "")).strip(),
                }
    return entities

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=str, default="data/raw/dataset/test")
    parser.add_argument("--model-dir", type=str, default="models/v3")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--output", type=str, default="output/matching_results.tsv")
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    # ── 1. Load Ensemble Config & Models ────────────────────────────────
    with open(model_dir / "training_summary_v3.json", "r") as f:
        summary = json.load(f)
    weights = summary["ensemble_weights"]
    threshold = summary["ensemble_threshold"]
    feature_names = get_feature_columns_v3()
    logger.info(f"Loaded ensemble config: weights={weights}, threshold={threshold:.2f}")

    import lightgbm as lgb
    from catboost import CatBoostClassifier
    import xgboost as xgb

    lgb_model = lgb.Booster(model_file=str(model_dir / "lightgbm_v3.txt"))
    cb_model = CatBoostClassifier()
    cb_model.load_model(str(model_dir / "catboost_v3.cbm"))
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(model_dir / "xgboost_v3.json"))
    logger.info("Loaded LightGBM, CatBoost, and XGBoost models.")

    # ── 2. Load Target Entities ─────────────────────────────────────────
    target_entities = {}
    target_entities.update(load_all_entities(test_dir / "test_source2.tsv", "S2"))
    target_entities.update(load_all_entities(test_dir / "test_source3.tsv", "S3"))
    logger.info("Total target entities: %d", len(target_entities))

    # ── 3. Build Blocking Index ─────────────────────────────────────────
    token_index = build_token_index(target_entities)
    country_index = build_country_index(target_entities)
    gc.collect()

    # ── 4. Stream S1 ────────────────────────────────────────────────────
    s1_path = test_dir / "test_source1.tsv"
    logger.info("Counting S1 entities...")
    total_s1 = sum(len(c) for c in pd.read_csv(s1_path, sep="\t", chunksize=200_000, dtype=str, keep_default_na=False))
    
    processed = 0
    matched_count = 0

    with open(output_path, "w") as out_f:
        out_f.write("source1_entity_id\tmatched_entity_ids\n")
        
        for chunk in pd.read_csv(s1_path, sep="\t", chunksize=args.chunk_size, dtype=str, keep_default_na=False):
            chunk_s1 = {}
            s1_ids_order = []
            for _, row in chunk.iterrows():
                eid = str(row.get("entity_id", "")).strip()
                if eid:
                    chunk_s1[eid] = {
                        "name": str(row.get("business_name", "")).strip(),
                        "address": str(row.get("business_address", "")).strip(),
                        "country": str(row.get("country", "")).strip(),
                    }
                    s1_ids_order.append(eid)

            all_pairs = []
            for s1_id in s1_ids_order:
                cands = retrieve_candidates(chunk_s1[s1_id], token_index, country_index, top_k=args.top_k)
                for t_id in cands:
                    all_pairs.append((s1_id, t_id))

            s1_matches = defaultdict(list)
            if all_pairs:
                features_df = extract_features_for_pairs(all_pairs, chunk_s1, target_entities)
                if not features_df.empty:
                    X = features_df[feature_names].values
                    
                    # Ensemble Predictions
                    p_lgb = lgb_model.predict(X)
                    p_cb = cb_model.predict_proba(X)[:, 1]
                    # XGB expects a DMatrix or DataFrame, passing 2D numpy array works in latest sklearn API but best to wrap
                    p_xgb = xgb_model.predict_proba(X)[:, 1]
                    
                    probs = (
                        weights["lightgbm"] * p_lgb +
                        weights["catboost"] * p_cb +
                        weights["xgboost"] * p_xgb
                    )
                    
                    for i, (s1_id, t_id) in enumerate(all_pairs):
                        if probs[i] >= threshold:
                            s1_matches[s1_id].append(t_id)

            for s1_id in s1_ids_order:
                matched = s1_matches.get(s1_id, [])
                out_f.write(f"{s1_id}\t{','.join(matched)}\n")
                if matched: matched_count += 1

            processed += len(s1_ids_order)
            logger.info("  Processed %d/%d (%.1f%%) - %d matched", processed, total_s1, 100*processed/total_s1, matched_count)

    elapsed = time.time() - total_start
    logger.info("DONE: Wrote %d rows (%d matched) in %.1f min", processed, matched_count, elapsed/60)

if __name__ == "__main__":
    main()
