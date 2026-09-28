"""Memory-efficient submission generation script for Amazon ML Challenge 2026.

Designed for Apple Silicon M2, 8GB RAM.  Processes 1.7M S1 entities in
streaming chunks, matches against S2+S3 using lightweight token-based
blocking, extracts Anmol's 29 pairwise features, runs LightGBM inference,
and writes the final matching_results.tsv ready for hackathon submission.

Usage:
    python3 scripts/generate_submission.py \
        --test-dir data/raw/dataset/test \
        --model models/lightgbm_model.txt \
        --threshold 0.57 \
        --output output/matching_results.tsv
"""
from __future__ import annotations

import argparse
import gc
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.features import build_pair_features

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("submission")

# ── text normalisation helpers ──────────────────────────────────────────

def _clean(value) -> str:
    if pd.isna(value):
        return ""
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

def build_token_index(
    entities: Dict[str, Dict[str, str]],
) -> Dict[str, Set[str]]:
    """Build token → set-of-entity-ids inverted index from business_name."""
    logger.info("Building token inverted index over %d target entities...", len(entities))
    idx: Dict[str, Set[str]] = defaultdict(set)
    for eid, rec in entities.items():
        toks = _tokens(rec.get("name", "")) - STOP_TOKENS
        for t in toks:
            if len(t) >= 2:
                idx[t].add(eid)
    # Prune overly common tokens (> 5000 entities) to avoid blowup
    pruned = {t: ids for t, ids in idx.items() if len(ids) <= 5000}
    logger.info(
        "Token index: %d unique tokens (%d pruned), %d postings",
        len(idx), len(idx) - len(pruned),
        sum(len(v) for v in pruned.values()),
    )
    return pruned


def build_country_index(
    entities: Dict[str, Dict[str, str]],
) -> Dict[str, Set[str]]:
    """Build country → set-of-entity-ids index."""
    idx: Dict[str, Set[str]] = defaultdict(set)
    for eid, rec in entities.items():
        c = _clean(rec.get("country", ""))
        if c:
            idx[c].add(eid)
    return idx


# ── candidate retrieval ─────────────────────────────────────────────────

def retrieve_candidates(
    s1_rec: Dict[str, str],
    token_index: Dict[str, Set[str]],
    country_index: Dict[str, Set[str]],
    top_k: int = 15,
    all_target_entities: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[str]:
    """Retrieve top_k candidate target entity IDs for a single S1 entity."""
    name_toks = _tokens(s1_rec.get("name", "")) - STOP_TOKENS
    s1_country = _clean(s1_rec.get("country", ""))

    candidate_scores: Dict[str, float] = defaultdict(float)
    for t in name_toks:
        if t in token_index and len(t) >= 2:
            for eid in token_index[t]:
                candidate_scores[eid] += 1.0

    if not candidate_scores:
        return []

    if s1_country and s1_country in country_index:
        same_country_ids = country_index[s1_country]
        for eid in candidate_scores:
            if eid in same_country_ids:
                candidate_scores[eid] += 0.5

    # Fast Top-K without fully sorting a huge dict
    from collections import Counter
    c = Counter(candidate_scores)
    return [eid for eid, _ in c.most_common(top_k)]


# ── feature extraction for a batch of (s1, target) pairs ────────────────

def extract_features_for_pairs(
    pairs: List[Tuple[str, str]],
    s1_entities: Dict[str, Dict[str, str]],
    target_entities: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    """Build Anmol's 29 pairwise features for a list of (s1_id, target_id) pairs."""
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
    if pair_df.empty:
        return pd.DataFrame()

    pair_df.columns = [
        "source_name", "target_name",
        "source_address", "target_address",
        "source_country", "target_country",
    ]

    features = build_pair_features(pair_df)
    return features


# ── main ─────────────────────────────────────────────────────────────────

def load_all_entities(tsv_path: Path, prefix: str) -> Dict[str, Dict[str, str]]:
    """Load all entities from a TSV into a dict keyed by entity_id."""
    logger.info("Loading %s from %s ...", prefix, tsv_path.name)
    t0 = time.time()
    entities: Dict[str, Dict[str, str]] = {}
    for chunk in pd.read_csv(tsv_path, sep="\t", chunksize=200_000, dtype=str, keep_default_na=False):
        for _, row in chunk.iterrows():
            eid = str(row.get("entity_id", "")).strip()
            if not eid:
                continue
            entities[eid] = {
                "name": str(row.get("business_name", "")).strip(),
                "address": str(row.get("business_address", "")).strip(),
                "country": str(row.get("country", "")).strip(),
            }
    logger.info("Loaded %d %s entities in %.1fs", len(entities), prefix, time.time() - t0)
    return entities


def main():
    parser = argparse.ArgumentParser(description="Generate final submission TSV")
    parser.add_argument("--test-dir", type=str, default="data/raw/dataset/test")
    parser.add_argument("--model", type=str, default="models/lightgbm_model.txt")
    parser.add_argument("--threshold", type=float, default=0.57)
    parser.add_argument("--top-k", type=int, default=15, help="Max candidates per S1 entity")
    parser.add_argument("--output", type=str, default="output/matching_results.tsv")
    parser.add_argument("--chunk-size", type=int, default=2500, help="S1 entities per chunk")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # ── Step 1: Load LightGBM model ─────────────────────────────────────
    import lightgbm as lgb
    model = lgb.Booster(model_file=args.model)
    feature_names = model.feature_name()
    logger.info("Loaded LightGBM model with %d features: %s", model.num_feature(), feature_names[:5])

    # ── Step 2: Load target entities (S2 + S3) ──────────────────────────
    s2_entities = load_all_entities(test_dir / "test_source2.tsv", "S2")
    s3_entities = load_all_entities(test_dir / "test_source3.tsv", "S3")

    target_entities: Dict[str, Dict[str, str]] = {}
    target_entities.update(s2_entities)
    target_entities.update(s3_entities)
    logger.info("Total target entities: %d", len(target_entities))
    del s2_entities, s3_entities
    gc.collect()

    # ── Step 3: Build lightweight blocking indices ──────────────────────
    token_index = build_token_index(target_entities)
    country_index = build_country_index(target_entities)
    gc.collect()

    # ── Step 4: Stream S1, block → infer → write directly to disk ───────
    s1_path = test_dir / "test_source1.tsv"

    logger.info("Counting S1 entities...")
    total_s1 = sum(len(c) for c in pd.read_csv(s1_path, sep="\t", chunksize=200_000, dtype=str, keep_default_na=False))
    logger.info("Total S1 entities: %d", total_s1)

    processed = 0
    matched_count = 0

    logger.info("Starting inference: chunk_size=%d, top_k=%d, threshold=%.2f", args.chunk_size, args.top_k, args.threshold)
    
    # Open file for writing streaming results
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
                    for fn in feature_names:
                        if fn not in features_df.columns:
                            features_df[fn] = 0.0
                    probs = model.predict(features_df[feature_names].values)
                    for i, (s1_id, t_id) in enumerate(all_pairs):
                        if probs[i] >= args.threshold:
                            s1_matches[s1_id].append(t_id)

            for s1_id in s1_ids_order:
                matched = s1_matches.get(s1_id, [])
                out_f.write(f"{s1_id}\t{','.join(matched)}\n")
                if matched:
                    matched_count += 1

            processed += len(s1_ids_order)
            if processed % args.chunk_size == 0 or processed == total_s1:
                logger.info(
                    "  Processed %d / %d S1 entities (%.1f%%), %d matched so far",
                    processed, total_s1, 100.0 * processed / total_s1, matched_count,
                )


    elapsed = time.time() - total_start
    logger.info("=== DONE === Wrote %d rows (%d matched) in %.1fs (%.1f min)",
                processed, matched_count, elapsed, elapsed / 60)
    logger.info("Submission file: %s", output_path.resolve())


if __name__ == "__main__":
    main()
