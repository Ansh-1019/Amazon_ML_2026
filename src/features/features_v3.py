"""V3 Pairwise Feature Extraction — Top 18 most impactful features.

Pruned from V2's 47 features down to the 18 most critical features
as identified by LightGBM feature importance (F0.5 = 0.993).
This drastically speeds up inference for final submission.
"""
from __future__ import annotations

import re
from typing import List, Optional

import numpy as np
import pandas as pd
import Levenshtein
import jellyfish

# ── Constants ────────────────────────────────────────────────────────────
_COMMON_BUSINESS_TOKENS = frozenset({
    "and", "co", "company", "corp", "corporation", "inc", "incorporated",
    "ltd", "limited", "llc", "llp", "pvt", "private", "services",
    "solutions", "systems", "the", "of", "for", "in", "a", "an",
    "group", "holdings", "enterprises", "international", "global",
    "associates", "partners", "consulting", "technologies", "tech",
})

# ── Text Normalization ───────────────────────────────────────────────────
def _clean(value) -> str:
    if pd.isna(value) or value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())

def _tokenize(value) -> List[str]:
    return _clean(value).split()

def _significant_tokens(value) -> List[str]:
    return [t for t in _tokenize(value) if t not in _COMMON_BUSINESS_TOKENS and len(t) > 1]

# ── Similarity Functions ───────────────────────────────────────────────
def _levenshtein_ratio(a: str, b: str) -> float:
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return Levenshtein.ratio(a, b)

def _jaro_winkler(a: str, b: str) -> float:
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return jellyfish.jaro_winkler_similarity(a, b)

def _jaccard(tokens_a: List[str], tokens_b: List[str]) -> float:
    set_a, set_b = set(tokens_a), set(tokens_b)
    if not set_a and not set_b: return 1.0
    if not set_a or not set_b: return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def _sorted_token_similarity(a: str, b: str) -> float:
    sorted_a = " ".join(sorted(_tokenize(a)))
    sorted_b = " ".join(sorted(_tokenize(b)))
    return _levenshtein_ratio(sorted_a, sorted_b)

def _partial_ratio(a: str, b: str) -> float:
    a_clean, b_clean = _clean(a), _clean(b)
    if not a_clean and not b_clean: return 1.0
    if not a_clean or not b_clean: return 0.0
    if len(a_clean) > len(b_clean):
        a_clean, b_clean = b_clean, a_clean
    if a_clean in b_clean: return 1.0
    
    best = 0.0
    needle_len = len(a_clean)
    for i in range(len(b_clean) - needle_len + 1):
        ratio = Levenshtein.ratio(a_clean, b_clean[i:i + needle_len])
        if ratio > best:
            best = ratio
            if best == 1.0: break
    return best

def _char_ngrams(text: str, n: int = 3) -> set:
    t = re.sub(r"\s+", "", _clean(text))
    if len(t) < n:
        return {t} if t else set()
    return {t[i:i+n] for i in range(len(t) - n + 1)}

def _char_ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    grams_a, grams_b = _char_ngrams(a, n), _char_ngrams(b, n)
    if not grams_a and not grams_b: return 1.0
    if not grams_a or not grams_b: return 0.0
    return len(grams_a & grams_b) / len(grams_a | grams_b)

def _length_ratio(a: str, b: str) -> float:
    la, lb = len(_clean(a)), len(_clean(b))
    if la == 0 and lb == 0: return 1.0
    if la == 0 or lb == 0: return 0.0
    return min(la, lb) / max(la, lb)

def _rare_token_overlap(a: str, b: str) -> float:
    toks_a, toks_b = _significant_tokens(a), _significant_tokens(b)
    if not toks_a and not toks_b:
        toks_a, toks_b = _tokenize(a), _tokenize(b)
    return _jaccard(toks_a, toks_b)

# ── Master Feature Builder ───────────────────────────────────────────────
def build_pair_features_v3(
    candidate_pairs: pd.DataFrame,
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    """Build the top 18 features for fast entity matching."""
    if candidate_pairs is None or candidate_pairs.empty:
        raise ValueError("candidate_pairs must be a non-empty DataFrame.")

    data = candidate_pairs.copy()
    n = len(data)
    
    name_left = name_right = addr_left = addr_right = ctry_left = ctry_right = None
    col_lower = {c.lower(): c for c in data.columns}
    for prefix_l, prefix_r in [
        ("source_", "target_"), ("left_", "right_"), ("entity1_", "entity2_"),
    ]:
        if f"{prefix_l}name" in col_lower:
            name_left = col_lower[f"{prefix_l}name"]
            name_right = col_lower.get(f"{prefix_r}name", name_right)
            addr_left = col_lower.get(f"{prefix_l}address", addr_left)
            addr_right = col_lower.get(f"{prefix_r}address", addr_right)
            ctry_left = col_lower.get(f"{prefix_l}country", ctry_left)
            ctry_right = col_lower.get(f"{prefix_r}country", ctry_right)
            break
            
    if name_left is None:
        for col in data.columns:
            cl = col.lower()
            if "name" in cl and "left" in cl: name_left = col
            elif "name" in cl and "right" in cl: name_right = col
            elif "address" in cl and "left" in cl: addr_left = col
            elif "address" in cl and "right" in cl: addr_right = col
            elif "country" in cl and "left" in cl: ctry_left = col
            elif "country" in cl and "right" in cl: ctry_right = col

    def _series_clean(series):
        return series.fillna("").astype(str).str.strip().str.lower().str.replace(
            r"[^a-z0-9\s]", " ", regex=True
        ).str.replace(r"\s+", " ", regex=True).str.strip()
    
    name_a = _series_clean(data[name_left]) if name_left else pd.Series([""] * n)
    name_b = _series_clean(data[name_right]) if name_right else pd.Series([""] * n)
    addr_a = _series_clean(data[addr_left]) if addr_left else pd.Series([""] * n)
    addr_b = _series_clean(data[addr_right]) if addr_right else pd.Series([""] * n)
    ctry_a = _series_clean(data[ctry_left]) if ctry_left else pd.Series([""] * n)
    ctry_b = _series_clean(data[ctry_right]) if ctry_right else pd.Series([""] * n)

    features = {}

    # 1. addr_char3_jaccard
    features["addr_char3_jaccard"] = np.array([
        _char_ngram_jaccard(a, b, 3) for a, b in zip(addr_a, addr_b)
    ], dtype=np.float32)

    # 2. addr_length_ratio
    features["addr_length_ratio"] = np.array([
        _length_ratio(a, b) for a, b in zip(addr_a, addr_b)
    ], dtype=np.float32)

    # 3. name_partial_ratio
    features["name_partial_ratio"] = np.array([
        _partial_ratio(a, b) for a, b in zip(name_a, name_b)
    ], dtype=np.float32)

    # 4. name_length_ratio
    features["name_length_ratio"] = np.array([
        _length_ratio(a, b) for a, b in zip(name_a, name_b)
    ], dtype=np.float32)

    # 5. all_fields_levenshtein
    features["all_fields_levenshtein"] = np.array([
        _levenshtein_ratio(f"{na} {aa} {ca}", f"{nb} {ab} {cb}")
        for na, nb, aa, ab, ca, cb in zip(name_a, name_b, addr_a, addr_b, ctry_a, ctry_b)
    ], dtype=np.float32)

    # 6. name_addr_combined_jaccard
    features["name_addr_combined_jaccard"] = np.array([
        _jaccard(_tokenize(f"{na} {aa}"), _tokenize(f"{nb} {ab}"))
        for na, nb, aa, ab in zip(name_a, name_b, addr_a, addr_b)
    ], dtype=np.float32)

    # 7. name_sorted_token_sim
    features["name_sorted_token_sim"] = np.array([
        _sorted_token_similarity(a, b) for a, b in zip(name_a, name_b)
    ], dtype=np.float32)

    # 8. name_char3_jaccard
    features["name_char3_jaccard"] = np.array([
        _char_ngram_jaccard(a, b, 3) for a, b in zip(name_a, name_b)
    ], dtype=np.float32)

    # 9. name_rare_token_overlap
    features["name_rare_token_overlap"] = np.array([
        _rare_token_overlap(a, b) for a, b in zip(name_a, name_b)
    ], dtype=np.float32)

    # 10. name_char4_jaccard
    features["name_char4_jaccard"] = np.array([
        _char_ngram_jaccard(a, b, 4) for a, b in zip(name_a, name_b)
    ], dtype=np.float32)

    # 11. addr_jaro_winkler
    features["addr_jaro_winkler"] = np.array([
        _jaro_winkler(a, b) for a, b in zip(addr_a, addr_b)
    ], dtype=np.float32)

    # 12. name_levenshtein
    features["name_levenshtein"] = np.array([
        _levenshtein_ratio(a, b) for a, b in zip(name_a, name_b)
    ], dtype=np.float32)

    # 13. pair_quality_score
    name_jw = np.array([_jaro_winkler(a, b) for a, b in zip(name_a, name_b)], dtype=np.float32)
    name_tok = np.array([_jaccard(_tokenize(a), _tokenize(b)) for a, b in zip(name_a, name_b)], dtype=np.float32)
    addr_lev = np.array([_levenshtein_ratio(a, b) for a, b in zip(addr_a, addr_b)], dtype=np.float32)
    ctry_exact = (ctry_a == ctry_b).astype(np.float32).values
    
    features["pair_quality_score"] = (
        features["name_levenshtein"] * 0.3 +
        name_jw * 0.2 +
        name_tok * 0.15 +
        addr_lev * 0.15 +
        ctry_exact * 0.2
    )

    # 14. name_max_sim
    features["name_max_sim"] = np.maximum.reduce([
        features["name_levenshtein"], name_jw, features["name_sorted_token_sim"], name_tok
    ])

    # 15. addr_levenshtein
    features["addr_levenshtein"] = addr_lev

    # 16. name_jaro_winkler (used in quality score, highly important)
    features["name_jaro_winkler"] = name_jw

    # 17. name_token_jaccard (used in quality score)
    features["name_token_jaccard"] = name_tok

    # 18. country_exact (often helps distinct same names in diff countries)
    features["country_exact"] = ctry_exact

    # ── Build final DataFrame ────────────────────────────────────────
    feature_df = pd.DataFrame(features, index=data.index)
    if target_col is not None and target_col in data.columns:
        feature_df[target_col] = data[target_col].astype(int)
    feature_df = feature_df.fillna(0.0)
    return feature_df

def get_feature_columns_v3() -> List[str]:
    return [
        "addr_char3_jaccard", "addr_length_ratio", "name_partial_ratio",
        "name_length_ratio", "all_fields_levenshtein", "name_addr_combined_jaccard",
        "name_sorted_token_sim", "name_char3_jaccard", "name_rare_token_overlap",
        "name_char4_jaccard", "addr_jaro_winkler", "name_levenshtein",
        "pair_quality_score", "name_max_sim", "addr_levenshtein",
        "name_jaro_winkler", "name_token_jaccard", "country_exact"
    ]
