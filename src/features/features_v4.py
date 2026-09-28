"""V4 Pairwise Feature Extraction — 24 High-Impact Features.

Extends V3's top-18 proven features with 6 new targeted signals designed
to address specific failure modes discovered in data analysis:

 New features (19-24):
  19. addr_num_jaccard          – Jaccard on numeric sequences in address
  20. name_acronym_match        – acronym of one name matches other (IBM↔International Business Machines)
  21. script_mismatch           – one name is non-ASCII, other is ASCII (multilingual transliteration)
  22. blocker_score             – normalized candidate retrieval score from the V4 blocker
  23. is_source2                – target is from Source2 vs Source3 (schema difference signal)
  24. addr_first_number_match   – binary: first digit cluster in address matches

AUC target: 0.9995+   Inference: <14s per 100k pairs on Apple M2.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Optional

import numpy as np
import pandas as pd
import Levenshtein
import jellyfish

# ── Constants ─────────────────────────────────────────────────────────────────
_COMMON_BUSINESS_TOKENS = frozenset({
    "and", "co", "company", "corp", "corporation", "inc", "incorporated",
    "ltd", "limited", "llc", "llp", "pvt", "private", "services",
    "solutions", "systems", "the", "of", "for", "in", "a", "an",
    "group", "holdings", "enterprises", "international", "global",
    "associates", "partners", "consulting", "technologies", "tech",
})

_ADDR_STOP_TOKENS = frozenset({
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln",
    "drive", "dr", "blvd", "boulevard", "court", "ct", "way", "wy",
    "near", "opp", "opposite", "floor", "fl", "building", "bldg",
    "suite", "ste", "unit", "no", "number", "#", "and", "the", "of",
    "main", "sector", "phase", "block", "plot", "flat",
})

_RE_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_RE_SPACES    = re.compile(r"\s+")
_RE_NUMBERS   = re.compile(r"\b(\d{3,})\b")   # 3+ digit clusters in addresses
_RE_FIRST_NUM = re.compile(r"\b(\d+)\b")


# ── Text Normalization ─────────────────────────────────────────────────────────
def _clean(value) -> str:
    if pd.isna(value) or value is None:
        return ""
    text = str(value).strip().lower()
    text = _RE_NON_ALNUM.sub(" ", text)
    return _RE_SPACES.sub(" ", text).strip()


def _tokenize(value) -> List[str]:
    return _clean(value).split()


def _significant_tokens(value) -> List[str]:
    return [t for t in _tokenize(value) if t not in _COMMON_BUSINESS_TOKENS and len(t) > 1]


def _is_non_ascii(value) -> bool:
    """True if string contains any non-ASCII / non-latin character."""
    try:
        text = str(value).encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


# ── Similarity primitives ──────────────────────────────────────────────────────
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
    n = len(a_clean)
    for i in range(len(b_clean) - n + 1):
        r = Levenshtein.ratio(a_clean, b_clean[i:i+n])
        if r > best:
            best = r
            if best == 1.0: break
    return best


def _char_ngrams(text: str, n: int = 3) -> set:
    t = re.sub(r"\s+", "", _clean(text))
    if len(t) < n:
        return {t} if t else set()
    return {t[i:i+n] for i in range(len(t) - n + 1)}


def _char_ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    ga, gb = _char_ngrams(a, n), _char_ngrams(b, n)
    if not ga and not gb: return 1.0
    if not ga or not gb: return 0.0
    return len(ga & gb) / len(ga | gb)


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


# ── V4 New Feature Primitives ─────────────────────────────────────────────────
def _addr_num_jaccard(addr_a: str, addr_b: str) -> float:
    """Jaccard on 3+ digit numeric clusters in addresses (zip codes, building numbers)."""
    nums_a = set(_RE_NUMBERS.findall(addr_a))
    nums_b = set(_RE_NUMBERS.findall(addr_b))
    if not nums_a and not nums_b: return 1.0
    if not nums_a or not nums_b: return 0.0
    return len(nums_a & nums_b) / len(nums_a | nums_b)


def _addr_first_number_match(addr_a: str, addr_b: str) -> float:
    """Binary match on the first digit cluster in the address (street/building number)."""
    m_a = _RE_FIRST_NUM.search(addr_a)
    m_b = _RE_FIRST_NUM.search(addr_b)
    if not m_a and not m_b: return 1.0
    if not m_a or not m_b: return 0.0
    return 1.0 if m_a.group(1) == m_b.group(1) else 0.0


def _name_acronym_match(name_a: str, name_b: str) -> float:
    """Detects acronym-to-expansion matches (IBM ↔ International Business Machines)."""
    toks_a = _significant_tokens(name_a)
    toks_b = _significant_tokens(name_b)
    if not toks_a or not toks_b:
        return 0.0
    # Check if name_a is a single token that matches initials of name_b
    if len(toks_a) == 1 and len(toks_b) >= 2:
        initials_b = "".join(t[0] for t in toks_b)
        if toks_a[0] == initials_b: return 1.0
    if len(toks_b) == 1 and len(toks_a) >= 2:
        initials_a = "".join(t[0] for t in toks_a)
        if toks_b[0] == initials_a: return 1.0
    # Partial: does the acronym candidate appear as a substring?
    if len(toks_a) == 1:
        initials_b = "".join(t[0] for t in toks_b)
        ratio = _levenshtein_ratio(toks_a[0], initials_b)
        if ratio >= 0.85: return ratio
    if len(toks_b) == 1:
        initials_a = "".join(t[0] for t in toks_a)
        ratio = _levenshtein_ratio(toks_b[0], initials_a)
        if ratio >= 0.85: return ratio
    return 0.0


def _script_mismatch(name_a: str, name_b: str) -> float:
    """Detects that one entity name uses a non-ASCII script (e.g., Devanagari, Tamil, Arabic)
    while the other is Latin-script. Strong signal for transliteration pairs.
    Returns 1.0 if mismatch (one is non-ASCII, other is ASCII), 0.0 otherwise."""
    a_non = _is_non_ascii(name_a)
    b_non = _is_non_ascii(name_b)
    return 1.0 if a_non != b_non else 0.0


# ── Master Feature Builder ─────────────────────────────────────────────────────
def build_pair_features_v4(
    candidate_pairs: pd.DataFrame,
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    """Build 24 pairwise features for entity matching (V4 — memory-efficient, fast).

    Input DataFrame columns (prefix-flexible):
        source_name, target_name, source_address, target_address,
        source_country, target_country

    Optional extra columns read from candidate_pairs:
        blocker_score  – float, normalized blocker retrieval score (0.0 if missing)
        is_source2     – int/float 0/1, 1 if target is from Source2 (0 if missing)
    """
    if candidate_pairs is None or candidate_pairs.empty:
        raise ValueError("candidate_pairs must be a non-empty DataFrame.")

    data = candidate_pairs.copy()
    n = len(data)

    # ── Column detection ──────────────────────────────────────────────────
    name_left = name_right = addr_left = addr_right = ctry_left = ctry_right = None
    col_lower = {c.lower(): c for c in data.columns}
    for prefix_l, prefix_r in [
        ("source_", "target_"), ("left_", "right_"), ("entity1_", "entity2_"),
    ]:
        if f"{prefix_l}name" in col_lower:
            name_left  = col_lower[f"{prefix_l}name"]
            name_right = col_lower.get(f"{prefix_r}name", name_right)
            addr_left  = col_lower.get(f"{prefix_l}address", addr_left)
            addr_right = col_lower.get(f"{prefix_r}address", addr_right)
            ctry_left  = col_lower.get(f"{prefix_l}country", ctry_left)
            ctry_right = col_lower.get(f"{prefix_r}country", ctry_right)
            break

    if name_left is None:
        for col in data.columns:
            cl = col.lower()
            if "name" in cl and "left" in cl:    name_left  = col
            elif "name" in cl and "right" in cl: name_right = col
            elif "address" in cl and "left" in cl:    addr_left  = col
            elif "address" in cl and "right" in cl:   addr_right = col
            elif "country" in cl and "left" in cl:    ctry_left  = col
            elif "country" in cl and "right" in cl:   ctry_right = col

    def _sc(series):
        return (series.fillna("").astype(str).str.strip().str.lower()
                .str.replace(r"[^a-z0-9\s]", " ", regex=True)
                .str.replace(r"\s+", " ", regex=True).str.strip())

    name_a = _sc(data[name_left])  if name_left  else pd.Series([""] * n, dtype=str)
    name_b = _sc(data[name_right]) if name_right else pd.Series([""] * n, dtype=str)
    addr_a = _sc(data[addr_left])  if addr_left  else pd.Series([""] * n, dtype=str)
    addr_b = _sc(data[addr_right]) if addr_right else pd.Series([""] * n, dtype=str)
    ctry_a = _sc(data[ctry_left])  if ctry_left  else pd.Series([""] * n, dtype=str)
    ctry_b = _sc(data[ctry_right]) if ctry_right else pd.Series([""] * n, dtype=str)

    # Raw names (before ASCII-only clean) needed for script mismatch
    raw_name_a = data[name_left].fillna("").astype(str)  if name_left  else pd.Series([""] * n)
    raw_name_b = data[name_right].fillna("").astype(str) if name_right else pd.Series([""] * n)
    # Raw addresses for numeric extraction
    raw_addr_a = data[addr_left].fillna("").astype(str)  if addr_left  else pd.Series([""] * n)
    raw_addr_b = data[addr_right].fillna("").astype(str) if addr_right else pd.Series([""] * n)

    features: dict = {}

    # ══════════════════════════════════════════════════════
    # V3 CORE FEATURES (1–18)
    # ══════════════════════════════════════════════════════

    # 1. addr_char3_jaccard
    features["addr_char3_jaccard"] = np.array(
        [_char_ngram_jaccard(a, b, 3) for a, b in zip(addr_a, addr_b)], dtype=np.float32
    )

    # 2. addr_length_ratio
    features["addr_length_ratio"] = np.array(
        [_length_ratio(a, b) for a, b in zip(addr_a, addr_b)], dtype=np.float32
    )

    # 3. name_partial_ratio
    features["name_partial_ratio"] = np.array(
        [_partial_ratio(a, b) for a, b in zip(name_a, name_b)], dtype=np.float32
    )

    # 4. name_length_ratio
    features["name_length_ratio"] = np.array(
        [_length_ratio(a, b) for a, b in zip(name_a, name_b)], dtype=np.float32
    )

    # 5. all_fields_levenshtein
    features["all_fields_levenshtein"] = np.array(
        [_levenshtein_ratio(f"{na} {aa} {ca}", f"{nb} {ab} {cb}")
         for na, nb, aa, ab, ca, cb in zip(name_a, name_b, addr_a, addr_b, ctry_a, ctry_b)],
        dtype=np.float32,
    )

    # 6. name_addr_combined_jaccard
    features["name_addr_combined_jaccard"] = np.array(
        [_jaccard(_tokenize(f"{na} {aa}"), _tokenize(f"{nb} {ab}"))
         for na, nb, aa, ab in zip(name_a, name_b, addr_a, addr_b)],
        dtype=np.float32,
    )

    # 7. name_sorted_token_sim
    features["name_sorted_token_sim"] = np.array(
        [_sorted_token_similarity(a, b) for a, b in zip(name_a, name_b)], dtype=np.float32
    )

    # 8. name_char3_jaccard
    features["name_char3_jaccard"] = np.array(
        [_char_ngram_jaccard(a, b, 3) for a, b in zip(name_a, name_b)], dtype=np.float32
    )

    # 9. name_rare_token_overlap
    features["name_rare_token_overlap"] = np.array(
        [_rare_token_overlap(a, b) for a, b in zip(name_a, name_b)], dtype=np.float32
    )

    # 10. name_char4_jaccard
    features["name_char4_jaccard"] = np.array(
        [_char_ngram_jaccard(a, b, 4) for a, b in zip(name_a, name_b)], dtype=np.float32
    )

    # 11. addr_jaro_winkler
    features["addr_jaro_winkler"] = np.array(
        [_jaro_winkler(a, b) for a, b in zip(addr_a, addr_b)], dtype=np.float32
    )

    # 12. name_levenshtein
    name_lev = np.array(
        [_levenshtein_ratio(a, b) for a, b in zip(name_a, name_b)], dtype=np.float32
    )
    features["name_levenshtein"] = name_lev

    # 13. pair_quality_score (composite weighted signal)
    name_jw  = np.array([_jaro_winkler(a, b) for a, b in zip(name_a, name_b)], dtype=np.float32)
    name_tok = np.array([_jaccard(_tokenize(a), _tokenize(b)) for a, b in zip(name_a, name_b)], dtype=np.float32)
    addr_lev = np.array([_levenshtein_ratio(a, b) for a, b in zip(addr_a, addr_b)], dtype=np.float32)
    ctry_exact = (ctry_a == ctry_b).astype(np.float32).values

    features["pair_quality_score"] = (
        name_lev * 0.3 + name_jw * 0.2 + name_tok * 0.15 + addr_lev * 0.15 + ctry_exact * 0.2
    )

    # 14. name_max_sim
    features["name_max_sim"] = np.maximum.reduce([
        name_lev, name_jw, features["name_sorted_token_sim"], name_tok
    ])

    # 15. addr_levenshtein
    features["addr_levenshtein"] = addr_lev

    # 16. name_jaro_winkler
    features["name_jaro_winkler"] = name_jw

    # 17. name_token_jaccard
    features["name_token_jaccard"] = name_tok

    # 18. country_exact
    features["country_exact"] = ctry_exact

    # ══════════════════════════════════════════════════════
    # V4 NEW FEATURES (19–24)
    # ══════════════════════════════════════════════════════

    # 19. addr_num_jaccard – shared 3+ digit clusters (zip codes, building numbers)
    features["addr_num_jaccard"] = np.array(
        [_addr_num_jaccard(a, b) for a, b in zip(raw_addr_a, raw_addr_b)], dtype=np.float32
    )

    # 20. name_acronym_match – IBM ↔ International Business Machines
    features["name_acronym_match"] = np.array(
        [_name_acronym_match(a, b) for a, b in zip(name_a, name_b)], dtype=np.float32
    )

    # 21. script_mismatch – one is non-ASCII (Tamil, Devanagari, Arabic), other is ASCII
    features["script_mismatch"] = np.array(
        [_script_mismatch(a, b) for a, b in zip(raw_name_a, raw_name_b)], dtype=np.float32
    )

    # 22. blocker_score – normalized retrieval score from multi-signal blocker
    if "blocker_score" in data.columns:
        bs = data["blocker_score"].fillna(0.0).astype(np.float32).values
        # Normalize to [0, 1]
        bmax = bs.max()
        features["blocker_score"] = bs / bmax if bmax > 0 else bs
    else:
        features["blocker_score"] = np.zeros(n, dtype=np.float32)

    # 23. is_source2 – target schema differs (S2 vs S3 may have different address formats)
    if "is_source2" in data.columns:
        features["is_source2"] = data["is_source2"].fillna(0).astype(np.float32).values
    elif "target_id" in data.columns:
        features["is_source2"] = data["target_id"].fillna("").astype(str).str.startswith("S2").astype(np.float32).values
    else:
        features["is_source2"] = np.zeros(n, dtype=np.float32)

    # 24. addr_first_number_match – building/street number exact match
    features["addr_first_number_match"] = np.array(
        [_addr_first_number_match(a, b) for a, b in zip(raw_addr_a, raw_addr_b)], dtype=np.float32
    )

    # ── Assemble final DataFrame ────────────────────────────────────────────
    feature_df = pd.DataFrame(features, index=data.index)
    if target_col is not None and target_col in data.columns:
        feature_df[target_col] = data[target_col].astype(int)
    feature_df = feature_df.fillna(0.0)
    return feature_df


def get_feature_columns_v4() -> List[str]:
    """Canonical ordered list of all 24 V4 feature names."""
    return [
        # V3 core (18)
        "addr_char3_jaccard", "addr_length_ratio", "name_partial_ratio",
        "name_length_ratio", "all_fields_levenshtein", "name_addr_combined_jaccard",
        "name_sorted_token_sim", "name_char3_jaccard", "name_rare_token_overlap",
        "name_char4_jaccard", "addr_jaro_winkler", "name_levenshtein",
        "pair_quality_score", "name_max_sim", "addr_levenshtein",
        "name_jaro_winkler", "name_token_jaccard", "country_exact",
        # V4 new (6)
        "addr_num_jaccard", "name_acronym_match", "script_mismatch",
        "blocker_score", "is_source2", "addr_first_number_match",
    ]
