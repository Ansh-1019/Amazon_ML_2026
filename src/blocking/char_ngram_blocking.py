"""Sparse character TF-IDF retrieval for candidate generation only."""
from collections.abc import Iterable
from dataclasses import dataclass
from heapq import nsmallest
import math

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.normalization import normalize_name


@dataclass(frozen=True)
class CharNgramIndex:
    """Aligned target IDs, fitted vectorizer and sparse L2-normalized rows.

    Treat the contained vectorizer/matrix as read-only. A None vectorizer and
    zero-column matrix represent an empty vocabulary, including after pruning.
    """
    entity_ids: tuple[str, ...]
    vectorizer: TfidfVectorizer | None
    matrix: csr_matrix


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def build_char_ngram_index(
    entity_ids: Iterable[str], names: Iterable[object], *,
    ngram_range: tuple[int, int] = (2, 5),
    min_df: int | float = 1, max_features: int | None = None,
) -> CharNgramIndex:
    """Fit target-only TF-IDF on normalize_name() output.

    Character n-grams retain normalized spaces/punctuation and cross word
    boundaries, with no boundary padding or language segmentation. Defaults
    use lengths 2–5, raw term counts, smoothed IDF, float64 and L2 normalization.
    min_df is a positive document count or fraction in (0, 1]; max_features is
    None or a positive vocabulary cap. Target documents are sorted by ID.

    Duplicate IDs contribute exactly one document: the lexically smallest
    nonempty normalized name. Other aliases are discarded, not concatenated.
    Empty names/None are excluded. Names shorter than the minimum n-gram may
    yield zero rows. No vocabulary (including after pruning) is a valid index.
    Aligned single-pass inputs are required; IDs must be strings.
    """
    if not isinstance(ngram_range, tuple) or len(ngram_range) != 2:
        raise ValueError("ngram_range must be a (minimum, maximum) tuple")
    for length in ngram_range:
        _positive_integer(length, "ngram_range lengths")
    if ngram_range[0] > ngram_range[1]:
        raise ValueError("ngram_range minimum must not exceed maximum")
    if isinstance(min_df, bool) or not isinstance(min_df, (int, float)):
        raise ValueError("min_df must be a positive integer or fraction in (0, 1]")
    if isinstance(min_df, int):
        _positive_integer(min_df, "min_df")
    elif not math.isfinite(min_df) or not 0 < min_df <= 1:
        raise ValueError("min_df fraction must be in (0, 1]")
    if max_features is not None:
        _positive_integer(max_features, "max_features")

    documents: dict[str, str] = {}
    for entity_id, name in zip(entity_ids, names, strict=True):
        if not isinstance(entity_id, str):
            raise TypeError("entity IDs must be strings")
        normalized = normalize_name(name)
        if normalized and (entity_id not in documents or normalized < documents[entity_id]):
            documents[entity_id] = normalized
    ids = tuple(sorted(documents))
    empty = CharNgramIndex(ids, None, csr_matrix((len(ids), 0), dtype=np.float64))
    if not ids or (isinstance(min_df, int) and min_df > len(ids)):
        return empty

    vectorizer = TfidfVectorizer(
        input="content", encoding="utf-8", decode_error="strict",
        analyzer="char", ngram_range=ngram_range,
        lowercase=False, strip_accents=None, preprocessor=None,
        tokenizer=None, token_pattern=None, stop_words=None,
        min_df=min_df, max_df=1.0, max_features=max_features,
        vocabulary=None, binary=False, dtype=np.float64,
        norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False,
    )
    try:
        matrix = vectorizer.fit_transform(documents[entity_id] for entity_id in ids).tocsr()
    except ValueError as exc:
        # sklearn reports these two legitimate empty-vocabulary conditions.
        if str(exc).startswith(("empty vocabulary;", "After pruning, no terms remain.")):
            return empty
        raise
    return CharNgramIndex(ids, vectorizer, matrix)


def generate_char_ngram_candidates(
    name: object, index: CharNgramIndex, *, top_k: int,
) -> list[str]:
    """Retrieve up to required positive top_k IDs by positive cosine overlap.

    Reuses fitted target vocabulary/IDF; ties use lexical entity ID. Returns
    IDs only, not final match scores or identity decisions. A single sparse
    query-by-target product is computed, never a dense/all-source pair matrix.
    Retrieval may still touch every target for common n-grams; top_k bounds
    output, not overlap-computation cost. Index contents are not mutated.
    """
    _positive_integer(top_k, "top_k")
    normalized = normalize_name(name)
    if not normalized or index.vectorizer is None:
        return []
    query = index.vectorizer.transform([normalized])
    if query.nnz == 0:
        return []
    similarities = (query @ index.matrix.T).tocsr()
    best = nsmallest(
        top_k,
        ((-float(score), index.entity_ids[row])
         for row, score in zip(similarities.indices, similarities.data)
         if score > 0),
    )
    return [entity_id for _, entity_id in best]
