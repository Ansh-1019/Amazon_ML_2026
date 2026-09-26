"""Reusable exact-field blocking over caller-normalized strings."""
from collections.abc import Iterable, Mapping
from decimal import Decimal
import math


ExactIndex = dict[str, tuple[str, ...]]


def _has_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, Decimal) and value.is_nan():
        return False
    if not isinstance(value, str):
        raise TypeError("exact values must be strings or missing")
    return value != ""


def build_exact_index(entity_ids: Iterable[str], values: Iterable[object]) -> ExactIndex:
    """Map exact strings to lexically sorted unique IDs from aligned iterables.

    None, float/Decimal NaN and empty strings create no bucket. Other non-string
    values raise TypeError; unequal input lengths raise ValueError. No trimming,
    case folding or normalization occurs, including for whitespace-only strings.
    Duplicate IDs may contribute several values but occur once per bucket.
    Keys are sorted too, making iteration independent of input order. Callers
    must preserve source identity in IDs, e.g. S2-1 and S3-1 are distinct.
    """
    postings: dict[str, set[str]] = {}
    for entity_id, value in zip(entity_ids, values, strict=True):
        if not isinstance(entity_id, str):
            raise TypeError("entity IDs must be strings")
        if _has_value(value):
            postings.setdefault(value, set()).add(entity_id)
    result: ExactIndex = {}
    for value in sorted(postings):
        result[value] = tuple(sorted(postings.pop(value)))
    return result


def generate_exact_candidates(value: object, index: Mapping[str, tuple[str, ...]]) -> list[str]:
    """Return a fresh list of exact candidates from a build_exact_index result.

    Missing/empty or unknown values return []. Value validation matches the
    builder; queries are not normalized. The index is never mutated. Lexical
    ID order is deterministic enumeration, not scoring or an identity decision.
    """
    if not _has_value(value):
        return []
    return list(index.get(value, ()))
