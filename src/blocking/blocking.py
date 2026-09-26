"""Shared-name-token blocking, with no scoring or identity decisions.

Use the same stopwords/min_token_length for build and query. Custom stopwords
replace defaults and must be normalized tokens. Caps and frequency pruning
trade recall for cost; tune them on validation data.
"""
from collections.abc import Iterable, Mapping
from heapq import merge

from src.data.normalization import tokenize_name

DEFAULT_STOPWORDS = frozenset("""
and the of for in on at to a an private limited ltd llc inc incorporated
corp corporation company co plc llp pllc lp pvt
""".split())
TokenIndex = dict[str, tuple[str, ...]]


def _validate_limit(value: int | None, name: str, minimum: int, *, optional: bool = True) -> None:
    if value is None and optional:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _tokens(name: object, stopwords: frozenset[str], minimum: int) -> set[str]:
    return {token for token in tokenize_name(name)
            if len(token) >= minimum and token not in stopwords}


def build_token_index(
    entity_ids: Iterable[str], names: Iterable[object], *,
    stopwords: Iterable[str] | None = None, min_token_length: int = 2,
    max_token_frequency: int | None = None,
) -> TokenIndex:
    """Build token -> sorted unique string IDs from aligned, single-pass inputs.

    Frequency counts distinct entity IDs per token, including across duplicate
    rows. Tokens ABOVE max_token_frequency are discarded permanently, releasing
    their postings during construction. None disables pruning: no universal safe
    threshold exists, so the default preserves recall. Supply an explicit limit
    for large datasets to bound each retained posting list.

    Unequal input lengths raise ValueError. Only tokens and IDs are retained.
    Duplicate IDs can contribute multiple names. min_token_length counts Unicode
    code points; there is no extra punctuation/script processing. At length 1,
    punctuation tokens are eligible. Custom stopwords replace DEFAULT_STOPWORDS.
    """
    _validate_limit(min_token_length, "min_token_length", 1, optional=False)
    _validate_limit(max_token_frequency, "max_token_frequency", 0)
    excluded = DEFAULT_STOPWORDS if stopwords is None else frozenset(stopwords)
    postings: dict[str, set[str]] = {}
    discarded: set[str] = set()
    for entity_id, name in zip(entity_ids, names, strict=True):
        if not isinstance(entity_id, str):
            raise TypeError("entity IDs must be strings")
        for token in _tokens(name, excluded, min_token_length):
            if token in discarded:
                continue
            ids = postings.setdefault(token, set())
            ids.add(entity_id)
            if max_token_frequency is not None and len(ids) > max_token_frequency:
                del postings[token]
                discarded.add(token)
    # Release sets as they are converted instead of retaining two full indexes.
    result: TokenIndex = {}
    for token in sorted(postings):
        result[token] = tuple(sorted(postings.pop(token)))
    return result


def generate_token_candidates(
    name: object, token_index: Mapping[str, tuple[str, ...]], *,
    stopwords: Iterable[str] | None = None, min_token_length: int = 2,
    max_candidates: int | None = None,
) -> list[str]:
    """Return unique matching IDs in lexical order, without ranking.

    Requires sorted postings as returned by build_token_index. None returns the
    whole union; zero returns nothing. A positive cap returns the lexically first
    max_candidates unique IDs (S2-10 precedes S2-2), independently of input order.
    Lazy merging avoids materializing the whole union for capped queries.
    Use the same filtering configuration as at build time.
    """
    _validate_limit(min_token_length, "min_token_length", 1, optional=False)
    _validate_limit(max_candidates, "max_candidates", 0)
    if max_candidates == 0:
        return []
    excluded = DEFAULT_STOPWORDS if stopwords is None else frozenset(stopwords)
    tokens = sorted(_tokens(name, excluded, min_token_length))
    candidates: list[str] = []
    for entity_id in merge(*(token_index[t] for t in tokens if t in token_index)):
        if not candidates or candidates[-1] != entity_id:
            candidates.append(entity_id)
            if max_candidates is not None and len(candidates) >= max_candidates:
                break
    return candidates
