"""Rare-name-token blocking with an explicit distinct-entity frequency bound."""
from collections.abc import Iterable, Mapping
from heapq import merge

from src.blocking import DEFAULT_STOPWORDS
from src.normalization import tokenize_name


RareTokenIndex = dict[str, tuple[str, ...]]


def _validate_integer(value: object, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _informative_tokens(name: object, excluded: frozenset[str], minimum: int) -> set[str]:
    return {token for token in tokenize_name(name)
            if len(token) >= minimum and token not in excluded}


def build_rare_token_index(
    entity_ids: Iterable[str], names: Iterable[object], *,
    max_token_frequency: int,
    stopwords: Iterable[str] | None = None, min_token_length: int = 2,
) -> RareTokenIndex:
    """Index tokens occurring in at most max_token_frequency distinct IDs.

    The required nonnegative threshold has no implicit default; zero excludes
    all tokens. Over-threshold postings are released and never reintroduced.
    Duplicate rows/tokens do not inflate counts. Inputs are single-pass aligned
    iterables; unequal lengths raise ValueError and non-string IDs TypeError.

    Uses M1 tokenization and M2 default stopwords/code-point length semantics.
    Custom stopwords replace defaults and must already be normalized tokens.
    None/empty names yield no tokens, following M1. Keys and unique ID tuples
    are sorted. No scoring or cross-block candidate union is performed.
    """
    _validate_integer(max_token_frequency, "max_token_frequency", 0)
    _validate_integer(min_token_length, "min_token_length", 1)
    excluded = DEFAULT_STOPWORDS if stopwords is None else frozenset(stopwords)
    postings: dict[str, set[str]] = {}
    common: set[str] = set()
    for entity_id, name in zip(entity_ids, names, strict=True):
        if not isinstance(entity_id, str):
            raise TypeError("entity IDs must be strings")
        for token in _informative_tokens(name, excluded, min_token_length):
            if token in common:
                continue
            bucket = postings.setdefault(token, set())
            bucket.add(entity_id)
            if len(bucket) > max_token_frequency:
                common.add(token)
                del postings[token]
    result: RareTokenIndex = {}
    for token in sorted(postings):
        result[token] = tuple(sorted(postings.pop(token)))
    return result


def generate_rare_token_candidates(
    name: object, token_index: Mapping[str, tuple[str, ...]], *,
    stopwords: Iterable[str] | None = None, min_token_length: int = 2,
    max_candidates: int | None = None,
) -> list[str]:
    """Return unique matching IDs in lexical order without mutating the index.

    Requires sorted postings from build_rare_token_index and the same stopword
    and length settings. Frequency filtering is fixed at build time. None is
    uncapped; zero returns []; positive caps choose the lexically first IDs,
    not better matches. Lazy merging avoids building an uncapped candidate set.
    """
    _validate_integer(min_token_length, "min_token_length", 1)
    if max_candidates is not None:
        _validate_integer(max_candidates, "max_candidates", 0)
    if max_candidates == 0:
        return []
    excluded = DEFAULT_STOPWORDS if stopwords is None else frozenset(stopwords)
    tokens = sorted(_informative_tokens(name, excluded, min_token_length))
    candidates: list[str] = []
    for entity_id in merge(*(token_index[token] for token in tokens if token in token_index)):
        if not candidates or candidates[-1] != entity_id:
            candidates.append(entity_id)
            if max_candidates is not None and len(candidates) >= max_candidates:
                break
    return candidates
