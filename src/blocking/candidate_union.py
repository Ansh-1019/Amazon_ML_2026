"""Deterministic composition of candidate IDs from arbitrary blocking layers."""
from collections.abc import Iterable


def union_candidates(*candidate_blocks: Iterable[str] | None) -> list[str]:
    """Return unique candidate IDs in lexical order across all supplied blocks.

    No blocks, empty blocks and None blocks contribute nothing. Each non-None
    block must be an iterable of strings, not a bare string/bytes value. Invalid
    IDs (including None) raise TypeError. ID text is preserved exactly, without
    source-prefix validation or normalization. Input collections are not mutated;
    iterators are consumed once. Pass a dynamic collection with *blocks.

    Uses O(U) storage and O(N + U log U) time for N input IDs and U unique IDs.
    Ordering carries no score, priority or match decision.
    """
    candidates: set[str] = set()
    for block in candidate_blocks:
        if block is None:
            continue
        if isinstance(block, (str, bytes)):
            raise TypeError("candidate blocks must be iterables of IDs, not strings or bytes")
        for entity_id in block:
            if not isinstance(entity_id, str):
                raise TypeError("candidate IDs must be strings")
            candidates.add(entity_id)
    return sorted(candidates)
