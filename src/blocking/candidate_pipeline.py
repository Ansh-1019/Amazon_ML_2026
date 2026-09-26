"""Thin single-record orchestration of arbitrary candidate-producing blocks."""
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from src.blocking.candidate_union import union_candidates


CandidateBlock = Callable[[object], Iterable[str] | None]


@dataclass
class CandidateResult:
    """Fresh per-call candidate lists and deterministic block provenance.

    candidates_by_block can be passed directly to M8 compare_candidate_blocks.
    Provenance tuples follow configured block order; candidate keys are lexical.
    """
    all_candidates: list[str]
    candidates_by_block: dict[str, list[str]]
    provenance: dict[str, tuple[str, ...]]

    @property
    def counts_by_block(self) -> dict[str, int]:
        """Unique candidate counts for each block in configuration order."""
        return {label: len(ids) for label, ids in self.candidates_by_block.items()}


class CandidatePipeline:
    """Run named callables sequentially in the supplied mapping's order.

    Configuration is snapshotted at construction. Each block receives exactly
    the same record object; closures/partials can bind indexes or adapt fields.
    Blocks must treat the record as read-only and produce deterministic outputs
    for reproducible results. No normalization or strategy logic lives here.
    """

    def __init__(self, blocks: Mapping[str, CandidateBlock]) -> None:
        if not isinstance(blocks, Mapping):
            raise TypeError("blocks must be a mapping of names to callables")
        configured = tuple(blocks.items())
        for label, block in configured:
            if not isinstance(label, str):
                raise TypeError("block names must be strings")
            if not callable(block):
                raise TypeError(f"block {label!r} must be callable")
        self._blocks = configured

    def generate(self, record: object) -> CandidateResult:
        """Run each block once, then compose candidates using M7.

        None/empty block outputs contribute no IDs. Generators are consumed
        once. M7 validates IDs and preserves their text. Block/iteration errors
        propagate immediately; later blocks are not run and no partial result
        is returned. Results are independent of previous calls.
        """
        by_block = {}
        for label, block in self._blocks:
            by_block[label] = union_candidates(block(record))
        candidates = union_candidates(*by_block.values())
        origins: dict[str, list[str]] = {entity_id: [] for entity_id in candidates}
        for label, ids in by_block.items():
            for entity_id in ids:
                origins[entity_id].append(label)
        provenance = {entity_id: tuple(labels) for entity_id, labels in origins.items()}
        return CandidateResult(candidates, by_block, provenance)
