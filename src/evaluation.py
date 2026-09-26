"""Streaming evaluation of candidate retrieval using M8 and M9."""
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from heapq import nsmallest

from src.candidate_diagnostics import (
    CandidateDiagnostics, aggregate_diagnostics, compare_candidate_blocks,
    diagnose_by_source, diagnose_candidates, split_truth_by_source,
)
from src.candidate_pipeline import CandidatePipeline


@dataclass(frozen=True)
class MissedPair:
    """A missed target for a zero-based position in the evaluation stream."""
    record_index: int
    target_id: str


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate retrieval metrics, optional block totals and bounded samples."""
    total_records: int
    total_candidate_pairs: int
    total_true_pairs: int
    total_captured_true_pairs: int
    total_missed_true_pairs: int
    pair_recall: float | None
    average_candidates_per_record: float
    candidate_count_percentiles: dict[str, float | int]
    maximum_candidates: int
    true_S2_pairs: int
    captured_S2_pairs: int
    S2_recall: float | None
    true_S3_pairs: int
    captured_S3_pairs: int
    S3_recall: float | None
    block_contributions: dict[str, dict[str, int]] | None
    missed_pairs: tuple[MissedPair, ...]


def evaluate_pipeline(
    pipeline: CandidatePipeline, records: Iterable[object],
    truth_getter: Callable[[object], Iterable[str] | None], *,
    include_block_contributions: bool = False, max_missed_pairs: int = 0,
) -> EvaluationResult:
    """Evaluate each record once without retaining records or candidate lists.

    truth_getter is called once per record; truth is deduplicated and validated
    by M8 (literal S2-/S3- prefixes). Each stream entry counts as one S1 row,
    including repeated records and zero-truth rows. Errors propagate.

    M8 retains one candidate-count integer per row for linear percentiles.
    Other retained state is aggregate counters plus an optional first-N missed
    sample in stream order, then lexical target-ID order within a row. Sample
    positions are zero-based; no source records or full missed sets are saved.
    Block marginals follow pipeline order. Disabled contributions are None;
    an empty stream returns {} when enabled, since no blocks were observed.
    """
    if not isinstance(include_block_contributions, bool):
        raise TypeError("include_block_contributions must be a bool")
    if (isinstance(max_missed_pairs, bool) or not isinstance(max_missed_pairs, int)
            or max_missed_pairs < 0):
        raise ValueError("max_missed_pairs must be a nonnegative integer")
    if not callable(truth_getter):
        raise TypeError("truth_getter must be callable")
    sources = {source: [0, 0, 0] for source in ("S2", "S3")}
    contributions = {} if include_block_contributions else None
    missed: list[MissedPair] = []

    def diagnostics() -> Iterable[CandidateDiagnostics]:
        for position, record in enumerate(records):
            result = pipeline.generate(record)
            groups = split_truth_by_source(truth_getter(record))
            truth = groups["S2"] + groups["S3"]
            diagnostic = diagnose_candidates(result.all_candidates, truth)
            for source, stats in diagnose_by_source(result.all_candidates, truth).items():
                totals = sources[source]
                totals[0] += stats.candidate_count
                totals[1] += stats.true_count
                totals[2] += stats.captured
            if contributions is not None:
                comparison = compare_candidate_blocks(result.candidates_by_block, truth)
                for label, marginal in comparison["marginal"].items():
                    totals = contributions.setdefault(label, {key: 0 for key in marginal})
                    for key, value in marginal.items():
                        totals[key] += value
                del comparison
            remaining = max_missed_pairs - len(missed)
            if remaining:
                candidates = set(result.all_candidates)
                selected = nsmallest(remaining, (target for target in truth if target not in candidates))
                missed.extend(MissedPair(position, target) for target in selected)
                del candidates, selected
            del result, record, truth, groups
            yield diagnostic

    aggregate = aggregate_diagnostics(diagnostics())
    s2 = CandidateDiagnostics(*sources["S2"])
    s3 = CandidateDiagnostics(*sources["S3"])
    return EvaluationResult(
        total_records=aggregate["s1_rows"],
        total_candidate_pairs=aggregate["candidate_pairs"],
        total_true_pairs=aggregate["true_pairs"],
        total_captured_true_pairs=aggregate["captured_true_pairs"],
        total_missed_true_pairs=aggregate["missed_true_pairs"],
        pair_recall=aggregate["pair_recall"],
        average_candidates_per_record=aggregate["average_candidates"],
        candidate_count_percentiles=aggregate["candidate_percentiles"],
        maximum_candidates=aggregate["maximum_candidates"],
        true_S2_pairs=s2.true_count, captured_S2_pairs=s2.captured, S2_recall=s2.recall,
        true_S3_pairs=s3.true_count, captured_S3_pairs=s3.captured, S3_recall=s3.recall,
        block_contributions=contributions, missed_pairs=tuple(missed),
    )
