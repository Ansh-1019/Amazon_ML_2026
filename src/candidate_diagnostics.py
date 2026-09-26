"""Candidate retrieval diagnostics; no matching or similarity computation."""
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from src.candidate_union import union_candidates


@dataclass(frozen=True)
class CandidateDiagnostics:
    """Unique candidate/true-pair counts for one S1 row, without retained IDs."""
    candidate_count: int
    true_count: int
    captured: int

    def __post_init__(self) -> None:
        for value in (self.candidate_count, self.true_count, self.captured):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("diagnostic counts must be nonnegative integers")
        if self.captured > min(self.candidate_count, self.true_count):
            raise ValueError("captured cannot exceed candidate_count or true_count")

    @property
    def missed(self) -> int:
        return self.true_count - self.captured

    @property
    def recall(self) -> float | None:
        return self.captured / self.true_count if self.true_count else None


def diagnose_candidates(
    candidates: Iterable[str] | None, truth: Iterable[str] | None,
) -> CandidateDiagnostics:
    """Count unique exact IDs for one row; None collections mean empty.

    Duplicate IDs count once. String IDs are preserved verbatim and validated
    using M7. Recall is None for zero truth, even when candidates are present.
    """
    candidate_ids = set(union_candidates(candidates))
    true_ids = set(union_candidates(truth))
    return CandidateDiagnostics(len(candidate_ids), len(true_ids), len(candidate_ids & true_ids))


def aggregate_diagnostics(records: Iterable[CandidateDiagnostics]) -> dict:
    """Consume records once; compute pair-weighted recall and count statistics.

    Retains one integer per S1 for exact linearly interpolated percentiles:
    position = (row_count - 1) * percentile / 100, as in NumPy's linear method.
    Candidate IDs are never retained. Empty input has zero volume/statistics
    and None recall. Unlabeled rows still contribute candidate volume.
    """
    counts: list[int] = []
    true_pairs = captured = 0
    for record in records:
        if not isinstance(record, CandidateDiagnostics):
            raise TypeError("records must contain CandidateDiagnostics")
        counts.append(record.candidate_count)
        true_pairs += record.true_count
        captured += record.captured
    counts.sort()

    def percentile(percent: float) -> float:
        if not counts:
            return 0.0
        position = (len(counts) - 1) * percent / 100
        lower = int(position)
        upper = min(lower + 1, len(counts) - 1)
        return counts[lower] + (counts[upper] - counts[lower]) * (position - lower)

    distribution = {f"{p}%": percentile(p) for p in (50, 90, 95, 99, 99.9)}
    maximum = counts[-1] if counts else 0
    distribution["maximum"] = maximum
    total = sum(counts)
    return {
        "s1_rows": len(counts), "candidate_pairs": total,
        "true_pairs": true_pairs, "captured_true_pairs": captured,
        "missed_true_pairs": true_pairs - captured,
        "pair_recall": captured / true_pairs if true_pairs else None,
        "average_candidates": total / len(counts) if counts else 0.0,
        "median_candidates": distribution["50%"], "maximum_candidates": maximum,
        "candidate_percentiles": distribution,
    }


def split_truth_by_source(truth: Iterable[str] | None) -> dict[str, tuple[str, ...]]:
    """Return sorted unique S2/S3 truth IDs; reject unsupported prefixes.

    Prefix checks are literal and case-sensitive. IDs are never rewritten.
    """
    groups: dict[str, list[str]] = {"S2": [], "S3": []}
    for entity_id in union_candidates(truth):
        if not entity_id.startswith(("S2-", "S3-")):
            raise ValueError(f"unsupported truth source prefix: {entity_id!r}")
        groups[entity_id[:2]].append(entity_id)
    return {source: tuple(ids) for source, ids in groups.items()}


def diagnose_by_source(
    candidates: Iterable[str] | None, truth: Iterable[str] | None,
) -> dict[str, CandidateDiagnostics]:
    """Compute S2/S3 diagnostics from literal prefixes in one S1 row.

    Other candidate prefixes do not contribute to either source; use overall
    diagnostics to include them. Unknown truth prefixes raise ValueError.
    """
    groups = split_truth_by_source(truth)
    candidate_ids = union_candidates(candidates)
    return {
        source: diagnose_candidates(
            (entity_id for entity_id in candidate_ids if entity_id.startswith(source + "-")),
            true_ids,
        ) for source, true_ids in groups.items()
    }


def compare_candidate_blocks(
    blocks: Mapping[str, Iterable[str] | None], truth: Iterable[str] | None,
) -> dict:
    """Report per-block, union and marginal diagnostics for one S1 row.

    Mapping iteration order defines incremental attribution; no strategy names
    are hard-coded. Each block is consumed once. Raw counts are deduplicated;
    new counts exclude all preceding blocks. Uses M7 for cumulative union.
    Storage is limited to this S1's truth, union, and current block IDs.
    """
    true_ids = set(union_candidates(truth))
    cumulative: list[str] = []
    reports = {}
    marginal = {}
    for label, candidates in blocks.items():
        if not isinstance(label, str):
            raise TypeError("block labels must be strings")
        current = union_candidates(candidates)
        report = diagnose_candidates(current, true_ids)
        new_ids = set(current).difference(cumulative)
        reports[label] = report
        marginal[label] = {
            "raw_candidate_count": report.candidate_count,
            "new_candidate_count": len(new_ids),
            "raw_captured": report.captured,
            "new_captured": len(new_ids & true_ids),
        }
        cumulative = union_candidates(cumulative, current)
    return {"blocks": reports, "union": diagnose_candidates(cumulative, true_ids),
            "marginal": marginal}
