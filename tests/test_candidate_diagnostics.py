import pytest

from src.blocking.candidate_diagnostics import (
    CandidateDiagnostics, aggregate_diagnostics, compare_candidate_blocks,
    diagnose_by_source, diagnose_candidates, split_truth_by_source,
)


@pytest.mark.parametrize("candidates,truth,expected", [
    (["S2-1", "S2-2", "S2-3"], ["S2-2", "S2-4"], (3, 2, 1, 1, .5)),
    ([], ["S2-1"], (0, 1, 0, 1, 0.0)),
    (["S2-1"], [], (1, 0, 0, 0, None)),
    ([], [], (0, 0, 0, 0, None)),
    (None, None, (0, 0, 0, 0, None)),
    (["S2-1", "S3-1"], ["S2-1", "S3-1"], (2, 2, 2, 0, 1.0)),
    (["S2-1"], ["S3-1"], (1, 1, 0, 1, 0.0)),
    (["S2-1", "S2-1"], ["S2-1", "S2-1"], (1, 1, 1, 0, 1.0)),
])
def test_per_row_counts(candidates, truth, expected):
    record = diagnose_candidates(candidates, truth)
    assert (record.candidate_count, record.true_count, record.captured,
            record.missed, record.recall) == expected


def test_exact_ids_generators_and_input_order():
    candidates = [" S2-1", "s2-1", "S3-1", "S2-1"]
    truth = ["S2-1", "S3-2"]
    first = diagnose_candidates(iter(candidates), (value for value in truth))
    assert first == diagnose_candidates(reversed(candidates), reversed(truth))
    assert first == CandidateDiagnostics(4, 2, 1)
    assert candidates == [" S2-1", "s2-1", "S3-1", "S2-1"]


def test_aggregate_exact_metrics_and_pair_weighted_recall():
    records = [CandidateDiagnostics(0, 1, 0), CandidateDiagnostics(2, 2, 2),
               CandidateDiagnostics(4, 0, 0), CandidateDiagnostics(6, 3, 1)]
    report = aggregate_diagnostics(record for record in records)
    assert report == {
        "s1_rows": 4, "candidate_pairs": 12, "true_pairs": 6,
        "captured_true_pairs": 3, "missed_true_pairs": 3, "pair_recall": .5,
        "average_candidates": 3.0, "median_candidates": 3.0, "maximum_candidates": 6,
        "candidate_percentiles": {"50%": 3.0, "90%": pytest.approx(5.4),
                                  "95%": pytest.approx(5.7), "99%": pytest.approx(5.94),
                                  "99.9%": pytest.approx(5.994), "maximum": 6},
    }
    assert aggregate_diagnostics(reversed(records)) == report


def test_empty_aggregate():
    report = aggregate_diagnostics(iter([]))
    assert report == {
        "s1_rows": 0, "candidate_pairs": 0, "true_pairs": 0,
        "captured_true_pairs": 0, "missed_true_pairs": 0, "pair_recall": None,
        "average_candidates": 0.0, "median_candidates": 0.0, "maximum_candidates": 0,
        "candidate_percentiles": {"50%": 0.0, "90%": 0.0, "95%": 0.0,
                                  "99%": 0.0, "99.9%": 0.0, "maximum": 0},
    }


def test_unlabeled_single_row_still_counts_volume():
    report = aggregate_diagnostics([diagnose_candidates(["S2-1", "S3-1"], [])])
    assert report["candidate_pairs"] == 2
    assert report["pair_recall"] is None
    assert report["captured_true_pairs"] == report["missed_true_pairs"] == 0
    assert all(value == 2 for value in report["candidate_percentiles"].values())


@pytest.mark.parametrize("truth,expected", [
    (["S2-2", "S2-1", "S2-1"], {"S2": ("S2-1", "S2-2"), "S3": ()}),
    (["S3-1"], {"S2": (), "S3": ("S3-1",)}),
    (["S3-5", "S2-1"], {"S2": ("S2-1",), "S3": ("S3-5",)}),
    ([], {"S2": (), "S3": ()}),
])
def test_source_split(truth, expected):
    assert split_truth_by_source(iter(truth)) == expected
    assert split_truth_by_source(reversed(truth)) == expected


def test_per_source_diagnostics():
    result = diagnose_by_source(iter(["S2-1", "S3-5", "S3-9", "other"]),
                                iter(["S2-1", "S2-3", "S3-5"]))
    assert list(result) == ["S2", "S3"]
    assert result == {"S2": CandidateDiagnostics(1, 2, 1),
                      "S3": CandidateDiagnostics(2, 1, 1)}
    assert result["S2"].recall == .5
    assert result["S3"].recall == 1.0
    assert diagnose_by_source([], ["S2-1"])["S3"].recall is None


@pytest.mark.parametrize("truth", [[" S2-1"], ["s2-1"], ["S4-1"], ["S20-1"]])
def test_unknown_truth_prefix_is_not_silently_dropped(truth):
    with pytest.raises(ValueError, match="prefix"):
        split_truth_by_source(truth)


def test_block_union_and_marginal_integration():
    blocks = {"M1": ["S2-1", "S2-9"], "M2": ["S2-3", "S3-5", "S3-8"],
              "M4": ["S2-1", "S3-10", "S2-1"]}
    report = compare_candidate_blocks(
        {label: iter(ids) for label, ids in blocks.items()}, iter(["S2-1", "S2-3", "S3-5"]),
    )
    assert report["blocks"] == {"M1": CandidateDiagnostics(2, 3, 1),
                                "M2": CandidateDiagnostics(3, 3, 2),
                                "M4": CandidateDiagnostics(2, 3, 1)}
    assert [record.recall for record in report["blocks"].values()] == [1 / 3, 2 / 3, 1 / 3]
    assert report["union"] == CandidateDiagnostics(6, 3, 3)
    assert report["union"].recall == 1.0
    assert report["marginal"] == {
        "M1": {"raw_candidate_count": 2, "new_candidate_count": 2, "raw_captured": 1, "new_captured": 1},
        "M2": {"raw_candidate_count": 3, "new_candidate_count": 3, "raw_captured": 2, "new_captured": 2},
        "M4": {"raw_candidate_count": 2, "new_candidate_count": 1, "raw_captured": 1, "new_captured": 0},
    }
    assert compare_candidate_blocks(blocks, ["S2-1", "S2-3", "S3-5"]) == report
    reversed_report = compare_candidate_blocks(dict(reversed(list(blocks.items()))),
                                               ["S2-1", "S2-3", "S3-5"])
    assert reversed_report["union"] == report["union"]
    assert reversed_report["marginal"]["M1"]["new_captured"] == 0
    assert blocks["M4"] == ["S2-1", "S3-10", "S2-1"]


def test_empty_blocks_and_zero_truth():
    report = compare_candidate_blocks({"custom": None, "other": [], "union": ["S2-1"]}, None)
    assert report["union"] == CandidateDiagnostics(1, 0, 0)
    assert report["union"].recall is None
    assert report["marginal"]["custom"]["new_candidate_count"] == 0
    assert compare_candidate_blocks({}, ["S2-1"]) == {
        "blocks": {}, "union": CandidateDiagnostics(0, 1, 0), "marginal": {},
    }


@pytest.mark.parametrize("counts", [(-1, 0, 0), (True, 0, 0), (1.5, 0, 0), (1, 2, 2), (2, 1, 2)])
def test_invalid_diagnostic_records(counts):
    with pytest.raises(ValueError):
        CandidateDiagnostics(*counts)


def test_invalid_input_types():
    with pytest.raises(TypeError):
        diagnose_candidates([None], [])
    with pytest.raises(TypeError):
        diagnose_candidates([], "S2-1")
    with pytest.raises(TypeError):
        aggregate_diagnostics([{}])
    with pytest.raises(TypeError):
        compare_candidate_blocks({1: []}, [])
