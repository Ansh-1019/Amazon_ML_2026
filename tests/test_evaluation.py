import weakref

import pytest

from src.candidate_pipeline import CandidatePipeline
from src.evaluation import MissedPair, evaluate_pipeline


def test_end_to_end_exact_synthetic_metrics():
    records = [
        {"id": "S1-A", "truth": ["S2-1", "S2-3", "S3-5", "S2-1"]},
        {"id": "S1-B", "truth": ["S2-2", "S3-6"]},
        {"id": "S1-C", "truth": []},
    ]
    outputs = {
        "M1": {"S1-A": ["S2-1", "S2-9", "S2-1"], "S1-B": ["S2-2"]},
        "M2": {"S1-A": ["S2-3", "S3-5", "S3-8"]},
        "M4": {"S1-A": ["S2-1", "S3-10"]},
        "M5": {"S1-C": ["S2-99"]},
        "M6": {"S1-B": ["S2-2", "S3-7"]},
    }
    pipeline = CandidatePipeline({label: (lambda record, table=table: iter(table.get(record["id"], [])))
                                  for label, table in outputs.items()})
    report = evaluate_pipeline(pipeline, iter(records), lambda r: iter(r["truth"]),
                               include_block_contributions=True, max_missed_pairs=3)
    assert report.total_records == 3
    assert report.total_candidate_pairs == 9
    assert report.total_true_pairs == 5
    assert report.total_captured_true_pairs == 4
    assert report.total_missed_true_pairs == 1
    assert report.pair_recall == .8
    assert report.average_candidates_per_record == 3
    assert report.maximum_candidates == 6
    assert report.candidate_count_percentiles == {
        "50%": 2, "90%": pytest.approx(5.2), "95%": pytest.approx(5.6),
        "99%": pytest.approx(5.92), "99.9%": pytest.approx(5.992), "maximum": 6,
    }
    assert (report.true_S2_pairs, report.captured_S2_pairs, report.S2_recall) == (3, 3, 1)
    assert (report.true_S3_pairs, report.captured_S3_pairs, report.S3_recall) == (2, 1, .5)
    expected = {"M1": (3, 3, 2, 2), "M2": (3, 3, 2, 2), "M4": (2, 1, 1, 0),
                "M5": (1, 1, 0, 0), "M6": (2, 1, 1, 0)}
    keys = ("raw_candidate_count", "new_candidate_count", "raw_captured", "new_captured")
    assert report.block_contributions == {label: dict(zip(keys, values)) for label, values in expected.items()}
    assert report.missed_pairs == (MissedPair(1, "S3-6"),)
    assert evaluate_pipeline(pipeline, records, lambda r: r["truth"],
                             include_block_contributions=True, max_missed_pairs=3) == report


@pytest.mark.parametrize("candidates,truth,counts,recall", [
    (["S2-1", "S2-1"], ["S2-1", "S2-1"], (1, 1, 1, 0), 1),
    (["S3-1"], ["S3-1"], (1, 1, 1, 0), 1),
    (["S2-1"], ["S2-1", "S3-1"], (1, 2, 1, 1), .5),
    ([], ["S2-1", "S3-1"], (0, 2, 0, 2), 0),
    (["S2-1"], [], (1, 0, 0, 0), None),
    ([], None, (0, 0, 0, 0), None),
])
def test_single_record_edge_cases(candidates, truth, counts, recall):
    report = evaluate_pipeline(CandidatePipeline({"block": lambda r: candidates}), [object()], lambda r: truth)
    assert (report.total_candidate_pairs, report.total_true_pairs,
            report.total_captured_true_pairs, report.total_missed_true_pairs) == counts
    assert report.pair_recall == recall
    assert report.block_contributions is None
    assert report.missed_pairs == ()
    assert all(value == counts[0] for value in report.candidate_count_percentiles.values())
    if truth and all(target.startswith("S2-") for target in truth):
        assert report.S3_recall is None
    if truth and all(target.startswith("S3-") for target in truth):
        assert report.S2_recall is None


def test_empty_stream_never_runs_callbacks():
    def forbidden(record):
        raise AssertionError("no records")
    report = evaluate_pipeline(CandidatePipeline({"block": forbidden}), iter(()), forbidden,
                               include_block_contributions=True)
    assert report.total_records == report.total_candidate_pairs == report.total_true_pairs == 0
    assert report.total_captured_true_pairs == report.total_missed_true_pairs == 0
    assert report.pair_recall is report.S2_recall is report.S3_recall is None
    assert report.average_candidates_per_record == report.maximum_candidates == 0
    assert all(value == 0 for value in report.candidate_count_percentiles.values())
    assert report.block_contributions == {}


@pytest.mark.parametrize("limit", [0, 1, 3, 10])
def test_first_n_sample_stream_then_lexical_order(limit):
    truth = ["S3-2", "S2-2", "S2-10", "S3-2"]
    pipeline = CandidatePipeline({})
    report = evaluate_pipeline(pipeline, ["A", "B"], lambda r: iter(truth), max_missed_pairs=limit)
    expected = tuple(MissedPair(i, target) for i in range(2) for target in ["S2-10", "S2-2", "S3-2"])
    assert report.missed_pairs == expected[:limit]
    assert report.total_true_pairs == report.total_missed_true_pairs == 6
    reverse = evaluate_pipeline(pipeline, ["A", "B"], lambda r: reversed(truth), max_missed_pairs=limit)
    assert reverse == report


def test_streaming_releases_previous_results_and_calls_truth_once():
    pipeline = CandidatePipeline({"block": lambda record: ["S2-1"]})
    generate = pipeline.generate
    previous = None
    seen = []

    def checked_generate(record):
        nonlocal previous
        assert previous is None or previous() is None
        result = generate(record)
        previous = weakref.ref(result)
        return result

    pipeline.generate = checked_generate

    def records():
        for i in range(5):
            assert len(seen) == i
            assert previous is None or previous() is None
            yield i

    def truth(record):
        seen.append(record)
        return (target for target in ["S2-1"])

    report = evaluate_pipeline(pipeline, records(), truth, include_block_contributions=True)
    assert seen == list(range(5))
    assert previous() is None
    assert report.total_records == report.total_captured_true_pairs == 5


@pytest.mark.parametrize("value", [-1, True, 1.5, None, "2"])
def test_invalid_sample_limits(value):
    with pytest.raises(ValueError, match="max_missed_pairs"):
        evaluate_pipeline(CandidatePipeline({}), [], lambda r: [], max_missed_pairs=value)


def test_errors_propagate_and_no_unknown_truth_is_dropped():
    with pytest.raises(ValueError, match="prefix"):
        evaluate_pipeline(CandidatePipeline({}), [1], lambda r: ["S4-1"])
    with pytest.raises(TypeError):
        evaluate_pipeline(CandidatePipeline({}), [], None)
    with pytest.raises(TypeError):
        evaluate_pipeline(CandidatePipeline({}), [], lambda r: [], include_block_contributions=1)

    def broken(record):
        raise RuntimeError("failure")

    with pytest.raises(RuntimeError, match="failure"):
        evaluate_pipeline(CandidatePipeline({"broken": broken}), [1], lambda r: [])
    with pytest.raises(RuntimeError, match="failure"):
        evaluate_pipeline(CandidatePipeline({}), [1], broken)
