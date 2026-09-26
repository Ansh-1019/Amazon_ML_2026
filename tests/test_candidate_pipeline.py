import pytest

from src.blocking.candidate_diagnostics import compare_candidate_blocks
from src.blocking.candidate_pipeline import CandidatePipeline, CandidateResult


def test_one_block_deduplicates_without_mutating_output():
    output = ["S2-2", "S2-1", "S2-1"]
    result = CandidatePipeline({"custom": lambda record: output}).generate("name")
    assert result == CandidateResult(
        ["S2-1", "S2-2"], {"custom": ["S2-1", "S2-2"]},
        {"S2-1": ("custom",), "S2-2": ("custom",)},
    )
    assert result.counts_by_block == {"custom": 2}
    result.all_candidates.append("other")
    result.candidates_by_block["custom"].append("new")
    assert output == ["S2-2", "S2-1", "S2-1"]


def test_no_blocks():
    result = CandidatePipeline({}).generate(None)
    assert result == CandidateResult([], {}, {})
    assert result.counts_by_block == {}


def test_all_blocks_empty():
    result = CandidatePipeline({"empty": lambda record: [], "missing": lambda record: None,
                                "generator": lambda record: iter(())}).generate(None)
    assert result == CandidateResult([], {"empty": [], "missing": [], "generator": []}, {})
    assert result.counts_by_block == {"empty": 0, "missing": 0, "generator": 0}


def test_call_order_input_identity_and_configuration_snapshot():
    calls = []
    record = {"name": "Raw NAME", "address": "12, Road"}

    def block(label):
        def generate(value):
            assert value is record
            calls.append(label)
            return ["S2-1"]
        return generate

    configuration = {"z": block("z"), "a": block("a")}
    pipeline = CandidatePipeline(configuration)
    configuration.clear()
    result = pipeline.generate(record)
    assert calls == ["z", "a"]
    assert list(result.candidates_by_block) == ["z", "a"]
    assert result.provenance == {"S2-1": ("z", "a")}
    assert record == {"name": "Raw NAME", "address": "12, Road"}


def test_input_adapters_with_bound_state():
    name_index = {"alpha": ["S2-1"]}
    address_index = {"12 road": ["S3-1"]}
    pipeline = CandidatePipeline({
        "name": lambda record: name_index.get(record["name"], []),
        "address": lambda record: address_index.get(record["address"], []),
    })
    assert pipeline.generate({"name": "alpha", "address": "12 road"}).all_candidates == ["S2-1", "S3-1"]


def test_ids_are_preserved_and_ordered_lexically():
    ids = ["S3-1", "S2-2", "S2-10", " S2-1 ", "s2-1", "", "é", "e\u0301"]
    result = CandidatePipeline({"arbitrary": lambda record: set(ids)}).generate(None)
    assert result.all_candidates == sorted(ids)
    assert list(result.provenance) == sorted(ids)


def test_multiple_blocks_synthetic_integration_and_m8():
    outputs = {
        "M1": ["S2-1", "S2-2", "S2-1"],
        "M2": ["S2-3", "S2-2"],
        "M4": ["S3-1", "S2-1"],
        "M5": [],
        "M6": ["S3-2", "S2-3", "S3-2"],
    }
    pipeline = CandidatePipeline({label: (lambda record, ids=ids: iter(ids))
                                  for label, ids in outputs.items()})
    expected = CandidateResult(
        ["S2-1", "S2-2", "S2-3", "S3-1", "S3-2"],
        {"M1": ["S2-1", "S2-2"], "M2": ["S2-2", "S2-3"],
         "M4": ["S2-1", "S3-1"], "M5": [], "M6": ["S2-3", "S3-2"]},
        {"S2-1": ("M1", "M4"), "S2-2": ("M1", "M2"),
         "S2-3": ("M2", "M6"), "S3-1": ("M4",), "S3-2": ("M6",)},
    )
    for _ in range(3):
        result = pipeline.generate("synthetic source")
        assert result == expected
        assert result.counts_by_block == {"M1": 2, "M2": 2, "M4": 2, "M5": 0, "M6": 2}
    report = compare_candidate_blocks(result.candidates_by_block, ["S2-1", "S3-2", "S3-9"])
    assert report["union"].candidate_count == 5
    assert report["union"].captured == 2
    assert report["union"].recall == 2 / 3


def test_results_do_not_leak_between_calls():
    pipeline = CandidatePipeline({"block": lambda record: [record]})
    first = pipeline.generate("S2-1")
    first.all_candidates.clear()
    first.candidates_by_block["block"].clear()
    first.provenance.clear()
    assert pipeline.generate("S3-1") == CandidateResult(
        ["S3-1"], {"block": ["S3-1"]}, {"S3-1": ("block",)},
    )


@pytest.mark.parametrize("delayed", [False, True])
def test_exceptions_propagate_and_later_blocks_do_not_run(delayed):
    failure = RuntimeError("block failed")
    calls = []

    def broken(record):
        raise failure

    def broken_iterator(record):
        yield "S2-1"
        raise failure

    pipeline = CandidatePipeline({"bad": broken_iterator if delayed else broken,
                                  "later": lambda record: calls.append("later")})
    with pytest.raises(RuntimeError) as exc:
        pipeline.generate(None)
    assert exc.value is failure
    assert calls == []


@pytest.mark.parametrize("output", [[None], [1], "S2-1", 1])
def test_invalid_block_output_is_rejected(output):
    with pytest.raises(TypeError):
        CandidatePipeline({"bad": lambda record: output}).generate(None)


@pytest.mark.parametrize("configuration", [None, [], {1: lambda record: []}, {"bad": []}])
def test_invalid_configuration(configuration):
    with pytest.raises(TypeError):
        CandidatePipeline(configuration)
