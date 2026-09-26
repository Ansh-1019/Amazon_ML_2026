from types import MappingProxyType

import pytest

from src.normalization import normalize_name, tokenize_name
from src.rare_blocking import build_rare_token_index, generate_rare_token_candidates


def test_basic_rare_and_common_tokens():
    index = build_rare_token_index(
        ["S2-1", "S2-2", "S3-1"],
        ["alpha services", "beta services", "gamma services"],
        max_token_frequency=2,
    )
    assert index == {"alpha": ("S2-1",), "beta": ("S2-2",), "gamma": ("S3-1",)}
    assert generate_rare_token_candidates("services", index) == []
    assert generate_rare_token_candidates("alpha gamma", index) == ["S2-1", "S3-1"]


def test_threshold_boundary_and_permanent_exclusion():
    ids = ["1", "2", "3"]
    names = ["shared"] * 3
    assert build_rare_token_index(ids, names, max_token_frequency=3) == {
        "shared": ("1", "2", "3"),
    }
    assert build_rare_token_index(
        ids + ["4", "1", "4", "5"], names + ["shared"] * 4,
        max_token_frequency=3,
    ) == {}


def test_distinct_entities_duplicates_and_repeated_tokens():
    index = build_rare_token_index(
        ["1", "1", "2", "2", "1"],
        ["alpha alpha", "alpha beta", "alpha", "alpha alpha", "gamma"],
        max_token_frequency=2,
    )
    assert index == {"alpha": ("1", "2"), "beta": ("1",), "gamma": ("1",)}
    assert generate_rare_token_candidates("alpha alpha beta gamma", index) == ["1", "2"]


@pytest.mark.parametrize("name", [None, "", " \t\n "])
def test_missing_and_empty_names(name):
    assert build_rare_token_index(["1"], [name], max_token_frequency=2) == {}
    assert generate_rare_token_candidates(name, {"alpha": ("1",)}) == []


def test_empty_unknown_and_zero_frequency():
    assert build_rare_token_index([], [], max_token_frequency=3) == {}
    assert build_rare_token_index(["1"], ["alpha"], max_token_frequency=0) == {}
    assert generate_rare_token_candidates("alpha", {}) == []
    assert generate_rare_token_candidates("unknown", {"alpha": ("1",)}) == []


def test_default_and_custom_stopwords():
    assert build_rare_token_index(
        ["1"], ["Alpha Pvt. Ltd."], max_token_frequency=1,
    ) == {"alpha": ("1",)}
    index = build_rare_token_index(
        ["1"], ["Alpha Limited"], max_token_frequency=1,
        stopwords=(word for word in ["alpha"]),
    )
    assert index == {"limited": ("1",)}
    assert generate_rare_token_candidates("Alpha Limited", index, stopwords={"alpha"}) == ["1"]
    assert build_rare_token_index(
        ["1"], ["Limited"], max_token_frequency=1, stopwords=[],
    ) == {"limited": ("1",)}
    assert generate_rare_token_candidates("Limited", {"limited": ("1",)}) == []


def test_minimum_length_and_punctuation_semantics():
    assert build_rare_token_index(["1"], ["R&D"], max_token_frequency=1) == {}
    index = build_rare_token_index(["1"], ["R&D"], max_token_frequency=1, min_token_length=1)
    assert index == {"&": ("1",), "d": ("1",), "r": ("1",)}
    assert generate_rare_token_candidates("R", index, min_token_length=1) == ["1"]
    assert generate_rare_token_candidates("R", index) == []


@pytest.mark.parametrize("cap,expected", [
    (None, ["S2-1", "S2-10", "S2-2", "S3-1"]), (0, []),
    (1, ["S2-1"]), (2, ["S2-1", "S2-10"]),
    (10, ["S2-1", "S2-10", "S2-2", "S3-1"]),
])
def test_caps_order_independence_and_no_mutation(cap, expected):
    ids = ["S2-2", "S2-10", "S2-1", "S3-1"]
    names = ["alpha common", "beta common", "alpha beta common", "beta common"]
    forward = build_rare_token_index(ids, names, max_token_frequency=3)
    reverse = build_rare_token_index(reversed(ids), reversed(names), max_token_frequency=3)
    assert list(forward.items()) == list(reverse.items())
    assert "common" not in forward
    for index in (forward, reverse):
        before = dict(index)
        for query in ("alpha beta", "beta alpha alpha"):
            for _ in range(3):
                result = generate_rare_token_candidates(query, MappingProxyType(index), max_candidates=cap)
                assert result == expected
                result.append("unrelated")
        assert index == before


def test_single_pass_generators():
    assert build_rare_token_index(
        (str(i) for i in [2, 1]), (name for name in ["alpha", "alpha"]),
        max_token_frequency=2,
    ) == {"alpha": ("1", "2")}


@pytest.mark.parametrize("ids,names", [(["1"], []), ([], ["alpha"]), (["1", "2"], [None])])
def test_misaligned_inputs(ids, names):
    with pytest.raises(ValueError):
        build_rare_token_index(iter(ids), iter(names), max_token_frequency=1)


@pytest.mark.parametrize("entity_id", [None, 1, True, b"1", []])
def test_invalid_ids(entity_id):
    with pytest.raises(TypeError, match="IDs must be strings"):
        build_rare_token_index([entity_id], [None], max_token_frequency=1)


@pytest.mark.parametrize("limit", [None, -1, 1.5, True, False, "3"])
def test_invalid_frequency_limits(limit):
    with pytest.raises(ValueError, match="max_token_frequency"):
        build_rare_token_index([], [], max_token_frequency=limit)


def test_frequency_limit_is_required():
    with pytest.raises(TypeError):
        build_rare_token_index([], [])


@pytest.mark.parametrize("limit", [-1, 1.5, True, False, "3"])
def test_invalid_candidate_limits(limit):
    with pytest.raises(ValueError, match="max_candidates"):
        generate_rare_token_candidates("alpha", {}, max_candidates=limit)


@pytest.mark.parametrize("limit", [None, 0, -1, 1.5, True, "2"])
def test_invalid_minimum_length(limit):
    with pytest.raises(ValueError, match="min_token_length"):
        build_rare_token_index([], [], max_token_frequency=1, min_token_length=limit)
    with pytest.raises(ValueError, match="min_token_length"):
        generate_rare_token_candidates("", {}, min_token_length=limit, max_candidates=0)


def test_existing_normalization_pipeline_integration():
    raw = ["ＣＡＦÉ Pvt. Ltd.", "Cafe\u0301 Private Limited", "श्री गणेश", "Other Ltd."]
    ids = ["S2-2", "S3-1", "S2-3", "S2-4"]
    normalized = [normalize_name(name) for name in raw]
    raw_index = build_rare_token_index(ids, raw, max_token_frequency=2)
    index = build_rare_token_index(ids, normalized, max_token_frequency=2)
    assert index == raw_index
    assert tokenize_name(raw[0]) == ["café", "private", "limited"]
    assert generate_rare_token_candidates(normalize_name("CAFÉ LTD."), index) == ["S2-2", "S3-1"]
    assert generate_rare_token_candidates("गणेश", index) == ["S2-3"]
    assert generate_rare_token_candidates("Cafe", index) == []
