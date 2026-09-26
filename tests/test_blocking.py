import pytest

from src.blocking import DEFAULT_STOPWORDS, build_token_index, generate_token_candidates


@pytest.fixture
def index():
    return build_token_index(
        ["S2-1", "S2-2", "S2-3"],
        ["alpha finance", "beta finance", "gamma health"],
    )


def test_basic_index_and_shared_tokens(index):
    assert index == {
        "alpha": ("S2-1",), "beta": ("S2-2",),
        "finance": ("S2-1", "S2-2"), "gamma": ("S2-3",),
        "health": ("S2-3",),
    }
    for query in ("alpha finance", "alpha finance group", "alpha alpha finance"):
        assert generate_token_candidates(query, index) == ["S2-1", "S2-2"]


@pytest.mark.parametrize("name", [None, "", "  ", "private limited company", "unknown", "x"])
def test_empty_or_unmatched_query(index, name):
    assert generate_token_candidates(name, index) == []


def test_stopwords_and_configuration():
    index = build_token_index(["1", "2"], ["Alpha Pvt. Ltd.", "Beta Company"])
    assert index == {"alpha": ("1",), "beta": ("2",)}
    assert generate_token_candidates("private limited company", index) == []
    assert build_token_index(["1"], [" ".join(sorted(DEFAULT_STOPWORDS))]) == {}
    custom = build_token_index(["1"], ["Alpha Limited"], stopwords={"alpha"})
    assert custom == {"limited": ("1",)}
    assert generate_token_candidates("Limited", custom, stopwords={"alpha"}) == ["1"]
    assert "limited" in build_token_index(["1"], ["Limited"], stopwords=set())


def test_duplicate_names_and_ids():
    index = build_token_index(
        ["2", "1", "1", "1"],
        ["alpha alpha finance", "alpha finance", "alpha finance", "health"],
    )
    assert index["alpha"] == ("1", "2")
    assert index["health"] == ("1",)
    assert generate_token_candidates("alpha alpha finance health", index) == ["1", "2"]


def test_determinism_and_cap():
    ids = ["S2-2", "S2-10", "S2-1", "S2-3"]
    names = ["alpha", "beta", "alpha beta", "beta"]
    forward = build_token_index(ids, names)
    reverse = build_token_index(reversed(ids), reversed(names))
    assert list(forward.items()) == list(reverse.items())
    for index in (forward, reverse):
        for query in ("alpha beta", "beta alpha alpha"):
            for _ in range(3):
                assert generate_token_candidates(query, index) == ["S2-1", "S2-10", "S2-2", "S2-3"]
                assert generate_token_candidates(query, index, max_candidates=2) == ["S2-1", "S2-10"]
            assert generate_token_candidates(query, index, max_candidates=0) == []
            assert len(generate_token_candidates(query, index, max_candidates=20)) == 4


def test_frequency_counts_distinct_ids_and_discards_permanently():
    ids = ["1", "1", "2", "3", "1"]
    names = ["services services alpha"] * 3 + ["services beta", "services"]
    index = build_token_index(iter(ids), iter(names), max_token_frequency=2)
    assert index == {"alpha": ("1", "2"), "beta": ("3",)}
    assert generate_token_candidates("services", index) == []
    assert "services" in build_token_index(ids, names)
    assert build_token_index(ids, names, max_token_frequency=0) == {}


def test_unicode_punctuation_and_one_character_tokens():
    index = build_token_index(
        ["1", "2", "3", "4"],
        ["Café-Tools", "श्री गणेश", "北京商贸有限公司", "R & D Solutions"],
    )
    assert generate_token_candidates("CAFE\u0301", index) == ["1"]
    assert generate_token_candidates("Cafe", index) == []
    assert generate_token_candidates("गणेश", index) == ["2"]
    assert generate_token_candidates("北京商贸有限公司", index) == ["3"]
    assert generate_token_candidates("R & D", index) == []
    assert generate_token_candidates("Solutions", index) == ["4"]
    short = build_token_index(["1"], ["R&D"], min_token_length=1)
    assert short == {"&": ("1",), "d": ("1",), "r": ("1",)}
    assert generate_token_candidates("R", short, min_token_length=1) == ["1"]


def test_empty_targets_and_missing_names():
    assert build_token_index([], []) == {}
    assert build_token_index(["1", "2"], [None, ""]) == {}
    assert generate_token_candidates("alpha", {}) == []


@pytest.mark.parametrize("ids,names", [(["1"], []), ([], ["alpha"])])
def test_misaligned_inputs(ids, names):
    with pytest.raises(ValueError):
        build_token_index(ids, names)


def test_invalid_id():
    with pytest.raises(TypeError, match="IDs must be strings"):
        build_token_index([None], ["alpha"])


@pytest.mark.parametrize("value", [-1, 1.5, True, "2"])
def test_invalid_limits(value):
    with pytest.raises(ValueError):
        build_token_index([], [], max_token_frequency=value)
    with pytest.raises(ValueError):
        generate_token_candidates("alpha", {}, max_candidates=value)


@pytest.mark.parametrize("value", [0, -1, None, 1.5, True])
def test_invalid_min_length(value):
    with pytest.raises(ValueError):
        build_token_index([], [], min_token_length=value)
    with pytest.raises(ValueError):
        generate_token_candidates("alpha", {}, min_token_length=value)
