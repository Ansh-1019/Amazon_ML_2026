from decimal import Decimal
from types import MappingProxyType

import pytest

from src.address import address_fingerprint
from src.exact_blocking import build_exact_index, generate_exact_candidates
from src.normalization import normalize_name


def test_basic_lookup_shared_values_and_duplicate_ids():
    index = build_exact_index(
        ["S2-2", "S2-1", "S2-1", "S2-1", "S3-1"],
        ["alpha", "alpha", "alpha", "beta", "alpha"],
    )
    assert index == {"alpha": ("S2-1", "S2-2", "S3-1"), "beta": ("S2-1",)}
    assert generate_exact_candidates("alpha", index) == ["S2-1", "S2-2", "S3-1"]
    assert generate_exact_candidates("beta", index) == ["S2-1"]
    assert generate_exact_candidates("unknown", index) == []


@pytest.mark.parametrize("missing", [None, "", float("nan"), Decimal("NaN"), Decimal("sNaN")])
def test_missing_values_never_form_buckets_or_candidates(missing):
    index = build_exact_index(["S2-1", "S2-2"], [missing, "alpha"])
    assert index == {"alpha": ("S2-2",)}
    assert generate_exact_candidates(missing, index) == []
    assert generate_exact_candidates(missing, {"": ("S2-bad",)}) == []


def test_empty_inputs():
    assert build_exact_index([], []) == {}
    assert generate_exact_candidates("alpha", {}) == []


def test_order_independence_and_repeated_queries():
    ids = ["S2-2", "S2-10", "S3-1", "S2-1"]
    values = ["beta", "beta", "alpha", "beta"]
    forward = build_exact_index(ids, values)
    reverse = build_exact_index(reversed(ids), reversed(values))
    assert list(forward.items()) == list(reverse.items()) == [
        ("alpha", ("S3-1",)), ("beta", ("S2-1", "S2-10", "S2-2")),
    ]
    for index in (forward, reverse):
        for _ in range(3):
            assert generate_exact_candidates("beta", index) == ["S2-1", "S2-10", "S2-2"]


def test_generator_and_iterator_inputs():
    ids = (f"S2-{i}" for i in [2, 1, 1])
    values = iter(["alpha", "alpha", "beta"])
    assert build_exact_index(ids, values) == {
        "alpha": ("S2-1", "S2-2"), "beta": ("S2-1",),
    }
    assert list(ids) == list(values) == []


@pytest.mark.parametrize("ids,values", [
    (["S2-1"], []), ([], ["alpha"]),
    (["S2-1", "S2-2"], [None]), (["S2-1"], ["", None]),
])
def test_misaligned_lengths(ids, values):
    with pytest.raises(ValueError):
        build_exact_index(iter(ids), iter(values))


@pytest.mark.parametrize("entity_id", [None, 1, True, 1.5, b"S2-1", [], {}])
def test_invalid_ids_are_rejected_even_with_missing_values(entity_id):
    for value in ("alpha", None, ""):
        with pytest.raises(TypeError, match="entity IDs must be strings"):
            build_exact_index([entity_id], [value])


@pytest.mark.parametrize("value", [0, False, 1.5, float("inf"), Decimal("1"), b"alpha", [], {}])
def test_invalid_values_fail_without_implicit_string_conversion(value):
    with pytest.raises(TypeError, match="exact values must be strings or missing"):
        build_exact_index(["S2-1"], [value])
    with pytest.raises(TypeError, match="exact values must be strings or missing"):
        generate_exact_candidates(value, {})


def test_no_normalization_is_performed():
    values = ["Alpha", "alpha", " alpha ", "alpha.", "  ", "NA", "Café", "Cafe\u0301"]
    ids = [f"S2-{i}" for i in range(len(values))]
    index = build_exact_index(ids, values)
    assert len(index) == len(values)
    for entity_id, value in zip(ids, values):
        assert generate_exact_candidates(value, index) == [entity_id]
    assert generate_exact_candidates("ALPHA", index) == []


def test_lookup_does_not_mutate_index_or_expose_postings():
    index = build_exact_index(["S2-2", "S2-1"], ["alpha", "alpha"])
    before = dict(index)
    readonly = MappingProxyType(index)
    candidates = generate_exact_candidates("alpha", readonly)
    candidates.append("S2-new")
    candidates.reverse()
    assert generate_exact_candidates("unknown", readonly) == []
    assert generate_exact_candidates(None, readonly) == []
    assert index == before
    assert generate_exact_candidates("alpha", readonly) == ["S2-1", "S2-2"]


def test_name_normalization_integration():
    raw_names = ["ACME Pvt. Ltd.", "Acme Private Limited", "Other Corp.", None, "  "]
    index = build_exact_index(
        ["S2-2", "S3-1", "S2-3", "S2-4", "S3-5"],
        (normalize_name(name) for name in raw_names),
    )
    assert generate_exact_candidates(normalize_name("acme pvt ltd"), index) == ["S2-2", "S3-1"]
    assert generate_exact_candidates(normalize_name("Other Corporation"), index) == ["S2-3"]
    assert generate_exact_candidates(normalize_name(None), index) == []
    assert "" not in index


def test_address_fingerprint_integration():
    raw_addresses = ["12A, Main St., Apt. 4B", "１２Ａ/Main/St/Apt/４Ｂ",
                     "12A Main St Apt 5B", None, "  "]
    index = build_exact_index(
        ["S2-2", "S3-1", "S2-3", "S2-4", "S3-5"],
        (address_fingerprint(address) for address in raw_addresses),
    )
    query = address_fingerprint(" 12a MAIN st apt 4b ")
    assert generate_exact_candidates(query, index) == ["S2-2", "S3-1"]
    assert generate_exact_candidates(address_fingerprint("12A Main St Apt 5B"), index) == ["S2-3"]
    assert generate_exact_candidates(address_fingerprint("13A Main St Apt 4B"), index) == []
    assert generate_exact_candidates(address_fingerprint(None), index) == []
    assert "" not in index
