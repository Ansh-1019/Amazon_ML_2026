import pytest

from src.address import address_fingerprint
from src.blocking import build_token_index, generate_token_candidates
from src.candidate_union import union_candidates
from src.exact_blocking import build_exact_index, generate_exact_candidates
from src.normalization import normalize_name


def test_one_block():
    assert union_candidates(["S2-2", "S2-1"]) == ["S2-1", "S2-2"]


def test_multiple_blocks_with_duplicates_within_and_across_blocks():
    assert union_candidates(
        ["S2-1", "S2-1", "S2-2"], ["S2-2", "S2-3"], ["S2-4", "S2-1"],
    ) == ["S2-1", "S2-2", "S2-3", "S2-4"]


@pytest.mark.parametrize("blocks", [(), ([],), ([], (), set()), (None,), (None, [], None)])
def test_no_candidates(blocks):
    assert union_candidates(*blocks) == []


def test_empty_and_none_blocks_among_nonempty_blocks():
    assert union_candidates(None, [], ["S3-1"], (), None, ["S2-1"]) == ["S2-1", "S3-1"]


def test_deterministic_lexical_order_independent_of_block_and_id_order():
    blocks = [["S3-1", "S2-2"], ["S2-10", "S2-1", "S2-2"]]
    expected = ["S2-1", "S2-10", "S2-2", "S3-1"]
    for _ in range(3):
        assert union_candidates(*blocks) == expected
        assert union_candidates(*(reversed(block) for block in reversed(blocks))) == expected
        assert union_candidates(*(set(block) for block in blocks)) == expected


def test_ids_and_unrelated_string_values_are_preserved_exactly():
    values = ["S2-1", "S3-1", "s2-1", " S2-1 ", "S2-01", "", "unrelated", "é", "e\u0301"]
    assert union_candidates(values, values) == sorted(values)


def test_generators_and_iterators_are_consumed_once():
    first = (entity_id for entity_id in ["S2-2", "S2-1", "S2-1"])
    second = iter(["S2-2", "S3-1"])
    blocks = (block for block in [first, None, second])
    assert union_candidates(*blocks) == ["S2-1", "S2-2", "S3-1"]
    assert list(first) == list(second) == list(blocks) == []


def test_inputs_are_not_mutated_and_result_is_independent():
    first = ["S2-2", "S2-1", "S2-1"]
    second = {"S3-1", "S2-2"}
    result = union_candidates(first, second)
    result.append("S2-new")
    assert first == ["S2-2", "S2-1", "S2-1"]
    assert second == {"S3-1", "S2-2"}
    assert union_candidates(first, second) == ["S2-1", "S2-2", "S3-1"]


def test_large_synthetic_collections():
    first = (f"S2-{i}" for i in range(20000))
    second = (f"S2-{i}" for i in range(10000, 30000))
    third = (f"S3-{i}" for i in range(1000))
    expected = sorted([f"S2-{i}" for i in range(30000)] + [f"S3-{i}" for i in range(1000)])
    assert union_candidates(first, second, third) == expected


@pytest.mark.parametrize("value", [None, 1, True, 1.5, b"S2-1", [], {}])
def test_invalid_candidate_ids_are_rejected(value):
    with pytest.raises(TypeError, match="candidate IDs must be strings"):
        union_candidates(["S2-1", value])


@pytest.mark.parametrize("block", ["S2-1", b"S2-1", 1])
def test_invalid_blocks_are_rejected(block):
    with pytest.raises(TypeError):
        union_candidates(block)


def test_normalized_name_token_and_exact_address_integration():
    ids = ["S2-1", "S2-2", "S3-1", "S3-2"]
    names = ["Acme Pvt. Ltd.", "Acme Trading", "Other Corp.", "Unrelated"]
    addresses = ["12, Main St.", "99 Oak Road", "12/Main/St", "88 Elm Road"]
    query_name = "ACME Private Limited"

    # M1 supplies name normalization; M4 supplies the reusable exact lookup.
    name_index = build_exact_index(ids, map(normalize_name, names))
    name_candidates = generate_exact_candidates(normalize_name(query_name), name_index)
    token_index = build_token_index(ids, names)
    token_candidates = generate_token_candidates(query_name, token_index)
    address_index = build_exact_index(ids, map(address_fingerprint, addresses))
    address_candidates = generate_exact_candidates(address_fingerprint("12 MAIN ST"), address_index)

    assert name_candidates == ["S2-1"]
    assert token_candidates == ["S2-1", "S2-2"]
    assert address_candidates == ["S2-1", "S3-1"]
    assert union_candidates(name_candidates, token_candidates, address_candidates) == [
        "S2-1", "S2-2", "S3-1",
    ]
