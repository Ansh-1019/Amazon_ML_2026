import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse

from src.blocking.char_ngram_blocking import build_char_ngram_index, generate_char_ngram_candidates
from src.data.normalization import normalize_name


def test_basic_index_and_obvious_nearest_candidates():
    index = build_char_ngram_index(
        ["S2-2", "S2-1", "S3-1"], ["alpine bakery", "alpha finance", "zzzz"],
    )
    assert index.entity_ids == ("S2-1", "S2-2", "S3-1")
    assert generate_char_ngram_candidates("alpha finance", index, top_k=1) == ["S2-1"]
    assert generate_char_ngram_candidates("alpha finance", index, top_k=10) == ["S2-1", "S2-2"]
    assert generate_char_ngram_candidates("alpha financ", index, top_k=1) == ["S2-1"]


def test_character_configuration_retains_spaces_and_punctuation_without_padding():
    index = build_char_ngram_index(["1"], ["AB-C D"], ngram_range=(2, 2))
    assert set(index.vectorizer.vocabulary_) == {"ab", "b-", "-c", "c ", " d"}
    assert generate_char_ngram_candidates("ab", index, top_k=1) == ["1"]
    longer = build_char_ngram_index(["1"], ["abc"], ngram_range=(3, 3))
    assert generate_char_ngram_candidates("ab", longer, top_k=1) == []


@pytest.mark.parametrize("value", [None, (), (2,), (1, 2, 3), [2, 5], (0, 2),
                                   (-1, 2), (3, 2), (True, 2), (1, 2.5), ("2", 3)])
def test_invalid_ngram_ranges(value):
    with pytest.raises(ValueError, match="ngram_range"):
        build_char_ngram_index([], [], ngram_range=value)


@pytest.mark.parametrize("query", [None, "", " \t\n ", "漢字", "x"])
def test_empty_missing_and_no_overlap_queries(query):
    index = build_char_ngram_index(["1"], ["alpha"])
    assert generate_char_ngram_candidates(query, index, top_k=2) == []


@pytest.mark.parametrize("names,kwargs", [
    ([], {}), ([None, "", "  "], {}), (["a", "b"], {}),
    (["alpha", "zzzz"], {"min_df": 2}), (["alpha"], {"min_df": 2}),
])
def test_empty_and_pruned_vocabularies(names, kwargs):
    index = build_char_ngram_index((str(i) for i in range(len(names))), iter(names), **kwargs)
    assert index.vectorizer is None
    assert index.matrix.shape[1] == 0
    assert issparse(index.matrix)
    assert generate_char_ngram_candidates("alpha", index, top_k=1) == []


def test_deterministic_ties_and_reversed_input():
    ids = ["S2-2", "S2-10", "S3-1", "S2-1"]
    names = ["alpha"] * 4
    forward = build_char_ngram_index(ids, names)
    reverse = build_char_ngram_index(reversed(ids), reversed(names))
    assert forward.vectorizer.vocabulary_ == reverse.vectorizer.vocabulary_
    assert (forward.matrix != reverse.matrix).nnz == 0
    for index in (forward, reverse):
        for _ in range(3):
            assert generate_char_ngram_candidates("alpha", index, top_k=2) == ["S2-1", "S2-10"]
            assert generate_char_ngram_candidates("alpha", index, top_k=20) == sorted(ids)


@pytest.mark.parametrize("top_k", [None, 0, -1, True, False, 1.5, "2"])
def test_invalid_top_k_even_for_empty_query(top_k):
    index = build_char_ngram_index([], [])
    with pytest.raises(ValueError, match="top_k"):
        generate_char_ngram_candidates(None, index, top_k=top_k)


def test_top_k_is_required():
    with pytest.raises(TypeError):
        generate_char_ngram_candidates("alpha", build_char_ngram_index([], []))


def test_duplicate_ids_have_one_document_and_deterministic_name_choice():
    ids = ["1", "1", "1", "1", "2"]
    names = ["zulu", None, "alpha", "ALPHA", "beta"]
    index = build_char_ngram_index(ids, names)
    baseline = build_char_ngram_index(["1", "2"], ["alpha", "beta"])
    reverse = build_char_ngram_index(reversed(ids), reversed(names))
    assert index.entity_ids == ("1", "2")
    for other in (baseline, reverse):
        assert index.vectorizer.vocabulary_ == other.vectorizer.vocabulary_
        np.testing.assert_array_equal(index.vectorizer.idf_, other.vectorizer.idf_)
        assert (index.matrix != other.matrix).nnz == 0
    assert generate_char_ngram_candidates("alpha", index, top_k=10) == ["1"]
    assert generate_char_ngram_candidates("zulu", index, top_k=10) == []


def test_unicode_formatting_and_normalization_integration():
    names = ["ＣＡＦÉ Pvt. Ltd.", "श्री गणेश", "北京商贸", "O’Neil–Smith"]
    ids = ["1", "2", "3", "4"]
    index = build_char_ngram_index(ids, names)
    normalized = build_char_ngram_index(ids, map(normalize_name, names))
    assert index.vectorizer.vocabulary_ == normalized.vectorizer.vocabulary_
    assert (index.matrix != normalized.matrix).nnz == 0
    for query, expected in [("  Cafe\u0301 PRIVATE LIMITED ", "1"), ("गणेश", "2"),
                            ("北京商贸", "3"), ("O'Neil-Smith", "4"), ("O Neil Smith", "4")]:
        assert generate_char_ngram_candidates(query, index, top_k=1) == [expected]
        assert generate_char_ngram_candidates(normalize_name(query), index, top_k=1) == [expected]


def test_query_uses_fitted_vocabulary_idf_and_sparse_cosine(monkeypatch):
    index = build_char_ngram_index(["1", "2"], ["alpha", "beta"])
    vocabulary = dict(index.vectorizer.vocabulary_)
    idf = index.vectorizer.idf_.copy()
    matrix = index.matrix.copy()
    assert isinstance(index.matrix, csr_matrix)
    assert index.matrix.dtype == np.float64
    np.testing.assert_allclose(index.matrix.multiply(index.matrix).sum(axis=1), 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("query must not fit or densify")

    monkeypatch.setattr(index.vectorizer, "fit", forbidden)
    monkeypatch.setattr(index.vectorizer, "fit_transform", forbidden)
    monkeypatch.setattr(csr_matrix, "toarray", forbidden)
    assert generate_char_ngram_candidates("alpha 漢字", index, top_k=1) == ["1"]
    assert generate_char_ngram_candidates("漢字", index, top_k=1) == []
    assert index.vectorizer.vocabulary_ == vocabulary
    np.testing.assert_array_equal(index.vectorizer.idf_, idf)
    assert (index.matrix != matrix).nnz == 0


def test_min_df_and_max_features():
    for min_df in (2, 1.0):
        index = build_char_ngram_index(["1", "2"], ["abcd", "abxy"],
                                       ngram_range=(2, 2), min_df=min_df)
        assert set(index.vectorizer.vocabulary_) == {"ab"}
    index = build_char_ngram_index(["1", "2"], ["ababc", "abxyz"], max_features=1)
    assert len(index.vectorizer.vocabulary_) == 1
    reverse = build_char_ngram_index(["2", "1"], ["abxyz", "ababc"], max_features=1)
    assert index.vectorizer.vocabulary_ == reverse.vectorizer.vocabulary_


@pytest.mark.parametrize("value", [None, True, 0, -1, 1.5, float("nan"), float("inf"), "1"])
def test_invalid_min_df(value):
    with pytest.raises(ValueError, match="min_df"):
        build_char_ngram_index([], [], min_df=value)


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "2"])
def test_invalid_max_features(value):
    with pytest.raises(ValueError, match="max_features"):
        build_char_ngram_index([], [], max_features=value)


@pytest.mark.parametrize("ids,names", [(["1"], []), ([], ["alpha"])])
def test_misaligned_inputs(ids, names):
    with pytest.raises(ValueError):
        build_char_ngram_index(iter(ids), iter(names))


@pytest.mark.parametrize("entity_id", [None, 1, True, b"1"])
def test_invalid_ids(entity_id):
    with pytest.raises(TypeError, match="IDs must be strings"):
        build_char_ngram_index([entity_id], [None])
