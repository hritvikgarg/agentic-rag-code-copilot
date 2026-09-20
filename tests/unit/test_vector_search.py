"""``VectorIndex.search``: ranking, top_k, FAISS ``-1`` padding, ties, input validation."""

import numpy as np
import pytest

from copilot.embeddings import HashEmbedder
from copilot.vectorstore import IndexExpectation, SearchHit, SearchInputError, VectorIndex
from copilot.vectorstore import faiss_backend as backend
from tests.vectorstore_helpers import build_test_index

DUP_BODY = "value = 1\nother = 2\n"
TIE_FILES = {
    "src/a.py": "def only_in_a():\n    return 'alpha'\n",
    "src/dup1.py": DUP_BODY,
    "src/dup2.py": DUP_BODY,
    "src/dup3.py": DUP_BODY,
    "src/z.py": "class Z:\n    pass\n",
}


@pytest.fixture
def built():
    index, chunks, embedder = build_test_index()
    return index, chunks, embedder


def _all_vectors(index: VectorIndex) -> np.ndarray:
    return np.stack([index.reconstruct(i) for i in range(index.count)])


def test_results_are_ranked_best_first_and_match_a_brute_force_ranking(built):
    index, _, embedder = built
    query = embedder.embed_query("def a return 1 function")
    hits = index.search(query, index.count)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    brute = _all_vectors(index) @ query
    expected = sorted(range(index.count), key=lambda i: (-float(brute[i]), i))
    assert [h.position for h in hits] == expected
    for hit in hits:
        assert hit.score == pytest.approx(float(brute[hit.position]), abs=1e-5)


def test_every_stored_vector_finds_itself_first_with_cosine_one(built):
    index, _, _ = built
    for position in range(index.count):
        (hit,) = index.search(index.reconstruct(position), 1)
        assert hit.position == position
        assert hit.score == pytest.approx(1.0, abs=1e-5)


def test_scores_are_cosine_similarities_in_range(built):
    index, _, embedder = built
    hits = index.search(embedder.embed_query("class B pass"), index.count)
    assert all(-1.0 - 1e-5 <= h.score <= 1.0 + 1e-5 for h in hits)


def test_top_k_one_returns_exactly_one_hit(built):
    index, _, embedder = built
    hits = index.search(embedder.embed_query("x = 1"), 1)
    assert len(hits) == 1 and isinstance(hits[0], SearchHit)


def test_top_k_equal_to_the_index_size_returns_every_vector_once(built):
    index, _, embedder = built
    hits = index.search(embedder.embed_query("x = 1"), index.count)
    assert sorted(h.position for h in hits) == list(range(index.count))


def test_top_k_above_the_index_size_returns_fewer_hits_and_no_padding(built):
    index, _, embedder = built
    hits = index.search(embedder.embed_query("x = 1"), index.count + 50)
    assert len(hits) == index.count
    assert all(0 <= h.position < index.count for h in hits)
    assert len({h.position for h in hits}) == index.count


def test_a_smaller_top_k_is_a_prefix_of_a_larger_one(built):
    index, _, embedder = built
    query = embedder.embed_query("some text more text")
    full = index.search(query, index.count)
    for k in range(1, index.count + 1):
        assert index.search(query, k) == full[:k]


def test_faiss_itself_pads_with_minus_one_and_search_never_exposes_it(built):
    """The premise (FAISS pads with -1) and the guarantee (we drop it), in one test."""
    index, _, embedder = built
    query = embedder.embed_query("anything")
    _, raw_positions = backend.search(index._faiss, query[np.newaxis, :], index.count + 3)
    assert -1 in raw_positions[0]  # FAISS returns -1 for "no more results"
    hits = index.search(query, index.count + 3)
    assert all(h.position >= 0 for h in hits)


def test_padded_positions_are_filtered_even_if_the_backend_returns_them(built, monkeypatch):
    """A backend that pads results must never make the last sidecar row appear."""
    index, _, embedder = built

    def padded(_faiss_index, _queries, k):
        scores = np.full((1, k), -3.4e38, dtype=np.float32)
        positions = np.full((1, k), -1, dtype=np.int64)
        scores[0, :2], positions[0, :2] = [0.9, 0.5], [1, 0]
        return scores, positions

    monkeypatch.setattr(backend, "search", padded)
    hits = index.search(embedder.embed_query("q"), index.count)
    assert [h.position for h in hits] == [1, 0]  # the -1 pads are gone, not mapped to row -1


def test_exact_ties_are_ordered_by_position(monkeypatch):
    index, _, embedder = build_test_index(TIE_FILES, style="raw")
    dup_positions = [r.position for r in index.records if r.file_path.startswith("src/dup")]
    assert len(dup_positions) == 3
    query = index.reconstruct(dup_positions[0])
    hits = index.search(query, 3)
    assert [h.position for h in hits] == dup_positions  # ascending position among equal scores
    assert len({h.score for h in hits}) == 1


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_a_tie_at_the_top_k_boundary_keeps_the_lowest_positions(top_k):
    index, _, _ = build_test_index(TIE_FILES, style="raw")
    dup_positions = sorted(r.position for r in index.records if r.file_path.startswith("src/dup"))
    query = index.reconstruct(dup_positions[0])
    hits = index.search(query, top_k)
    assert [h.position for h in hits] == dup_positions[:top_k]


def test_the_result_is_identical_on_repeated_calls_and_after_save_and_load(tmp_path):
    index, _, embedder = build_test_index(TIE_FILES, style="raw")
    query = embedder.embed_query("value other")
    first = index.search(query, 4)
    assert index.search(query, 4) == first
    path = index.save(tmp_path / "indexes")
    loaded = VectorIndex.load(path, expected=IndexExpectation())
    assert loaded.search(query, 4) == first


def test_a_float64_query_is_accepted(built):
    index, _, embedder = built
    query = embedder.embed_query("def b return 2")
    assert index.search(query.astype(np.float64), 3) == index.search(query, 3)


def test_the_query_array_is_not_modified(built):
    index, _, embedder = built
    query = embedder.embed_query("def b return 2")
    before = query.copy()
    index.search(query, 3)
    assert np.array_equal(query, before)


def _query(dim: int, value: float = 1.0) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[0] = value
    return v


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(lambda d: [0.1] * d, id="list"),
        pytest.param(lambda d: None, id="none"),
        pytest.param(lambda d: "text", id="str"),
        pytest.param(lambda d: np.ones(d, dtype=np.int64), id="integer-dtype"),
        pytest.param(lambda d: np.ones((1, d), dtype=np.float32) / np.sqrt(d), id="two-d"),
        pytest.param(lambda d: _query(d + 1), id="wrong-dimension"),
        pytest.param(lambda d: _query(d - 1), id="short-dimension"),
        pytest.param(lambda d: np.zeros(d, dtype=np.float32), id="zero-vector"),
        pytest.param(lambda d: _query(d, 2.0), id="not-unit-length"),
        pytest.param(lambda d: _query(d, float("nan")), id="nan"),
        pytest.param(lambda d: _query(d, float("inf")), id="inf"),
    ],
)
def test_a_malformed_query_vector_is_refused(built, bad):
    index, _, _ = built
    with pytest.raises(SearchInputError):
        index.search(bad(index.dimension), 3)


@pytest.mark.parametrize("top_k", [0, -1, 2.5, "3", None, True])
def test_an_invalid_top_k_is_refused(built, top_k):
    index, _, embedder = built
    with pytest.raises(SearchInputError):
        index.search(embedder.embed_query("x"), top_k)


def test_search_input_errors_are_value_errors_and_vector_store_errors():
    from copilot.vectorstore import VectorStoreError

    assert issubclass(SearchInputError, ValueError)
    assert issubclass(SearchInputError, VectorStoreError)


def test_a_query_from_an_embedder_of_another_dimension_is_refused(built):
    index, _, _ = built
    other = HashEmbedder(dimension=64).embed_query("x")
    with pytest.raises(SearchInputError, match="dimension"):
        index.search(other, 1)
