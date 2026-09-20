"""Query / top_k validation and the embedder-compatibility check of ``Retriever``."""

import numpy as np
import pytest

from copilot.embeddings import HashEmbedder
from copilot.models.embedding import EmbeddingModelInfo
from copilot.retrieval import (
    MAX_QUERY_CHARS,
    MAX_TOP_K,
    ChunkSource,
    EmbedderMismatchError,
    QueryError,
    Retriever,
    retrieve,
    validate_query,
    validate_top_k,
)
from tests.vectorstore_helpers import build_test_index


class CountingEmbedder(HashEmbedder):
    """A fake embedder that records how often it is asked to embed a query."""

    def __init__(self, dimension: int = 32) -> None:
        super().__init__(dimension=dimension)
        self.queries: list[str] = []

    def embed_query(self, text: str) -> np.ndarray:
        self.queries.append(text)
        return super().embed_query(text)


@pytest.fixture
def parts():
    embedder = CountingEmbedder(dimension=32)
    index, chunks, _ = build_test_index(embedder=embedder)
    source = ChunkSource({c.chunk_id: c for c in chunks}, repository_name="repo")
    return index, source, embedder, chunks


@pytest.fixture
def retriever(parts):
    index, source, embedder, _ = parts
    return Retriever(index=index, embedder=embedder, source=source)


@pytest.mark.parametrize("query", ["", " ", "   ", "\n", "\t \n", " "])
def test_an_empty_or_blank_query_is_refused(query):
    with pytest.raises(QueryError, match="empty|whitespace"):
        validate_query(query)


@pytest.mark.parametrize("query", [None, 123, 1.5, b"bytes", ["a"], {"q": 1}, object()])
def test_a_non_string_query_is_refused(query):
    with pytest.raises(QueryError, match="str"):
        validate_query(query)


def test_an_over_long_query_is_refused_and_the_limit_itself_is_allowed():
    assert validate_query("a" * MAX_QUERY_CHARS) == "a" * MAX_QUERY_CHARS
    with pytest.raises(QueryError, match="too long"):
        validate_query("a" * (MAX_QUERY_CHARS + 1))


def test_a_valid_query_is_returned_stripped():
    assert validate_query("  where is x?\n") == "where is x?"


@pytest.mark.parametrize("top_k", [0, -1, MAX_TOP_K + 1, 10_000])
def test_out_of_range_top_k_is_refused(top_k):
    with pytest.raises(QueryError, match="between 1 and"):
        validate_top_k(top_k)


@pytest.mark.parametrize("top_k", [1.0, 2.5, "5", None, True, False, [3]])
def test_non_integer_top_k_is_refused(top_k):
    with pytest.raises(QueryError, match="integer"):
        validate_top_k(top_k)


@pytest.mark.parametrize("top_k", [1, 2, 5, MAX_TOP_K])
def test_valid_top_k_is_accepted(top_k):
    assert validate_top_k(top_k) == top_k


def test_the_documented_bound_matches_the_settings_bound():
    from copilot.config import Settings

    with pytest.raises(ValueError):
        Settings(_env_file=None, retrieval_top_k=MAX_TOP_K + 1)
    assert Settings(_env_file=None, retrieval_top_k=MAX_TOP_K).retrieval_top_k == MAX_TOP_K


def test_query_errors_are_value_errors_and_retrieval_errors():
    from copilot.retrieval import RetrievalError

    assert issubclass(QueryError, ValueError) and issubclass(QueryError, RetrievalError)


@pytest.mark.parametrize("query", ["", "  ", None, 5])
def test_an_invalid_query_never_reaches_the_embedder(retriever, parts, query):
    embedder = parts[2]
    with pytest.raises(QueryError):
        retriever.retrieve(query, 3)
    assert embedder.queries == []


@pytest.mark.parametrize("top_k", [0, MAX_TOP_K + 1, "3", True])
def test_an_invalid_top_k_never_reaches_the_embedder(retriever, parts, top_k):
    embedder = parts[2]
    with pytest.raises(QueryError):
        retriever.retrieve("class B", top_k)
    assert embedder.queries == []


def test_the_query_is_embedded_stripped(retriever, parts):
    retriever.retrieve("  class B pass \n", 1)
    assert parts[2].queries == ["class B pass"]


def test_top_k_defaults_to_the_configured_value(parts):
    index, source, embedder, _ = parts
    retriever = Retriever(index=index, embedder=embedder, source=source, default_top_k=2)
    assert len(retriever.retrieve("some text")) == 2


def test_top_k_larger_than_the_index_returns_every_chunk(retriever, parts):
    index = parts[0]
    results = retriever.retrieve("some text", MAX_TOP_K)
    assert len(results) == index.count < MAX_TOP_K
    assert [r.rank for r in results] == list(range(1, index.count + 1))


def test_top_k_one_returns_one_result(retriever):
    (only,) = retriever.retrieve("class B pass", 1)
    assert only.rank == 1


def test_an_embedder_of_another_dimension_is_refused(parts):
    index, source, _, _ = parts
    with pytest.raises(EmbedderMismatchError, match="dimension 64"):
        Retriever(index=index, embedder=HashEmbedder(dimension=64), source=source)


class RenamedEmbedder(HashEmbedder):
    """Same dimension as the index, but it claims to be a different model."""

    @property
    def info(self) -> EmbeddingModelInfo:
        return super().info.model_copy(update={"model_id": "other/model"})


def test_an_embedder_of_another_model_is_refused_even_with_the_same_dimension(parts):
    index, source, _, _ = parts
    with pytest.raises(EmbedderMismatchError, match="other/model"):
        Retriever(index=index, embedder=RenamedEmbedder(dimension=32), source=source)


class UnnormalisedEmbedder(HashEmbedder):
    @property
    def info(self) -> EmbeddingModelInfo:
        return super().info.model_copy(update={"normalized": False})


def test_an_unnormalised_embedder_is_refused(parts):
    index, source, _, _ = parts
    with pytest.raises(EmbedderMismatchError, match="normalised"):
        Retriever(index=index, embedder=UnnormalisedEmbedder(dimension=32), source=source)


class BadVectorEmbedder(CountingEmbedder):
    def embed_query(self, text: str) -> np.ndarray:
        return np.full(32, 5.0, dtype=np.float32)  # finite but far from unit length


def test_an_unusable_query_vector_from_the_embedder_is_reported(parts):
    index, source, _, _ = parts
    bad = BadVectorEmbedder(dimension=32)
    with pytest.raises(EmbedderMismatchError, match="unusable query vector"):
        Retriever(index=index, embedder=bad, source=source).retrieve("x", 1)


def test_a_source_of_another_repository_is_refused(parts):
    index, _, embedder, chunks = parts
    other = ChunkSource({c.chunk_id: c for c in chunks}, repository_name="different")
    with pytest.raises(Exception, match="different repository"):
        Retriever(index=index, embedder=embedder, source=other)


def test_the_one_shot_function_validates_before_opening_anything(tmp_path):
    with pytest.raises(QueryError):
        retrieve("", 3, tmp_path / "no-repo", tmp_path / "no-index", embedder=HashEmbedder())
    with pytest.raises(QueryError):
        retrieve("ok", 0, tmp_path / "no-repo", tmp_path / "no-index", embedder=HashEmbedder())
