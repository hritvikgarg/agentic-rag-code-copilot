import numpy as np
import pytest

from copilot.embeddings import Embedder, EmbeddingInputError, HashEmbedder
from copilot.embeddings.base import TokenCountingEmbedder

TEXTS = ["def connect(db_url): pass", "class UserService: ...", "# Install with uv sync"]


@pytest.fixture
def embedder():
    return HashEmbedder(dimension=64)


def test_conforms_to_the_interfaces(embedder):
    assert isinstance(embedder, Embedder)
    assert isinstance(embedder, TokenCountingEmbedder)


def test_model_info_describes_the_fake_honestly(embedder):
    info = embedder.info
    assert (info.model_id, info.dimension, info.normalized) == ("fake/hash-embedder", 64, True)
    assert info.runtime == "python-hash"
    assert info.max_input_tokens is None


def test_one_input_gives_one_vector(embedder):
    assert embedder.embed_documents(["x = 1"]).shape == (1, 64)


def test_multiple_inputs_preserve_order(embedder):
    batch = embedder.embed_documents(TEXTS)
    for i, text in enumerate(TEXTS):
        assert np.array_equal(batch[i], embedder.embed_documents([text])[0])
    reversed_batch = embedder.embed_documents(TEXTS[::-1])
    assert np.array_equal(reversed_batch[0], batch[2])


def test_dimension_is_stable_and_values_finite_and_unit_length(embedder):
    for size in (1, 2, 7, 50):
        vectors = embedder.embed_documents([f"text number {i}" for i in range(size)])
        assert vectors.shape == (size, 64)
        assert np.isfinite(vectors).all()
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
        assert vectors.dtype == np.float32


def test_repeated_input_is_deterministic(embedder):
    assert np.array_equal(embedder.embed_documents(TEXTS), embedder.embed_documents(TEXTS))
    assert np.array_equal(HashEmbedder(64).embed_documents(TEXTS), embedder.embed_documents(TEXTS))


def test_identical_texts_in_one_batch_get_identical_vectors(embedder):
    a, b = embedder.embed_documents(["same text", "same text"])
    assert np.array_equal(a, b)


def test_query_and_document_embeddings_share_dimension_and_space(embedder):
    query = embedder.embed_query(TEXTS[0])
    assert query.shape == (64,)
    assert np.array_equal(query, embedder.embed_documents([TEXTS[0]])[0])  # symmetric model


def test_shared_words_mean_higher_cosine_than_unrelated_text(embedder):
    query = embedder.embed_query("database connection url")
    related, unrelated = embedder.embed_documents(
        ["open the database connection using the url", "render the html template footer"]
    )
    assert float(query @ related) > float(query @ unrelated)


def test_empty_batch_is_defined(embedder):
    out = embedder.embed_documents([])
    assert out.shape == (0, 64)


@pytest.mark.parametrize("bad", ["", "   ", None, 3])
def test_invalid_query_is_rejected(embedder, bad):
    with pytest.raises(EmbeddingInputError):
        embedder.embed_query(bad)


def test_invalid_documents_are_rejected(embedder):
    with pytest.raises(EmbeddingInputError):
        embedder.embed_documents("a bare string")
    with pytest.raises(EmbeddingInputError):
        embedder.embed_documents(["ok", ""])


def test_punctuation_only_text_still_gets_a_valid_vector(embedder):
    vec = embedder.embed_query("!!! ???")
    assert np.isclose(np.linalg.norm(vec), 1.0)


def test_too_small_dimension_is_rejected():
    with pytest.raises(ValueError):
        HashEmbedder(dimension=4)


def test_count_tokens_uses_the_project_estimator(embedder):
    assert embedder.count_tokens(["def f(x):", "y"]) == [6, 1]
