"""The real fastembed + onnxruntime + tokenizers path, using a tiny synthetic ONNX model.

**This is not the Jina model.** ``tests/fixtures/stub_onnx_model.py`` builds a 768-dimensional
stand-in (fixed random lookup table, character-level tokenizer, ``model_max_length`` 32) that is
loaded through fastembed under the registry name of the real Jina code model. It proves our
adapter works with fastembed's real loading, tokenizer truncation/padding, mean pooling and
normalisation, offline. It says nothing about embedding quality or the real tokenizer.
"""

import numpy as np
import pytest

pytest.importorskip("onnx")

from fastembed import TextEmbedding  # noqa: E402

from copilot.chunking import LineChunker  # noqa: E402
from copilot.embeddings import (  # noqa: E402
    EmbeddingInputError,
    FastEmbedEmbedder,
    embed_chunks,
)
from tests.chunking_helpers import make_file  # noqa: E402
from tests.fixtures.stub_onnx_model import (  # noqa: E402
    DIMENSION,
    MODEL_MAX_LENGTH,
    build_stub_model_dir,
)

MODEL_ID = "jinaai/jina-embeddings-v2-base-code"


class LocalStubTextEmbedding:
    """Looks like the TextEmbedding class but loads the stub directory."""

    stub_dir: str = ""

    list_supported_models = staticmethod(TextEmbedding.list_supported_models)

    def __new__(cls, **kwargs):
        return TextEmbedding(specific_model_path=cls.stub_dir, **kwargs)


@pytest.fixture(scope="module")
def stub_dir(tmp_path_factory):
    return build_stub_model_dir(tmp_path_factory.mktemp("stub-model"))


@pytest.fixture
def embedder(stub_dir, tmp_path):
    LocalStubTextEmbedding.stub_dir = str(stub_dir)
    return FastEmbedEmbedder(
        MODEL_ID,
        cache_dir=tmp_path / "cache",
        batch_size=2,
        model_factory=lambda: LocalStubTextEmbedding,
    )


def test_registry_facts_used_by_the_adapter():
    entry = next(m for m in TextEmbedding.list_supported_models() if m["model"] == MODEL_ID)
    assert entry["dim"] == 768 and entry["model_file"] == "onnx/model.onnx"


def test_info_from_the_real_runtime(embedder):
    info = embedder.info
    assert (info.model_id, info.dimension, info.runtime) == (MODEL_ID, DIMENSION, "fastembed-onnx")
    assert info.max_input_tokens == MODEL_MAX_LENGTH  # read from the tokenizer's truncation
    assert info.runtime_version is not None


def test_runtime_limits_read_the_model_files(embedder):
    assert embedder.runtime_limits() == {
        "tokenizer_truncation_max_length": MODEL_MAX_LENGTH,
        "config_max_position_embeddings": MODEL_MAX_LENGTH,
        "tokenizer_config_model_max_length": MODEL_MAX_LENGTH,
    }


def test_single_and_multiple_inputs_shape_finite_unit_norm(embedder):
    one = embedder.embed_documents(["def f(): pass"])
    many = embedder.embed_documents(["a", "def f(): pass", "class X: ...", "y = 1", "zz"])
    assert one.shape == (1, DIMENSION) and many.shape == (5, DIMENSION)
    assert np.isfinite(many).all() and one.dtype == np.float32
    assert np.allclose(np.linalg.norm(many, axis=1), 1.0, atol=1e-5)


def test_batching_and_padding_do_not_change_vectors_or_order(embedder):
    texts = ["short", "a considerably longer piece of text", "mid length", "x", "another one!"]
    together = embedder.embed_documents(texts)  # 5 texts, batch size 2 -> 3 padded batches
    for i, text in enumerate(texts):
        alone = embedder.embed_documents([text])[0]
        assert np.allclose(together[i], alone, atol=1e-5), text


def test_repeated_input_is_deterministic(embedder):
    a = embedder.embed_documents(["same text", "other"])
    b = embedder.embed_documents(["same text", "other"])
    assert np.array_equal(a, b)


def test_different_texts_give_different_vectors(embedder):
    a, b = embedder.embed_documents(["alpha beta", "gamma delta"])
    assert not np.allclose(a, b)


def test_query_and_document_share_dimension_and_space(embedder):
    doc = embedder.embed_documents(["find the login"])[0]
    query = embedder.embed_query("find the login")
    assert query.shape == doc.shape == (DIMENSION,)
    assert np.allclose(query, doc, atol=1e-6)  # symmetric model: no query prefix


def test_input_beyond_the_model_limit_is_silently_truncated_by_fastembed(embedder):
    within = "x" * (MODEL_MAX_LENGTH - 2)  # fills the limit exactly, with [CLS] and [SEP]
    longer = within + "y" * 50
    a, b = embedder.embed_documents([within, longer])
    assert np.allclose(a, b, atol=1e-6)  # the extra characters never reached the model


def test_count_tokens_sees_the_full_length_and_does_not_disturb_embeddings(embedder):
    text = "z" * 100
    before = embedder.embed_documents([text])[0]
    assert embedder.count_tokens([text, "ab"]) == [102, 4]  # chars + [CLS] + [SEP], untruncated
    after = embedder.embed_documents([text])[0]
    assert np.array_equal(before, after)  # fastembed's own tokenizer still truncates identically
    assert embedder.count_tokens([text])[0] > MODEL_MAX_LENGTH


def test_blank_input_is_rejected(embedder):
    with pytest.raises(EmbeddingInputError):
        embedder.embed_documents(["fine", "  "])


def test_chunks_to_vectors_end_to_end(embedder):
    file = make_file("".join(f"value_{i} = {i}\n" for i in range(30)))
    chunks = LineChunker(size_lines=10, overlap_lines=2, max_tokens=512).chunk_file(file)
    result = embed_chunks(embedder, chunks, style="prefixed")
    assert result.vectors.shape == (len(chunks), DIMENSION)
    assert result.chunk_ids == tuple(c.chunk_id for c in chunks)
    assert result.model.model_id == MODEL_ID
