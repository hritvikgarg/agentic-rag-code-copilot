import numpy as np
import pytest

from copilot.chunking import LineChunker
from copilot.embeddings import REPRESENTATION_VERSION, HashEmbedder, embed_chunks
from copilot.embeddings.chunk_embeddings import ChunkEmbeddings
from copilot.embeddings.representation import embedding_text
from tests.chunking_helpers import make_file

TEXT = "".join(f"def f{i}(x):\n    return x + {i}\n" for i in range(20))


@pytest.fixture
def chunks():
    chunker = LineChunker(size_lines=8, overlap_lines=2, max_tokens=512)
    return chunker.chunk_file(make_file(TEXT))


def test_vectors_pair_with_chunk_ids_in_order(chunks):
    embedder = HashEmbedder(32)
    result = embed_chunks(embedder, chunks)
    assert result.chunk_ids == tuple(c.chunk_id for c in chunks)
    assert result.vectors.shape == (len(chunks), 32)
    for i, chunk in enumerate(chunks):  # row i really is the embedding of chunk i's text
        expected = embedder.embed_documents([embedding_text(chunk, "prefixed")])[0]
        assert np.array_equal(result.vectors[i], expected)


def test_provenance_is_recorded(chunks):
    result = embed_chunks(HashEmbedder(32), chunks, style="raw")
    assert result.text_style == "raw"
    assert result.representation_version == REPRESENTATION_VERSION
    assert result.model.model_id == "fake/hash-embedder" and result.model.dimension == 32


def test_raw_and_prefixed_styles_give_different_vectors(chunks):
    embedder = HashEmbedder(64)
    raw = embed_chunks(embedder, chunks, style="raw").vectors
    prefixed = embed_chunks(embedder, chunks, style="prefixed").vectors
    assert not np.array_equal(raw, prefixed)


def test_result_holds_ids_and_vectors_but_no_chunk_content(chunks):
    result = embed_chunks(HashEmbedder(32), chunks)
    assert not hasattr(result, "chunks") and not hasattr(result, "texts")
    assert "def f0" not in repr(result)


def test_empty_chunk_list_is_defined():
    result = embed_chunks(HashEmbedder(32), [])
    assert result.chunk_ids == () and result.vectors.shape == (0, 32)


def test_duplicate_chunk_ids_are_rejected(chunks):
    with pytest.raises(ValueError, match="unique"):
        embed_chunks(HashEmbedder(32), [chunks[0], chunks[0]])


def test_shape_mismatch_is_rejected(chunks):
    embedder = HashEmbedder(32)
    with pytest.raises(ValueError, match="shape"):
        ChunkEmbeddings(
            chunk_ids=("a",),
            vectors=np.zeros((2, 32), dtype=np.float32),
            model=embedder.info,
            text_style="raw",
            representation_version=1,
        )
