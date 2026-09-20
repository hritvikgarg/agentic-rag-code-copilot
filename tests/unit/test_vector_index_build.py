"""Building a FAISS-backed index: construction, mapping, ordering contract."""

import json

import numpy as np
import pytest

from copilot.embeddings import HashEmbedder
from copilot.embeddings.representation import embedding_text
from copilot.vectorstore import ChunkRecord, VectorIndex
from copilot.vectorstore.manifest import CHUNKS_FILE, INDEX_FILE
from copilot.vectorstore.records import chunk_ids_digest, decode_records, encode_records
from tests.vectorstore_helpers import build_test_index, make_chunker, make_files


def test_construction_counts_and_metadata():
    index, chunks, embedder = build_test_index()
    assert isinstance(index, VectorIndex)
    assert index.count == len(chunks) == len(index.records) == 6
    assert index.dimension == embedder.info.dimension == 32
    assert index.manifest.vector_count == index.manifest.chunk_count == len(chunks)
    assert index.spec.index_type == "IndexFlatIP"
    assert index.spec.metric == "inner_product"
    assert index.spec.embedding_normalized is True


def test_position_i_is_chunk_i_in_canonical_order():
    index, chunks, _ = build_test_index()
    assert [r.position for r in index.records] == list(range(len(chunks)))
    assert index.chunk_ids == tuple(c.chunk_id for c in chunks)
    keys = [(c.file_path, c.chunk_index) for c in chunks]
    assert keys == sorted(keys)
    for position, chunk in enumerate(chunks):
        assert index.record_at(position).chunk_id == chunk.chunk_id
        assert index.position_of(chunk.chunk_id) == position


def test_each_stored_vector_is_the_embedding_of_its_own_chunk():
    index, chunks, embedder = build_test_index()
    for position, chunk in enumerate(chunks):
        expected = embedder.embed_documents([embedding_text(chunk, "prefixed")])[0]
        assert np.array_equal(index.reconstruct(position), expected)


def test_raw_style_stores_different_vectors_than_prefixed():
    prefixed, _, _ = build_test_index(style="prefixed")
    raw, _, _ = build_test_index(style="raw")
    assert not np.array_equal(prefixed.reconstruct(0), raw.reconstruct(0))


def test_ordering_does_not_depend_on_the_order_files_are_supplied():
    contents = {f.relative_path: f.content for f in make_files()}
    forward, _, _ = build_test_index(contents)
    backward, _, _ = build_test_index(dict(reversed(list(contents.items()))))
    assert forward.chunk_ids == backward.chunk_ids
    assert forward.index_id == backward.index_id
    for i in range(forward.count):
        assert np.array_equal(forward.reconstruct(i), backward.reconstruct(i))


def test_low_level_self_search_finds_each_vector_at_its_own_position():
    index, _, _ = build_test_index()
    vectors = np.stack([index.reconstruct(i) for i in range(index.count)])
    scores, positions = index._search_positions(vectors, 1)
    assert positions[:, 0].tolist() == list(range(index.count))
    assert np.allclose(scores[:, 0], 1.0, atol=1e-5)  # cosine == inner product for unit vectors


def test_inner_product_equals_cosine_for_unit_vectors():
    index, _, _ = build_test_index()
    a, b = index.reconstruct(0), index.reconstruct(1)
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    scores, positions = index._search_positions(a[None, :], index.count)
    assert scores[0][positions[0].tolist().index(1)] == pytest.approx(cosine, abs=1e-6)


def test_records_carry_citation_metadata_but_no_source_text():
    index, chunks, _ = build_test_index()
    record = index.record_at(1)
    assert isinstance(record, ChunkRecord)
    chunk = chunks[1]
    assert (record.file_path, record.start_line, record.end_line) == (
        chunk.file_path,
        chunk.start_line,
        chunk.end_line,
    )
    assert record.content_sha256 == chunk.content_sha256
    assert record.source_sha256 == chunk.source_sha256
    assert "content" not in record.model_dump()
    sidecar = index._artifact_bytes[CHUNKS_FILE].decode()
    for chunk in chunks:
        assert chunk.content not in sidecar  # no source text duplicated into the sidecar
    assert "some text" not in sidecar and "return 1" not in sidecar


def test_position_lookups_fail_clearly():
    index, _, _ = build_test_index()
    with pytest.raises(IndexError):
        index.record_at(index.count)
    with pytest.raises(IndexError):
        index.record_at(-1)
    with pytest.raises(KeyError):
        index.position_of("0" * 16)


def test_fragment_chunks_are_recorded_with_their_fragment_fields():
    long_line = "x = '" + "abcdefgh " * 80 + "'\n"
    index, chunks, _ = build_test_index({"long.py": long_line}, chunker=make_chunker(max_tokens=64))
    fragments = [r for r in index.records if r.fragment_index is not None]
    assert fragments and all(r.fragment_count and r.start_line == r.end_line for r in fragments)
    assert index.count == len(chunks) > 1


def test_records_encode_decode_round_trip_and_are_canonical_bytes():
    index, _, _ = build_test_index()
    data = encode_records(index.records)
    assert decode_records(data) == list(index.records)
    assert data.endswith(b"\n") and b"\r" not in data
    assert encode_records(decode_records(data)) == data
    first = data.split(b"\n")[0].decode()
    assert first == first.strip() and '": ' not in first  # compact separators
    assert list(json.loads(first)) == sorted(json.loads(first))


def test_chunk_id_digest_detects_order_and_content():
    ids = ["a" * 16, "b" * 16, "c" * 16]
    assert chunk_ids_digest(ids) == chunk_ids_digest(list(ids))
    assert chunk_ids_digest(ids) != chunk_ids_digest(list(reversed(ids)))
    assert chunk_ids_digest(ids) != chunk_ids_digest(ids[:-1])


def test_manifest_artifact_entries_describe_the_bytes():
    index, _, _ = build_test_index()
    assert set(index.manifest.artifacts) == {INDEX_FILE, CHUNKS_FILE}
    for name, data in index._artifact_bytes.items():
        assert index.manifest.artifacts[name].size_bytes == len(data)


def test_larger_dimension_works():
    index, chunks, _ = build_test_index(embedder=HashEmbedder(dimension=768))
    assert index.dimension == 768 and index.count == len(chunks)
