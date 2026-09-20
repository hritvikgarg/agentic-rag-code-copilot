"""``ChunkSource.get`` (hash verification) and ``create_chunker_from_params``."""

import pytest

from copilot.chunking import (
    ChunkingError,
    StrategyNotImplementedError,
    UnknownChunkingStrategyError,
    create_chunker_from_params,
)
from copilot.models.chunk import sha256_text
from copilot.retrieval import ChunkContentMismatchError, ChunkNotFoundError, ChunkSource
from tests.vectorstore_helpers import build_test_index

PARAMS = {"size_lines": 4, "overlap_lines": 1, "max_tokens": 512}


@pytest.fixture
def indexed():
    index, chunks, _ = build_test_index()
    return index, chunks


def test_get_returns_the_chunk_with_the_recorded_identity_and_text(indexed):
    index, chunks = indexed
    source = ChunkSource({c.chunk_id: c for c in chunks}, repository_name="repo")
    assert len(source) == len(chunks)
    for record, chunk in zip(index.records, chunks, strict=True):
        got = source.get(record)
        assert got is chunk
        assert sha256_text(got.content) == record.content_sha256


def test_a_chunk_missing_from_the_source_is_reported(indexed):
    index, chunks = indexed
    source = ChunkSource({c.chunk_id: c for c in chunks[1:]}, repository_name="repo")
    with pytest.raises(ChunkNotFoundError, match=index.records[0].chunk_id):
        source.get(index.records[0])


def _replace(chunks, position, **changes):
    tampered = list(chunks)
    tampered[position] = chunks[position].model_copy(update=changes)  # keeps the same chunk id
    return ChunkSource({c.chunk_id: c for c in tampered}, repository_name="repo")


def test_changed_content_with_the_same_chunk_id_is_refused(indexed):
    index, chunks = indexed
    new_text = chunks[0].content + "\ninjected = True"
    source = _replace(chunks, 0, content=new_text, content_sha256=sha256_text(new_text))
    with pytest.raises(ChunkContentMismatchError, match="no longer matches"):
        source.get(index.records[0])


def test_a_changed_source_hash_is_refused(indexed):
    index, chunks = indexed
    source = _replace(chunks, 0, source_sha256="0" * 64)
    with pytest.raises(ChunkContentMismatchError):
        source.get(index.records[0])


def test_a_moved_line_range_is_refused(indexed):
    index, chunks = indexed
    source = _replace(
        chunks, 0, start_line=chunks[0].start_line + 1, end_line=chunks[0].end_line + 1
    )
    with pytest.raises(ChunkContentMismatchError):
        source.get(index.records[0])


def test_a_different_file_path_is_refused(indexed):
    index, chunks = indexed
    source = _replace(chunks, 0, file_path="src/elsewhere.py")
    with pytest.raises(ChunkContentMismatchError):
        source.get(index.records[0])


def test_the_error_messages_carry_no_source_text(indexed):
    index, chunks = indexed
    new_text = "SECRET_SENTINEL = 1"
    source = _replace(chunks, 0, content=new_text, content_sha256=sha256_text(new_text))
    with pytest.raises(ChunkContentMismatchError) as caught:
        source.get(index.records[0])
    assert "SECRET_SENTINEL" not in str(caught.value)


def test_the_chunker_is_recreated_from_the_recorded_parameters():
    chunker = create_chunker_from_params("line", 1, PARAMS)
    assert chunker.name == "line" and chunker.version == 1 and chunker.params() == PARAMS


def test_a_different_recorded_algorithm_version_is_refused():
    with pytest.raises(ChunkingError, match="version"):
        create_chunker_from_params("line", 99, PARAMS)


@pytest.mark.parametrize(
    "params",
    [{}, {"size_lines": 4}, {**PARAMS, "size_lines": "many"}, {**PARAMS, "size_lines": 0}],
)
def test_missing_or_invalid_recorded_parameters_are_refused(params):
    with pytest.raises(ChunkingError):
        create_chunker_from_params("line", 1, params)


def test_unknown_and_reserved_strategies_are_refused():
    with pytest.raises(UnknownChunkingStrategyError):
        create_chunker_from_params("nope", 1, PARAMS)
    with pytest.raises(StrategyNotImplementedError):
        create_chunker_from_params("ast", 1, PARAMS)
