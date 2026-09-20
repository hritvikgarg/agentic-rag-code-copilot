import pytest
from pydantic import ValidationError

from copilot.models import Chunk, ChunkType
from copilot.models.chunk import sha256_text


def make_chunk(**overrides):
    content = overrides.pop("content", "x = 1")
    values = {
        "chunk_id": "0123456789abcdef",
        "repository_name": "repo",
        "file_path": "src/a.py",
        "source_sha256": "a" * 64,
        "language": "python",
        "chunk_type": ChunkType.LINE_WINDOW,
        "chunking_strategy": "line",
        "chunking_version": 1,
        "chunk_index": 0,
        "start_line": 1,
        "end_line": 1,
        "content": content,
        "content_sha256": sha256_text(content),
        "token_estimate": 3,
    }
    values.update(overrides)
    return Chunk(**values)


def test_valid_chunk_has_no_symbol_or_fragment_metadata_by_default():
    chunk = make_chunk()
    assert chunk.symbol_name is None
    assert chunk.qualified_name is None
    assert chunk.parent_class is None
    assert chunk.fragment_index is None and chunk.fragment_count is None
    assert chunk.is_fragment is False


def test_valid_fragment():
    chunk = make_chunk(fragment_index=1, fragment_count=3)
    assert chunk.is_fragment is True


def test_content_is_not_in_repr():
    assert "x = 1" not in repr(make_chunk())


@pytest.mark.parametrize(
    "overrides",
    [
        {"start_line": 0},
        {"start_line": 5, "end_line": 4},
        {"file_path": "/etc/passwd"},
        {"file_path": "../escape.py"},
        {"chunk_id": "short"},
        {"source_sha256": "xyz"},
        {"content_sha256": "0" * 64},  # does not match content
        {"chunk_index": -1},
        {"chunking_version": 0},
        {"fragment_index": 0},  # count missing
        {"fragment_count": 2},  # index missing
        {"fragment_index": 2, "fragment_count": 2},  # index out of range
        {"fragment_index": 0, "fragment_count": 2, "start_line": 1, "end_line": 2},
    ],
)
def test_invalid_chunks_are_rejected(overrides):
    with pytest.raises(ValidationError):
        make_chunk(**overrides)


def test_chunk_is_immutable():
    with pytest.raises(ValidationError):
        make_chunk().start_line = 2
