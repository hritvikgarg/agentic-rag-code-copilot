import pytest
from pydantic import ValidationError

from copilot.models import Chunk, ChunkType
from copilot.models.chunk import make_chunk_id, sha256_text


def make_chunk(**overrides):
    content = overrides.pop("content", "x = 1")
    values = {
        "chunk_id": "0123456789abcdef",
        "repository_name": "repo",
        "file_path": "src/a.py",
        "language": "python",
        "chunk_type": ChunkType.CODE_WINDOW,
        "chunking_strategy": "line",
        "chunk_index": 0,
        "start_line": 1,
        "end_line": 1,
        "content": content,
        "content_sha256": sha256_text(content),
        "token_estimate": 3,
    }
    values.update(overrides)
    return Chunk(**values)


def test_valid_chunk_has_no_symbol_metadata_by_default():
    chunk = make_chunk()
    assert chunk.symbol_name is None
    assert chunk.qualified_name is None
    assert chunk.parent_class is None
    assert chunk.heading is None
    assert chunk.line_fragment is False


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
        {"content_sha256": "0" * 64},  # does not match content
        {"chunk_index": -1},
        {"line_fragment": True, "start_line": 1, "end_line": 2},
    ],
)
def test_invalid_chunks_are_rejected(overrides):
    with pytest.raises(ValidationError):
        make_chunk(**overrides)


def test_chunk_is_immutable():
    with pytest.raises(ValidationError):
        make_chunk().start_line = 2


def test_chunk_id_is_deterministic_and_sensitive_to_every_field():
    base = {
        "repository_name": "r",
        "chunking_strategy": "line",
        "file_path": "a.py",
        "chunk_index": 0,
        "start_line": 1,
        "end_line": 2,
        "content_sha256": "0" * 64,
    }
    first = make_chunk_id(**base)
    assert first == make_chunk_id(**base)
    assert len(first) == 16
    for key, other in [
        ("repository_name", "r2"),
        ("chunking_strategy", "ast"),
        ("file_path", "b.py"),
        ("chunk_index", 1),
        ("start_line", 2),
        ("end_line", 3),
        ("content_sha256", "1" * 64),
    ]:
        assert make_chunk_id(**{**base, key: other}) != first, key
