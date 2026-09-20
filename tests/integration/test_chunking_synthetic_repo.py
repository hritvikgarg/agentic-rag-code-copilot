"""Chunking the ingested synthetic repository against hand-written expectations."""

import pytest

from copilot.chunking import LineChunker, chunk_repository
from copilot.ingestion import IngestionPolicy, ingest_repository
from tests.chunking_helpers import assert_chunk_invariants
from tests.fixtures.synthetic_repo import EXPECTED_ACCEPTED, SYNTHETIC_MAX_FILE_SIZE

MAX_TOKENS = 512
CHUNKER = LineChunker(size_lines=4, overlap_lines=1, max_tokens=MAX_TOKENS)

# Hand-derived from the fixture contents (see tests/fixtures/synthetic_repo.py), NOT from output:
# file -> [(start_line, end_line, chunk_type, heading)]
GOLDEN = {
    "README.md": [(1, 3, "doc_section", "Synthetic repo")],
    "docs/guide.md": [(1, 3, "doc_section", "Guide")],
    "src/app/__init__.py": [],  # empty file: nothing to index
    "src/app/main.py": [(1, 4, "code_window", None), (4, 7, "code_window", None)],
    "src/app/crlf_module.py": [(1, 4, "code_window", None), (4, 6, "code_window", None)],
    "src/app/utf8_bom.py": [(1, 1, "code_window", None)],
    "src/app/utf16_module.py": [(1, 2, "code_window", None)],
    "config/ci.yml": [(1, 2, "config_window", None)],
    "config/settings.json": [(1, 1, "config_window", None)],
    "web/index.js": [(1, 1, "code_window", None)],
}


@pytest.fixture
def ingestion(synthetic_repo):
    return ingest_repository(
        synthetic_repo, IngestionPolicy(max_file_size_bytes=SYNTHETIC_MAX_FILE_SIZE)
    )


@pytest.fixture
def chunked(ingestion):
    return chunk_repository(ingestion, CHUNKER)


def test_golden_chunk_boundaries(chunked):
    by_file: dict[str, list] = {}
    for c in chunked.chunks:
        by_file.setdefault(c.file_path, []).append(
            (c.start_line, c.end_line, c.chunk_type.value, c.heading)
        )
    for path, expected in GOLDEN.items():
        assert by_file.get(path, []) == expected, path


def test_invariants_hold_for_every_file(ingestion, chunked):
    for file in ingestion.files:
        chunks = [c for c in chunked.chunks if c.file_path == file.relative_path]
        assert_chunk_invariants(file, chunks, max_tokens=MAX_TOKENS)


def test_only_ingested_files_are_chunked_and_no_skipped_content_leaks(ingestion, chunked):
    assert {c.file_path for c in chunked.chunks} <= set(EXPECTED_ACCEPTED)
    joined = "\n".join(c.content for c in chunked.chunks)
    for forbidden in ("FAKE_SETTING", "fake pem placeholder", "fake ssh key placeholder"):
        assert forbidden not in joined


def test_chunking_is_deterministic_end_to_end(ingestion):
    first = chunk_repository(ingestion, CHUNKER)
    second = chunk_repository(ingestion, CHUNKER)
    assert first.chunks == second.chunks
    assert len({c.chunk_id for c in first.chunks}) == len(first.chunks)  # unique repo-wide


def test_output_is_ordered_by_file_then_position(chunked):
    keys = [(c.file_path, c.chunk_index) for c in chunked.chunks]
    assert keys == sorted(keys)


def test_only_the_empty_file_produces_no_chunks(chunked):
    assert chunked.stats.files_without_chunks == 1
    assert chunked.stats.files_chunked == len(EXPECTED_ACCEPTED) - 1


def test_multi_window_real_files_end_to_end(tmp_path):
    """Longer generated repo: many functions, a doc with fenced code, one giant line."""
    body = "\n".join(f"def f{i}(x):\n    return x + {i}\n" for i in range(60))
    (tmp_path / "long.py").write_text(body)
    (tmp_path / "wide.js").write_text(
        "var a = [" + ",".join(str(i) for i in range(3000)) + "];\nnext();\n"
    )
    (tmp_path / "GUIDE.md").write_text("# G\n```\n# not heading\n```\n## H\nx\n")
    ingestion = ingest_repository(tmp_path, IngestionPolicy())
    chunker = LineChunker(size_lines=30, overlap_lines=5, max_tokens=200)
    result = chunk_repository(ingestion, chunker)
    for file in ingestion.files:
        chunks = [c for c in result.chunks if c.file_path == file.relative_path]
        assert chunks, file.relative_path
        assert_chunk_invariants(file, chunks, max_tokens=200)
    assert result.stats.line_fragment_count > 5
    assert result.stats.tokens_max <= 200
