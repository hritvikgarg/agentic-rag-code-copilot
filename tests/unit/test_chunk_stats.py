import pytest

from copilot.chunking import LineChunker, chunk_repository
from copilot.chunking.stats import compute_stats, percentile
from copilot.ingestion import IngestionPolicy, ingest_repository


def test_percentile_nearest_rank():
    values = list(range(1, 101))
    assert percentile(values, 95) == 95.0
    assert percentile(values, 50) == 50.0
    assert percentile([7], 95) == 7.0
    assert percentile([], 95) == 0.0


def test_empty_stats_are_zero():
    stats = compute_stats([], strategy="line", total_files=3, elapsed_seconds=0.0)
    assert (stats.chunk_count, stats.files_chunked, stats.files_without_chunks) == (0, 0, 3)
    assert (stats.tokens_mean, stats.tokens_median, stats.tokens_max) == (0.0, 0.0, 0)


def test_stats_on_a_tiny_repository(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\ny = 2\nz = 3\n")
    (tmp_path / "empty.py").write_text("")
    (tmp_path / "doc.md").write_text("# T\ntext\n")
    ingestion = ingest_repository(tmp_path, IngestionPolicy())
    result = chunk_repository(
        ingestion, LineChunker(size_lines=10, overlap_lines=2, max_tokens=512)
    )
    stats = result.stats
    assert stats.strategy == "line"
    assert stats.chunk_count == 2
    assert stats.files_chunked == 2
    assert stats.files_without_chunks == 1  # empty.py
    assert stats.by_chunk_type == {"code_window": 1, "doc_section": 1}
    assert stats.by_language == {"markdown": 1, "python": 1}
    assert stats.tokens_max == 9  # "x = 1" is 3 tokens (x, =, 1) x 3 lines
    assert stats.tokens_mean == pytest.approx((9 + 3) / 2)  # doc chunk "# T\ntext" = 3 tokens
    assert "chunks=2" in stats.summary()
    assert result.chunks == tuple(sorted(result.chunks, key=lambda c: (c.file_path, c.chunk_index)))
