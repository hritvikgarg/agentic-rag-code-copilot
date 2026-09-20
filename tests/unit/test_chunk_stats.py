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
    stats = compute_stats(
        [], strategy="line", total_files=3, source_lines_total=0, elapsed_seconds=0.0
    )
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
    assert stats.by_chunk_type == {"line_window": 2}
    assert stats.by_language == {"markdown": 1, "python": 1}
    assert stats.tokens_max == 9  # "x = 1" is 3 tokens (x, =, 1) x 3 lines
    assert stats.tokens_mean == pytest.approx((9 + 3) / 2)  # doc chunk "# T\ntext" = 3 tokens
    assert "chunks=2" in stats.summary()
    assert result.chunks == tuple(sorted(result.chunks, key=lambda c: (c.file_path, c.chunk_index)))


def chunk_text(text, *, size, overlap, max_tokens=512):
    from tests.chunking_helpers import make_file

    file = make_file(text)
    chunker = LineChunker(size_lines=size, overlap_lines=overlap, max_tokens=max_tokens)
    chunks = chunker.chunk_file(file)
    return compute_stats(
        chunks,
        strategy="line",
        total_files=1,
        source_lines_total=text.count("\n"),
        elapsed_seconds=0.0,
    )


def test_overlap_duplicates_are_reported_separately_from_unique_lines():
    # 25 lines, size 10, overlap 3 -> windows (1,10) (8,17) (15,24) (22,25): 10+10+10+4 line slots
    stats = chunk_text("x = 1\n" * 25, size=10, overlap=3)
    assert stats.chunk_count == stats.whole_line_chunk_count == 4
    assert stats.source_lines_total == 25
    assert stats.unique_source_lines_represented == 25
    assert stats.overlap_duplicated_lines == 34 - 25
    assert (stats.fragment_chunk_count, stats.fragmented_line_count) == (0, 0)


def test_no_overlap_means_no_duplicated_lines():
    stats = chunk_text("x = 1\n" * 25, size=10, overlap=0)
    assert stats.unique_source_lines_represented == 25
    assert stats.overlap_duplicated_lines == 0


def test_fragments_are_not_counted_as_extra_physical_lines():
    long_line = "v = [" + ", ".join(["0"] * 300) + "]"
    text = "a = 1\n" * 4 + long_line + "\n" + "b = 2\n" * 4  # 9 physical lines, line 5 is long
    stats = chunk_text(text, size=10, overlap=2, max_tokens=40)
    assert stats.source_lines_total == 9
    assert stats.fragmented_line_count == 1
    assert stats.fragment_chunk_count > 3  # many fragments...
    assert stats.unique_source_lines_represented == 9  # ...but line 5 is counted once
    assert stats.chunk_count == stats.whole_line_chunk_count + stats.fragment_chunk_count
    # hand-computed: whole-line windows are (1,4) and (6,9); the long line 5 is only fragments
    assert stats.whole_line_chunk_count == 2
    assert stats.overlap_duplicated_lines == 0


def test_unrepresented_lines_are_blank_lines_in_dropped_windows():
    text = "a = 1\n" + "\n" * 20 + "b = 2\n"  # 22 lines, 20 of them blank
    stats = chunk_text(text, size=5, overlap=1)
    assert stats.source_lines_total == 22
    assert stats.unique_source_lines_represented < 22
