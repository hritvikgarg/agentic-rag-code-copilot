"""Chunk statistics: counts and token-size distribution."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Sequence

from copilot.models.chunk import Chunk, ChunkStats


def percentile(sorted_values: Sequence[int], q: float) -> float:
    """Nearest-rank percentile of already-sorted values (``q`` in 0..100; 0.0 when empty)."""
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil(q / 100 * len(sorted_values)))
    return float(sorted_values[rank - 1])


def compute_stats(
    chunks: Sequence[Chunk], *, strategy: str, total_files: int, elapsed_seconds: float
) -> ChunkStats:
    """Summarise ``chunks`` produced from ``total_files`` files."""
    tokens = sorted(c.token_estimate for c in chunks)
    files_chunked = len({c.file_path for c in chunks})
    return ChunkStats(
        strategy=strategy,
        files_chunked=files_chunked,
        files_without_chunks=total_files - files_chunked,
        chunk_count=len(chunks),
        line_fragment_count=sum(c.line_fragment for c in chunks),
        tokens_mean=statistics.fmean(tokens) if tokens else 0.0,
        tokens_median=float(statistics.median(tokens)) if tokens else 0.0,
        tokens_p95=percentile(tokens, 95),
        tokens_max=tokens[-1] if tokens else 0,
        by_chunk_type=dict(sorted(Counter(c.chunk_type.value for c in chunks).items())),
        by_language=dict(sorted(Counter(c.language for c in chunks).items())),
        elapsed_seconds=elapsed_seconds,
    )
