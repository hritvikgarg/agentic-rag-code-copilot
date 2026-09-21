"""Shared builders for the Milestone 6 tests (no model, no network)."""

from __future__ import annotations

import hashlib

from copilot.models.chunk import ChunkType
from copilot.retrieval import RetrievalResult


def make_result(
    text: str,
    *,
    rank: int = 1,
    path: str = "src/pkg/mod.py",
    start: int = 10,
    chunk_id: str | None = None,
    score: float = 0.5,
    end: int | None = None,
) -> RetrievalResult:
    """A ``RetrievalResult`` with a distinct content hash and chunk id per text."""
    digest = hashlib.sha256(f"{path}:{start}:{text}".encode()).hexdigest()
    return RetrievalResult(
        rank=rank,
        score=score,
        position=rank - 1,
        chunk_id=chunk_id or digest[:16],
        repository_name="demo",
        file_path=path,
        language="python",
        chunk_type=ChunkType.LINE_WINDOW,
        chunk_index=rank - 1,
        start_line=start,
        end_line=end if end is not None else start + text.rstrip("\n").count("\n"),
        token_estimate=max(1, len(text) // 4),
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        text=text,
    )


class FakeRetriever:
    """Returns fixed results (or raises) and records the calls."""

    def __init__(self, results=(), error: Exception | None = None) -> None:
        self.results = list(results)
        self.error = error
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, top_k: int | None = None):
        self.calls.append((query, top_k))
        if self.error:
            raise self.error
        return self.results[: top_k or len(self.results)]
