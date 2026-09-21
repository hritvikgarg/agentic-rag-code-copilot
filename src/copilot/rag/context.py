"""Deterministic, bounded evidence context for one RAG prompt.

Format of one block (text is the chunk exactly as materialised from the repository)::

    BEGIN SOURCE 1 [3fa9c1d2]
    file: src/pkg/ingest.py
    lines: 12-37
    chunk_id: 0123456789abcdef
    content:
    <chunk text>
    END SOURCE 1 [3fa9c1d2]

The 8-hex tag is derived from the included chunks' content hashes, so repository text cannot
forge a matching ``END`` marker (it cannot know the hash of the text that contains it).

Selection rules (all deterministic):
1. candidates are ordered by retrieval rank;
2. a chunk whose ``chunk_id`` or ``(file, start, end)`` was already taken is dropped as duplicate;
3. blocks are added in rank order while the running estimated-token total stays within budget;
   a block that does not fit is dropped whole (never cut) and later, smaller ones may still fit;
4. included blocks are numbered ``Source 1..n`` contiguously in that order.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from copilot.rag.citations import Citation
from copilot.retrieval import RetrievalResult
from copilot.utils.tokens import estimate_tokens

SEPARATOR = "\n\n"
_TAG_PLACEHOLDER = "0" * 8


class DroppedEvidence(BaseModel):
    """A retrieved chunk that did not enter the context, and why."""

    model_config = ConfigDict(frozen=True)

    rank: int
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    reason: Literal["duplicate", "budget"]


class BuiltContext(BaseModel):
    """The evidence text plus the metadata of exactly what entered it."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(repr=False)
    tag: str
    citations: tuple[Citation, ...]
    included: tuple[RetrievalResult, ...] = Field(repr=False)  # same order as ``citations``
    dropped: tuple[DroppedEvidence, ...]
    estimated_tokens: int


def format_block(number: int, result: RetrievalResult, tag: str) -> str:
    """One evidence block (see the module docstring)."""
    text = result.text if result.text.endswith("\n") else result.text + "\n"
    return (
        f"BEGIN SOURCE {number} [{tag}]\n"
        f"file: {result.file_path}\n"
        f"lines: {result.start_line}-{result.end_line}\n"
        f"chunk_id: {result.chunk_id}\n"
        f"content:\n"
        f"{text}"
        f"END SOURCE {number} [{tag}]"
    )


def _context_tag(results: Sequence[RetrievalResult]) -> str:
    digest = hashlib.sha256()
    for result in results:
        digest.update(result.content_sha256.encode("ascii"))
    return digest.hexdigest()[:8]


def build_context(results: Sequence[RetrievalResult], max_tokens: int) -> BuiltContext:
    """Select and format evidence within ``max_tokens`` estimated tokens.

    ``results`` may come in any order; they are ranked by ``(rank, position)`` first.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    ordered = sorted(results, key=lambda r: (r.rank, r.position))
    taken_ids: set[str] = set()
    taken_spans: set[tuple[str, int, int]] = set()
    included: list[RetrievalResult] = []
    dropped: list[DroppedEvidence] = []
    used = 0
    for result in ordered:
        span = (result.file_path, result.start_line, result.end_line)

        def drop(reason: Literal["duplicate", "budget"], result: RetrievalResult = result) -> None:
            dropped.append(
                DroppedEvidence(
                    rank=result.rank,
                    chunk_id=result.chunk_id,
                    file_path=result.file_path,
                    start_line=result.start_line,
                    end_line=result.end_line,
                    reason=reason,
                )  # fmt: skip
            )

        if result.chunk_id in taken_ids or span in taken_spans:
            drop("duplicate")
            continue
        block = format_block(len(included) + 1, result, _TAG_PLACEHOLDER)
        cost = estimate_tokens(block) + (estimate_tokens(SEPARATOR) if included else 0)
        if used + cost > max_tokens:
            drop("budget")
            continue
        used += cost
        included.append(result)
        taken_ids.add(result.chunk_id)
        taken_spans.add(span)

    tag = _context_tag(included)
    text = SEPARATOR.join(format_block(i, r, tag) for i, r in enumerate(included, start=1))
    citations = tuple(
        Citation(
            source_number=i,
            file_path=r.file_path,
            start_line=r.start_line,
            end_line=r.end_line,
            chunk_id=r.chunk_id,
            score=r.score,
            rank=r.rank,
        )  # fmt: skip
        for i, r in enumerate(included, start=1)
    )
    return BuiltContext(
        text=text,
        tag=tag,
        citations=citations,
        included=tuple(included),
        dropped=tuple(dropped),
        estimated_tokens=estimate_tokens(text),
    )
