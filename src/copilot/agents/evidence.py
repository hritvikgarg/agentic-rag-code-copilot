"""Accumulating retrieved evidence across rounds, and arranging it for the prompt.

Three pure functions, no I/O:

``merge_evidence``   add one round's results to what is already known, without losing anything
``arrange_evidence`` order (and re-rank) the accumulated evidence for the final prompt
``evidence_text``    the lowercase text the assessor looks in (chunk text, path, symbol names)

Identity. A chunk is "the same" when its ``chunk_id`` matches (stable, content-derived) or when it
covers the same ``(file, start_line, end_line)``. Merging never changes a result's metadata; only
``arrange_evidence`` rewrites ``rank`` (so ``build_context`` keeps the order chosen here). The
``score`` stays the cosine similarity of the query that found the chunk, which is not comparable
across different queries, so scores are never used to order evidence from different rounds.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from copilot.agents.state import RetrievalPlan
from copilot.agents.terms import term_in
from copilot.retrieval import RetrievalResult


def _span(result: RetrievalResult) -> tuple[str, int, int]:
    return (result.file_path, result.start_line, result.end_line)


def merge_evidence(
    existing: Sequence[RetrievalResult],
    new: Iterable[RetrievalResult],
    *,
    max_chunks: int,
    per_file_cap: int | None,
) -> tuple[tuple[RetrievalResult, ...], int]:
    """``(accumulated evidence, number of newly accepted chunks)``.

    Existing evidence is never dropped or reordered. A new chunk is accepted when it is not a
    duplicate, the total stays within ``max_chunks`` and its file is below ``per_file_cap``.
    """
    merged = list(existing)
    seen_ids = {r.chunk_id for r in merged}
    seen_spans = {_span(r) for r in merged}
    per_file = Counter(r.file_path for r in merged)
    accepted = 0
    for result in new:
        if len(merged) >= max_chunks:
            break
        if result.chunk_id in seen_ids or _span(result) in seen_spans:
            continue
        if per_file_cap is not None and per_file[result.file_path] >= per_file_cap:
            continue
        merged.append(result)
        seen_ids.add(result.chunk_id)
        seen_spans.add(_span(result))
        per_file[result.file_path] += 1
        accepted += 1
    return tuple(merged), accepted


def _mentions_symbol(result: RetrievalResult, symbols_lower: Sequence[str]) -> bool:
    haystack = " ".join(
        part.lower()
        for part in (result.symbol_name, result.qualified_name, result.file_path)
        if part
    )
    return any(symbol in haystack for symbol in symbols_lower)


def arrange_evidence(
    evidence: Sequence[RetrievalResult],
    plan: RetrievalPlan,
    *,
    symbols: Sequence[str],
    terms: Sequence[str],
) -> tuple[RetrievalResult, ...]:
    """The evidence in prompt order with ``rank`` rewritten to 1..n. Stable and deterministic.

    * ``symbol_lookup`` (``prefers_symbol_matches``): chunks whose symbol name, qualified name or
      file path contains a symbol or key term from the question come first, because a lookup
      question wants the defining chunk at the top of a budget-limited context.
    * ``architecture`` (``diversify_files``): one chunk per file first (best first), then the rest,
      so the context shows breadth before depth.
    * everything else keeps discovery order (round 1's ranking first).
    """
    ordered = list(evidence)
    if plan.prefers_symbol_matches:
        needles = [s.lower() for s in (*symbols, *terms)]
        ordered.sort(key=lambda r: 0 if _mentions_symbol(r, needles) else 1)  # stable
    if plan.diversify_files:
        first: list[RetrievalResult] = []
        rest: list[RetrievalResult] = []
        seen_files: set[str] = set()
        for result in ordered:
            (rest if result.file_path in seen_files else first).append(result)
            seen_files.add(result.file_path)
        ordered = first + rest
    return tuple(r.model_copy(update={"rank": i}) for i, r in enumerate(ordered, start=1))


def evidence_text(evidence: Sequence[RetrievalResult]) -> str:
    """Lowercase text of all evidence for lexical matching: path, symbol names and chunk text."""
    parts: list[str] = []
    for result in evidence:
        parts.append(result.file_path)
        parts.extend(p for p in (result.symbol_name, result.qualified_name) if p)
        parts.append(result.text)
    return "\n".join(parts).lower()


def mentions_any(evidence: Sequence[RetrievalResult], needles: Sequence[str]) -> bool:
    """Does any evidence chunk mention any of ``needles`` (a symbol, its stem, a term)?"""
    return any(term_in(n.lower(), evidence_text([r])) for r in evidence for n in needles)
