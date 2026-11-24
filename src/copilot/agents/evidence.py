"""Selecting retrieved evidence across rounds, and arranging it for the prompt.

Three pure functions, no I/O:

``merge_evidence``   select which of one round's candidates join the accumulated evidence
``arrange_evidence`` order the accumulated evidence for the final prompt (ranks are NOT rewritten)
``evidence_text``    the lowercase text the assessor looks in (chunk text, path, symbol names)

Budgets. Three limits are applied at three different points and reported separately:
the **candidate** budget (how many chunks the retriever is asked for; see ``planning.py``), the
**evidence** budget (``max_chunks`` and the per-file cap, enforced here) and the **context**
budget (the token limit applied by ``build_context``). Keeping them apart is what makes starvation
(few candidates, or all from one file) distinguishable from truncation (evidence that did not fit
the prompt).

Identity. A chunk is "the same" when its ``chunk_id`` matches (stable, content-derived) or when it
covers the same ``(file, start_line, end_line)``. Existing evidence is never dropped or reordered
and no result is modified: ``rank``, ``score`` and ``position`` always keep the values of the
retrieval that first supplied the chunk. Those values are only meaningful *within one round* (one
query); scores and ranks from different refined queries are never compared with each other.

Selection (``prefer_distinct_files``, used for architecture questions). Candidates are first put in
a deterministic order, ``(rank, position, chunk_id)``. Pass 1 takes, in that order, the best
candidate of every file that is not yet represented in the evidence; pass 2 fills the remaining
capacity with the best remaining candidates, still honouring the per-file cap. So relevance order
decides *which* chunk of a file and *which* file comes first, and diversity decides only that a
new file beats a second chunk of an already represented one. Without the flag candidates are taken
in ``(rank, position, chunk_id)`` order (the Milestone 8 behaviour).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict

from copilot.agents.state import RetrievalPlan
from copilot.agents.terms import term_in
from copilot.retrieval import RetrievalResult


class MergeResult(BaseModel):
    """The outcome of merging one round's candidates into the evidence, with safe counts."""

    model_config = ConfigDict(frozen=True)

    evidence: tuple[RetrievalResult, ...]  # accumulated evidence, selection order
    accepted: tuple[RetrievalResult, ...]  # the candidates added by this call
    candidates: int  # candidates offered
    candidate_files: int  # distinct files among them
    rejected_duplicate: int  # already in the evidence, or repeated within the batch
    rejected_per_file_cap: int  # their file had reached the per-file cap
    rejected_capacity: int  # not selected because the evidence budget was full
    evidence_files: int  # distinct files in the accumulated evidence

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)


def _span(result: RetrievalResult) -> tuple[str, int, int]:
    return (result.file_path, result.start_line, result.end_line)


def merge_evidence(
    existing: Sequence[RetrievalResult],
    new: Iterable[RetrievalResult],
    *,
    max_chunks: int,
    per_file_cap: int | None,
    prefer_distinct_files: bool = False,
) -> MergeResult:
    """Add the non-duplicate ``new`` candidates to ``existing`` within the evidence budget."""
    candidates = sorted(new, key=lambda r: (r.rank, r.position, r.chunk_id))
    merged = list(existing)
    known_ids = {r.chunk_id for r in merged}
    known_spans = {_span(r) for r in merged}
    per_file = Counter(r.file_path for r in merged)

    unique: list[RetrievalResult] = []
    duplicates = 0
    for result in candidates:
        if result.chunk_id in known_ids or _span(result) in known_spans:
            duplicates += 1
            continue
        unique.append(result)
        known_ids.add(result.chunk_id)
        known_spans.add(_span(result))

    taken: set[str] = set()
    accepted: list[RetrievalResult] = []

    def take(result: RetrievalResult) -> bool:
        if len(merged) >= max_chunks:
            return False
        if per_file_cap is not None and per_file[result.file_path] >= per_file_cap:
            return False
        merged.append(result)
        accepted.append(result)
        taken.add(result.chunk_id)
        per_file[result.file_path] += 1
        return True

    if prefer_distinct_files:
        represented = {path for path, count in per_file.items() if count}
        for result in unique:  # pass 1: the best chunk of each file not represented yet
            if result.file_path not in represented and take(result):
                represented.add(result.file_path)
    for result in unique:  # pass 2 (or the only pass): fill the remaining capacity in order
        if result.chunk_id not in taken:
            take(result)

    capped = full = 0
    for result in unique:
        if result.chunk_id in taken:
            continue
        if per_file_cap is not None and per_file[result.file_path] >= per_file_cap:
            capped += 1
        else:
            full += 1
    return MergeResult(
        evidence=tuple(merged),
        accepted=tuple(accepted),
        candidates=len(candidates),
        candidate_files=len({r.file_path for r in candidates}),
        rejected_duplicate=duplicates,
        rejected_per_file_cap=capped,
        rejected_capacity=full,
        evidence_files=len({r.file_path for r in merged}),
    )


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
    """The evidence in prompt order. Stable and deterministic; no result is modified.

    ``rank``, ``score`` and ``position`` stay the values of the retrieval that supplied each chunk
    (see the module docstring), so a citation's retrieval rank is never confused with its place in
    the prompt. The order is conveyed to ``build_context`` with ``preserve_order=True``.

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
    return tuple(ordered)


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
