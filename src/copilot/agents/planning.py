"""Intent-aware retrieval planning: how many chunks, what to prefer, and what to ask next.

The plan is where the router's decision *changes behaviour* (all deterministic, no LLM):

=============  =====  ==========  =========================  ===============================
intent         top-k  file cap    context order              refinement order
=============  =====  ==========  =========================  ===============================
symbol_lookup  base   none        symbol/path matches first  symbol, missing, focused
code_explain   base   none        discovery order            missing, focused, symbol
architecture   2x     2 per file  one chunk per file first   missing, focused
debugging      base   none        discovery order            missing, symbol, focused
general        base   none        discovery order            missing, focused
=============  =====  ==========  =========================  ===============================

("missing" = question terms the evidence lacks, "focused" = the question's content words,
"symbol" = identifiers named in the question.)

A refined query is assembled only from words that are in the user's question (its key terms and
identifiers) plus one fixed intent hint word ("definition", "overview", ...). Nothing is taken from
retrieved text, so repository content cannot steer the search, and no path or symbol is invented.
A query that would repeat an earlier one (same set of words) is never issued.
"""

from __future__ import annotations

from collections.abc import Sequence

from copilot.agents.intent import Intent, IntentDecision
from copilot.agents.state import AgentConfig, QueryKind, RetrievalPlan
from copilot.agents.terms import join_terms, normalize_query
from copilot.retrieval.retriever import MAX_QUERY_CHARS, MAX_TOP_K

_HINTS: dict[Intent, str | None] = {
    Intent.SYMBOL_LOOKUP: "definition",
    Intent.CODE_EXPLAIN: "implementation",
    Intent.ARCHITECTURE: "overview structure",
    Intent.DEBUGGING: "error handling",
    Intent.GENERAL: None,
}

_REFINEMENT_ORDER: dict[Intent, tuple[QueryKind, ...]] = {
    Intent.SYMBOL_LOOKUP: (QueryKind.SYMBOL, QueryKind.MISSING_TERMS, QueryKind.FOCUSED_TERMS),
    Intent.CODE_EXPLAIN: (QueryKind.MISSING_TERMS, QueryKind.FOCUSED_TERMS, QueryKind.SYMBOL),
    Intent.ARCHITECTURE: (QueryKind.MISSING_TERMS, QueryKind.FOCUSED_TERMS),
    Intent.DEBUGGING: (QueryKind.MISSING_TERMS, QueryKind.SYMBOL, QueryKind.FOCUSED_TERMS),
    Intent.GENERAL: (QueryKind.MISSING_TERMS, QueryKind.FOCUSED_TERMS),
}


def build_plan(decision: IntentDecision, config: AgentConfig) -> RetrievalPlan:
    """The retrieval plan for ``decision.intent`` (see the table in the module docstring)."""
    intent = decision.intent
    architecture = intent is Intent.ARCHITECTURE
    return RetrievalPlan(
        intent=intent,
        top_k=min(config.top_k * 2, MAX_TOP_K) if architecture else config.top_k,
        per_file_cap=2 if architecture else None,
        prefers_symbol_matches=intent is Intent.SYMBOL_LOOKUP,
        diversify_files=architecture,
        hint=_HINTS[intent],
    )


def _compose(words: Sequence[str], hint: str | None) -> str:
    text = join_terms(words)
    if hint:
        text = f"{text} {hint}"
    return text[:MAX_QUERY_CHARS].strip()


def next_query(
    plan: RetrievalPlan,
    decision: IntentDecision,
    terms: Sequence[str],
    missing_terms: Sequence[str],
    asked: Sequence[str],
) -> tuple[QueryKind, str] | None:
    """The next refined query, or ``None`` when every candidate would repeat an earlier query.

    ``asked`` holds ``normalize_query`` forms of everything already searched (including the
    user's original question).
    """
    asked_set = set(asked)
    for kind in _REFINEMENT_ORDER[plan.intent]:
        if kind is QueryKind.SYMBOL:
            words: Sequence[str] = decision.symbols
        elif kind is QueryKind.MISSING_TERMS:
            words = (*missing_terms, *decision.symbols[:1])
        else:
            words = terms
        if not words:
            continue
        query = _compose(words, plan.hint)
        if query and normalize_query(query) not in asked_set:
            return kind, query
    return None
