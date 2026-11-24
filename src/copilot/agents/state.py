"""Typed graph state, configuration and the structured trace of the agentic workflow.

LangGraph passes one ``AgentState`` dict through the nodes; every node returns only the keys it
changes. ``trace`` is the only key with a *reducer* (``operator.add``): nodes append events, and
LangGraph concatenates them instead of overwriting. Everything else is last-write-wins.

The trace is built for explanation and debugging, so it is deliberately limited to counts, labels,
timings and hashes. It never contains the question's or a refined query's text, retrieved source
text, prompts, model output, provider error bodies, API keys or secret-scanner findings.
"""

from __future__ import annotations

import operator
from enum import StrEnum
from typing import Annotated, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from copilot.agents.intent import Intent, IntentDecision
from copilot.config.settings import Settings
from copilot.rag.citations import Citation
from copilot.rag.models import RagAnswer
from copilot.retrieval import RetrievalResult

MAX_ROUNDS_LIMIT = 5


class AgentConfig(BaseModel):
    """Agent-specific limits. Independent of the baseline RAG configuration (``top_k`` of the
    baseline retriever is only the *starting point* for the per-round chunk count)."""

    model_config = ConfigDict(frozen=True)

    max_rounds: int = Field(
        default=3, ge=1, le=MAX_ROUNDS_LIMIT
    )  # first retrieval counts as round 1
    max_evidence_chunks: int = Field(default=15, ge=1, le=50)  # accumulated evidence cap
    min_term_coverage: float = Field(default=0.5, ge=0.0, le=1.0)
    architecture_min_files: int = Field(default=2, ge=1, le=10)
    top_k: int = Field(default=5, ge=1, le=50)  # chunks requested per retrieval round (base)

    @classmethod
    def from_settings(cls, settings: Settings, *, max_rounds: int | None = None) -> AgentConfig:
        return cls(
            max_rounds=settings.agent_max_retrieval_rounds if max_rounds is None else max_rounds,
            max_evidence_chunks=settings.agent_max_evidence_chunks,
            min_term_coverage=settings.agent_min_term_coverage,
            architecture_min_files=settings.agent_architecture_min_files,
            top_k=settings.retrieval_top_k,
        )


class QueryKind(StrEnum):
    """Where a round's search text came from. Only the first is the user's own wording."""

    ORIGINAL = "original"
    MISSING_TERMS = "missing_terms"  # the question terms the evidence does not mention yet
    FOCUSED_TERMS = "focused_terms"  # the question's content words without the boilerplate
    SYMBOL = "symbol"  # identifier(s) that appear in the question


class RetrievalPlan(BaseModel):
    """What the intent changes about retrieval (see ``copilot.agents.planning``)."""

    model_config = ConfigDict(frozen=True)

    intent: Intent
    top_k: int  # chunks requested per round
    per_file_cap: int | None  # max accumulated chunks from one file (None: unlimited)
    prefers_symbol_matches: bool  # order symbol/path matches first when building the context
    diversify_files: bool  # spread the context across files before filling from the best file
    hint: str | None  # fixed intent word appended to refined queries (never a path or symbol)


class AssessmentOutcome(StrEnum):
    EMPTY = "empty"  # nothing retrieved so far
    POTENTIALLY_SUFFICIENT = "potentially_sufficient"  # the heuristic's checks passed
    INSUFFICIENT_REFINABLE = "insufficient_refinable"  # checks failed; another round is possible
    EXHAUSTED = "exhausted"  # checks failed (or evidence is empty) and no round is possible


class Assessment(BaseModel):
    """Result of the deterministic evidence check. ``potentially`` is deliberate: the check is
    lexical, not semantic, and cannot prove the evidence answers the question."""

    model_config = ConfigDict(frozen=True)

    outcome: AssessmentOutcome
    reason: str  # short machine label, e.g. "term_coverage_below_threshold"
    term_coverage: float | None  # fraction of question terms found in the evidence (None: no terms)
    missing_terms: tuple[str, ...] = ()
    distinct_files: int = 0
    evidence_count: int = 0
    stop_reason: str | None = None  # why no further round is possible, when it is not


class TerminationReason(StrEnum):
    SUFFICIENT = "evidence_potentially_sufficient"
    EXHAUSTED_ROUNDS = "max_rounds_reached"
    NO_NEW_EVIDENCE = "no_new_evidence"
    NO_NEW_QUERY = "no_new_query"  # every refinement would repeat an earlier query
    NO_EVIDENCE = "no_evidence"
    RETRIEVAL_FAILED = "retrieval_failed"
    GENERATION_FAILED = "generation_failed"


class TraceEvent(BaseModel):
    """One node execution. All fields are safe to print and to log."""

    model_config = ConfigDict(frozen=True)

    node: str
    round: int = 0
    intent: str | None = None
    duration_seconds: float = 0.0
    # --- query metadata (never the text) ---
    query_kind: str | None = None
    query_chars: int | None = None
    query_terms: int | None = None
    query_sha256_12: str | None = None
    # --- retrieval ---
    top_k: int | None = None
    retrieved: int | None = None
    new_accepted: int | None = None
    evidence_total: int | None = None
    # --- assessment / routing ---
    assessment: str | None = None
    term_coverage: float | None = None
    missing_terms: int | None = None
    routing: str | None = None
    termination_reason: str | None = None
    rule_id: str | None = None
    # --- generation ---
    generation_status: str | None = None
    note: str | None = None


class FailureInfo(BaseModel):
    """A failure kept as data instead of an exception: the type name and a safe message."""

    model_config = ConfigDict(frozen=True)

    stage: str  # "retrieval" | "generation"
    error_type: str
    message: str
    blocked_by_secret_gate: bool = False


class AgentState(TypedDict, total=False):
    """The graph state. ``total=False``: a key exists once the node that owns it has run."""

    # input
    query: str
    config: AgentConfig
    # routing and planning
    intent: Intent
    decision: IntentDecision
    plan: RetrievalPlan
    terms: tuple[str, ...]
    current_query: str
    current_kind: QueryKind
    asked_queries: tuple[str, ...]  # normalised, to refuse repeats
    # retrieval
    round: int  # rounds started (the attempt count)
    max_rounds: int
    results: tuple[RetrievalResult, ...]  # the latest round only
    evidence: tuple[RetrievalResult, ...]  # accumulated across rounds
    new_accepted: int
    retrieval_seconds: float
    # assessment and outcome
    assessment: Assessment
    termination: TerminationReason
    answer: RagAnswer | None
    citations: tuple[Citation, ...]  # only from the evidence that entered the final context
    abstention_reason: str | None
    failure: FailureInfo | None
    trace: Annotated[list[TraceEvent], operator.add]
