"""The LangGraph workflow: classify -> retrieve -> merge -> assess -> (refine | generate | abstain).

::

    START -> classify_intent -> retrieve -> merge_evidence -> assess_evidence -> ...
    retrieve        --failure-----------------------------------> abstain -> END
    assess_evidence --potentially sufficient / exhausted+evidence -> generate -> END
    assess_evidence --insufficient, round + new query available --> retrieve  (the loop)
    assess_evidence --no evidence and no round or query left -----> abstain -> END

Where the control flow lives. Every decision is a plain Python function over the typed state
(``route_after_retrieval``, ``route_after_assessment``) wired in with ``add_conditional_edges``, so
the graph can be read, drawn (``AgentWorkflow.mermaid``) and tested without any model. The only
node that can call a model is ``generate``; it goes through the existing guarded RAG path
(``RagService.answer_from_evidence``), so the secret gate runs on the exact outbound text and
cannot be bypassed. Nothing in the graph reads files, writes files, runs commands, calls tools or
reaches the web: it is read-only apart from the one model call.

Retrieved text is untrusted data. It is only (a) matched lexically by the assessor and (b) placed
in the delimited context of the grounded prompt. It never selects the next query, the intent, a
node or a tool, so text such as "ignore previous instructions" inside a chunk has no control path.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, computed_field

from copilot.agents.assessment import assess_evidence
from copilot.agents.evidence import arrange_evidence, merge_evidence
from copilot.agents.intent import IntentDecision, classify_intent
from copilot.agents.planning import build_plan, next_query
from copilot.agents.state import (
    AgentConfig,
    AgentState,
    Assessment,
    AssessmentOutcome,
    FailureInfo,
    QueryKind,
    SourceProvenance,
    TerminationReason,
    TraceEvent,
)
from copilot.agents.terms import key_terms, normalize_query
from copilot.config.settings import Settings, get_settings
from copilot.llm import LLMClient, LLMError
from copilot.rag.citations import Citation
from copilot.rag.models import AnswerStatus, RagAnswer
from copilot.rag.service import RagService, SupportsRetrieve
from copilot.retrieval import validate_query
from copilot.retrieval.errors import RetrievalError
from copilot.security import RepositorySecretRiskError

logger = logging.getLogger(__name__)

Route = Literal["retrieve", "generate", "abstain"]

_GATE_MESSAGE = (
    "the outbound content was refused by the secret scanner; nothing was sent to the model"
)


class AgentResult(BaseModel):
    """What one workflow run returns. Safe to print; ``answer`` carries the RAG answer."""

    model_config = ConfigDict(frozen=True)

    question: str
    decision: IntentDecision
    rounds: int = Field(ge=0)
    max_rounds: int
    evidence_count: int
    assessment: Assessment | None
    termination: TerminationReason
    answer: RagAnswer | None  # None when abstaining or when generation failed
    citations: tuple[Citation, ...]
    # One entry per citation (same order and ``source_number``): the retrieval round, rank and
    # query-specific score that first supplied the chunk, and its place in the agent's evidence
    # order. ``Citation.rank`` and ``.score`` keep their baseline meaning (the supplying retrieval).
    provenance: tuple[SourceProvenance, ...] = ()
    abstention_reason: str | None
    failure: FailureInfo | None
    retrieval_seconds: float
    trace: tuple[TraceEvent, ...]

    @property
    def generated(self) -> bool:
        return self.answer is not None and self.answer.status is AnswerStatus.ANSWERED

    @computed_field  # type: ignore[prop-decorator]
    @property
    def evidence_status(self) -> Literal["potentially_sufficient", "partial", "none"]:
        """How much to trust that retrieval found what the question needs (a heuristic, see
        ``assessment.py``). ``partial`` means evidence exists but the heuristic was NOT satisfied
        within the budget: an answer built on it must not be presented as evidence-complete."""
        if self.evidence_count == 0:
            return "none"
        if (
            self.assessment is not None
            and self.assessment.outcome is AssessmentOutcome.POTENTIALLY_SUFFICIENT
        ):
            return "potentially_sufficient"
        return "partial"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def warnings(self) -> tuple[str, ...]:
        """Caveats every consumer (CLI, UI, evaluation) should show next to the answer."""
        found: list[str] = []
        if self.evidence_status == "partial":
            reason = self.assessment.reason if self.assessment else "unknown"
            found.append(
                "partial_evidence: the evidence check was not satisfied "
                f"({reason}; stopped by {self.termination.value}); the answer rests on "
                "incomplete repository evidence"
            )
        if self.answer is not None and self.answer.unknown_source_numbers:
            numbers = ", ".join(str(n) for n in self.answer.unknown_source_numbers)
            found.append(f"unknown_source_labels: the answer mentions unsupplied sources {numbers}")
        return tuple(found)


def _fingerprint(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------- routing (pure functions)


def route_after_retrieval(state: AgentState) -> Literal["merge_evidence", "abstain"]:
    """A retrieval failure ends the run; otherwise the results are merged."""
    return "abstain" if state.get("failure") else "merge_evidence"


def route_after_assessment(state: AgentState) -> Route:
    """The conditional edge out of ``assess_evidence`` (a function of the assessment only)."""
    assessment = state["assessment"]
    outcome = assessment.outcome
    if outcome is AssessmentOutcome.POTENTIALLY_SUFFICIENT:
        return "generate"
    if outcome is AssessmentOutcome.INSUFFICIENT_REFINABLE:
        return "retrieve"
    if outcome is AssessmentOutcome.EMPTY:
        return "retrieve" if assessment.stop_reason is None else "abstain"
    return "generate"  # EXHAUSTED: partial evidence; the grounded prompt states what is missing


# --------------------------------------------------------------------------------- the workflow


class AgentWorkflow:
    """Builds the graph once; ``run`` executes it for one question.

    ``retriever`` is any ``SupportsRetrieve`` (the baseline ``Retriever`` in production, a fake in
    tests). ``llm`` is wrapped in the secret-gating ``GuardedLLMClient`` by ``RagService``.
    """

    def __init__(
        self,
        retriever: SupportsRetrieve,
        llm: LLMClient,
        *,
        settings: Settings | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._config = config or AgentConfig.from_settings(self._settings)
        self._retriever = retriever
        self._rag = RagService(retriever, llm, settings=self._settings)
        self._graph = self._build()

    # ------------------------------------------------------------------------------- nodes

    def _classify(self, state: AgentState) -> dict:
        started = time.perf_counter()
        query = state["query"]
        config = state["config"]
        decision = classify_intent(query)
        plan = build_plan(decision, config)
        terms = key_terms(query)
        event = TraceEvent(
            node="classify_intent",
            intent=decision.intent.value,
            rule_id=decision.rule_id,
            query_terms=len(terms),
            top_k=plan.top_k,
            note="matched=" + ",".join(i.value for i in decision.matched)
            if decision.matched
            else None,
            duration_seconds=round(time.perf_counter() - started, 4),
        )
        return {
            "intent": decision.intent,
            "decision": decision,
            "plan": plan,
            "terms": terms,
            "current_query": query,  # round 1 always searches the user's own wording
            "current_kind": QueryKind.ORIGINAL,
            "asked_queries": (),
            "round": 0,
            "max_rounds": config.max_rounds,
            "evidence": (),
            "evidence_origin": {},
            "selection_order": (),
            "results": (),
            "new_accepted": 0,
            "retrieval_seconds": 0.0,
            "trace": [event],
        }

    def _retrieve(self, state: AgentState) -> dict:
        started = time.perf_counter()
        query = state["current_query"]
        kind = state["current_kind"]
        plan = state["plan"]
        rnd = state["round"] + 1
        base = {
            "node": "retrieve",
            "round": rnd,
            "intent": state["intent"].value,
            "query_kind": kind.value,
            "query_chars": len(query),
            "query_terms": len(query.split()),
            "query_sha256_12": _fingerprint(query),
            "top_k": plan.top_k,
        }
        asked = (*state["asked_queries"], normalize_query(query))
        try:
            validate_query(query)
            results = tuple(self._retriever.retrieve(query, plan.top_k))
        except RetrievalError as exc:
            elapsed = round(time.perf_counter() - started, 4)
            logger.warning("agent retrieval failed in round %d (%s)", rnd, type(exc).__name__)
            return {
                "round": rnd,
                "asked_queries": asked,
                "failure": FailureInfo(
                    stage="retrieval", error_type=type(exc).__name__, message=str(exc)
                ),
                "termination": TerminationReason.RETRIEVAL_FAILED,
                "trace": [TraceEvent(**base, duration_seconds=elapsed, note="retrieval_failed")],
            }
        elapsed = time.perf_counter() - started
        return {
            "round": rnd,
            "asked_queries": asked,
            "results": results,
            "retrieval_seconds": round(state["retrieval_seconds"] + elapsed, 4),
            "trace": [
                TraceEvent(
                    **base,
                    retrieved=len(results),
                    candidate_files=len({r.file_path for r in results}),
                    duration_seconds=round(elapsed, 4),
                )
            ],
        }

    def _merge(self, state: AgentState) -> dict:
        started = time.perf_counter()
        config = state["config"]
        plan = state["plan"]
        merged = merge_evidence(
            state["evidence"],
            state["results"],
            max_chunks=config.max_evidence_chunks,
            per_file_cap=plan.per_file_cap,
            prefer_distinct_files=plan.diversify_files,
        )
        # Provenance: the FIRST retrieval that supplied a chunk (later duplicates are rejected).
        origin = dict(state.get("evidence_origin", {}))
        for result in merged.accepted:
            origin[result.chunk_id] = (state["round"], state["current_kind"].value)
        event = TraceEvent(
            node="merge_evidence",
            round=state["round"],
            intent=state["intent"].value,
            retrieved=merged.candidates,
            candidate_files=merged.candidate_files,
            new_accepted=merged.accepted_count,
            rejected_duplicate=merged.rejected_duplicate,
            rejected_per_file_cap=merged.rejected_per_file_cap,
            rejected_capacity=merged.rejected_capacity,
            evidence_total=len(merged.evidence),
            evidence_files=merged.evidence_files,
            duration_seconds=round(time.perf_counter() - started, 4),
        )
        return {
            "evidence": merged.evidence,
            "evidence_origin": origin,
            "new_accepted": merged.accepted_count,
            "trace": [event],
        }

    def _assess(self, state: AgentState) -> dict:
        started = time.perf_counter()
        config = state["config"]
        decision = state["decision"]
        terms = state["terms"]
        evidence = state["evidence"]
        rnd = state["round"]

        stop: TerminationReason | None = None
        if rnd >= state["max_rounds"]:
            stop = TerminationReason.EXHAUSTED_ROUNDS
        elif rnd >= 2 and state["new_accepted"] == 0:
            stop = TerminationReason.NO_NEW_EVIDENCE

        assessment = assess_evidence(evidence, decision, terms, config, refinable=True)
        update: dict = {}
        if assessment.outcome is not AssessmentOutcome.POTENTIALLY_SUFFICIENT:
            if stop is None:
                missing = assessment.missing_terms or (terms if not evidence else ())
                candidate = next_query(
                    state["plan"], decision, terms, missing, state["asked_queries"]
                )
                if candidate is None:
                    stop = TerminationReason.NO_NEW_QUERY
                else:
                    update["current_kind"], update["current_query"] = candidate
            if stop is not None:
                assessment = assess_evidence(
                    evidence, decision, terms, config, refinable=False, stop_reason=stop.value
                )

        route = route_after_assessment({"assessment": assessment})  # type: ignore[typeddict-item]
        if route == "retrieve":
            termination = None
        elif assessment.outcome is AssessmentOutcome.POTENTIALLY_SUFFICIENT:
            termination = TerminationReason.SUFFICIENT
        elif not evidence:
            termination = TerminationReason.NO_EVIDENCE
        else:
            termination = stop
        if termination is not None:
            update["termination"] = termination

        event = TraceEvent(
            node="assess_evidence",
            round=rnd,
            intent=state["intent"].value,
            evidence_total=len(evidence),
            new_accepted=state["new_accepted"],
            assessment=assessment.outcome.value,
            term_coverage=None
            if assessment.term_coverage is None
            else round(assessment.term_coverage, 3),
            missing_terms=len(assessment.missing_terms),
            routing="refine" if route == "retrieve" else route,
            termination_reason=termination.value if termination else None,
            note=assessment.reason,
            duration_seconds=round(time.perf_counter() - started, 4),
        )
        update["assessment"] = assessment
        update["trace"] = [event]
        return update

    def _generate(self, state: AgentState) -> dict:
        started = time.perf_counter()
        arranged = arrange_evidence(
            state["evidence"],
            state["plan"],
            symbols=state["decision"].symbols,
            terms=state["terms"],
        )
        base = {"node": "generate", "round": state["round"], "intent": state["intent"].value}
        selection_order = tuple(r.chunk_id for r in arranged)
        try:
            answer = self._rag.answer_from_evidence(
                state["query"],
                arranged,
                retrieval_seconds=state["retrieval_seconds"],
                preserve_order=True,  # keep the agent's order; ranks stay retrieval provenance
            )
        except RepositorySecretRiskError:
            failure = FailureInfo(
                stage="generation",
                error_type="RepositorySecretRiskError",
                message=_GATE_MESSAGE,
                blocked_by_secret_gate=True,
            )
            return self._generation_failed(base, failure, "blocked_by_secret_gate", started)
        except LLMError as exc:
            failure = FailureInfo(
                stage="generation", error_type=type(exc).__name__, message=str(exc)
            )
            return self._generation_failed(base, failure, "provider_error", started)

        abstained = answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE
        event = TraceEvent(
            **base,
            evidence_total=len(arranged),
            context_chunks=len(answer.sources),
            context_dropped_budget=sum(d.reason == "budget" for d in answer.dropped_evidence),
            generation_status="abstained_before_model" if abstained else "answered",
            note=answer.insufficient_reason,
            duration_seconds=round(time.perf_counter() - started, 4),
        )
        return {
            "answer": answer,
            "citations": answer.sources,
            "selection_order": selection_order,
            "abstention_reason": answer.insufficient_reason if abstained else None,
            "trace": [event],
        }

    @staticmethod
    def _generation_failed(base: dict, failure: FailureInfo, status: str, started: float) -> dict:
        event = TraceEvent(
            **base,
            generation_status=status,
            note=failure.error_type,
            duration_seconds=round(time.perf_counter() - started, 4),
        )
        return {
            "answer": None,
            "citations": (),
            "failure": failure,
            "termination": TerminationReason.GENERATION_FAILED,
            "trace": [event],
        }

    def _abstain(self, state: AgentState) -> dict:
        failure = state.get("failure")
        reason = "retrieval_failed" if failure else "no_results"
        event = TraceEvent(
            node="abstain",
            round=state["round"],
            intent=state["intent"].value,
            generation_status="not_attempted",
            termination_reason=state["termination"].value if "termination" in state else None,
            note=reason,
        )
        return {"answer": None, "citations": (), "abstention_reason": reason, "trace": [event]}

    # -------------------------------------------------------------------------------- graph

    def _build(self):
        graph = StateGraph(AgentState)
        graph.add_node("classify_intent", self._classify)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("merge_evidence", self._merge)
        graph.add_node("assess_evidence", self._assess)
        graph.add_node("generate", self._generate)
        graph.add_node("abstain", self._abstain)

        graph.add_edge(START, "classify_intent")
        graph.add_edge("classify_intent", "retrieve")
        graph.add_conditional_edges(
            "retrieve",
            route_after_retrieval,
            {"merge_evidence": "merge_evidence", "abstain": "abstain"},
        )
        graph.add_edge("merge_evidence", "assess_evidence")
        graph.add_conditional_edges(
            "assess_evidence",
            route_after_assessment,
            {"retrieve": "retrieve", "generate": "generate", "abstain": "abstain"},
        )
        graph.add_edge("generate", END)
        graph.add_edge("abstain", END)
        return graph.compile()

    def mermaid(self) -> str:
        """The compiled graph as a Mermaid flowchart (used by ``docs/agent.md``)."""
        return self._graph.get_graph().draw_mermaid()

    @staticmethod
    def _provenance(final: AgentState) -> tuple[SourceProvenance, ...]:
        """One ``SourceProvenance`` per citation, joined by ``chunk_id`` (never by position)."""
        by_id = {r.chunk_id: r for r in final.get("evidence", ())}
        order = {cid: i for i, cid in enumerate(final.get("selection_order", ()), start=1)}
        origin = final.get("evidence_origin", {})
        entries: list[SourceProvenance] = []
        for citation in final.get("citations", ()):
            result = by_id[citation.chunk_id]
            round_no, kind = origin[citation.chunk_id]
            entries.append(
                SourceProvenance(
                    source_number=citation.source_number,
                    chunk_id=citation.chunk_id,
                    file_path=citation.file_path,
                    start_line=citation.start_line,
                    end_line=citation.end_line,
                    retrieval_round=round_no,
                    retrieval_rank=result.rank,
                    query_kind=kind,
                    score=result.score,
                    selection_order=order[citation.chunk_id],
                )
            )
        return tuple(entries)

    # ---------------------------------------------------------------------------------- run

    def run(self, question: str, *, max_rounds: int | None = None) -> AgentResult:
        """Answer ``question``. Raises ``QueryError`` for a blank/oversized question and
        ``ValidationError`` for ``max_rounds`` outside 1..5; everything else becomes data."""
        text = validate_query(question)
        config = (
            self._config
            if max_rounds is None
            else self._config.model_copy(update={"max_rounds": max_rounds})
        )
        config = AgentConfig.model_validate(config.model_dump())  # re-validate the limits
        final: AgentState = self._graph.invoke(
            {"query": text, "config": config, "trace": []},
            {"recursion_limit": 3 * config.max_rounds + 8},
        )
        return AgentResult(
            provenance=self._provenance(final),
            question=text,
            decision=final["decision"],
            rounds=final["round"],
            max_rounds=final["max_rounds"],
            evidence_count=len(final["evidence"]),
            assessment=final.get("assessment"),
            termination=final["termination"],
            answer=final.get("answer"),
            citations=final.get("citations", ()),
            abstention_reason=final.get("abstention_reason"),
            failure=final.get("failure"),
            retrieval_seconds=final["retrieval_seconds"],
            trace=tuple(final["trace"]),
        )
