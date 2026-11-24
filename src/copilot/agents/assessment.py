"""Deterministic evidence assessment: has retrieval *probably* found enough to answer from?

This is a **lexical heuristic, not a relevance judge**. It answers a narrow question: do the
question's own content words (and, for some intents, an identifier or several files) appear in the
accumulated evidence? It cannot tell whether the evidence is *correct* or *answers* the question,
which is why the positive outcome is called ``potentially_sufficient``.

Checks, by intent (all must pass):

* every intent: ``term_coverage >= min_term_coverage`` where coverage is the fraction of the
  question's key terms (suffix-stripped, so "authentication" matches "authenticate") that occur in
  any evidence chunk's text, path or symbol names. No terms -> the check is skipped.
* symbol_lookup: if the question names an identifier, some chunk must mention it.
* architecture: the evidence must span at least ``architecture_min_files`` distinct files.

There is deliberately **no similarity-score threshold**: cosine scores are not calibrated and not
comparable across queries or models (see ``copilot.retrieval``), so any universal cutoff would be
arbitrary. The thresholds above are configuration (``COPILOT_AGENT_*``), not tuned constants.

Known weaknesses: a question whose words are common in the repository passes trivially; a question
phrased with synonyms the code never uses fails even when the right chunk was retrieved; stemming
is crude. Exhaustion with partial evidence still lets the grounded prompt answer, because the RAG
prompt itself instructs the model to state insufficient evidence.
"""

from __future__ import annotations

from collections.abc import Sequence

from copilot.agents.evidence import evidence_text, mentions_any
from copilot.agents.intent import Intent, IntentDecision
from copilot.agents.state import AgentConfig, Assessment, AssessmentOutcome
from copilot.agents.terms import coverage
from copilot.retrieval import RetrievalResult


def assess_evidence(
    evidence: Sequence[RetrievalResult],
    decision: IntentDecision,
    terms: Sequence[str],
    config: AgentConfig,
    *,
    refinable: bool,
    stop_reason: str | None = None,
) -> Assessment:
    """Assess the accumulated evidence.

    ``refinable`` is decided by the caller (rounds left, a fresh query exists, the last round added
    something); ``stop_reason`` explains a ``False``.
    """
    files = len({r.file_path for r in evidence})
    if not evidence:
        return Assessment(
            outcome=AssessmentOutcome.EMPTY,
            reason="no_evidence",
            term_coverage=None,
            evidence_count=0,
            stop_reason=None if refinable else stop_reason,
        )

    fraction, missing = coverage(terms, evidence_text(evidence))
    problems: list[str] = []
    if fraction is not None and fraction < config.min_term_coverage:
        problems.append("term_coverage_below_threshold")
    symbol_missing = (
        decision.intent is Intent.SYMBOL_LOOKUP
        and bool(decision.symbols)
        and not mentions_any(evidence, decision.symbols)
    )
    if symbol_missing:
        problems.append("symbol_not_in_evidence")
    if decision.intent is Intent.ARCHITECTURE and files < config.architecture_min_files:
        problems.append("too_few_files_for_architecture")

    common = {
        "term_coverage": fraction,
        "missing_terms": missing,
        "distinct_files": files,
        "evidence_count": len(evidence),
    }
    if not problems:
        return Assessment(
            outcome=AssessmentOutcome.POTENTIALLY_SUFFICIENT, reason="checks_passed", **common
        )
    outcome = AssessmentOutcome.INSUFFICIENT_REFINABLE if refinable else AssessmentOutcome.EXHAUSTED
    return Assessment(
        outcome=outcome,
        reason="+".join(problems),
        stop_reason=None if refinable else stop_reason,
        **common,
    )
