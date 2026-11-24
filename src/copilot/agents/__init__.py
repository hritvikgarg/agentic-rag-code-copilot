"""Agentic RAG workflow (Milestone 8): a LangGraph graph over the existing retriever and RAG path.

``AgentWorkflow(retriever, llm).run(question)`` classifies the question deterministically, retrieves
in up to a few bounded rounds, assesses the accumulated evidence with a lexical heuristic, and then
either generates a grounded answer through the guarded RAG service or abstains. See
``docs/agent.md``. The baseline ``copilot.rag`` path is unchanged and independent of this package.
"""

from copilot.agents.intent import Intent, IntentDecision, classify_intent
from copilot.agents.state import (
    AgentConfig,
    Assessment,
    AssessmentOutcome,
    FailureInfo,
    TerminationReason,
    TraceEvent,
)
from copilot.agents.workflow import AgentResult, AgentWorkflow

__all__ = [
    "AgentConfig",
    "AgentResult",
    "AgentWorkflow",
    "Assessment",
    "AssessmentOutcome",
    "FailureInfo",
    "Intent",
    "IntentDecision",
    "TerminationReason",
    "TraceEvent",
    "classify_intent",
]
