"""Grounded answer generation (Milestone 6): plain baseline, bounded context, citations, RAG.

See ``docs/rag.md``. No agents, no graph: a single synchronous retrieve -> gate -> generate flow.
"""

from copilot.rag.citations import Citation, extract_source_numbers, resolve_source_numbers
from copilot.rag.context import BuiltContext, DroppedEvidence, build_context, format_block
from copilot.rag.models import AnswerStatus, GenerationInfo, PlainAnswer, RagAnswer
from copilot.rag.prompts import (
    PLAIN_PROMPT_VERSION,
    PLAIN_SYSTEM_PROMPT,
    RAG_PROMPT_VERSION,
    RAG_SYSTEM_PROMPT,
)
from copilot.rag.service import RagService, SupportsRetrieve, answer_plain, answer_with_rag

__all__ = [
    "PLAIN_PROMPT_VERSION",
    "PLAIN_SYSTEM_PROMPT",
    "RAG_PROMPT_VERSION",
    "RAG_SYSTEM_PROMPT",
    "AnswerStatus",
    "BuiltContext",
    "Citation",
    "DroppedEvidence",
    "GenerationInfo",
    "PlainAnswer",
    "RagAnswer",
    "RagService",
    "SupportsRetrieve",
    "answer_plain",
    "answer_with_rag",
    "build_context",
    "extract_source_numbers",
    "resolve_source_numbers",
    "format_block",
]
