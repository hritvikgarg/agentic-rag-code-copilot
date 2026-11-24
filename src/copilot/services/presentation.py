"""UI-independent presentation helpers (pure functions, unit-tested without Streamlit).

Keeping these out of ``ui/streamlit_app.py`` means the rules that decide what a user is *told*
(warnings about truncation, unknown citations, insufficient evidence) are testable and cannot
drift between front-ends.
"""

from __future__ import annotations

from copilot.chunking.errors import ChunkingError
from copilot.embeddings.errors import EmbeddingError
from copilot.ingestion.errors import IngestionError
from copilot.llm import LLMError
from copilot.rag.models import AnswerStatus, RagAnswer
from copilot.retrieval.errors import RetrievalError
from copilot.security import SecurityError
from copilot.services.copilot_service import ServiceStateError
from copilot.vectorstore.errors import VectorStoreError

# Exceptions whose messages are fixed, secret-free and meant to be read by a user. Anything else
# is a bug and must NOT be shown raw (it could carry paths); the UI shows a generic message.
USER_FACING_ERRORS: tuple[type[Exception], ...] = (
    SecurityError,
    LLMError,
    RetrievalError,
    VectorStoreError,
    EmbeddingError,
    IngestionError,
    ChunkingError,
    ServiceStateError,
    ValueError,
)


def parse_ignore_directories(text: str) -> list[str]:
    """``"data, tests ,submission"`` -> ``["data", "tests", "submission"]`` (commas or newlines)."""
    parts = text.replace("\n", ",").split(",")
    seen: list[str] = []
    for part in parts:
        name = part.strip().strip("/\\")
        if name and name not in seen:
            seen.append(name)
    return seen


def answer_warnings(answer: RagAnswer) -> list[str]:
    """Plain-language caveats a reader must see next to an answer (never empty strings)."""
    warnings: list[str] = []
    if answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE:
        reason = {
            "no_results": "retrieval returned nothing for this question",
            "context_budget": "no retrieved chunk fit the context budget",
        }.get(answer.insufficient_reason or "", "the retrieved evidence was insufficient")
        warnings.append(
            f"Insufficient repository evidence: {reason}. The model was not called, so nothing "
            "below is a guess."
        )
        return warnings
    if answer.unknown_source_numbers:
        numbers = ", ".join(str(n) for n in answer.unknown_source_numbers)
        warnings.append(
            f"The answer mentions source numbers that were not supplied ({numbers}); treat those "
            "references as invalid."
        )
    if not answer.cited_source_numbers:
        warnings.append(
            "The answer does not cite any source by number. The sources listed below were "
            "retrieved and supplied to the model, but the answer text is not tied to them."
        )
    generation = answer.generation
    if generation is not None and generation.finish_reason == "MAX_TOKENS":
        warnings.append(
            "The model stopped at its output limit; the answer may be cut off. Raise "
            "COPILOT_LLM_MAX_OUTPUT_TOKENS (hidden reasoning tokens share that budget)."
        )
    if answer.dropped_evidence:
        warnings.append(
            f"{len(answer.dropped_evidence)} retrieved chunk(s) did not fit the context budget "
            "and were not shown to the model."
        )
    return warnings
