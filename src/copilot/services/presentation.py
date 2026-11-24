"""UI-independent presentation helpers (pure functions, unit-tested without Streamlit).

Keeping these out of ``ui/streamlit_app.py`` means the rules that decide what a user is *told*
(warnings about truncation, unknown citations, insufficient evidence) are testable and cannot
drift between front-ends.
"""

from __future__ import annotations

import re

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


# --- literal text in Markdown widgets ----------------------------------------------------------
#
# Streamlit renders most widget text (expander labels, st.error/st.success/st.caption bodies,
# st.markdown) as Markdown. A repository path such as ``src/pkg/__init__.py`` therefore shows up
# as ``src/pkg/<b>init</b>.py`` (the double underscores read as bold). Two helpers keep literal text
# literal. Verified by rendering real Streamlit 1.67 in a browser (see docs/ui.md): ``st.selectbox``
# option labels are NOT Markdown, so they must not be escaped (the backslashes would show).

_PUNCTUATION = re.compile(r"([!-/:-@\[-`{-~])")


def literal(text: str) -> str:
    """An inline Markdown code span that displays ``text`` exactly as written.

    Use it for a fragment inside a Markdown string (a path, a repository name, an index id) where
    surrounding prose should still render normally. The fence is longer than any backtick run in
    ``text`` so the text can never close the span early; a leading/trailing backtick is padded with
    a space, which CommonMark strips. Newlines become spaces (a code span cannot contain them).
    """
    flat = " ".join(text.split("\n")) if text else ""
    if not flat.strip():
        return ""
    longest = max((len(run) for run in re.findall(r"`+", flat)), default=0)
    fence = "`" * (longest + 1)
    pad = " " if flat.startswith("`") or flat.endswith("`") else ""
    return f"{fence}{pad}{flat}{pad}{fence}"


def escape_markdown(text: str) -> str:
    """Backslash-escape every ASCII punctuation character so Markdown shows ``text`` verbatim.

    Use it for a whole message that is not meant to contain formatting (an error text). CommonMark
    allows a backslash before any ASCII punctuation, which also stops ``:emoji:``, ``$latex$`` and
    ``:red[colour]`` directives from being interpreted.
    """
    return _PUNCTUATION.sub(r"\\\1", text)


def source_title(label: str, location: str, score: float, *, cited: bool) -> str:
    """The expander title for one retrieved source; the location is shown literally."""
    suffix = " (cited in the answer)" if cited else ""
    return f"{escape_markdown(label)} - {literal(location)} - score {score:.3f}{suffix}"
