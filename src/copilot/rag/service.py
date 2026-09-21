"""Plain-LLM baseline and the basic synchronous RAG service (no agents, no graph).

RAG flow (``RagService.answer``)::

    validate question -> retrieve top-k (verified chunks) -> build bounded context
      -> [insufficient evidence? stop: the model is NOT called]
      -> secret-scan the exact question + the exact chunks that entered the context
      -> build the outbound request -> GuardedLLMClient.generate
           (scans the exact outbound strings again, then and only then calls the provider)
      -> answer + authoritative citations from retrieval metadata

The client given to a service is always wrapped in ``GuardedLLMClient``: there is no way to
reach the provider without the secret gate, and no argument that disables it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from copilot.config import Settings, get_settings
from copilot.embeddings.base import Embedder
from copilot.llm import GuardedLLMClient, LLMClient, LLMRequest
from copilot.rag.citations import resolve_source_numbers
from copilot.rag.context import build_context
from copilot.rag.models import AnswerStatus, GenerationInfo, PlainAnswer, RagAnswer
from copilot.rag.prompts import (
    PLAIN_PROMPT_VERSION,
    PLAIN_SYSTEM_PROMPT,
    RAG_PROMPT_VERSION,
    RAG_SYSTEM_PROMPT,
    build_plain_user_prompt,
    build_rag_user_prompt,
)
from copilot.retrieval import RetrievalResult, Retriever, validate_query, validate_top_k
from copilot.security import ScanTarget, assert_safe_for_external_llm

logger = logging.getLogger(__name__)

QUESTION_TARGET_PATH = "user-question.txt"


class SupportsRetrieve(Protocol):
    """What the RAG service needs from a retriever (``Retriever`` satisfies it)."""

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]: ...


def answer_plain(question: str, llm: LLMClient, *, settings: Settings | None = None) -> PlainAnswer:
    """The plain baseline: the question goes to the LLM with no repository context.

    The outbound text is scanned by the secret gate inside ``GuardedLLMClient`` before the
    provider is called. There is no repository grounding: answers about a specific repository
    come from the model's general knowledge (and may be invented).
    """
    settings = settings or get_settings()
    text = validate_query(question)
    request = LLMRequest(
        system_instruction=PLAIN_SYSTEM_PROMPT,
        user_prompt=build_plain_user_prompt(text),
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_output_tokens,
        prompt_version=PLAIN_PROMPT_VERSION,
    )
    response = GuardedLLMClient(llm).generate(request)
    return PlainAnswer(
        question=text,
        answer=response.text,
        prompt_version=PLAIN_PROMPT_VERSION,
        generation=GenerationInfo.from_response(response),
    )


class RagService:
    """Retrieve -> bound context -> gate -> LLM -> answer with retrieval-derived citations."""

    def __init__(
        self,
        retriever: SupportsRetrieve,
        llm: LLMClient,
        *,
        settings: Settings | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = GuardedLLMClient(llm)  # the gate cannot be skipped
        self._settings = settings or get_settings()

    def answer(self, question: str, top_k: int | None = None) -> RagAnswer:
        settings = self._settings
        text = validate_query(question)
        k = validate_top_k(top_k if top_k is not None else settings.retrieval_top_k)

        started = time.perf_counter()
        results: Sequence[RetrievalResult] = self._retriever.retrieve(text, k)
        retrieval_seconds = round(time.perf_counter() - started, 3)

        context = build_context(results, settings.rag_context_max_tokens)
        common = {
            "question": text,
            "retrieved_count": len(results),
            "prompt_version": RAG_PROMPT_VERSION,
            "retrieval_seconds": retrieval_seconds,
            "dropped_evidence": context.dropped,
        }
        if not context.citations:
            reason = "no_results" if not results else "context_budget"
            logger.info("RAG: insufficient evidence (%s); the model is not called", reason)
            return RagAnswer(
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                answer=None,
                insufficient_reason=reason,
                sources=(),
                context_estimated_tokens=0,
                **common,
            )

        # Gate 1 (diagnostics): the question and each chunk that will leave, with real file:line.
        assert_safe_for_external_llm([ScanTarget(QUESTION_TARGET_PATH, text), *context.included])

        request = LLMRequest(
            system_instruction=RAG_SYSTEM_PROMPT,
            user_prompt=build_rag_user_prompt(text, context.text, len(context.citations)),
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_output_tokens,
            prompt_version=RAG_PROMPT_VERSION,
        )
        started = time.perf_counter()
        response = self._llm.generate(request)  # Gate 2: exact outbound text, then the provider
        generation_seconds = round(time.perf_counter() - started, 3)

        cited, unknown = resolve_source_numbers(response.text, context.citations)
        if unknown:
            logger.warning("RAG: the answer mentions source numbers that were not supplied")
        return RagAnswer(
            status=AnswerStatus.ANSWERED,
            answer=response.text,
            sources=context.citations,
            cited_source_numbers=cited,
            unknown_source_numbers=unknown,
            context_estimated_tokens=context.estimated_tokens,
            generation=GenerationInfo.from_response(response),
            generation_seconds=generation_seconds,
            **common,
        )


def answer_with_rag(
    question: str,
    repo_path: str | Path,
    index: str | Path,
    top_k: int | None = None,
    *,
    llm: LLMClient,
    embedder: Embedder,
    settings: Settings | None = None,
    ignore_directories: Sequence[str] = (),
) -> RagAnswer:
    """One-shot RAG: open the index (verifying the repository), then answer."""
    settings = settings or get_settings()
    validate_query(question)  # fail on bad input before loading anything
    retriever = Retriever.open(
        index,
        repo_path,
        embedder=embedder,
        settings=settings,
        ignore_directories=ignore_directories,
    )
    return RagService(retriever, llm, settings=settings).answer(question, top_k)
