"""Typed answers of the plain baseline and the RAG service."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from copilot.llm import LLMResponse
from copilot.rag.citations import Citation
from copilot.rag.context import DroppedEvidence


class AnswerStatus(StrEnum):
    """Structural outcome. ``ANSWERED`` means the model produced text, not that it is correct."""

    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class GenerationInfo(BaseModel):
    """Safe metadata of one model call (no request text, no credentials)."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_seconds: float | None = None

    @classmethod
    def from_response(cls, response: LLMResponse) -> GenerationInfo:
        return cls(
            provider=response.provider,
            model=response.model,
            finish_reason=response.finish_reason,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_seconds=response.latency_seconds,
        )


class PlainAnswer(BaseModel):
    """The plain-LLM baseline result: NO repository grounding, NO citations."""

    model_config = ConfigDict(frozen=True)

    status: AnswerStatus = AnswerStatus.ANSWERED
    question: str
    answer: str
    prompt_version: str
    generation: GenerationInfo


class RagAnswer(BaseModel):
    """The RAG result. ``sources`` are the authoritative citations (retrieval metadata)."""

    model_config = ConfigDict(frozen=True)

    status: AnswerStatus
    question: str
    answer: str | None  # None when status is INSUFFICIENT_EVIDENCE (the model was not called)
    insufficient_reason: str | None = None  # "no_results" | "context_budget"
    sources: tuple[Citation, ...]  # exactly what was sent to the model, numbered Source 1..n
    cited_source_numbers: tuple[int, ...] = ()  # supplied sources the prose mentions
    unknown_source_numbers: tuple[int, ...] = ()  # labels in the prose with no supplied source
    dropped_evidence: tuple[DroppedEvidence, ...] = ()
    retrieved_count: int = Field(ge=0)
    context_estimated_tokens: int = Field(ge=0)
    prompt_version: str
    generation: GenerationInfo | None = None
    retrieval_seconds: float = Field(ge=0.0)
    generation_seconds: float | None = Field(default=None, ge=0.0)
