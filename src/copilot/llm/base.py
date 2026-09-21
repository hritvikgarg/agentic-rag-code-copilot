"""The minimal provider-independent interface the RAG layer depends on."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from copilot.llm.models import LLMRequest, LLMResponse


@runtime_checkable
class LLMClient(Protocol):
    """Anything that turns an ``LLMRequest`` into an ``LLMResponse``.

    Implementations must raise ``copilot.llm.errors.LLMError`` subclasses for failures and must
    never include request text or credentials in an exception message.
    """

    provider: str
    model: str

    def generate(self, request: LLMRequest) -> LLMResponse: ...
