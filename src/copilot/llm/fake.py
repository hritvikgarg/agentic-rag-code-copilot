"""Deterministic in-memory client for tests and dry runs. Never touches the network."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from copilot.llm.models import LLMRequest, LLMResponse


class FakeLLMClient:
    """Returns canned text and records every request so tests can inspect what was "sent".

    ``reply`` may be a string (always returned), a sequence (returned in order, the last one
    repeats) or a callable ``request -> text``. ``error`` (if given) is raised on every call
    after the request has been recorded.
    """

    provider = "fake"

    def __init__(
        self,
        reply: str | Sequence[str] | Callable[[LLMRequest], str] = "fake answer",
        *,
        model: str = "fake-model",
        error: Exception | None = None,
    ) -> None:
        self.model = model
        self._reply = reply
        self._error = error
        self.requests: list[LLMRequest] = []

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def _next_text(self, request: LLMRequest) -> str:
        if callable(self._reply):
            return self._reply(request)
        if isinstance(self._reply, str):
            return self._reply
        return self._reply[min(len(self.requests) - 1, len(self._reply) - 1)]

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return LLMResponse(
            text=self._next_text(request),
            provider=self.provider,
            model=self.model,
            finish_reason="STOP",
            prompt_tokens=request.estimated_tokens,
            completion_tokens=0,
            total_tokens=request.estimated_tokens,
            latency_seconds=0.0,
        )

    def __repr__(self) -> str:
        return f"FakeLLMClient(model={self.model!r}, calls={self.call_count})"
