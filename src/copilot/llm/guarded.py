"""The choke point that puts the secret gate immediately before every external request.

``GuardedLLMClient`` wraps a client and scans the *exact* outbound strings of each request with
``assert_safe_for_external_llm`` before delegating. If the gate raises, the wrapped client is
not called. The RAG and plain services always wrap the client they are given, and the factory
returns a guarded client, so no code path can reach a hosted model without the scan. There is
no parameter, flag or setting that disables it.
"""

from __future__ import annotations

from copilot.llm.base import LLMClient
from copilot.llm.models import LLMRequest, LLMResponse
from copilot.security import assert_safe_for_external_llm


class GuardedLLMClient:
    """``LLMClient`` wrapper: scan first, call only if the scan is clean."""

    def __init__(self, inner: LLMClient) -> None:
        # Unwrap so a double wrap does not scan twice; the innermost client is what is guarded.
        self._inner: LLMClient = inner._inner if isinstance(inner, GuardedLLMClient) else inner

    @property
    def provider(self) -> str:
        return self._inner.provider

    @property
    def model(self) -> str:
        return self._inner.model

    def generate(self, request: LLMRequest) -> LLMResponse:
        assert_safe_for_external_llm(request.scan_targets())  # raises RepositorySecretRiskError
        return self._inner.generate(request)

    def __repr__(self) -> str:
        return f"GuardedLLMClient(provider={self.provider!r}, model={self.model!r})"
