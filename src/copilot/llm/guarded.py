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
from copilot.llm.runtime import RuntimeIdentity
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

    def runtime_identity(
        self, *, temperature: float, max_output_tokens: int
    ) -> RuntimeIdentity | None:
        """The wrapped client's runtime identity, or ``None`` for hosted providers (Gemini).

        A metadata call only (model digest, server version); it carries no request text.
        """
        identity = getattr(self._inner, "runtime_identity", None)
        if identity is None:
            return None
        return identity(temperature=temperature, max_output_tokens=max_output_tokens)

    def describe_runtime(self) -> dict | None:
        """Metadata-only description of a local runtime (``None`` for hosted providers)."""
        describe = getattr(self._inner, "describe_runtime", None)
        return describe() if describe is not None else None

    def generate(self, request: LLMRequest) -> LLMResponse:
        assert_safe_for_external_llm(request.scan_targets())  # raises RepositorySecretRiskError
        return self._inner.generate(request)

    def __repr__(self) -> str:
        return f"GuardedLLMClient(provider={self.provider!r}, model={self.model!r})"
