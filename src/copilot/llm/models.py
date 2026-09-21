"""Typed request/response of the provider-independent LLM interface.

``LLMRequest`` holds *exactly* the text that will leave the machine: ``outbound_text()`` returns
it and ``scan_targets()`` hands the same two strings to the secret gate. The texts are excluded
from ``repr`` so logging a request object cannot dump repository content. ``LLMResponse`` carries
generated text plus safe metadata only.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from copilot.security import ScanTarget
from copilot.utils.tokens import estimate_tokens

SYSTEM_TARGET_PATH = "llm-request/system-instruction.txt"
USER_TARGET_PATH = "llm-request/user-prompt.txt"


class LLMRequest(BaseModel):
    """One generation request. No API key, no provider-specific fields."""

    model_config = ConfigDict(frozen=True)

    system_instruction: str = Field(min_length=1, repr=False)
    user_prompt: str = Field(min_length=1, repr=False)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, ge=1)
    prompt_version: str = Field(min_length=1)  # e.g. "rag-v1"; recorded for reproducibility

    def outbound_text(self) -> str:
        """The exact text sent to the provider (system instruction, blank line, user prompt)."""
        return f"{self.system_instruction}\n\n{self.user_prompt}"

    def scan_targets(self) -> tuple[ScanTarget, ...]:
        """The exact outbound strings as secret-gate targets (relative pseudo paths)."""
        return (
            ScanTarget(SYSTEM_TARGET_PATH, self.system_instruction),
            ScanTarget(USER_TARGET_PATH, self.user_prompt),
        )

    @property
    def estimated_tokens(self) -> int:
        """Estimated size of the outbound text (the project's chunk token heuristic)."""
        return estimate_tokens(self.outbound_text())


class LLMResponse(BaseModel):
    """A provider's answer plus safe metadata."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    provider: str
    model: str
    finish_reason: str | None = None  # provider's own label, e.g. "STOP" or "MAX_TOKENS"
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_seconds: float | None = Field(default=None, ge=0.0)
