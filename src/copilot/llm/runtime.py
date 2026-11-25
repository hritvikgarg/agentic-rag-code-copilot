"""Deterministic identity of a local model runtime (Milestone 11B).

A hosted model is identified by its name; a *local* model is only identified by its name plus the
exact weights file (digest), the quantisation, the server version and every setting that changes
the generated text. ``RuntimeIdentity`` collects those, and ``fingerprint()`` hashes them into one
stable string that is part of a local judge's cache key, its judgement records, its judgement file
metadata and its resume checks.

This is identity, not a determinism claim: the same fingerprint on different hardware, drivers or
inference-library builds may still produce different text. Within one machine and one runtime it
guarantees the converse - a changed model file, quantisation, server version, context size, seed,
temperature, output budget or structured-output mode changes the fingerprint, so cached answers
and half-finished runs are never silently reused across them.

``keep_alive`` is deliberately excluded: it controls how long the model stays in memory, not what
it generates.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field


class RuntimeIdentity(BaseModel):
    """Everything that identifies a local model run, as read from the runtime itself."""

    model_config = ConfigDict(frozen=True)

    provider: str = Field(min_length=1)
    model_tag: str = Field(min_length=1)
    model_digest: str = Field(min_length=8)
    quantization: str = Field(min_length=1)
    parameter_size: str = Field(min_length=1)
    runtime_version: str = Field(min_length=1)
    context_size: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    temperature: float = Field(ge=0.0)
    seed: int
    structured_output: str = Field(min_length=1)  # "json" or "none"

    def canonical_json(self) -> str:
        """Stable serialisation (sorted keys, no whitespace) that the fingerprint is taken over."""
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )

    def fingerprint(self) -> str:
        """SHA-256 (hex) of ``canonical_json()``."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
