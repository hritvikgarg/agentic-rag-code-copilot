"""``judge-preflight``: is the local judge runtime present and as expected? (Milestone 11B)

Metadata calls only (version, installed models, running models): it never pulls a model, never
sends repository content and fails clearly when the server is down or the model is not installed.
``warm_up`` additionally loads the model with one constant, repository-free prompt (through the
secret gate like every other request) so GPU residency can be inspected afterwards.

Which commands need a running Ollama server: ``judge-preflight``, ``judge`` (always - even when
every answer would come from the cache, because the runtime fingerprint is read from the server),
``judge-controls`` and ``judge-validate --repeat``. Offline: ``report``, ``judge-compare``,
``audit-worksheet``, ``audit-agree`` and ``judge-validate`` over existing artifacts.
"""

from __future__ import annotations

from typing import Any

from copilot.llm import GuardedLLMClient, LLMError, LLMRequest

WARM_UP_SYSTEM = "You are a connectivity check. Reply with JSON only."
WARM_UP_USER = 'Reply with exactly this JSON object: {"status": "ready"}'


def run_preflight(
    llm: GuardedLLMClient,
    *,
    temperature: float,
    max_output_tokens: int,
    expect_quantization: str | None = None,
    require_gpu: bool = False,
    warm_up: bool = False,
) -> dict[str, Any]:
    """Raises ``LLMError`` when the runtime cannot be reached or the model is not installed."""
    description = llm.describe_runtime()
    if description is None:
        raise LLMError("judge-preflight only applies to a local (ollama) judge")
    checks: list[dict[str, Any]] = []
    warnings: list[str] = list(description.get("notes", []))
    if warm_up:
        request = LLMRequest(
            system_instruction=WARM_UP_SYSTEM,
            user_prompt=WARM_UP_USER,
            temperature=temperature,
            max_output_tokens=min(max_output_tokens, 64),
            prompt_version="judge-preflight-warmup",
        )
        response = llm.generate(request)
        description = llm.describe_runtime() or description  # re-read residency after loading
        description["warm_up"] = {
            "latency_seconds": response.latency_seconds,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
        }
    model = description["model"]
    if expect_quantization is not None:
        checks.append(
            {
                "check": "quantization",
                "expected": expect_quantization,
                "actual": model["quantization"],
                "ok": model["quantization"].lower() == expect_quantization.lower(),
            }
        )
    loaded = description["loaded"]
    if loaded is None:
        warnings.append(
            "the model is not loaded right now, so GPU residency is unknown; run with --warm-up "
            "(or `ollama run`) and check again"
        )
        if require_gpu:
            checks.append({"check": "gpu_residency", "expected": 1.0, "actual": None, "ok": False})
    else:
        fraction = loaded["gpu_fraction"]
        if fraction is None or fraction < 1.0:
            warnings.append(
                f"the model is only {fraction if fraction is not None else 'unknown'} resident on "
                "the GPU; part of it runs on the CPU, which is much slower"
            )
        if require_gpu:
            checks.append(
                {
                    "check": "gpu_residency",
                    "expected": 1.0,
                    "actual": fraction,
                    "ok": fraction is not None and fraction >= 1.0,
                }
            )
    identity = llm.runtime_identity(temperature=temperature, max_output_tokens=max_output_tokens)
    return {
        **description,
        "runtime_identity": identity.model_dump(mode="json") if identity else None,
        "runtime_fingerprint": identity.fingerprint() if identity else None,
        "checks": checks,
        "warnings": warnings,
        "ok": all(c["ok"] for c in checks),
    }
