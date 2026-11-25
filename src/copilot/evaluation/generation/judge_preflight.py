"""``judge-preflight``: is the local judge runtime present and as expected? (Milestone 11B)

Metadata calls only (version, installed models, running models): it never pulls a model, never
sends repository content and fails clearly when the server is down or the model is not installed.
``warm_up`` additionally loads the model with one constant, repository-free prompt (through the
secret gate like every other request) so GPU residency can be inspected afterwards.

``run_hosted_preflight`` is the offline counterpart for a hosted judge: it makes **no provider
call**. It checks the selection and configuration, states the self-preference risk, and computes the
call budget of every validation stage so the operator can compare it with the quota before spending
any of it.

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


# --------------------------------------------------------------------------- hosted (offline)

STAGES = (
    ("controls_first_pass", "controls"),
    ("controls_repeat", "controls"),
    ("pilot_judging", "pilot"),
    ("pilot_repeat", "pilot"),
    ("primary_judging", "primary"),
)


def hosted_call_budget(
    *, controls: int, pilot_answers: int, primary_answers: int, max_attempts: int
) -> dict[str, Any]:
    """Upper bounds on hosted judge calls per stage, without and with the retry bound.

    ``max_calls`` is one attempt per item. ``max_calls_with_retries`` multiplies by
    ``max_attempts`` (the retry policy's total attempts per request; only HTTP 429/503/504 and
    timeouts are retried, so it is reached only if every request keeps failing transiently). Items
    whose answer is a rule-scored RAG abstention and items already in the judge cache need no
    call, so the real number can only be lower. Each *resume* re-attempts failed items and spends
    calls again; that is outside these per-run bounds.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    size = {"controls": controls, "pilot": pilot_answers, "primary": primary_answers}
    stages = {
        name: {
            "max_calls": size[kind],
            "max_calls_with_retries": size[kind] * max_attempts,
        }
        for name, kind in STAGES
    }
    through_pilot = [n for n, _ in STAGES if n != "primary_judging"]
    stages["validation_and_pilot_total"] = {
        "max_calls": sum(stages[n]["max_calls"] for n in through_pilot),
        "max_calls_with_retries": sum(stages[n]["max_calls_with_retries"] for n in through_pilot),
    }
    return {"retry_attempts_per_call": max_attempts, "stages": stages}


def run_hosted_preflight(
    *,
    provider: str,
    judge_model: str,
    generation_model: str | None,
    api_key_present: bool,
    budget: dict[str, Any],
    judgements_dir: str,
    cache_dir: str | None,
    profile_version: str,
    profile_sha256: str,
    spec_sha256: str,
    controls_sha256: str,
    controls: int,
    prompt_version: str,
    rubric_version: str,
) -> dict[str, Any]:
    """Offline checks for a hosted judge. Never constructs a client and never calls a provider."""
    checks: list[dict[str, Any]] = [
        {"check": "judge_model_named_explicitly", "ok": bool(judge_model.strip())},
        {"check": "judge_provider_is_gemini", "ok": provider == "gemini", "actual": provider},
        {
            "check": "api_key_configured",
            "ok": api_key_present,
            "note": "presence only; never shown",
        },
    ]
    warnings: list[str] = []
    if not api_key_present:
        warnings.append("no GEMINI_API_KEY is configured; the first hosted call would fail")
    if generation_model is None:
        same: bool | None = None
        warnings.append(
            "the generation model is unknown (pass --confirm-model PROVIDER/MODEL); the "
            "self-preference check could not be made"
        )
    else:
        same = generation_model == f"{provider}/{judge_model}"
        if same:
            warnings.append(
                "the judge model IS the generation model: LLM judges can favour their own "
                "model's text (self-preference bias); results must state this"
            )
        else:
            warnings.append(
                "the judge model differs from the generation model; both are Gemini-family, "
                "which reduces but does not remove self-preference"
            )
    return {
        "provider": provider,
        "judge_model": judge_model,
        "generation_model": generation_model,
        "same_model_as_generator": same,
        "judge_prompt_version": prompt_version,
        "rubric_version": rubric_version,
        "validation_spec_sha256": spec_sha256,
        "hosted_profile": {"version": profile_version, "sha256": profile_sha256},
        "controls": {"model_controls": controls, "sha256": controls_sha256},
        "judgements_dir": judgements_dir,
        "cache_dir": cache_dir,
        "budget": budget,
        "checks": checks,
        "warnings": warnings,
        "hosted_calls_made": 0,
        "ok": all(c["ok"] for c in checks),
    }
