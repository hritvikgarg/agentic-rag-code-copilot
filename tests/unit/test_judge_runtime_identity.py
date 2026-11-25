"""Milestone 11B: a local judge's runtime fingerprint isolates cache, records and resume.

``LocalFake`` is a FakeLLMClient that also reports a RuntimeIdentity, like ``OllamaLLMClient``.
Nothing here touches a network.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from copilot.evaluation.generation.judge import JudgeCache, LLMJudge, judge_cache_key
from copilot.evaluation.generation.judging import (
    JudgementFile,
    JudgementMismatchError,
    load_judgements,
)
from copilot.llm import (
    FakeLLMClient,
    GuardedLLMClient,
    LLMConnectionError,
    LLMContextLimitError,
    RuntimeIdentity,
)
from copilot.llm.retry import RetryPolicy
from tests.security import secret_helpers as h
from tests.unit.test_generation_judge import GOOD, NO_SLEEP, _question, _reference
from tests.unit.test_generation_judging import (
    _generation,
    _judge_all,
    _judge_reply,
)

FAST = RetryPolicy(initial_delay_seconds=0.0, max_attempts=2)

# Frozen from the code as it was BEFORE local judges existed (commit 91a1671). Changing either
# value means every cached Gemini judgement would silently stop being found.
GEMINI_GOLDEN = {
    "0" * 64: "db28448a68abdaa1c7d85c6780b1764e5ebd964906c279e3de1f4607953c45e4",
    "ab" * 32: "04104d1cc09c9e29107c4c7d235d97bdeeb4b6144104b0781b96638c250b825d",
}


def identity(**kw) -> RuntimeIdentity:
    base = {
        "provider": "local",
        "model_tag": "llama3.1:8b",
        "model_digest": "a" * 64,
        "quantization": "Q4_K_M",
        "parameter_size": "8.0B",
        "runtime_version": "0.12.3",
        "context_size": 8192,
        "max_output_tokens": 1024,
        "temperature": 0.0,
        "seed": 20260,
        "structured_output": "json",
    }
    base.update(kw)
    return RuntimeIdentity(**base)


class LocalFake(FakeLLMClient):
    provider = "local"

    def __init__(self, reply=GOOD, *, ident: RuntimeIdentity | None = None, id_error=None, **kw):
        super().__init__(reply, model="llama3.1:8b", **kw)
        self.ident = ident or identity()
        self.id_error = id_error
        self.identity_calls = 0

    def runtime_identity(self, *, temperature: float, max_output_tokens: int):
        self.identity_calls += 1
        if self.id_error is not None:
            raise self.id_error
        return self.ident.model_copy(
            update={"temperature": temperature, "max_output_tokens": max_output_tokens}
        )


def judge_for(llm, cache=None, **kw) -> LLMJudge:
    return LLMJudge(llm, max_output_tokens=1024, retry_policy=FAST, cache=cache, **NO_SLEEP, **kw)


# ---------------------------------------------------------------- Gemini compatibility


@pytest.mark.parametrize(("sha", "expected"), GEMINI_GOLDEN.items())
def test_hosted_judge_cache_keys_are_unchanged(sha, expected):
    key = judge_cache_key(
        provider="gemini",
        model="gemini-3.6-flash",
        temperature=0.0,
        max_output_tokens=4096,
        input_sha256=sha,
    )
    assert key == expected
    same = judge_cache_key(
        provider="gemini", model="gemini-3.6-flash", temperature=0.0, max_output_tokens=4096,
        input_sha256=sha, runtime_fingerprint=None,
    )  # fmt: skip
    assert same == expected


def test_a_fingerprint_changes_the_key():
    args = {
        "provider": "p",
        "model": "m",
        "temperature": 0.0,
        "max_output_tokens": 1,
        "input_sha256": "x",
    }
    base = judge_cache_key(**args)
    assert judge_cache_key(**args, runtime_fingerprint="f1") != base
    assert judge_cache_key(**args, runtime_fingerprint="f1") != judge_cache_key(
        **args, runtime_fingerprint="f2"
    )


def test_a_hosted_style_judge_has_no_fingerprint_and_the_old_key(tmp_path):
    llm = FakeLLMClient(GOOD)
    judge = judge_for(llm, JudgeCache(tmp_path / "c"))
    record, _ = judge.judge(_question(), "a() returns one.", _reference(_question()), blind_id="A")
    assert record.state == "successful" and record.runtime_fingerprint is None
    sha = record.input_sha256
    key = judge_cache_key(
        provider="fake",
        model="fake-model",
        temperature=0.0,
        max_output_tokens=1024,
        input_sha256=sha,
    )
    assert (tmp_path / "c" / f"{key}.json").is_file()


def test_older_judgement_files_without_runtime_fields_still_load(tmp_path):
    run, ctx = _generation(tmp_path)
    out = _judge_all(run, ctx, judge_for(FakeLLMClient(_judge_reply)), tmp_path)
    raw = json.loads(out.model_dump_json())
    raw.pop("judge_runtime_fingerprint"), raw.pop("judge_runtime")
    path = tmp_path / "old.json"
    path.write_text(json.dumps(raw))
    loaded = load_judgements(path)
    assert loaded.judge_runtime_fingerprint is None and loaded.judge_runtime is None


# ---------------------------------------------------------------- cache isolation


def _judge_once(llm, cache):
    judge = judge_for(llm, cache)
    return judge.judge(_question(), "a() returns one.", _reference(_question()), blind_id="A")


def test_the_same_runtime_reuses_the_cache_and_a_different_one_does_not(tmp_path):
    cache = JudgeCache(tmp_path / "c")
    first = LocalFake()
    record, reused = _judge_once(first, cache)
    assert record.state == "successful" and reused is False
    assert record.runtime_fingerprint == identity().fingerprint()

    again = LocalFake()
    assert _judge_once(again, cache)[1] is True and again.call_count == 0

    for changed in (
        identity(model_digest="b" * 64),
        identity(quantization="Q8_0"),
        identity(runtime_version="0.13.0"),
        identity(context_size=4096),
        identity(seed=1),
        identity(structured_output="none"),
    ):
        other = LocalFake(ident=changed)
        assert _judge_once(other, cache)[1] is False, changed
        assert other.call_count == 1


def test_a_planted_entry_with_the_wrong_fingerprint_is_never_served(tmp_path):
    cache = JudgeCache(tmp_path / "c")
    donor = LocalFake(ident=identity(model_digest="c" * 64))
    record, _ = _judge_once(donor, cache)
    # Copy the donor's record under the key the *real* runtime would look up.
    judge = judge_for(LocalFake(), cache)
    request = judge.request_for(_question(), "a() returns one.", _reference(_question()))
    sha = hashlib.sha256(request.outbound_text().encode("utf-8")).hexdigest()
    key = judge_cache_key(
        provider="local", model="llama3.1:8b", temperature=0.0, max_output_tokens=1024,
        input_sha256=sha, runtime_fingerprint=identity().fingerprint(),
    )  # fmt: skip
    cache.put(key, record.model_copy(update={"input_sha256": sha}))
    _, reused = judge.judge(_question(), "a() returns one.", _reference(_question()), blind_id="B")
    assert reused is False  # fingerprint inside the record did not match


def test_the_secret_gate_runs_before_the_identity_lookup_cache_and_provider(tmp_path):
    llm = LocalFake()
    judge = judge_for(llm, JudgeCache(tmp_path / "c"))
    record, reused = judge.judge(
        _question(), f"token {h.github_token()}", _reference(_question()), blind_id="S"
    )
    assert (record.state, reused) == ("failed", False)
    assert "RepositorySecretRiskError" in (record.error or "")
    assert llm.identity_calls == 0 and llm.call_count == 0
    assert h.github_token() not in record.model_dump_json()
    assert not list((tmp_path / "c").glob("*.json")) if (tmp_path / "c").exists() else True


def test_an_unidentifiable_runtime_cannot_judge(tmp_path):
    llm = LocalFake(id_error=LLMConnectionError("could not reach Ollama at http://127.0.0.1:11434"))
    record, _ = _judge_once(llm, JudgeCache(tmp_path / "c"))
    assert record.state == "failed" and "LLMConnectionError" in (record.error or "")
    assert llm.call_count == 0


def test_the_guard_still_wraps_a_local_client_for_identity():
    llm = LocalFake()
    assert (
        GuardedLLMClient(llm)
        .runtime_identity(temperature=0.0, max_output_tokens=7)
        .max_output_tokens
        == 7
    )


# ---------------------------------------------------------------- judgement file and resume


def test_the_judgement_file_records_the_runtime_identity(tmp_path):
    run, ctx = _generation(tmp_path)
    llm = LocalFake(_judge_reply)
    out = _judge_all(run, ctx, judge_for(llm), tmp_path)
    expected = identity()
    assert out.judge_runtime_fingerprint == expected.fingerprint()
    assert out.judge_runtime is not None
    assert out.judge_runtime["model_digest"] == "a" * 64
    assert out.judge_runtime["quantization"] == "Q4_K_M" and out.judge_runtime["seed"] == 20260
    assert all(
        i.record.runtime_fingerprint == expected.fingerprint() for i in out.items if i.record
    )
    reloaded = load_judgements(tmp_path / "j.json")
    assert reloaded.judge_runtime_fingerprint == out.judge_runtime_fingerprint


def test_resume_with_the_same_runtime_is_accepted_and_reuses_items(tmp_path):
    run, ctx = _generation(tmp_path)
    first = _judge_all(run, ctx, judge_for(LocalFake(_judge_reply)), tmp_path)
    second_llm = LocalFake(_judge_reply)
    second = _judge_all(run, ctx, judge_for(second_llm), tmp_path, resume_from=first)
    assert second_llm.call_count == 0 and second.counts()["judged"] == first.counts()["judged"]


@pytest.mark.parametrize(
    "changed",
    [
        {"model_digest": "d" * 64},
        {"quantization": "Q8_0"},
        {"parameter_size": "70B"},
        {"runtime_version": "0.13.0"},
        {"context_size": 16384},
        {"seed": 99},
        {"structured_output": "none"},
    ],
)
def test_resume_with_a_different_runtime_is_refused(tmp_path, changed):
    run, ctx = _generation(tmp_path)
    first = _judge_all(run, ctx, judge_for(LocalFake(_judge_reply)), tmp_path)
    other = LocalFake(_judge_reply, ident=identity(**changed))
    with pytest.raises(JudgementMismatchError, match="judge_runtime_fingerprint"):
        _judge_all(run, ctx, judge_for(other), tmp_path, resume_from=first)
    assert other.call_count == 0


def test_a_hosted_file_cannot_be_resumed_by_a_local_judge_and_vice_versa(tmp_path):
    run, ctx = _generation(tmp_path)
    hosted = _judge_all(
        run, ctx, judge_for(FakeLLMClient(_judge_reply, model="llama3.1:8b")), tmp_path
    )
    with pytest.raises(JudgementMismatchError):
        _judge_all(run, ctx, judge_for(LocalFake(_judge_reply)), tmp_path, resume_from=hosted)
    local = _judge_all(run, ctx, judge_for(LocalFake(_judge_reply)), tmp_path)
    with pytest.raises(JudgementMismatchError):
        _judge_all(
            run, ctx, judge_for(FakeLLMClient(_judge_reply, model="llama3.1:8b")), tmp_path,
            resume_from=local,
        )  # fmt: skip


def test_context_failures_are_recorded_and_not_counted_as_valid_judgements(tmp_path):
    run, ctx = _generation(tmp_path)
    llm = LocalFake(error=LLMContextLimitError("the prompt is estimated at up to 9000 tokens"))
    out = _judge_all(run, ctx, judge_for(llm), tmp_path, max_consecutive_rate_limit_failures=None)
    counts = out.counts()
    assert counts["judged"] == 0 and counts["failed"] == 6
    assert out.context_failures() == 6
    assert isinstance(out, JudgementFile)
    assert all(i.assessment is None for i in out.items if i.state == "failed")
