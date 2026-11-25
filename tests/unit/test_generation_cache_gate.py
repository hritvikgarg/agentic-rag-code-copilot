"""Milestone 11A corrections: the secret gate runs BEFORE any cache can serve a response.

FakeLLMClient only. A "planted" cache entry simulates a stale or tampered cache: it must never be
served for a request the gate would block, and blocked input must cost zero provider calls and
leave no copy of the sensitive text in any new artifact.
"""

from __future__ import annotations

import hashlib

import pytest

from copilot.evaluation.generation.execution import MeteredLLM, ResponseCache, response_cache_key
from copilot.evaluation.generation.judge import JudgeCache, LLMJudge, judge_cache_key
from copilot.llm import FakeLLMClient, GuardedLLMClient, LLMRequest, LLMResponse
from copilot.security import RepositorySecretRiskError
from tests.security import secret_helpers as h
from tests.unit.test_generation_judge import GOOD, NO_SLEEP, _question, _reference

SAFE_ANSWER = "a() returns one."


def _secret_answer() -> str:
    return f"Use this token {h.github_token()} to call it."


def _planted(judge: LLMJudge, cache: JudgeCache, answer: str) -> str:
    """Write a *successful* record under exactly the key this request would use."""
    request = judge.request_for(_question(), answer, _reference(_question()))
    sha = hashlib.sha256(request.outbound_text().encode("utf-8")).hexdigest()
    key = judge_cache_key(
        provider=judge.llm.provider, model=judge.llm.model, temperature=judge.temperature,
        max_output_tokens=judge.max_output_tokens, input_sha256=sha,
    )  # fmt: skip
    good = LLMJudge(FakeLLMClient(GOOD)).judge(
        _question(), SAFE_ANSWER, _reference(_question()), blind_id="X"
    )[0]
    cache.put(key, good.model_copy(update={"input_sha256": sha}))
    return key


def _judge(tmp_path, reply=GOOD):
    llm = FakeLLMClient(reply)
    cache = JudgeCache(tmp_path / "jc")
    return LLMJudge(llm, cache=cache, **NO_SLEEP), llm, cache


def test_safe_input_misses_then_hits_the_cache(tmp_path):
    judge, llm, _ = _judge(tmp_path)
    first, reused1 = judge.judge(_question(), SAFE_ANSWER, _reference(_question()), blind_id="A")
    second, reused2 = judge.judge(_question(), SAFE_ANSWER, _reference(_question()), blind_id="B")
    assert (first.state, reused1) == ("successful", False)
    assert (second.state, reused2) == ("successful", True)
    assert llm.call_count == 1


def test_secret_input_with_a_planted_cache_entry_is_blocked_not_served(tmp_path):
    judge, llm, cache = _judge(tmp_path)
    answer = _secret_answer()
    key = _planted(judge, cache, answer)
    assert cache.get(key) is not None  # the planted entry really is readable
    record, reused = judge.judge(_question(), answer, _reference(_question()), blind_id="A")
    assert record.state == "failed" and reused is False
    assert "RepositorySecretRiskError" in (record.error or "")
    assert llm.call_count == 0


def test_secret_input_without_an_entry_is_blocked_with_zero_calls(tmp_path):
    judge, llm, cache = _judge(tmp_path)
    record, reused = judge.judge(
        _question(), _secret_answer(), _reference(_question()), blind_id="A"
    )
    assert record.state == "failed" and reused is False and llm.call_count == 0
    assert not list((tmp_path / "jc").glob("*.json"))  # nothing cached for blocked input


def test_blocked_input_is_not_persisted_in_any_new_artifact(tmp_path):
    judge, llm, cache = _judge(tmp_path)
    answer = _secret_answer()
    record, _ = judge.judge(_question(), answer, _reference(_question()), blind_id="A")
    token = h.github_token()
    assert token not in record.model_dump_json() and token not in (record.raw_output or "")
    for path in (tmp_path / "jc").rglob("*"):
        if path.is_file():
            assert token not in path.read_text(encoding="utf-8")
    assert llm.call_count == 0


def test_a_blocked_failed_record_is_never_cached_and_does_not_poison_safe_input(tmp_path):
    judge, llm, cache = _judge(tmp_path)
    judge.judge(_question(), _secret_answer(), _reference(_question()), blind_id="A")
    ok, reused = judge.judge(_question(), SAFE_ANSWER, _reference(_question()), blind_id="B")
    assert ok.state == "successful" and reused is False and llm.call_count == 1


# --- generation cache (MeteredLLM) isolation -------------------------------------------------


def _request(text: str) -> LLMRequest:
    return LLMRequest(
        system_instruction="sys", user_prompt=text, temperature=0.0, max_output_tokens=64,
        prompt_version="t-1",
    )  # fmt: skip


def _plant(cache: ResponseCache, metered: MeteredLLM, request: LLMRequest) -> None:
    key, _ = response_cache_key(metered.provider, metered.model, request)
    cache.put(key, LLMResponse(text="planted", provider="fake", model="fake-model"))


@pytest.mark.parametrize("guarded", [False, True])
def test_metered_llm_blocks_secret_input_even_with_a_planted_entry(tmp_path, guarded):
    inner = FakeLLMClient("fresh")
    cache = ResponseCache(tmp_path / "gc")
    metered = MeteredLLM(inner, cache)
    request = _request(f"please use {h.github_token()}")
    _plant(cache, metered, request)
    client = GuardedLLMClient(metered) if guarded else metered
    with pytest.raises(RepositorySecretRiskError):
        client.generate(request)
    assert inner.call_count == 0 and metered.provider_calls == 0 and metered.cache_hits == 0


def test_metered_llm_safe_input_still_hits_and_misses_normally(tmp_path):
    inner = FakeLLMClient("fresh")
    metered = MeteredLLM(inner, ResponseCache(tmp_path / "gc"))
    request = _request("a perfectly safe question")
    assert metered.generate(request).text == "fresh"
    assert metered.generate(request).text == "fresh"
    assert (inner.call_count, metered.provider_calls, metered.cache_hits) == (1, 1, 1)


def test_blocked_generation_input_is_not_written_to_the_cache(tmp_path):
    cache_dir = tmp_path / "gc"
    metered = MeteredLLM(FakeLLMClient("fresh"), ResponseCache(cache_dir))
    with pytest.raises(RepositorySecretRiskError):
        metered.generate(_request(f"please use {h.github_token()}"))
    assert not cache_dir.exists() or not list(cache_dir.glob("*.json"))
