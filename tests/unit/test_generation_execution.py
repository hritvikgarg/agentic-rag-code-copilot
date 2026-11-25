"""Milestone 11A: the resumable multi-repository runner, response cache and dry-run plan.

Everything uses ``FakeLLMClient``; a socket guard proves no test can reach a network.
"""

from __future__ import annotations

import json
import socket

import pytest

from copilot.config import Settings
from copilot.evaluation.benchmark import Region
from copilot.evaluation.generation.compat import frozen_target  # noqa: F401  (import check)
from copilot.evaluation.generation.execution import (
    CircuitBreaker,
    MeteredLLM,
    RepositoryContext,
    ResponseCache,
    RunMismatchError,
    SideRecord,
    load_run,
    plan_run,
    response_cache_key,
    run_path,
    run_repositories,
    run_repository,
    settings_for,
)
from copilot.evaluation.generation.protocol import (
    PrimaryProtocol,
    RepositorySpec,
    default_protocol,
    dump_protocol,
    load_protocol,
    protocol_hash,
)
from copilot.evaluation.generation.schema import Answerability, GenerationQuestion, KeyFact
from copilot.llm import FakeLLMClient, LLMProviderError, LLMRateLimitError
from copilot.llm.models import LLMRequest
from copilot.llm.retry import RetryPolicy
from tests.rag_helpers import FakeRetriever, make_result
from tests.security import secret_helpers as h

NO_SLEEP = lambda s: None  # noqa: E731
FAST = RetryPolicy(initial_delay_seconds=0.0, max_attempts=2)
SETTINGS = Settings(_env_file=None)
REGION = Region(file_path="src/a.py", start_line=1, end_line=5)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def _q(qid: str, repo: str = "r1", answerable: bool = True) -> GenerationQuestion:
    if answerable:
        return GenerationQuestion(
            id=qid, repository_id=repo, question=f"What does {qid} do?", category="c",
            answerability=Answerability.ANSWERABLE, acceptable_answer="it does x",
            key_facts=(KeyFact(id="f1", statement="does exactly x", regions=(REGION,)),),
        )  # fmt: skip
    return GenerationQuestion(
        id=qid, repository_id=repo, question=f"Who owns {qid}?", category="c",
        answerability=Answerability.UNANSWERABLE, unanswerable_rationale="not in repo",
        expected_abstention="says so",
    )  # fmt: skip


def _protocol(model: str = "fake-model", **over) -> PrimaryProtocol:
    spec = RepositorySpec(
        repository_id="r1", kind="synthetic-authored", repository_name="r1", source_root="x",
        benchmark_path="b.jsonl", benchmark_digest="a" * 64,
    )  # fmt: skip
    protocol = default_protocol((spec,)).with_model("fake", model)
    return protocol.model_copy(update=over) if over else protocol


def _ctx(repo="r1", n=3, retriever=None, index_id="idx1") -> RepositoryContext:
    questions = tuple(_q(f"{repo}-q{i}", repo, answerable=i < n - 1) for i in range(n))
    spec = RepositorySpec(
        repository_id=repo, kind="synthetic-authored", repository_name=repo, source_root="x",
        benchmark_path="b.jsonl", benchmark_digest="a" * 64,
    )  # fmt: skip
    retriever = retriever or FakeRetriever(
        [make_result("def f():\n    pass\n", rank=1, path="src/a.py", start=1)]
    )
    return RepositoryContext(spec, questions, "b" * 64, retriever, index_id)


def _run(ctx, tmp_path, llm=None, protocol=None, **kw):
    protocol = protocol or _protocol()
    llm = llm or FakeLLMClient("answer [Source 1]")
    kw.setdefault("retry_policy", FAST)
    kw.setdefault("retry_sleep", NO_SLEEP)
    return run_repository(
        ctx, protocol, llm, settings_for(protocol, SETTINGS),
        checkpoint_path=tmp_path / "run.json", **kw,
    )  # fmt: skip


def _go(ctx, protocol, llm, path, **kw):
    kw.setdefault("retry_policy", FAST)
    kw.setdefault("retry_sleep", NO_SLEEP)
    return run_repository(
        ctx, protocol, llm, settings_for(protocol, SETTINGS), checkpoint_path=path, **kw
    )


# --------------------------------------------------------------------------- protocol


def test_protocol_roundtrip_hash_and_model_confirmation(tmp_path):
    base = default_protocol(_protocol().repositories)
    assert not base.model_confirmed and base.llm_model is None
    assert base.plain_prompt_version == "plain-v2" and base.rag_prompt_version == "rag-v1"
    assert base.max_output_tokens == 8192 and base.temperature == 0.0 and base.top_k == 5
    assert "deferred" in base.agentic_arm and base.primary_endpoint.name.startswith("paired")
    path = tmp_path / "p.json"
    path.write_text(dump_protocol(base))
    assert load_protocol(path) == base and protocol_hash(load_protocol(path)) == protocol_hash(base)
    one, two = base.with_model("gemini", "m1"), base.with_model("gemini", "m2")
    assert len({protocol_hash(base), protocol_hash(one), protocol_hash(two)}) == 3
    with pytest.raises(ValueError):
        base.with_model("", "m")


# --------------------------------------------------------------------------- a normal run


def test_run_records_both_arms_usage_provenance_and_test_label(tmp_path):
    fake = FakeLLMClient("an answer [Source 1]")
    run = _run(_ctx(), tmp_path, fake)
    assert fake.call_count == 6 and run.test_artifact is True
    assert run.side_counts() == {
        "plain": {"successful": 3, "failed": 0, "not_attempted": 0},
        "rag": {"successful": 3, "failed": 0, "not_attempted": 0},
    }
    first = run.entries[0]
    assert first.plain.prompt_version == "plain-v2" and first.rag.prompt_version == "rag-v1"
    assert first.plain.sources == () and first.plain.input_sha256
    assert first.rag.sources and first.rag.cited_source_numbers == (1,)
    assert first.rag.first_relevant_source == 1  # region src/a.py:1-5 overlaps the chunk
    assert first.plain.prompt_tokens is not None and first.rag.latency_seconds is not None
    assert run.entries[-1].answerable is False
    assert json.loads((tmp_path / "run.json").read_text())["test_artifact"] is True


def test_both_arms_receive_identical_question_text_and_plain_has_no_evidence(tmp_path):
    fake = FakeLLMClient("a")
    _run(_ctx(n=2), tmp_path, fake)
    plain, rag = fake.requests[:2]
    assert plain.prompt_version == "plain-v2" and rag.prompt_version == "rag-v1"
    assert "What does r1-q0 do?" in plain.user_prompt and "What does r1-q0 do?" in rag.user_prompt
    assert "def f()" in rag.user_prompt and "def f()" not in plain.user_prompt
    assert plain.max_output_tokens == rag.max_output_tokens == 8192
    assert plain.temperature == rag.temperature == 0.0


def test_rag_without_evidence_is_a_system_abstention_and_not_a_model_call(tmp_path):
    fake = FakeLLMClient("a")
    run = _run(_ctx(n=1, retriever=FakeRetriever([])), tmp_path, fake)
    side = run.entries[0].rag
    assert side.system_insufficient and side.answer is None and fake.call_count == 1  # plain only
    assert side.state == "successful"


def test_the_client_must_be_the_confirmed_model(tmp_path):
    with pytest.raises(RunMismatchError, match="not the model confirmed"):
        _run(_ctx(), tmp_path, FakeLLMClient("a", model="another"))
    unconfirmed = default_protocol(_protocol().repositories)
    with pytest.raises(RunMismatchError, match="not been confirmed"):
        _run(_ctx(), tmp_path, protocol=unconfirmed)


def test_the_secret_gate_blocks_before_the_provider_and_is_recorded(tmp_path):
    fake = FakeLLMClient("a")
    retriever = FakeRetriever([make_result(f"token = {h.github_token('x')}\n", rank=1)])
    run = _run(_ctx(n=1, retriever=retriever), tmp_path, fake)
    rag = run.entries[0].rag
    assert rag.state == "failed" and "RepositorySecretRiskError" in rag.error
    assert fake.call_count == 1  # only the (clean) plain request reached the provider
    assert h.github_token("x") not in (tmp_path / "run.json").read_text()


# --------------------------------------------------------------------------- failure, resume


def test_failed_sides_are_recorded_and_retried_on_resume_while_successes_are_reused(tmp_path):
    from copilot.llm.models import LLMResponse

    class Flaky(FakeLLMClient):
        def generate(self, request):
            self.requests.append(request)
            if len(self.requests) == 2:
                raise LLMProviderError("boom", status_code=400)
            return LLMResponse(text="ok", provider="fake", model=self.model)

    first = _run(_ctx(n=2), tmp_path, Flaky())
    assert first.side_counts()["rag"] == {"successful": 1, "failed": 1, "not_attempted": 0}
    snapshot = (tmp_path / "run.json").read_text()
    again = FakeLLMClient("fresh")
    second = _run(_ctx(n=2), tmp_path, again, resume_from=load_run(tmp_path / "run.json"))
    assert again.call_count == 1  # only the failed side was asked
    assert second.side_counts()["rag"]["successful"] == 2
    kept = json.loads(snapshot)["entries"]
    assert second.entries[1].plain == type(second.entries[1].plain).model_validate(kept[1]["plain"])
    assert second.entries[0].plain.answer == "ok" and second.entries[0].rag.answer == "fresh"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda ctx, p: (ctx, p.model_copy(update={"top_k": 3})),
        lambda ctx, p: (ctx, p.model_copy(update={"max_output_tokens": 1024})),
        lambda ctx, p: (ctx, p.model_copy(update={"plain_prompt_version": "plain-v1"})),
        lambda ctx, p: (ctx.__class__(ctx.spec, ctx.questions, "c" * 64, ctx.retriever, "idx1"), p),
        lambda ctx, p: (ctx.__class__(ctx.spec, ctx.questions, "b" * 64, ctx.retriever, "idx2"), p),
        lambda ctx, p: (
            ctx.__class__(ctx.spec, ctx.questions[:2], "b" * 64, ctx.retriever, "idx1"),
            p,
        ),
    ],
)
def test_resume_refuses_a_different_experiment_before_any_call(tmp_path, mutate):
    ctx, protocol = _ctx(), _protocol()
    _run(ctx, tmp_path, protocol=protocol)
    existing = load_run(tmp_path / "run.json")
    new_ctx, new_protocol = mutate(ctx, protocol)
    fake = FakeLLMClient("a")
    with pytest.raises(RunMismatchError, match="cannot resume"):
        _run(new_ctx, tmp_path, fake, protocol=new_protocol, resume_from=existing)
    assert fake.call_count == 0


def test_resume_refuses_a_different_model_and_changed_question_text(tmp_path):
    ctx = _ctx()
    _run(ctx, tmp_path)
    existing = load_run(tmp_path / "run.json")
    other = _protocol("model-b")
    with pytest.raises(RunMismatchError, match="llm_model|protocol_hash"):
        _run(
            ctx, tmp_path, FakeLLMClient("a", model="model-b"), protocol=other, resume_from=existing
        )
    edited = ctx.questions[0].model_copy(update={"question": "Something else?"})
    changed = RepositoryContext(
        ctx.spec, (edited, *ctx.questions[1:]), "b" * 64, ctx.retriever, "idx1"
    )
    with pytest.raises(RunMismatchError, match="question text changed"):
        _run(changed, tmp_path, resume_from=existing)


# --------------------------------------------------------------------------- 429 breaker


def test_circuit_breaker_stops_after_consecutive_429s_and_marks_not_attempted(tmp_path):
    fake = FakeLLMClient(error=LLMRateLimitError("slow"))
    breaker = CircuitBreaker(2)
    run = _run(_ctx(n=3), tmp_path, fake, breaker=breaker)
    assert breaker.tripped and run.stop_reason and "429" in run.stop_reason
    counts = run.side_counts()
    assert counts["plain"]["failed"] == 1 and counts["rag"]["failed"] == 1
    assert counts["plain"]["not_attempted"] == 2 and counts["rag"]["not_attempted"] == 2
    assert fake.call_count == 2 * FAST.max_attempts  # bounded retries, then the breaker


def test_a_success_resets_the_streak_and_other_errors_do_not_count():
    breaker = CircuitBreaker(2)
    breaker.record(LLMRateLimitError("x"))
    breaker.record(None)
    breaker.record(LLMRateLimitError("x"))
    breaker.record(LLMProviderError("y", status_code=400))
    assert not breaker.tripped
    breaker.record(LLMRateLimitError("x"))
    breaker.record(LLMRateLimitError("x"))
    assert breaker.tripped
    with pytest.raises(ValueError):
        CircuitBreaker(0)


def test_multi_repository_run_stops_everywhere_when_the_breaker_trips(tmp_path):
    fake = FakeLLMClient(error=LLMRateLimitError("slow"))
    runs = run_repositories(
        [_ctx("r1", 2), _ctx("r2", 2)], _protocol(), fake, SETTINGS, output_dir=tmp_path,
        retry_policy=FAST, retry_sleep=NO_SLEEP, max_consecutive_rate_limit_failures=2,
    )  # fmt: skip
    assert fake.call_count == 2 * FAST.max_attempts
    assert runs["r2"].side_counts()["plain"]["not_attempted"] == 2  # recorded, not dropped
    assert run_path(tmp_path, "r2").is_file() and runs["r2"].stop_reason


# --------------------------------------------------------------------------- caches


def test_cache_key_covers_model_settings_prompt_and_exact_input():
    def req(**over):
        base = dict(system_instruction="s", user_prompt="u", temperature=0.0,
                    max_output_tokens=10, prompt_version="v")  # fmt: skip
        base.update(over)
        return LLMRequest(**base)

    keys = {
        response_cache_key("p", "m", req())[0],
        response_cache_key("p", "m2", req())[0],
        response_cache_key("p2", "m", req())[0],
        response_cache_key("p", "m", req(user_prompt="u2"))[0],
        response_cache_key("p", "m", req(system_instruction="s2"))[0],
        response_cache_key("p", "m", req(max_output_tokens=11))[0],
        response_cache_key("p", "m", req(temperature=0.5))[0],
        response_cache_key("p", "m", req(prompt_version="v2"))[0],
    }
    assert len(keys) == 8


def test_the_generation_cache_serves_a_rerun_without_provider_calls(tmp_path):
    cache = ResponseCache(tmp_path / "cache")
    first = FakeLLMClient("cached answer")
    _run(_ctx(n=2), tmp_path / "a", first, cache=cache) if (
        tmp_path / "a"
    ).mkdir() is None else None
    second = FakeLLMClient("should not be used")
    out = tmp_path / "b"
    out.mkdir()
    run = run_repository(
        _ctx(n=2), _protocol(), second, settings_for(_protocol(), SETTINGS),
        checkpoint_path=out / "run.json", cache=cache, retry_policy=FAST, retry_sleep=NO_SLEEP,
    )  # fmt: skip
    assert first.call_count == 4 and second.call_count == 0
    assert all(e.plain.origin == "cache" and e.rag.origin == "cache" for e in run.entries)
    assert run.entries[0].plain.answer == "cached answer"


def test_cache_never_serves_another_model_and_never_stores_failures(tmp_path):
    cache = ResponseCache(tmp_path / "cache")
    ok = FakeLLMClient("one", model="m1")
    p1 = _protocol("m1")
    _go(_ctx(n=1), p1, ok, tmp_path / "x.json", cache=cache)
    other = FakeLLMClient("two", model="m2")
    p2 = _protocol("m2")
    run = _go(_ctx(n=1), p2, other, tmp_path / "y.json", cache=cache)
    assert other.call_count == 2 and run.entries[0].plain.answer == "two"
    failing = MeteredLLM(FakeLLMClient(error=LLMProviderError("x", status_code=400)), cache)
    with pytest.raises(LLMProviderError):
        failing.generate(
            LLMRequest(system_instruction="s", user_prompt="fresh", prompt_version="v")
        )
    assert len(list((tmp_path / "cache").glob("*.json"))) == 4  # 2 + 2 successes only


def test_cache_hits_do_not_extend_or_reset_the_429_streak(tmp_path):
    cache = ResponseCache(tmp_path / "cache")
    p = _protocol()
    warm = FakeLLMClient("a")
    _go(_ctx(n=1), p, warm, tmp_path / "w.json", cache=cache)
    breaker = CircuitBreaker(2)
    breaker.record(LLMRateLimitError("x"))
    _go(_ctx(n=1), p, FakeLLMClient("z"), tmp_path / "h.json", cache=cache, breaker=breaker)
    assert breaker.streak == 1 and not breaker.tripped


# --------------------------------------------------------------------------- historical reuse


def test_verified_historical_sides_are_reused_and_not_asked_again(tmp_path):
    ctx = _ctx(n=2)
    historical = {
        (ctx.questions[0].id, "rag"): SideRecord(
            arm="rag", state="successful", prompt_version="rag-v1", status="answered",
            answer="old rag", origin="historical", origin_file="old.json",
        ),
    }  # fmt: skip
    fake = FakeLLMClient("new")
    run = _run(ctx, tmp_path, fake, historical=historical)
    assert fake.call_count == 3  # 4 sides - 1 reused
    assert run.entries[0].rag.answer == "old rag" and run.entries[0].rag.origin == "historical"
    assert run.entries[0].plain.origin == "generated"


# --------------------------------------------------------------------------- dry-run plan


def test_plan_counts_calls_without_a_provider_and_accounts_for_reuse(tmp_path):
    contexts = [_ctx("r1", 5), _ctx("r2", 3)]
    plan = plan_run(contexts, _protocol(), output_dir=tmp_path)
    assert [line.questions for line in plan.lines] == [5, 3]
    assert plan.generation_calls == 16 and plan.judge_calls == 16
    assert plan.lines[0].answerable == 4 and plan.lines[0].unanswerable == 1
    assert plan.artifacts[0].endswith("generation-r1.json")
    side = SideRecord(
        arm="rag", state="successful", prompt_version="rag-v1", status="answered",
        answer="x", origin="historical",
    )  # fmt: skip
    reuse = {("r1-q0", "rag"): side}
    plan2 = plan_run(contexts, _protocol(), output_dir=tmp_path, historical={"r1": reuse})
    assert plan2.generation_calls == 15 and plan2.lines[0].reused_sides == 1
    assert not list(tmp_path.iterdir())  # planning writes nothing


def test_plan_after_runs_counts_only_the_sides_still_to_do(tmp_path):
    ctx = _ctx("r1", 3)
    _run(ctx, tmp_path, FakeLLMClient(error=LLMRateLimitError("x")), breaker=CircuitBreaker(2))
    (tmp_path / "run.json").rename(run_path(tmp_path, "r1"))
    # 2 failed + 4 not attempted sides all still need a call
    assert plan_run([ctx], _protocol(), output_dir=tmp_path).generation_calls == 6
    run_path(tmp_path, "r1").unlink()
    _run(ctx, tmp_path, FakeLLMClient("fine"))
    (tmp_path / "run.json").rename(run_path(tmp_path, "r1"))
    assert plan_run([ctx], _protocol(), output_dir=tmp_path).generation_calls == 0
