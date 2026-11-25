"""Milestone 11A: judging a generation run - blindness, rules, resume, failures, breaker."""

from __future__ import annotations

import json

import pytest

from copilot.evaluation.generation.execution import GenerationRun
from copilot.evaluation.generation.judge import JudgeCache, LLMJudge
from copilot.evaluation.generation.judging import (
    RULE_EVALUATOR_ID,
    JudgementMismatchError,
    blind_id,
    judge_run,
    load_judgements,
    rule_assessment,
)
from copilot.evaluation.generation.reference import FactReference, ReferenceMaterial
from copilot.evaluation.generation.rubric import score_assessment
from copilot.llm import FakeLLMClient, LLMRateLimitError
from copilot.llm.retry import RetryPolicy
from tests.rag_helpers import FakeRetriever
from tests.unit.test_generation_execution import _ctx, _run

FAST = RetryPolicy(initial_delay_seconds=0.0, max_attempts=2)


def _judge_reply(request) -> str:
    if "Key fact" in request.user_prompt:
        return json.dumps(
            {"fact_verdicts": {"f1": "supported"}, "groundedness": 2, "abstained": False}
        )
    return json.dumps({"fact_verdicts": {}, "groundedness": 2, "abstained": True})


def _refs(ctx):
    out = {}
    for q in ctx.questions:
        facts = tuple(
            FactReference(f.id, f.statement, (("src/a.py:1-5", "def f(): pass"),))
            for f in q.key_facts
        )
        out[q.id] = ReferenceMaterial(
            q.id, q.is_answerable, facts, q.unanswerable_rationale, q.expected_abstention,
            q.acceptable_answer,
        )  # fmt: skip
    return out


def _generation(tmp_path, ctx=None, retriever=None) -> tuple[GenerationRun, object]:
    ctx = ctx or _ctx(n=3, retriever=retriever)
    return _run(ctx, tmp_path, FakeLLMClient("an answer [Source 1]")), ctx


def _judge(fake=None, cache=None, model="judge-m"):
    fake = fake or FakeLLMClient(_judge_reply, model=model)
    return LLMJudge(fake, retry_policy=FAST, retry_sleep=lambda s: None, cache=cache), fake


def _judge_all(run, ctx, judge, tmp_path, **kw):
    return judge_run(
        run, {q.id: q for q in ctx.questions}, _refs(ctx), judge,
        checkpoint_path=tmp_path / "j.json", **kw,
    )  # fmt: skip


def test_every_successful_side_is_judged_blind_and_scored(tmp_path):
    run, ctx = _generation(tmp_path)
    judge, fake = _judge()
    out = _judge_all(run, ctx, judge, tmp_path)
    assert out.counts() == {"judged": 6, "rule": 0, "failed": 0, "skipped": 0}
    assert out.test_artifact is True and out.same_model_as_generator is False
    q = ctx.questions[0]
    item = out.by_key()[(q.id, "rag")]
    assert score_assessment(q, item.assessment).score == 2
    assert item.blind_id != out.by_key()[(q.id, "plain")].blind_id
    for request in fake.requests:  # nothing that identifies the arm or retrieval reaches the judge
        text = request.outbound_text().lower()
        assert "plain-v2" not in text and "rag-v1" not in text and "[source" not in text
        assert "retrieved" not in text and "arm" not in text.split("begin answer")[0]


def test_blind_ids_are_stable_and_do_not_reveal_the_arm():
    a, b = blind_id(1, "r", "q", "plain"), blind_id(1, "r", "q", "rag")
    assert a != b and a == blind_id(1, "r", "q", "plain") and blind_id(2, "r", "q", "plain") != a
    assert "plain" not in a and "rag" not in b


def test_system_abstention_is_scored_by_rule_and_not_judged(tmp_path):
    ctx = _ctx(n=2, retriever=FakeRetriever([]))
    run = _run(ctx, tmp_path, FakeLLMClient("plain answer"))
    judge, fake = _judge()
    out = _judge_all(run, ctx, judge, tmp_path)
    rag_answerable = out.by_key()[(ctx.questions[0].id, "rag")]
    rag_unanswerable = out.by_key()[(ctx.questions[1].id, "rag")]
    assert rag_answerable.state == rag_unanswerable.state == "rule"
    assert rag_answerable.assessment.evaluator_id == RULE_EVALUATOR_ID
    assert score_assessment(ctx.questions[0], rag_answerable.assessment).score == 0  # false refusal
    assert score_assessment(ctx.questions[1], rag_unanswerable.assessment).score == 2
    assert fake.call_count == 2  # the two plain answers only
    assert rule_assessment(ctx.questions[1]).fact_verdicts == {}


def test_failed_and_unattempted_sides_are_skipped_with_a_reason(tmp_path):
    ctx = _ctx(n=2)
    from copilot.evaluation.generation.execution import CircuitBreaker

    run = _run(
        ctx, tmp_path, FakeLLMClient(error=LLMRateLimitError("x")), breaker=CircuitBreaker(2)
    )
    judge, fake = _judge()
    out = _judge_all(run, ctx, judge, tmp_path)
    assert out.counts()["skipped"] == 4 and out.counts()["judged"] == 0 and fake.call_count == 0
    assert all(i.skipped_reason for i in out.items)


def test_malformed_judge_output_is_a_failed_item_never_a_default_score(tmp_path):
    run, ctx = _generation(tmp_path)
    judge, _ = _judge(FakeLLMClient("this is not json"))
    out = _judge_all(run, ctx, judge, tmp_path, max_consecutive_rate_limit_failures=None)
    assert out.counts()["failed"] == 6 and out.counts()["judged"] == 0
    assert all(i.assessment is None and i.record.raw_output for i in out.items)


def test_resume_reuses_successes_retries_failures_and_makes_no_extra_calls(tmp_path):
    run, ctx = _generation(tmp_path)
    judge, fake = _judge()
    first = _judge_all(run, ctx, judge, tmp_path)
    again, fake2 = _judge()
    second = _judge_all(run, ctx, again, tmp_path, resume_from=load_judgements(tmp_path / "j.json"))
    assert fake.call_count == 6 and fake2.call_count == 0
    assert [i.assessment for i in second.items] == [i.assessment for i in first.items]
    # a broken first pass is repaired by a later one
    broken, _ = _judge(FakeLLMClient("junk"))
    path = tmp_path / "j2.json"
    bad = judge_run(
        run, {q.id: q for q in ctx.questions}, _refs(ctx), broken, checkpoint_path=path,
        max_consecutive_rate_limit_failures=None,
    )  # fmt: skip
    fixed, fake3 = _judge(model="judge-m")
    bad = bad.model_copy(update={"judge_model": "judge-m"})
    repaired = judge_run(
        run, {q.id: q for q in ctx.questions}, _refs(ctx), fixed, checkpoint_path=path,
        resume_from=bad,
    )  # fmt: skip
    assert repaired.counts()["judged"] == 6 and fake3.call_count == 6


def test_a_different_judge_or_run_cannot_be_resumed_into_the_same_file(tmp_path):
    run, ctx = _generation(tmp_path)
    judge, _ = _judge()
    first = _judge_all(run, ctx, judge, tmp_path)
    other, _ = _judge(model="other-judge")
    with pytest.raises(JudgementMismatchError, match="judge"):
        _judge_all(run, ctx, other, tmp_path, resume_from=first)


def test_judge_cache_reuses_identical_judgements_across_runs(tmp_path):
    run, ctx = _generation(tmp_path)
    cache = JudgeCache(tmp_path / "jc")
    judge, fake = _judge(cache=cache)
    _judge_all(run, ctx, judge, tmp_path)
    judge2, fake2 = _judge(cache=cache)
    out = judge_run(
        run, {q.id: q for q in ctx.questions}, _refs(ctx), judge2,
        checkpoint_path=tmp_path / "other.json",
    )  # fmt: skip
    assert fake2.call_count == 0 and all(i.reused for i in out.items if i.state == "judged")


def test_judge_rate_limit_breaker_stops_judging_and_marks_the_rest_skipped(tmp_path):
    run, ctx = _generation(tmp_path)
    judge, fake = _judge(FakeLLMClient(error=LLMRateLimitError("slow")))
    out = _judge_all(run, ctx, judge, tmp_path, max_consecutive_rate_limit_failures=2)
    counts = out.counts()
    assert counts["failed"] == 2 and counts["skipped"] == 4
    assert fake.call_count == 2 * FAST.max_attempts


def test_same_model_judge_is_flagged(tmp_path):
    run, ctx = _generation(tmp_path)
    judge, _ = _judge(model="fake-model")  # same provider ("fake") and model as the generator
    assert _judge_all(run, ctx, judge, tmp_path).same_model_as_generator is True
