"""Milestone 7.1: the consecutive-429 circuit breaker, side states and resumable checkpoints.

No hosted calls and no real waiting: a scripted fake client plays a fixed sequence of outcomes
and ``sleep`` is a no-op. ``max_attempts=1`` makes every side exactly one provider call, so the
call order is deterministic: q01 plain, q01 rag, q02 plain, q02 rag, ...
"""

import json

import pytest

from copilot.config import Settings
from copilot.evaluation import Benchmark, BenchmarkQuestion, Region
from copilot.evaluation.comparison import (
    ComparisonEntry,
    ComparisonRun,
    ManualRating,
    PlainRecord,
    RagRecord,
    dump_comparison,
    load_comparison,
    run_comparison,
    side_state,
)
from copilot.llm import FakeLLMClient, LLMProviderError, LLMRateLimitError
from copilot.llm.retry import RetryPolicy
from tests.rag_helpers import FakeRetriever, make_result

SETTINGS = Settings(_env_file=None)
NO_RETRY = RetryPolicy(max_attempts=1)
RATE = "rate"  # outcome marker: HTTP 429
OK = "ok"  # outcome marker: a successful answer


def bench(n: int) -> list[BenchmarkQuestion]:
    return list(
        Benchmark(
            questions=tuple(
                BenchmarkQuestion(
                    id=f"q{i:02d}",
                    question=f"question number {i}?",
                    relevant=(Region(file_path="src/ids.py", start_line=7, end_line=9),),
                )
                for i in range(1, n + 1)
            )
        ).questions
    )


class Scripted(FakeLLMClient):
    """Plays ``outcomes`` in order, one per provider call; a missing tail repeats ``OK``."""

    def __init__(self, outcomes: list[object]) -> None:
        super().__init__("scripted answer")
        self._outcomes = list(outcomes)

    def generate(self, request):
        position = len(self.requests)
        outcome = self._outcomes[position] if position < len(self._outcomes) else OK
        if outcome == RATE:
            self.requests.append(request)
            raise LLMRateLimitError("Gemini rate or quota limit reached (HTTP 429)")
        if isinstance(outcome, Exception):
            self.requests.append(request)
            raise outcome
        return super().generate(request)


def retriever():
    return FakeRetriever(
        [make_result("def make_chunk_id():\n    pass\n", rank=1, path="src/ids.py", start=7)]
    )


def go(questions, outcomes, **kwargs):
    llm = Scripted(outcomes)
    kwargs.setdefault("retry_policy", NO_RETRY)
    run = run_comparison(
        questions,
        retriever(),
        llm,
        top_k=3,
        settings=SETTINGS,
        benchmark_name="b",
        sleep=lambda _s: None,
        **kwargs,
    )
    return run, llm


def states(run: ComparisonRun) -> list[tuple[str, str]]:
    return [(side_state(e.plain), side_state(e.rag)) for e in run.entries]


# ---- side states -----------------------------------------------------------------------


def test_side_state_distinguishes_successful_failed_and_not_attempted():
    assert side_state(PlainRecord(answer="a")) == "successful"
    assert side_state(PlainRecord(error="LLMError: x")) == "failed"
    assert side_state(PlainRecord(not_attempted=True)) == "not_attempted"
    assert side_state(RagRecord(status="answered", answer="a")) == "successful"
    assert side_state(RagRecord(status="insufficient_evidence")) == "successful"
    assert side_state(RagRecord(status="error", error="LLMError: x")) == "failed"
    assert side_state(RagRecord(status="not_attempted")) == "not_attempted"


def test_a_file_written_before_milestone_7_1_still_loads(tmp_path):
    """Additive schema: no ``not_attempted`` / ``stop_reason`` keys means 'as before'."""
    run, _ = go(bench(1), [OK, OK])
    data = json.loads(dump_comparison(run))
    del data["stop_reason"]
    del data["entries"][0]["plain"]["not_attempted"]
    path = tmp_path / "old.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    old = load_comparison(path)
    assert old.stop_reason is None
    assert states(old) == [("successful", "successful")]


# ---- circuit breaker -------------------------------------------------------------------


def test_breaker_stops_after_three_consecutive_rate_limited_sides(tmp_path):
    path = tmp_path / "compare.json"
    run, llm = go(bench(5), [RATE] * 20, checkpoint_path=path)

    assert llm.call_count == 3  # q01 plain, q01 rag, q02 plain - nothing after the third
    assert run.stop_reason is not None
    assert "3 consecutive" in run.stop_reason and "HTTP 429" in run.stop_reason
    assert "--resume" in run.stop_reason
    assert states(run) == [
        ("failed", "failed"),
        ("failed", "not_attempted"),  # the tripping side failed; its sibling was never asked
        ("not_attempted", "not_attempted"),
        ("not_attempted", "not_attempted"),
        ("not_attempted", "not_attempted"),
    ]
    assert len(run.entries) == 5  # every requested question is in the file


def test_breaker_message_does_not_claim_a_specific_quota():
    run, _ = go(bench(3), [RATE] * 10)
    text = (run.stop_reason or "").lower()
    assert "daily" not in text and "per day" not in text and "exhausted" not in text
    assert "did not say which limit" in text


def test_threshold_is_configurable():
    run, llm = go(bench(5), [RATE] * 20, max_consecutive_rate_limit_failures=2)
    assert llm.call_count == 2
    assert "2 consecutive" in (run.stop_reason or "")


def test_breaker_can_be_disabled_with_none():
    run, llm = go(bench(5), [RATE] * 20, max_consecutive_rate_limit_failures=None)
    assert llm.call_count == 10  # every side attempted and recorded as failed
    assert run.stop_reason is None
    assert all(s == ("failed", "failed") for s in states(run))


def test_threshold_below_one_is_rejected_before_any_call():
    llm = Scripted([])
    with pytest.raises(ValueError, match=">= 1"):
        run_comparison(
            bench(1),
            retriever(),
            llm,
            settings=SETTINGS,
            max_consecutive_rate_limit_failures=0,
        )
    assert llm.call_count == 0


def test_a_success_resets_the_counter():
    # 429 429 ok | 429 429 ok | 429 429 ok : never three in a row
    run, llm = go(bench(3), [RATE, RATE, OK] * 2)
    assert llm.call_count == 6
    assert run.stop_reason is None


def test_a_different_failure_also_ends_the_streak():
    # 429, 429, HTTP 400, 429, 429, ok: the 400 interrupts, so the streak never reaches 3
    run, llm = go(bench(3), [RATE, RATE, LLMProviderError("bad", status_code=400), RATE, RATE])
    assert llm.call_count == 6
    assert run.stop_reason is None


def test_only_sides_that_exhausted_retries_count():
    """A 429 that is retried and then succeeds is a success, not a failed side."""
    run, llm = go(
        bench(3),
        [RATE, OK] * 6,  # each of the 6 sides: one 429, then success
        retry_policy=RetryPolicy(initial_delay_seconds=0, max_attempts=2),
    )
    assert run.stop_reason is None
    assert all(s == ("successful", "successful") for s in states(run))
    assert llm.call_count == 12


def test_the_breaker_trips_on_the_third_side_even_after_earlier_successes():
    run, llm = go(bench(4), [OK, OK, RATE, RATE, RATE])  # q01 fine, then q02 plain/rag, q03 plain
    assert llm.call_count == 5
    assert states(run)[0] == ("successful", "successful")
    assert states(run)[1] == ("failed", "failed")
    assert states(run)[2] == ("failed", "not_attempted")
    assert run.entries[0].plain.answer == "scripted answer"  # successful answer kept


# ---- checkpoint preservation ------------------------------------------------------------


def test_the_tripping_run_leaves_a_complete_atomic_checkpoint(tmp_path):
    path = tmp_path / "compare.json"
    run, _ = go(bench(4), [OK, OK, RATE, RATE, RATE], checkpoint_path=path)
    on_disk = load_comparison(path)
    assert dump_comparison(on_disk) == dump_comparison(run)  # the final state, byte for byte
    assert on_disk.stop_reason == run.stop_reason
    assert on_disk.entries[0].plain.answer == "scripted answer"  # earlier success preserved
    assert [p.name for p in tmp_path.iterdir()] == ["compare.json"]  # no temp files left behind


def test_the_checkpoint_is_written_after_every_hosted_side(tmp_path):
    path = tmp_path / "compare.json"
    seen: list[list[tuple[str, str]]] = []

    class Spy(Scripted):
        def generate(self, request):
            if path.exists():
                seen.append(states(load_comparison(path)))
            return super().generate(request)

    run_comparison(
        bench(2),
        retriever(),
        Spy([]),
        top_k=3,
        settings=SETTINGS,
        benchmark_name="b",
        retry_policy=NO_RETRY,
        sleep=lambda _s: None,
        checkpoint_path=path,
    )
    # before the 2nd call the file already holds the 1st side; the 4th call sees three successes
    assert seen[0] == [("successful", "not_attempted"), ("not_attempted", "not_attempted")]
    assert seen[-1][0] == ("successful", "successful")


# ---- resume ----------------------------------------------------------------------------


def test_resume_processes_the_unattempted_sides_and_keeps_successes_and_ratings(tmp_path):
    first, first_llm = go(bench(4), [OK, OK, RATE, RATE, RATE])
    assert first.stop_reason is not None

    # a human rated q01 while the run was stopped
    rated = first.entries[0].model_copy(
        update={
            "plain_rating": ManualRating(grounded_correctness=2, notes="checked"),
            "rag_rating": ManualRating(grounded_correctness=1),
        }
    )
    stopped = first.model_copy(update={"entries": [rated, *first.entries[1:]]})
    path = tmp_path / "compare.json"
    path.write_text(dump_comparison(stopped), encoding="utf-8")

    resumed, llm = go(bench(4), [], resume_from=load_comparison(path), checkpoint_path=path)

    # q01 (2 successful sides) reused: only the 6 other sides were asked
    assert llm.call_count == 6
    assert resumed.stop_reason is None
    assert all(s == ("successful", "successful") for s in states(resumed))
    assert resumed.entries[0].plain == first.entries[0].plain  # unchanged, byte for byte
    assert resumed.entries[0].rag == first.entries[0].rag
    assert resumed.entries[0].plain_rating == ManualRating(grounded_correctness=2, notes="checked")
    assert resumed.entries[0].rag_rating == ManualRating(grounded_correctness=1)
    assert load_comparison(path).stop_reason is None  # the file no longer says "stopped"


def test_resume_can_be_stopped_again_without_losing_what_the_first_resume_finished():
    first, _ = go(bench(5), [OK, OK, RATE, RATE, RATE])
    # second attempt: q02 plain succeeds, then three 429s in a row
    second, llm = go(bench(5), [OK, RATE, RATE, RATE], resume_from=first)
    assert llm.call_count == 4
    assert second.stop_reason is not None
    assert states(second)[0] == ("successful", "successful")
    assert states(second)[1] == ("successful", "failed")  # q02 plain recovered this time


def test_reused_sides_are_not_hosted_calls_so_they_do_not_reset_the_streak():
    q1, q2 = bench(2)
    prior = ComparisonRun(
        benchmark_name="b",
        llm_provider="fake",
        llm_model="fake-model",
        plain_prompt_version=_plain_version(),
        rag_prompt_version=_rag_version(),
        top_k=3,
        temperature=SETTINGS.llm_temperature,
        max_output_tokens=SETTINGS.llm_max_output_tokens,
        entries=[
            _entry(q1, PlainRecord(not_attempted=True), RagRecord(status="not_attempted")),
            _entry(q2, PlainRecord(answer="kept"), RagRecord(status="not_attempted")),
        ],
    )
    # hosted calls: q01 plain (429), q01 rag (429), [q02 plain reused], q02 rag (429) -> trips
    run, llm = go([q1, q2], [RATE] * 10, resume_from=prior)
    assert llm.call_count == 3
    assert run.stop_reason is not None
    assert run.entries[1].plain.answer == "kept"


def test_an_unreached_question_keeps_what_the_earlier_file_already_said():
    """When the breaker trips, later questions are not blanked: failed stays failed."""
    q1, q2, q3 = bench(3)
    failed_before = _entry(q3, PlainRecord(error="LLMRateLimitError: old"), old_failed_rag())
    prior = ComparisonRun(
        benchmark_name="b",
        llm_provider="fake",
        llm_model="fake-model",
        plain_prompt_version=_plain_version(),
        rag_prompt_version=_rag_version(),
        top_k=3,
        temperature=SETTINGS.llm_temperature,
        max_output_tokens=SETTINGS.llm_max_output_tokens,
        entries=[
            _entry(q1, PlainRecord(not_attempted=True), RagRecord(status="not_attempted")),
            _entry(q2, PlainRecord(not_attempted=True), RagRecord(status="not_attempted")),
            failed_before,
        ],
    )
    run, _ = go([q1, q2, q3], [RATE] * 10, resume_from=prior)
    assert run.stop_reason is not None
    assert run.entries[2] == failed_before  # untouched, still "failed" (not rewritten)


# ---- helpers (kept at the bottom so the tests read first) ---------------------------------


def old_failed_rag() -> RagRecord:
    return RagRecord(status="error", error="LLMRateLimitError: old")


def _entry(q: BenchmarkQuestion, plain: PlainRecord, rag: RagRecord) -> ComparisonEntry:
    return ComparisonEntry(
        question_id=q.id,
        question=q.question,
        expected_regions=tuple(f"{r.file_path}:{r.start_line}-{r.end_line}" for r in q.relevant),
        plain=plain,
        rag=rag,
    )


def _plain_version() -> str:
    from copilot.rag.prompts import PLAIN_PROMPT_VERSION

    return PLAIN_PROMPT_VERSION


def _rag_version() -> str:
    from copilot.rag.prompts import RAG_PROMPT_VERSION

    return RAG_PROMPT_VERSION


# ---- the command line ---------------------------------------------------------------------


class _Spec:
    repository_name = "repo"


class _Index:
    spec = _Spec()
    index_id = "idx1"


class _CliRetriever(FakeRetriever):
    index = _Index()

    @classmethod
    def open(cls, *args, **kwargs):
        return cls(
            [make_result("def make_chunk_id():\n    pass\n", rank=1, path="src/ids.py", start=7)]
        )


def _cli(monkeypatch, tmp_path, outcomes, extra=()):
    from copilot.evaluation import __main__ as cli

    benchmark = Benchmark(questions=tuple(bench(4)))
    llm = Scripted(outcomes)
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(cli, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr(cli, "load_benchmark", lambda _p: benchmark)
    monkeypatch.setattr(cli, "verify_benchmark", lambda *a, **k: None)
    monkeypatch.setattr(cli, "create_llm_client", lambda _s: llm)
    monkeypatch.setattr(cli, "create_embedder", lambda _s: None)
    monkeypatch.setattr(cli, "Retriever", _CliRetriever)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    out = tmp_path / "compare.json"
    argv = [
        "compare", "bench.jsonl", "--repo", str(tmp_path), "--index", str(tmp_path),
        "--output", str(out), "--retry-max-attempts", "1", *extra,
    ]  # fmt: skip
    return cli.main(argv), out, llm


def test_cli_stops_with_exit_3_a_clear_message_and_a_resumable_file(monkeypatch, tmp_path, capsys):
    code, out, llm = _cli(monkeypatch, tmp_path, [OK, OK] + [RATE] * 10)
    captured = capsys.readouterr()
    assert code == 3
    assert llm.call_count == 5  # q01 ok/ok, then three 429s
    assert "STOPPED EARLY" in captured.err and "--resume" in captured.err
    assert "run INCOMPLETE: stopped early" in captured.out
    assert "plain: 1 successful (0 reused from the existing file), 2 failed, 1 not attempted" in (
        captured.out
    )
    assert "rag:   1 successful (0 reused from the existing file), 1 failed, 2 not attempted" in (
        captured.out
    )
    assert "Fill in plain_rating" not in captured.out  # nothing to rate yet
    saved = load_comparison(out)
    assert saved.stop_reason is not None
    assert states(saved)[0] == ("successful", "successful")


def test_cli_threshold_flag_and_disable(monkeypatch, tmp_path):
    code, _out, llm = _cli(
        monkeypatch, tmp_path, [RATE] * 20, extra=["--max-consecutive-rate-limit-failures", "2"]
    )
    assert code == 3 and llm.call_count == 2

    code, _out, llm = _cli(
        monkeypatch, tmp_path, [RATE] * 20, extra=["--max-consecutive-rate-limit-failures", "0"]
    )
    assert code == 0 and llm.call_count == 8  # breaker off: every side attempted, all failed


def test_cli_resume_after_a_stop_finishes_the_run(monkeypatch, tmp_path, capsys):
    code, out, _ = _cli(monkeypatch, tmp_path, [OK, OK] + [RATE] * 10)
    assert code == 3
    code, out2, llm = _cli(monkeypatch, tmp_path, [], extra=["--resume"])
    assert code == 0 and out2 == out
    assert llm.call_count == 6  # q01 reused; the other six sides asked
    assert "run complete" in capsys.readouterr().out
    assert load_comparison(out).stop_reason is None
