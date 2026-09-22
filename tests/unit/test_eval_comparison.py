"""Plain-vs-RAG comparison harness, serialization, rubric schema and rating summary."""

import json

import pytest
from pydantic import ValidationError

from copilot.config import Settings
from copilot.evaluation import Benchmark, BenchmarkQuestion, Region
from copilot.evaluation import __main__ as eval_cli
from copilot.evaluation.__main__ import _side_counts
from copilot.evaluation.comparison import (
    COMPARISON_SCHEMA,
    RUBRIC,
    ComparisonEntry,
    ComparisonMismatchError,
    ComparisonRun,
    ManualRating,
    PlainRecord,
    RagRecord,
    SourceRecord,
    dump_comparison,
    format_summary,
    load_comparison,
    run_comparison,
    select_questions,
    summarize_ratings,
    write_comparison,
)
from copilot.llm import FakeLLMClient, LLMConfigurationError, LLMProviderError, LLMRateLimitError
from copilot.llm.retry import RetryPolicy
from tests.rag_helpers import FakeRetriever, make_result
from tests.security import secret_helpers as h

SETTINGS = Settings(_env_file=None)


def question(qid: str, text: str, file: str = "src/ids.py", start=7, end=9) -> BenchmarkQuestion:
    return BenchmarkQuestion(
        id=qid, question=text, relevant=(Region(file_path=file, start_line=start, end_line=end),)
    )


BENCH = Benchmark(
    questions=(
        question("q01", "How are chunk ids made?"),
        question("q02", "Where is logging set up?", "src/log.py", 1, 5),
        question("q03", "What is ingestion?", "src/ing.py", 1, 5),
    )
)


def retriever():
    return FakeRetriever(
        [make_result("def make_chunk_id():\n    pass\n", rank=1, path="src/ids.py", start=7)]
    )


def test_select_questions_by_limit_id_and_errors():
    assert [q.id for q in select_questions(BENCH, limit=2)] == ["q01", "q02"]
    assert [q.id for q in select_questions(BENCH, ids=["q03", "q01"])] == ["q03", "q01"]
    assert len(select_questions(BENCH)) == 3
    with pytest.raises(ValueError, match="unknown question id"):
        select_questions(BENCH, ids=["nope"])


def test_run_records_paired_answers_evidence_status_and_latency():
    fake = FakeLLMClient("answer [Source 1]")
    run = run_comparison(
        list(BENCH.questions[:2]), retriever(), fake, top_k=3, settings=SETTINGS, benchmark_name="b"
    )
    assert (
        run.schema_version == COMPARISON_SCHEMA and run.llm_model == "fake-model" and run.top_k == 3
    )
    assert fake.call_count == 4  # plain + rag per question
    first, second = run.entries
    assert first.question_id == "q01" and first.expected_regions == ("src/ids.py:7-9",)
    assert first.plain.answer == "answer [Source 1]" and first.plain.latency_seconds is not None
    assert first.rag.status == "answered" and first.rag.cited_source_numbers == (1,)
    assert first.rag.sources[0].file_path == "src/ids.py" and first.rag.first_relevant_source == 1
    assert (
        second.rag.first_relevant_source is None
    )  # objective: retrieved evidence missed the region
    assert first.plain_rating is None and first.rag_rating is None  # nothing is rated automatically
    assert all(e.rag.latency_seconds is not None for e in run.entries)


def test_the_plain_request_has_no_repository_context_and_the_rag_request_does():
    fake = FakeLLMClient()
    run_comparison([BENCH.questions[0]], retriever(), fake, settings=SETTINGS)
    plain_request, rag_request = fake.requests
    assert "SOURCE" not in plain_request.user_prompt and "BEGIN SOURCE" in rag_request.user_prompt


def test_insufficient_evidence_is_recorded_without_calling_the_model_for_rag():
    fake = FakeLLMClient()
    run = run_comparison([BENCH.questions[0]], FakeRetriever([]), fake, settings=SETTINGS)
    assert (
        run.entries[0].rag.status == "insufficient_evidence" and fake.call_count == 1
    )  # plain only


def test_a_provider_error_is_recorded_and_the_run_continues():
    class FlakyClient(FakeLLMClient):
        def generate(self, request):
            if self.call_count == 0:
                self.requests.append(request)
                raise LLMProviderError("Gemini request failed (HTTP 500)", status_code=500)
            return super().generate(request)

    run = run_comparison(
        list(BENCH.questions[:2]), retriever(), FlakyClient("ok"), settings=SETTINGS
    )
    assert "LLMProviderError" in run.entries[0].plain.error and run.entries[0].plain.answer is None
    assert run.entries[0].rag.status == "answered" and run.entries[1].plain.answer == "ok"


def test_a_gate_refusal_is_recorded_without_the_secret_and_without_calling_the_model():
    secret = h.github_token("cmp")
    leaky = FakeRetriever([make_result(f"T = '{secret}'\n", path="src/ids.py", start=7)])
    fake = FakeLLMClient()
    run = run_comparison([BENCH.questions[0]], leaky, fake, settings=SETTINGS)
    entry = run.entries[0]
    assert entry.rag.status == "error" and "RepositorySecretRiskError" in entry.rag.error
    assert fake.call_count == 1  # the (clean) plain question only
    assert not h.contains_secret(dump_comparison(run), secret)


def test_delay_is_applied_after_each_request():
    sleeps: list[float] = []
    run_comparison(
        list(BENCH.questions[:2]),
        retriever(),
        FakeLLMClient(),
        settings=SETTINGS,
        delay_seconds=1.5,
        sleep=sleeps.append,
    )
    assert sleeps == [1.5] * 4


def test_serialization_round_trips_and_is_stable(tmp_path):
    run = run_comparison(list(BENCH.questions), retriever(), FakeLLMClient("x"), settings=SETTINGS)
    path = tmp_path / "out" / "cmp.json"
    write_comparison(run, path)
    assert load_comparison(path) == run
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n") and json.loads(text)["schema_version"] == COMPARISON_SCHEMA
    assert set(json.loads(text)["rubric"]) == set(RUBRIC)


def test_the_rubric_schema_validates_ratings():
    assert set(RUBRIC) == {
        "grounded_correctness", "citation_correctness", "hallucinated_claims",
        "completeness", "appropriate_abstention",
    }  # fmt: skip
    ManualRating(grounded_correctness=2, hallucinated_claims=0, appropriate_abstention=None)
    for bad in (
        {"grounded_correctness": 3},
        {"completeness": -1},
        {"hallucinated_claims": -1},
        {"unknown": 1},
    ):
        with pytest.raises(ValidationError):
            ManualRating(**bad)
    assert not ManualRating().is_rated and ManualRating(notes="x").is_rated is False


def test_the_summary_counts_only_rated_entries_and_declares_no_winner():
    run = run_comparison(list(BENCH.questions), retriever(), FakeLLMClient(), settings=SETTINGS)
    empty = summarize_ratings(run)
    assert empty.plain.n_rated == 0 and empty.plain.mean_grounded_correctness is None
    assert empty.rag_status_counts == {"answered": 3} and empty.rag_first_relevant_source_hits == 1
    run.entries[0].plain_rating = ManualRating(
        grounded_correctness=0, hallucinated_claims=2, completeness=0
    )
    run.entries[0].rag_rating = ManualRating(
        grounded_correctness=2, citation_correctness=2, hallucinated_claims=0, completeness=1
    )
    run.entries[1].rag_rating = ManualRating(grounded_correctness=1, appropriate_abstention=False)
    s = summarize_ratings(run)
    assert (s.plain.n_rated, s.rag.n_rated, s.rag.n_total) == (1, 2, 3)
    assert s.rag.mean_grounded_correctness == 1.5 and s.plain.mean_grounded_correctness == 0.0
    assert s.plain.total_hallucinated_claims == 2 and s.rag.total_hallucinated_claims == 0
    assert s.rag.mean_citation_correctness == 2.0 and s.plain.mean_citation_correctness is None
    assert s.rag.abstention_inappropriate == 1
    text = format_summary(s).lower()
    assert "no system is declared better" in text and "rated 2/3" in text
    assert "winner" not in text.replace("no winner", "")


def test_the_output_contains_no_absolute_paths():
    run = run_comparison(list(BENCH.questions), retriever(), FakeLLMClient(), settings=SETTINGS)
    assert "\\\\" not in dump_comparison(run)  # no Windows-style separators
    for entry in run.entries:
        assert all(
            not s.file_path.startswith("/") and "\\" not in s.file_path for s in entry.rag.sources
        )


# --- CLI ----------------------------------------------------------------------------------
def test_summarize_cli_reads_a_rated_file(tmp_path, capsys, monkeypatch):
    run = run_comparison(list(BENCH.questions[:1]), retriever(), FakeLLMClient(), settings=SETTINGS)
    run.entries[0].rag_rating = ManualRating(grounded_correctness=2)
    path = tmp_path / "c.json"
    write_comparison(run, path)
    monkeypatch.setattr(eval_cli, "setup_logging", lambda *a, **k: None)
    assert eval_cli.main(["summarize", str(path)]) == 0
    assert "rated 1/1" in capsys.readouterr().out
    assert eval_cli.main(["summarize", str(tmp_path / "missing.json")]) == 2


def test_compare_cli_help_lists_the_resume_and_retry_flags(capsys):
    with pytest.raises(SystemExit) as excinfo:
        eval_cli.main(["compare", "--help"])
    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    for flag in (
        "--resume",
        "--retry-initial-delay",
        "--retry-multiplier",
        "--retry-max-delay",
        "--retry-max-attempts",
    ):
        assert flag in help_text


# --- Resume / retry / checkpoint (robustness to transient hosted-provider failures) --------


def base_run(
    entries: list[ComparisonEntry], *, top_k: int = 3, benchmark_name: str = "b"
) -> ComparisonRun:
    """A ``ComparisonRun`` whose scalar fields match what ``run_comparison`` produces below,
    so it is a valid ``resume_from`` for the ``FakeLLMClient`` / ``retriever()`` / ``SETTINGS``
    combination used throughout this file (unless a test deliberately mismatches one field)."""
    return ComparisonRun(
        benchmark_name=benchmark_name,
        repository_name=None,
        index_id=None,
        llm_provider="fake",
        llm_model="fake-model",
        plain_prompt_version="plain-v1",
        rag_prompt_version="rag-v1",
        top_k=top_k,
        temperature=SETTINGS.llm_temperature,
        max_output_tokens=SETTINGS.llm_max_output_tokens,
        entries=entries,
    )


def entry_for(
    q: BenchmarkQuestion,
    *,
    plain: PlainRecord,
    rag: RagRecord,
    plain_rating: ManualRating | None = None,
    rag_rating: ManualRating | None = None,
) -> ComparisonEntry:
    return ComparisonEntry(
        question_id=q.id,
        question=q.question,
        expected_regions=tuple(f"{r.file_path}:{r.start_line}-{r.end_line}" for r in q.relevant),
        plain=plain,
        rag=rag,
        plain_rating=plain_rating,
        rag_rating=rag_rating,
    )


def plain_ok(text: str = "prior plain answer") -> PlainRecord:
    return PlainRecord(answer=text, latency_seconds=0.1)


def plain_failed(
    msg: str = "LLMRateLimitError: Gemini rate or quota limit reached (HTTP 429)",
) -> PlainRecord:
    return PlainRecord(error=msg, latency_seconds=0.1)


def rag_ok(text: str = "prior rag answer") -> RagRecord:
    return RagRecord(status="answered", answer=text, latency_seconds=0.1)


def rag_failed(
    msg: str = "LLMRateLimitError: Gemini rate or quota limit reached (HTTP 429)",
) -> RagRecord:
    return RagRecord(status="error", error=msg, latency_seconds=0.1)


def test_resume_reuses_a_successful_plain_side_and_only_calls_rag():
    q = BENCH.questions[0]
    prior = base_run([entry_for(q, plain=plain_ok(), rag=rag_failed())])
    fake = FakeLLMClient("new rag answer")
    run = run_comparison(
        [q], retriever(), fake, top_k=3, settings=SETTINGS, benchmark_name="b", resume_from=prior
    )
    assert fake.call_count == 1  # only the failed RAG side was re-asked
    assert run.entries[0].plain == prior.entries[0].plain  # untouched
    assert run.entries[0].rag.status == "answered" and run.entries[0].rag.answer == "new rag answer"


def test_resume_reuses_a_successful_rag_side_and_only_calls_plain():
    q = BENCH.questions[0]
    prior = base_run([entry_for(q, plain=plain_failed(), rag=rag_ok())])
    fake = FakeLLMClient("new plain answer")
    run = run_comparison(
        [q], retriever(), fake, top_k=3, settings=SETTINGS, benchmark_name="b", resume_from=prior
    )
    assert fake.call_count == 1  # only the failed plain side was re-asked
    assert run.entries[0].plain.answer == "new plain answer"
    assert run.entries[0].rag == prior.entries[0].rag  # untouched


def test_resume_with_both_sides_already_successful_makes_zero_provider_calls():
    q = BENCH.questions[0]
    prior = base_run([entry_for(q, plain=plain_ok("p"), rag=rag_ok("r"))])
    fake = FakeLLMClient("must never be used")
    run = run_comparison(
        [q], retriever(), fake, top_k=3, settings=SETTINGS, benchmark_name="b", resume_from=prior
    )
    assert fake.call_count == 0
    assert run.entries[0].plain.answer == "p" and run.entries[0].rag.answer == "r"


def test_resume_with_both_sides_failed_recomputes_both():
    q = BENCH.questions[0]
    prior = base_run([entry_for(q, plain=plain_failed(), rag=rag_failed())])
    fake = FakeLLMClient("recovered")
    run = run_comparison(
        [q], retriever(), fake, top_k=3, settings=SETTINGS, benchmark_name="b", resume_from=prior
    )
    assert fake.call_count == 2  # plain and rag both re-asked
    assert run.entries[0].plain.answer == "recovered" and run.entries[0].rag.answer == "recovered"


def test_a_429_is_retried_within_run_comparison_and_recorded_as_a_success():
    class RateLimitedOnce(FakeLLMClient):
        def generate(self, request):
            if self.call_count == 0:
                self.requests.append(request)
                raise LLMRateLimitError("Gemini rate or quota limit reached (HTTP 429)")
            return super().generate(request)

    sleeps: list[float] = []
    fake = RateLimitedOnce("ok")
    run = run_comparison(
        [BENCH.questions[0]],
        retriever(),
        fake,
        settings=SETTINGS,
        retry_policy=RetryPolicy(initial_delay_seconds=1.0, multiplier=2.0),
        sleep=sleeps.append,
    )
    assert run.entries[0].plain.answer == "ok" and run.entries[0].plain.error is None
    assert sleeps == [1.0]  # exactly one backoff wait, no real time passed


def test_a_transient_503_is_retried_within_run_comparison_and_recorded_as_a_success():
    class UnavailableOnce(FakeLLMClient):
        def generate(self, request):
            if self.call_count == 0:
                self.requests.append(request)
                raise LLMProviderError("Gemini request failed (HTTP 503)", status_code=503)
            return super().generate(request)

    sleeps: list[float] = []
    run = run_comparison(
        [BENCH.questions[0]],
        retriever(),
        UnavailableOnce("ok"),
        settings=SETTINGS,
        retry_policy=RetryPolicy(initial_delay_seconds=2.0),
        sleep=sleeps.append,
    )
    assert run.entries[0].rag.status in ("answered", "insufficient_evidence")
    assert sleeps  # at least one backoff wait happened


def test_a_configuration_error_is_recorded_without_any_retry():
    fake = FakeLLMClient(error=LLMConfigurationError("no GEMINI_API_KEY"))
    sleeps: list[float] = []
    run = run_comparison(
        [BENCH.questions[0]], retriever(), fake, settings=SETTINGS, sleep=sleeps.append
    )
    assert "LLMConfigurationError" in run.entries[0].plain.error
    assert "LLMConfigurationError" in run.entries[0].rag.error
    assert fake.call_count == 2  # plain once, rag once - neither retried
    assert sleeps == []


def test_a_secret_gate_refusal_is_not_retried():
    secret = h.github_token("cmp-retry")
    leaky = FakeRetriever([make_result(f"T = '{secret}'\n", path="src/ids.py", start=7)])
    fake = FakeLLMClient()
    sleeps: list[float] = []
    run = run_comparison([BENCH.questions[0]], leaky, fake, settings=SETTINGS, sleep=sleeps.append)
    assert "RepositorySecretRiskError" in run.entries[0].rag.error
    assert fake.call_count == 1  # the clean plain question only; rag never reached the model
    assert sleeps == []


def test_retry_exhaustion_records_the_error_and_the_run_continues_to_the_next_question():
    class AlwaysRateLimited(FakeLLMClient):
        def generate(self, request):
            self.requests.append(request)
            raise LLMRateLimitError("Gemini rate or quota limit reached (HTTP 429)")

    sleeps: list[float] = []
    run = run_comparison(
        list(BENCH.questions[:2]),
        retriever(),
        AlwaysRateLimited(),
        settings=SETTINGS,
        retry_policy=RetryPolicy(initial_delay_seconds=0.1, multiplier=1.0, max_attempts=2),
        sleep=sleeps.append,
    )
    assert "LLMRateLimitError" in run.entries[0].plain.error
    assert "LLMRateLimitError" in run.entries[0].rag.error
    # the run did not stop after q01's exhaustion - q02 was still attempted and recorded
    assert run.entries[1].question_id == "q02"
    assert "LLMRateLimitError" in run.entries[1].plain.error


def test_checkpoint_survives_a_simulated_interruption(tmp_path):
    class InterruptOnThirdCall(FakeLLMClient):
        def generate(self, request):
            if self.call_count >= 2:
                raise KeyboardInterrupt()
            return super().generate(request)

    path = tmp_path / "compare.json"
    with pytest.raises(KeyboardInterrupt):
        run_comparison(
            list(BENCH.questions[:2]),
            retriever(),
            InterruptOnThirdCall("ok"),
            settings=SETTINGS,
            checkpoint_path=path,
        )
    checkpoint = load_comparison(path)
    assert len(checkpoint.entries) == 1  # q01 (2 successful calls) checkpointed before the crash
    assert checkpoint.entries[0].question_id == "q01"
    assert checkpoint.entries[0].plain.answer == "ok" and checkpoint.entries[0].rag.answer == "ok"


def test_resume_refuses_an_incompatible_run_before_any_provider_call():
    q = BENCH.questions[0]
    prior = base_run([entry_for(q, plain=plain_ok(), rag=rag_ok())], top_k=99)  # mismatched top_k
    fake = FakeLLMClient()
    with pytest.raises(ComparisonMismatchError, match="top_k"):
        run_comparison(
            [q],
            retriever(),
            fake,
            top_k=3,
            settings=SETTINGS,
            benchmark_name="b",
            resume_from=prior,
        )
    assert fake.call_count == 0


def test_resume_refuses_when_the_question_id_set_differs():
    q1, q2 = BENCH.questions[0], BENCH.questions[1]
    prior = base_run([entry_for(q1, plain=plain_ok(), rag=rag_ok())])
    fake = FakeLLMClient()
    with pytest.raises(ComparisonMismatchError, match="question_id"):
        run_comparison(
            [q2],
            retriever(),
            fake,
            top_k=3,
            settings=SETTINGS,
            benchmark_name="b",
            resume_from=prior,
        )
    assert fake.call_count == 0


def test_resume_preserves_successful_records_and_ratings_exactly():
    q = BENCH.questions[0]
    src = SourceRecord(
        source_number=1, file_path="src/ids.py", start_line=7, end_line=9,
        chunk_id="abc123", score=0.9, rank=1,
    )  # fmt: skip
    prior_rag = RagRecord(
        status="answered", answer="prior rag", sources=(src,), cited_source_numbers=(1,),
        retrieved_count=1, first_relevant_source=1, retrieval_seconds=0.01, latency_seconds=0.2,
    )  # fmt: skip
    prior_plain = PlainRecord(answer="prior plain", latency_seconds=0.15)
    rating = ManualRating(grounded_correctness=2, hallucinated_claims=0)
    prior = base_run(
        [entry_for(q, plain=prior_plain, rag=prior_rag, plain_rating=rating, rag_rating=rating)]
    )
    fake = FakeLLMClient("must not be used")
    run = run_comparison(
        [q], retriever(), fake, top_k=3, settings=SETTINGS, benchmark_name="b", resume_from=prior
    )
    assert fake.call_count == 0
    assert run.entries[0].plain == prior_plain
    assert run.entries[0].rag == prior_rag
    assert run.entries[0].plain_rating == rating and run.entries[0].rag_rating == rating


def test_retry_progress_messages_never_contain_the_question_or_a_secret():
    secret = h.github_token("retrylog")
    sensitive_question = f"tell me about {secret}"

    class RateLimitedOnce(FakeLLMClient):
        def generate(self, request):
            if self.call_count == 0:
                self.requests.append(request)
                raise LLMRateLimitError("Gemini rate or quota limit reached (HTTP 429)")
            return super().generate(request)

    messages: list[str] = []
    q = question("qsec", sensitive_question)
    run_comparison(
        [q],
        retriever(),
        RateLimitedOnce("ok"),
        settings=SETTINGS,
        retry_policy=RetryPolicy(initial_delay_seconds=0.0),
        progress=messages.append,
        sleep=lambda _: None,
    )
    joined = " ".join(messages)
    assert secret not in joined and sensitive_question not in joined


def test_side_counts_reports_successful_reused_and_failed():
    q = BENCH.questions[0]
    prior_by_id = {q.id: entry_for(q, plain=plain_ok(), rag=rag_failed())}
    current = base_run([entry_for(q, plain=plain_ok(), rag=rag_ok())])  # rag recomputed -> now ok
    assert _side_counts(current, prior_by_id, "plain") == (1, 1, 0)  # ok, reused, failed
    assert _side_counts(current, prior_by_id, "rag") == (1, 0, 0)  # ok now, but not reused
