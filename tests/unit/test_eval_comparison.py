"""Plain-vs-RAG comparison harness, serialization, rubric schema and rating summary."""

import json

import pytest
from pydantic import ValidationError

from copilot.config import Settings
from copilot.evaluation import Benchmark, BenchmarkQuestion, Region
from copilot.evaluation import __main__ as eval_cli
from copilot.evaluation.comparison import (
    COMPARISON_SCHEMA,
    RUBRIC,
    ManualRating,
    dump_comparison,
    format_summary,
    load_comparison,
    run_comparison,
    select_questions,
    summarize_ratings,
    write_comparison,
)
from copilot.llm import FakeLLMClient, LLMProviderError
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
