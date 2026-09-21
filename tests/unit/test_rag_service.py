"""Plain baseline and RAG service with a fake LLM and a fake retriever."""

import logging

import pytest

from copilot.config import Settings
from copilot.llm import FakeLLMClient, LLMProviderError
from copilot.rag import (
    PLAIN_PROMPT_VERSION,
    RAG_PROMPT_VERSION,
    AnswerStatus,
    RagService,
    answer_plain,
)
from copilot.retrieval import QueryError
from copilot.security import RepositorySecretRiskError
from tests.rag_helpers import FakeRetriever, make_result
from tests.security import secret_helpers as h

SETTINGS = Settings(_env_file=None, rag_context_max_tokens=5000)


def results():
    return [
        make_result(
            "def make_chunk_id(p, s, e):\n    return sha(p, s, e)\n",
            rank=1,
            path="src/ids.py",
            start=7,
        ),
        make_result("def other():\n    pass\n", rank=2, path="src/other.py", start=1),
    ]


# --- plain baseline -----------------------------------------------------------------------
def test_plain_sends_only_the_question_and_returns_a_typed_answer():
    fake = FakeLLMClient("plain reply")
    answer = answer_plain("  How are chunk ids made?  ", fake, settings=SETTINGS)
    assert answer.answer == "plain reply" and answer.question == "How are chunk ids made?"
    assert answer.prompt_version == PLAIN_PROMPT_VERSION and answer.generation.model == "fake-model"
    (request,) = fake.requests
    assert request.user_prompt == "How are chunk ids made?"  # no repository context at all
    assert "SOURCE" not in request.outbound_text() and "file:" not in request.outbound_text()
    assert request.temperature == 0.0 and request.max_output_tokens == 1024


def test_plain_is_deterministic_with_a_fake_client():
    a = answer_plain("q?", FakeLLMClient("x"), settings=SETTINGS)
    b = answer_plain("q?", FakeLLMClient("x"), settings=SETTINGS)
    assert a == b


@pytest.mark.parametrize("bad", ["", "   ", "x" * 2001, None, 5])
def test_plain_validates_the_question_before_any_call(bad):
    fake = FakeLLMClient()
    with pytest.raises(QueryError):
        answer_plain(bad, fake, settings=SETTINGS)  # type: ignore[arg-type]
    assert fake.call_count == 0


def test_plain_uses_the_generation_settings():
    fake = FakeLLMClient()
    answer_plain(
        "q?", fake, settings=Settings(_env_file=None, llm_temperature=0.3, llm_max_output_tokens=77)
    )
    assert (fake.requests[0].temperature, fake.requests[0].max_output_tokens) == (0.3, 77)


# --- RAG ----------------------------------------------------------------------------------
def test_rag_puts_the_retrieved_evidence_in_the_request_and_returns_its_citations():
    retriever, fake = FakeRetriever(results()), FakeLLMClient("It is a hash [Source 1].")
    answer = RagService(retriever, fake, settings=SETTINGS).answer("How are chunk ids made?", 2)
    assert retriever.calls == [("How are chunk ids made?", 2)]
    (request,) = fake.requests
    assert request.prompt_version == RAG_PROMPT_VERSION
    assert "def make_chunk_id(p, s, e):" in request.user_prompt
    assert "BEGIN SOURCE 1" in request.user_prompt and "file: src/ids.py" in request.user_prompt
    assert request.user_prompt.rstrip().endswith("Question:\nHow are chunk ids made?")
    assert answer.status is AnswerStatus.ANSWERED and answer.answer == "It is a hash [Source 1]."
    assert [(s.file_path, s.start_line, s.end_line) for s in answer.sources] == [
        ("src/ids.py", 7, 8),
        ("src/other.py", 1, 2),
    ]
    assert answer.cited_source_numbers == (1,) and answer.unknown_source_numbers == ()
    assert answer.retrieved_count == 2 and answer.generation.provider == "fake"


def test_rag_reports_source_numbers_the_model_invented(caplog):
    fake = FakeLLMClient("See [Source 1] and [Source 7].")
    with caplog.at_level(logging.WARNING):
        answer = RagService(FakeRetriever(results()), fake, settings=SETTINGS).answer("q?")
    assert answer.cited_source_numbers == (1,) and answer.unknown_source_numbers == (7,)
    assert [s.source_number for s in answer.sources] == [1, 2]  # authoritative list unchanged
    assert any("not supplied" in r.getMessage() for r in caplog.records)


def test_rag_uses_the_default_top_k_from_settings():
    retriever = FakeRetriever(results())
    RagService(
        retriever, FakeLLMClient(), settings=Settings(_env_file=None, retrieval_top_k=4)
    ).answer("q?")
    assert retriever.calls[0][1] == 4


@pytest.mark.parametrize("bad", ["", "  "])
def test_rag_validates_the_question_before_retrieval(bad):
    retriever, fake = FakeRetriever(results()), FakeLLMClient()
    with pytest.raises(QueryError):
        RagService(retriever, fake, settings=SETTINGS).answer(bad)
    assert retriever.calls == [] and fake.call_count == 0


def test_no_results_is_insufficient_evidence_and_the_model_is_not_called():
    fake = FakeLLMClient()
    answer = RagService(FakeRetriever([]), fake, settings=SETTINGS).answer("q?")
    assert answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE and answer.answer is None
    assert answer.insufficient_reason == "no_results" and answer.sources == ()
    assert fake.call_count == 0


def test_evidence_that_cannot_fit_the_budget_is_insufficient_evidence():
    fake = FakeLLMClient()
    tiny = Settings(_env_file=None, rag_context_max_tokens=200)
    answer = RagService(FakeRetriever([make_result("line\n" * 500)]), fake, settings=tiny).answer(
        "q?"
    )
    assert answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert answer.insufficient_reason == "context_budget" and fake.call_count == 0
    assert [d.reason for d in answer.dropped_evidence] == ["budget"]


def test_budget_drops_are_reported_on_an_answered_result():
    big = make_result("line\n" * 500, rank=1, path="src/big.py")
    small = make_result("x = 1\n", rank=2, path="src/small.py")
    answer = RagService(
        FakeRetriever([big, small]),
        FakeLLMClient(),
        settings=Settings(_env_file=None, rag_context_max_tokens=300),
    ).answer("q?")
    assert answer.status is AnswerStatus.ANSWERED
    assert [s.file_path for s in answer.sources] == ["src/small.py"]
    assert [(d.rank, d.reason) for d in answer.dropped_evidence] == [(1, "budget")]


def test_a_duplicate_chunk_enters_the_context_once():
    dup = make_result("same\n", rank=2, path="src/ids.py", start=7, chunk_id=results()[0].chunk_id)
    fake = FakeLLMClient()
    answer = RagService(FakeRetriever([*results(), dup]), fake, settings=SETTINGS).answer("q?", 5)
    assert len(answer.sources) == 2
    assert fake.requests[0].user_prompt.count("BEGIN SOURCE") == 2


def test_the_rag_system_prompt_states_the_grounding_rules():
    fake = FakeLLMClient()
    RagService(FakeRetriever(results()), fake, settings=SETTINGS).answer("q?")
    system = fake.requests[0].system_instruction.lower()
    for phrase in ("untrusted", "insufficient", "[source n]", "general knowledge", "do not invent"):
        assert phrase in system


def test_provider_errors_propagate_as_typed_errors():
    fake = FakeLLMClient(
        error=LLMProviderError("Gemini request failed (HTTP 500)", status_code=500)
    )
    with pytest.raises(LLMProviderError):
        RagService(FakeRetriever(results()), fake, settings=SETTINGS).answer("q?")


def test_retriever_failures_propagate_and_the_model_is_not_called():
    fake = FakeLLMClient()
    with pytest.raises(RuntimeError):
        RagService(
            FakeRetriever(error=RuntimeError("index broken")), fake, settings=SETTINGS
        ).answer("q?")
    assert fake.call_count == 0


# --- security: the gate sits in front of every request ---------------------------------------
@pytest.mark.parametrize("kind", ["chunk", "question"])
def test_a_secret_in_the_chunk_or_the_question_never_reaches_the_client(kind, caplog):
    secret = h.github_token("svc")
    chunk_text = f"TOKEN = '{secret}'\n" if kind == "chunk" else "x = 1\n"
    question = "What does this do?" if kind == "chunk" else f"Why does {secret} fail?"
    fake = FakeLLMClient()
    service = RagService(
        FakeRetriever([make_result(chunk_text, path="src/cfg.py", start=3)]),
        fake,
        settings=SETTINGS,
    )
    with caplog.at_level(logging.DEBUG), pytest.raises(RepositorySecretRiskError) as info:
        service.answer(question)
    assert fake.call_count == 0
    assert not h.contains_secret(str(info.value), secret)
    assert not h.contains_secret(caplog.text, secret)
    if kind == "chunk":
        assert "src/cfg.py" in str(info.value)  # diagnostics carry file:line, not the value


def test_a_secret_in_the_plain_question_never_reaches_the_client():
    secret = h.aws_access_key("plain")
    fake = FakeLLMClient()
    with pytest.raises(RepositorySecretRiskError) as info:
        answer_plain(f"my key is {secret}, what should I do?", fake, settings=SETTINGS)
    assert fake.call_count == 0 and not h.contains_secret(str(info.value), secret)


def test_a_secret_in_an_unincluded_chunk_does_not_block_the_request_but_is_not_sent():
    secret = h.github_token("dropped")
    leaky = make_result(f"T = '{secret}'\n" + "pad\n" * 600, rank=1, path="src/leak.py")
    ok = make_result("x = 1\n", rank=2, path="src/ok.py")
    fake = FakeLLMClient()
    RagService(
        FakeRetriever([leaky, ok]),
        fake,
        settings=Settings(_env_file=None, rag_context_max_tokens=300),
    ).answer("q?")
    assert fake.call_count == 1 and secret not in fake.requests[0].outbound_text()


def test_the_services_always_wrap_the_client_and_expose_no_bypass():
    import inspect

    secret = h.github_token("bypass")
    for name in ("answer_plain", "answer_with_rag"):
        from copilot import rag

        params = inspect.signature(getattr(rag, name)).parameters
        assert not any(
            ("force" in p or "skip" in p or "bypass" in p or "unsafe" in p) for p in params
        )
    params = inspect.signature(RagService.__init__).parameters
    assert not any(("force" in p or "skip" in p or "bypass" in p or "unsafe" in p) for p in params)
    fake = FakeLLMClient()
    with pytest.raises(RepositorySecretRiskError):
        RagService(FakeRetriever(results()), fake, settings=SETTINGS).answer(f"key {secret}")
    assert fake.call_count == 0


def test_the_gate_runs_before_the_client_on_every_request_in_a_sequence():
    secret = h.github_token("seq")
    fake = FakeLLMClient()
    service = RagService(FakeRetriever(results()), fake, settings=SETTINGS)
    service.answer("clean question")
    with pytest.raises(RepositorySecretRiskError):
        service.answer(f"dirty {secret}")
    service.answer("another clean question")
    assert fake.call_count == 2
    assert all(secret not in r.outbound_text() for r in fake.requests)
