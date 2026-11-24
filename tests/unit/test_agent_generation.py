"""Generation goes through the existing guarded RAG path; citations come from the evidence."""

from copilot.agents import AssessmentOutcome, TerminationReason
from copilot.config import Settings
from copilot.llm import FakeLLMClient, LLMProviderError, LLMRateLimitError
from copilot.rag import RAG_PROMPT_VERSION, RAG_SYSTEM_PROMPT, RagService
from tests.agent_helpers import SETTINGS, ScriptedRetriever, chunk, workflow

QUESTION = "Explain how authentication and sessions work"
EVIDENCE = [
    chunk("authentication entrypoint", path="src/auth.py", start=10, rank=1, score=0.8),
    chunk("session store", path="src/session.py", start=40, rank=2, score=0.6),
]


def test_the_agent_sends_the_same_prompt_as_the_baseline_for_the_same_evidence():
    baseline_llm = FakeLLMClient("baseline")
    RagService(ScriptedRetriever([EVIDENCE]), baseline_llm, settings=SETTINGS).answer(QUESTION, 5)
    wf, agent_llm = workflow(ScriptedRetriever([EVIDENCE]))
    result = wf.run(QUESTION)
    assert result.rounds == 1
    base, agent = baseline_llm.requests[0], agent_llm.requests[0]
    assert agent.system_instruction == base.system_instruction == RAG_SYSTEM_PROMPT
    assert agent.prompt_version == base.prompt_version == RAG_PROMPT_VERSION
    assert agent.user_prompt == base.user_prompt
    assert (agent.temperature, agent.max_output_tokens) == (
        base.temperature,
        base.max_output_tokens,
    )


def test_the_prompt_contains_the_evidence_of_every_round():
    wf, llm = workflow(
        ScriptedRetriever([[chunk("logging helper", path="src/log.py")], [EVIDENCE[0]]])
    )
    result = wf.run(QUESTION)
    prompt = llm.requests[0].user_prompt
    assert "src/log.py" in prompt and "src/auth.py" in prompt and result.rounds == 2


def test_citations_are_exactly_the_evidence_that_entered_the_prompt():
    wf, llm = workflow(ScriptedRetriever([EVIDENCE]))
    result = wf.run(QUESTION)
    assert result.citations == result.answer.sources
    got = [(c.file_path, c.start_line, c.end_line, c.chunk_id) for c in result.citations]
    want = [(r.file_path, r.start_line, r.end_line, r.chunk_id) for r in EVIDENCE]
    assert got == want
    assert [c.source_number for c in result.citations] == [1, 2]
    assert [c.score for c in result.citations] == [0.8, 0.6]  # scores of the retrieving query
    for citation in result.citations:  # every cited chunk's text really was in the prompt
        assert f"file: {citation.file_path}" in llm.requests[0].user_prompt


def test_a_label_the_model_invents_never_becomes_a_citation():
    wf, _ = workflow(ScriptedRetriever([EVIDENCE]), FakeLLMClient("see [Source 9] and [Source 1]"))
    result = wf.run(QUESTION)
    assert result.answer.unknown_source_numbers == (9,)
    assert result.answer.cited_source_numbers == (1,)
    assert len(result.citations) == 2  # unchanged by the made-up label


def test_generation_metadata_and_counts_are_preserved():
    wf, llm = workflow(ScriptedRetriever([EVIDENCE]), FakeLLMClient("ok", model="m-1"))
    result = wf.run(QUESTION)
    info = result.answer.generation
    assert (info.provider, info.model) == ("fake", "m-1")
    assert result.answer.prompt_version == RAG_PROMPT_VERSION
    assert result.answer.retrieved_count == result.evidence_count == 2
    assert result.answer.retrieval_seconds >= 0 and result.answer.generation_seconds is not None
    assert [e.generation_status for e in result.trace if e.node == "generate"] == ["answered"]


def test_a_context_budget_abstention_does_not_call_the_model():
    tiny = Settings(_env_file=None, rag_context_max_tokens=200)
    big = chunk("authentication sessions " * 400, path="src/big.py")
    wf, llm = workflow(ScriptedRetriever([[big]]), settings=tiny)
    result = wf.run(QUESTION)
    assert llm.call_count == 0 and result.citations == () and not result.generated
    assert result.abstention_reason == "context_budget"
    assert [e.generation_status for e in result.trace if e.node == "generate"] == [
        "abstained_before_model"
    ]


def test_provider_errors_are_preserved_as_typed_failures():
    wf, llm = workflow(
        ScriptedRetriever([EVIDENCE]), FakeLLMClient(error=LLMRateLimitError("quota (HTTP 429)"))
    )
    result = wf.run(QUESTION)
    assert llm.call_count == 1
    assert result.failure.error_type == "LLMRateLimitError" and "429" in result.failure.message
    assert not result.failure.blocked_by_secret_gate
    assert result.termination is TerminationReason.GENERATION_FAILED
    assert result.answer is None and result.citations == () and not result.generated
    assert [e.generation_status for e in result.trace if e.node == "generate"] == ["provider_error"]

    wf, _ = workflow(ScriptedRetriever([EVIDENCE]), FakeLLMClient(error=LLMProviderError("boom")))
    assert wf.run(QUESTION).failure.error_type == "LLMProviderError"


def test_partial_evidence_after_exhaustion_is_flagged_not_hidden():
    wf, _ = workflow(ScriptedRetriever([[chunk("nothing relevant", path="src/x.py")]]))
    result = wf.run("authentication")
    assert result.generated
    assert result.assessment.outcome is AssessmentOutcome.EXHAUSTED
    assert result.termination is TerminationReason.NO_NEW_QUERY


def test_partial_evidence_is_machine_readable_for_every_consumer():
    wf, _ = workflow(ScriptedRetriever([[chunk("nothing relevant", path="src/x.py")]]))
    result = wf.run("authentication")
    assert result.evidence_status == "partial" and result.generated
    assert len(result.warnings) == 1 and result.warnings[0].startswith("partial_evidence:")
    dumped = result.model_dump()  # serialised results (UI, evaluation) carry the same fields
    assert dumped["evidence_status"] == "partial" and dumped["warnings"] == result.warnings


def test_sufficient_and_absent_evidence_have_their_own_status_and_no_partial_warning():
    wf, _ = workflow(ScriptedRetriever([EVIDENCE]))
    ok = wf.run(QUESTION)
    assert ok.evidence_status == "potentially_sufficient" and ok.warnings == ()
    wf, llm = workflow(ScriptedRetriever([[]]))
    none = wf.run(QUESTION)
    assert none.evidence_status == "none" and none.warnings == () and llm.call_count == 0


def test_unknown_source_labels_are_surfaced_as_a_warning():
    wf, _ = workflow(ScriptedRetriever([EVIDENCE]), FakeLLMClient("see [Source 7]"))
    assert any(w.startswith("unknown_source_labels") for w in wf.run(QUESTION).warnings)
