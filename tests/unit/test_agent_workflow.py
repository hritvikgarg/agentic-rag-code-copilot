"""The graph's control flow: rounds, refinement, termination and abstention (offline)."""

import pytest
from pydantic import ValidationError

from copilot.agents import AssessmentOutcome, Intent, TerminationReason
from copilot.agents.workflow import route_after_assessment, route_after_retrieval
from copilot.retrieval import QueryError
from copilot.retrieval.errors import MaterializationError
from tests.agent_helpers import MULTI_ROUNDS, Q_MULTI, ScriptedRetriever, chunk, workflow

NODES = ["classify_intent", "retrieve", "merge_evidence", "assess_evidence"]


def nodes(result):
    return [e.node for e in result.trace]


def test_one_round_when_the_first_evidence_looks_sufficient():
    question = "Explain how authentication works"
    retriever = ScriptedRetriever([[chunk("authentication code", path="src/auth.py")]])
    wf, llm = workflow(retriever)
    result = wf.run(question)
    assert retriever.queries == [question]  # round 1 is always the user's own wording
    assert result.rounds == 1 and result.termination is TerminationReason.SUFFICIENT
    assert result.generated and llm.call_count == 1
    assert nodes(result) == [*NODES, "generate"]


def test_an_insufficient_round_triggers_a_different_refined_query_and_evidence_accumulates():
    question = "Explain how authentication and sessions work"
    retriever = ScriptedRetriever(
        [
            [chunk("logging helper", path="src/log.py")],
            [chunk("authentication and session code", path="src/auth.py")],
        ]
    )
    wf, llm = workflow(retriever)
    result = wf.run(question)
    assert result.rounds == 2 and len(retriever.calls) == 2
    assert retriever.queries[0] == question and retriever.queries[1] != question
    assert result.evidence_count == 2  # round 1's chunk is kept, not replaced
    assert {c.file_path for c in result.citations} == {"src/log.py", "src/auth.py"}
    assert result.termination is TerminationReason.SUFFICIENT
    assert nodes(result) == [*NODES, "retrieve", "merge_evidence", "assess_evidence", "generate"]
    routing = [e.routing for e in result.trace if e.node == "assess_evidence"]
    assert routing == ["refine", "generate"]
    assert llm.call_count == 1


def test_three_rounds_each_adding_new_evidence_until_sufficient():
    retriever = ScriptedRetriever(MULTI_ROUNDS)
    wf, _ = workflow(retriever)
    result = wf.run(Q_MULTI)
    assert result.rounds == 3 and result.termination is TerminationReason.SUFFICIENT
    assert result.evidence_count == 3
    queries = retriever.queries
    assert queries[0] == Q_MULTI and len(set(queries)) == 3
    assigns = [e.assessment for e in result.trace if e.node == "assess_evidence"]
    assert assigns == ["insufficient_refinable", "insufficient_refinable", "potentially_sufficient"]
    # coverage rises as evidence accumulates across rounds
    coverage = [e.term_coverage for e in result.trace if e.node == "assess_evidence"]
    assert coverage == sorted(coverage) and coverage[0] < coverage[-1]


def test_the_round_budget_stops_the_loop_and_partial_evidence_still_generates():
    retriever = ScriptedRetriever(MULTI_ROUNDS)
    wf, llm = workflow(retriever)
    result = wf.run(Q_MULTI, max_rounds=2)
    assert len(retriever.calls) == 2 and result.rounds == 2
    assert result.termination is TerminationReason.EXHAUSTED_ROUNDS
    assert result.assessment.outcome is AssessmentOutcome.EXHAUSTED
    assert result.assessment.stop_reason == "max_rounds_reached"
    assert result.generated and llm.call_count == 1 and result.evidence_count == 2


def test_one_round_means_no_refinement():
    retriever = ScriptedRetriever(MULTI_ROUNDS)
    wf, _ = workflow(retriever)
    result = wf.run(Q_MULTI, max_rounds=1)
    assert len(retriever.calls) == 1
    assert result.termination is TerminationReason.EXHAUSTED_ROUNDS


def test_a_refinement_that_finds_nothing_new_stops_the_loop():
    same = [chunk("logging helper", path="src/log.py")]
    retriever = ScriptedRetriever([same, same, MULTI_ROUNDS[0]])  # a 3rd call would find more
    wf, _ = workflow(retriever)
    result = wf.run(Q_MULTI)
    assert len(retriever.calls) == 2  # stopped; the third (productive) call never happened
    assert result.termination is TerminationReason.NO_NEW_EVIDENCE
    assert result.evidence_count == 1  # the duplicate was not counted twice
    merged = [e for e in result.trace if e.node == "merge_evidence"]
    assert [e.new_accepted for e in merged] == [1, 0]


def test_when_every_refinement_would_repeat_a_query_the_loop_stops():
    retriever = ScriptedRetriever([[chunk("logging helper", path="src/log.py")]])
    wf, _ = workflow(retriever)
    result = wf.run("authentication")  # one term, no hint: every refinement equals the original
    assert len(retriever.calls) == 1
    assert result.termination is TerminationReason.NO_NEW_QUERY
    assert result.assessment.stop_reason == "no_new_query"


def test_no_query_is_ever_issued_twice():
    retriever = ScriptedRetriever(
        lambda q, k: [chunk(f"unrelated {len(q)}", path=f"src/f{len(q)}.py", start=len(q))]
    )
    wf, _ = workflow(retriever, max_rounds=5)
    wf.run(Q_MULTI, max_rounds=5)
    normalized = [" ".join(sorted(set(q.lower().split()))) for q in retriever.queries]
    assert len(normalized) == len(set(normalized))


def test_nothing_found_abstains_without_calling_the_model():
    retriever = ScriptedRetriever([[]])
    wf, llm = workflow(retriever)
    result = wf.run("Explain how authentication works")
    assert len(retriever.calls) == 2  # the empty first round is retried once with a new query
    assert result.termination is TerminationReason.NO_EVIDENCE
    assert result.answer is None and result.citations == () and not result.generated
    assert result.abstention_reason == "no_results"
    assert llm.call_count == 0
    assert nodes(result)[-1] == "abstain" and "generate" not in nodes(result)


def test_empty_first_round_then_evidence_recovers():
    retriever = ScriptedRetriever([[], [chunk("authentication code", path="src/auth.py")]])
    wf, llm = workflow(retriever)
    result = wf.run("Explain how authentication works")
    assert result.generated and result.rounds == 2 and llm.call_count == 1


def test_a_retrieval_failure_abstains_and_is_reported_as_data():
    retriever = ScriptedRetriever([[]], error=MaterializationError("source changed"))
    wf, llm = workflow(retriever)
    result = wf.run("Explain how authentication works")
    assert (
        result.failure.stage == "retrieval" and result.failure.error_type == "MaterializationError"
    )
    assert result.termination is TerminationReason.RETRIEVAL_FAILED
    assert result.abstention_reason == "retrieval_failed" and llm.call_count == 0
    assert nodes(result) == ["classify_intent", "retrieve", "abstain"]


def test_the_intent_changes_the_retrieval_request_and_the_refinement():
    arch = ScriptedRetriever([[chunk("architecture notes", path="docs/a.md")]])
    general = ScriptedRetriever([[chunk("authentication notes", path="docs/a.md")]])
    wf_a, _ = workflow(arch)
    wf_g, _ = workflow(general)
    ra = wf_a.run("Explain the architecture of the indexer")
    rg = wf_g.run("tell me something about authentication")
    assert ra.decision.intent is Intent.ARCHITECTURE and rg.decision.intent is Intent.GENERAL
    assert arch.calls[0][1] == 10 and general.calls[0][1] == 5  # top-k differs by intent
    # architecture needs >= 2 files, so one file triggers a refinement that "general" does not need
    assert ra.rounds >= 2 and rg.rounds == 1
    assert ra.assessment is not None
    assert (
        "too_few_files_for_architecture"
        in [e.note for e in ra.trace if e.node == "assess_evidence"][0]
    )


def test_the_accumulated_evidence_never_exceeds_its_cap():
    many = [chunk(f"filler {i}", path=f"src/f{i}.py", start=i + 1, rank=i + 1) for i in range(8)]
    retriever = ScriptedRetriever([many])
    wf, llm = workflow(retriever, max_evidence_chunks=4)
    result = wf.run("Explain how filler works")
    assert result.evidence_count == 4 and len(result.citations) <= 4


@pytest.mark.parametrize("bad", [0, 6, -1])
def test_invalid_round_limits_are_rejected_before_anything_runs(bad):
    retriever = ScriptedRetriever([[]])
    wf, llm = workflow(retriever)
    with pytest.raises(ValidationError):
        wf.run("Explain how authentication works", max_rounds=bad)
    assert retriever.calls == [] and llm.call_count == 0


@pytest.mark.parametrize("bad", ["", "   \n", "x" * 2001, None])
def test_a_bad_question_fails_before_retrieval(bad):
    retriever = ScriptedRetriever([[]])
    wf, _ = workflow(retriever)
    with pytest.raises(QueryError):
        wf.run(bad)
    assert retriever.calls == []


def test_the_route_functions_are_pure_functions_of_the_state():
    from copilot.agents.state import Assessment as A

    def route(outcome, stop=None):
        return route_after_assessment(
            {"assessment": A(outcome=outcome, reason="r", term_coverage=None, stop_reason=stop)}
        )

    out = AssessmentOutcome
    assert route(out.POTENTIALLY_SUFFICIENT) == "generate"
    assert route(out.INSUFFICIENT_REFINABLE) == "retrieve"
    assert route(out.EXHAUSTED, "max_rounds_reached") == "generate"
    assert route(out.EMPTY) == "retrieve"
    assert route(out.EMPTY, "no_new_evidence") == "abstain"
    assert route_after_retrieval({}) == "merge_evidence"
    assert route_after_retrieval({"failure": object()}) == "abstain"


def test_the_compiled_graph_has_exactly_the_documented_nodes_and_edges():
    wf, _ = workflow(ScriptedRetriever([[]]))
    graph = wf._graph.get_graph()
    assert set(graph.nodes) == {
        "__start__",
        "classify_intent",
        "retrieve",
        "merge_evidence",
        "assess_evidence",
        "generate",
        "abstain",
        "__end__",
    }
    edges = {(e.source, e.target, e.conditional) for e in graph.edges}
    assert ("assess_evidence", "retrieve", True) in edges  # the loop
    assert ("assess_evidence", "generate", True) in edges
    assert ("assess_evidence", "abstain", True) in edges
    assert ("retrieve", "abstain", True) in edges
    assert ("generate", "__end__", False) in edges and ("abstain", "__end__", False) in edges
