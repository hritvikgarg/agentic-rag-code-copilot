"""Milestone 8.1: candidate selection for architecture questions, and citation provenance."""

import itertools

import pytest

from copilot.agents import TerminationReason
from copilot.agents.evidence import merge_evidence
from copilot.config import Settings
from copilot.llm import FakeLLMClient
from copilot.rag.context import build_context
from tests.agent_helpers import SETTINGS, ScriptedRetriever, chunk, workflow

ARCH_Q = "Explain the architecture of this repository"
PLAN = "docs/ARCHITECTURE_PLAN.md"


class CorpusRetriever:
    """A ranked corpus: each call returns its first ``top_k`` chunks, ranked 1..n."""

    def __init__(self, corpus):
        self.corpus = list(corpus)
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query, top_k=None):
        self.calls.append((query, top_k))
        window = self.corpus[: top_k or len(self.corpus)]
        return ranked(window)


def ranked(items):
    """Assign retrieval ranks 1..n (and positions) in list order, as a retriever would."""
    return [c.model_copy(update={"rank": i, "position": i - 1}) for i, c in enumerate(items, 1)]


def plan_doc_chunks(n):
    return [chunk(f"architecture section {i}", path=PLAN, start=i * 20 + 1) for i in range(n)]


def other_chunks(n, prefix="src/pkg/mod"):
    return [chunk(f"architecture of module {i}", path=f"{prefix}{i}.py", start=1) for i in range(n)]


def retrieve_event(result, round_no=1):
    return next(e for e in result.trace if e.node == "retrieve" and e.round == round_no)


def merge_event(result, round_no=1):
    return next(e for e in result.trace if e.node == "merge_evidence" and e.round == round_no)


# --- the Demo 2 starvation scenario -----------------------------------------------------------
def test_demo2_one_dominant_file_in_the_top_ten_no_longer_starves_the_selection():
    """Before: top-10 were all from one file -> 2 chunks kept, 1 file, partial. Now: pool of 50."""
    corpus = plan_doc_chunks(10) + other_chunks(12)  # the real ranking: the doc first, then others
    retriever = CorpusRetriever(corpus)
    wf, llm = workflow(retriever)
    result = wf.run(ARCH_Q)
    assert retriever.calls[0][1] == 50  # the candidate budget is separate from the baseline's 5
    assert result.rounds == 1 and result.termination is TerminationReason.SUFFICIENT
    assert result.evidence_status == "potentially_sufficient" and result.warnings == ()
    files = {c.file_path for c in result.citations}
    assert len(files) > 2 and PLAN in files
    event = merge_event(result)
    requested = retrieve_event(result)
    assert (requested.top_k, requested.retrieved, requested.candidate_files) == (50, 22, 13)
    assert (event.retrieved, event.candidate_files) == (22, 13)
    assert event.evidence_files == 13  # every candidate file is represented
    # 13 files (pass 1) + the doc's second chunk (pass 2); its other 8 chunks hit the file cap
    assert event.new_accepted == 14 and event.rejected_per_file_cap == 8
    assert event.rejected_capacity == 0 and event.rejected_duplicate == 0


def test_a_dominant_file_followed_by_relevant_files_keeps_relevance_order_between_files():
    corpus = [
        *plan_doc_chunks(6),
        chunk("architecture b", path="b.py"),
        chunk("architecture c", path="c.py"),
        chunk("architecture d", path="d.py"),
    ]
    wf, _ = workflow(CorpusRetriever(corpus), max_evidence_chunks=15)
    result = wf.run(ARCH_Q)
    paths = [c.file_path for c in result.citations]
    assert paths[:4] == [PLAN, "b.py", "c.py", "d.py"]  # one per file, in rank order
    assert paths.count(PLAN) == 2  # the per-file cap still holds (second chunk added after)
    # the doc's second chunk is its second-best candidate, not an arbitrary one
    doc_lines = [c.start_line for c in result.citations if c.file_path == PLAN]
    assert doc_lines == [1, 21]


def test_diverse_selection_never_introduces_files_that_were_not_retrieved():
    corpus = plan_doc_chunks(4) + other_chunks(3)
    result = workflow(CorpusRetriever(corpus))[0].run(ARCH_Q)
    offered = {c.file_path for c in corpus}
    assert {c.file_path for c in result.citations} <= offered
    assert {c.chunk_id for c in result.citations} <= {c.chunk_id for c in corpus}


# --- genuinely limited repositories -----------------------------------------------------------
def test_a_repository_with_a_single_source_file_is_reported_as_partial_not_hidden():
    retriever = CorpusRetriever([chunk(f"architecture part {i}", path="only.py", start=i * 10 + 1)
                                 for i in range(6)])  # fmt: skip
    wf, llm = workflow(retriever)
    result = wf.run(ARCH_Q)
    first = merge_event(result, 1)
    assert (first.retrieved, first.candidate_files, first.new_accepted) == (6, 1, 2)
    assert first.rejected_per_file_cap == 4 and first.rejected_capacity == 0
    assert result.termination is TerminationReason.NO_NEW_EVIDENCE  # round 2 found nothing new
    second = merge_event(result, 2)
    assert second.new_accepted == 0 and second.rejected_duplicate == 2
    assert result.evidence_status == "partial"
    assert result.warnings and result.warnings[0].startswith("partial_evidence:")
    assert result.generated and llm.call_count == 1  # still answers, flagged, from 2 chunks
    assert len(result.citations) == 2


def test_fewer_chunks_in_the_index_than_the_candidate_budget():
    retriever = CorpusRetriever(other_chunks(3))
    result = workflow(retriever)[0].run(ARCH_Q)
    event = merge_event(result)
    requested = retrieve_event(result)
    assert (requested.top_k, requested.retrieved) == (50, 3)  # requested 50, the index had 3
    assert event.retrieved == 3
    assert event.new_accepted == 3 and result.evidence_status == "potentially_sufficient"


# --- determinism, caps and duplicates across rounds -------------------------------------------
def test_selection_is_deterministic_whatever_order_the_candidates_arrive_in():
    items = [chunk(f"t{i}", path=f"f{i % 3}.py", start=i + 1, rank=1 + i // 2) for i in range(6)]
    outcomes = {
        tuple(
            r.chunk_id
            for r in merge_evidence(
                [], perm, max_chunks=4, per_file_cap=2, prefer_distinct_files=True
            ).evidence
        )
        for perm in itertools.permutations(items)
    }
    assert len(outcomes) == 1


def test_ties_are_broken_by_rank_then_position_then_chunk_id():
    same_rank = [chunk(f"tie{i}", path=f"f{i}.py", rank=1) for i in range(4)]
    out = merge_evidence([], same_rank[::-1], max_chunks=10, per_file_cap=None).evidence
    assert [r.chunk_id for r in out] == [
        r.chunk_id for r in sorted(same_rank, key=lambda r: (r.rank, r.position, r.chunk_id))
    ]


def test_the_per_file_cap_holds_across_rounds_and_duplicates_are_not_recounted():
    a1, a2, a3, a4 = (chunk(f"architecture a{i}", path="a.py", start=i * 10 + 1) for i in range(4))
    b1 = chunk("architecture b", path="b.py")
    retriever = ScriptedRetriever([[a1, a2], [a1, a3, a4, b1]])
    result = workflow(retriever)[0].run(ARCH_Q)
    assert len(retriever.calls) == 2 and result.rounds == 2
    round2 = merge_event(result, 2)
    assert round2.rejected_duplicate == 1  # a1 again
    assert round2.rejected_per_file_cap == 2  # a3, a4: a.py already holds 2
    assert round2.new_accepted == 1  # b1
    per_file = [c.file_path for c in result.citations]
    assert per_file.count("a.py") == 2 and per_file.count("b.py") == 1
    assert len({c.chunk_id for c in result.citations}) == len(result.citations)


# --- the final context budget -----------------------------------------------------------------
def test_context_truncation_is_distinguishable_from_candidate_starvation():
    tiny = Settings(_env_file=None, rag_context_max_tokens=450)
    big = [chunk("architecture " + f"word{i} " * 100, path=f"src/m{i}.py") for i in range(6)]
    wf, llm = workflow(CorpusRetriever(big), settings=tiny)
    result = wf.run(ARCH_Q)
    merge = merge_event(result)
    generate = next(e for e in result.trace if e.node == "generate")
    # evidence selection kept everything (no starvation)...
    assert merge.new_accepted == 6 and merge.rejected_per_file_cap == 0
    # ...but the prompt could not hold it all (context truncation)
    assert generate.context_chunks == len(result.citations) < 6
    assert generate.context_dropped_budget == 6 - generate.context_chunks > 0
    assert result.answer.context_estimated_tokens <= 450
    assert llm.call_count == 1


def test_starvation_shows_no_context_truncation():
    result = workflow(CorpusRetriever([chunk("architecture x", path="only.py")]))[0].run(ARCH_Q)
    generate = next(e for e in result.trace if e.node == "generate")
    assert generate.context_dropped_budget == 0 and generate.context_chunks == 1


# --- other intents are unchanged --------------------------------------------------------------
def test_other_intents_keep_their_candidate_budget_and_have_no_file_cap():
    same_file = [chunk(f"authentication step {i}", path="a.py", start=i * 10 + 1) for i in range(8)]
    retriever = CorpusRetriever(same_file)
    result = workflow(retriever)[0].run("Explain how authentication works")
    assert retriever.calls[0][1] == 5  # the baseline top-k, not the architecture pool
    assert merge_event(result).new_accepted == 5 and merge_event(result).rejected_per_file_cap == 0


def test_baseline_context_building_is_unchanged_by_default():
    a = chunk("first", path="a.py", rank=2)
    b = chunk("second", path="b.py", rank=1)
    assert [c.file_path for c in build_context([a, b], 5000).citations] == ["b.py", "a.py"]
    kept = build_context([a, b], 5000, preserve_order=True)
    assert [c.file_path for c in kept.citations] == ["a.py", "b.py"]
    assert [c.rank for c in kept.citations] == [2, 1]  # rank untouched either way


# --- citation provenance ----------------------------------------------------------------------
def test_reordering_does_not_corrupt_citation_provenance_or_source_numbers():
    ids = chunk("def make_chunk_id(): ...", path="src/ids.py", start=3, score=0.31)
    docs = chunk("notes about identifiers", path="docs/notes.md", start=1, score=0.92)
    other = chunk("something else entirely", path="src/other.py", start=5, score=0.55)
    ids = ids.model_copy(update={"symbol_name": "make_chunk_id"})
    retrieved = [docs, other, ids]  # retrieval ranks 1, 2, 3
    ranked = [r.model_copy(update={"rank": i}) for i, r in enumerate(retrieved, start=1)]
    wf, llm = workflow(ScriptedRetriever([ranked]))
    result = wf.run("Where is make_chunk_id defined?")
    # symbol lookup moved the defining chunk to the front of the prompt...
    assert [c.file_path for c in result.citations][0] == "src/ids.py"
    assert [c.source_number for c in result.citations] == [1, 2, 3]
    prompt = llm.requests[0].user_prompt
    positions = [prompt.index(f"BEGIN SOURCE {n} [") for n in (1, 2, 3)]
    assert positions == sorted(positions)
    by_path = {r.file_path: r for r in ranked}
    for citation, prov in zip(result.citations, result.provenance, strict=True):
        original = by_path[citation.file_path]
        # ...but every citation still carries its ORIGINAL retrieval rank and score
        assert (citation.rank, citation.score) == (original.rank, original.score)
        assert (prov.retrieval_rank, prov.score, prov.retrieval_round) == (
            original.rank,
            original.score,
            1,
        )
        assert (prov.source_number, prov.chunk_id) == (citation.source_number, citation.chunk_id)
        assert (
            f"file: {citation.file_path}\nlines: {citation.start_line}-{citation.end_line}\n"
            in (prompt[prompt.index(f"BEGIN SOURCE {citation.source_number} [") :])
        )
    assert [p.selection_order for p in result.provenance] == [1, 2, 3]
    assert result.citations[0].rank == 3  # the defining chunk was retrieved third


def test_a_chunk_seen_in_two_rounds_keeps_the_provenance_of_its_first_retrieval():
    shared = chunk("architecture shared", path="a.py", start=1)
    other = chunk("architecture other", path="b.py", start=1)
    first = [shared.model_copy(update={"rank": 3, "score": 0.30})]
    second = [
        shared.model_copy(update={"rank": 1, "score": 0.90}),
        other.model_copy(update={"rank": 2, "score": 0.20}),
    ]
    result = workflow(ScriptedRetriever([first, second]))[0].run(ARCH_Q)
    prov = {p.file_path: p for p in result.provenance}
    assert (prov["a.py"].retrieval_round, prov["a.py"].retrieval_rank, prov["a.py"].score) == (
        1,
        3,
        0.30,
    )
    assert (prov["b.py"].retrieval_round, prov["b.py"].retrieval_rank) == (2, 2)
    assert prov["a.py"].query_kind == "original" and prov["b.py"].query_kind != "original"


def test_ranks_from_different_rounds_are_never_used_to_interleave_evidence():
    """Round 1 and round 2 both have a rank-1 chunk; evidence stays in discovery (round) order."""
    r1 = [chunk("architecture one", path="a.py", rank=1, score=0.2),
          chunk("architecture two", path="a.py", start=50, rank=2, score=0.1)]  # fmt: skip
    r2 = [chunk("architecture three", path="b.py", rank=1, score=0.99)]
    result = workflow(ScriptedRetriever([r1, r2]))[0].run(ARCH_Q)
    assert [p.retrieval_round for p in result.provenance] == [1, 2, 1]
    # architecture shows one chunk per file first: a.py (round 1), b.py (round 2), then a.py again
    assert [c.file_path for c in result.citations] == ["a.py", "b.py", "a.py"]
    assert result.provenance[1].score == 0.99 and result.provenance[0].score == 0.2


def test_abstention_has_no_provenance_and_no_model_call():
    wf, llm = workflow(ScriptedRetriever([[]]))
    result = wf.run(ARCH_Q)
    assert result.provenance == () and result.citations == () and llm.call_count == 0


# --- no hosted calls --------------------------------------------------------------------------
def test_the_whole_scenario_makes_no_network_attempt(_no_outbound_network):
    corpus = plan_doc_chunks(10) + other_chunks(5)
    fake = FakeLLMClient("ok [Source 1]")
    result = workflow(CorpusRetriever(corpus), fake, settings=SETTINGS)[0].run(ARCH_Q)
    assert result.generated and fake.call_count == 1
    assert _no_outbound_network == []


@pytest.mark.parametrize("bad", [0, 51])
def test_the_candidate_pool_setting_is_validated(bad, monkeypatch):
    from pydantic import ValidationError

    monkeypatch.setenv("COPILOT_AGENT_ARCHITECTURE_CANDIDATES", str(bad))
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_a_tight_evidence_budget_goes_to_distinct_files_before_a_second_chunk_of_one_file():
    corpus = ranked(
        [
            *plan_doc_chunks(4),
            chunk("architecture b", path="b.py"),
            chunk("architecture c", path="c.py"),
        ]
    )
    tight = merge_evidence(
        [], corpus, max_chunks=3, per_file_cap=2, prefer_distinct_files=True
    ).evidence
    assert [r.file_path for r in tight] == [PLAN, "b.py", "c.py"]
    plain = merge_evidence([], corpus, max_chunks=3, per_file_cap=2).evidence
    assert [r.file_path for r in plain] == [PLAN, PLAN, "b.py"]  # what rank order alone would do
    result = workflow(CorpusRetriever(corpus), max_evidence_chunks=3)[0].run(ARCH_Q)
    assert {c.file_path for c in result.citations} == {PLAN, "b.py", "c.py"}
    event = merge_event(result)
    # the doc's 3 other chunks were not selected because the evidence budget (3) was full
    assert event.rejected_capacity == 3 and event.rejected_per_file_cap == 0
