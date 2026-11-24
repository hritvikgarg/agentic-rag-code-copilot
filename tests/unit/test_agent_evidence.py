"""Evidence accumulation, ordering and the lexical assessment."""

import pytest

from copilot.agents import AgentConfig, AssessmentOutcome, classify_intent
from copilot.agents.assessment import assess_evidence
from copilot.agents.evidence import arrange_evidence, merge_evidence
from copilot.agents.planning import build_plan
from copilot.agents.terms import key_terms
from tests.agent_helpers import chunk

CONFIG = AgentConfig(top_k=5)


# --- merge ------------------------------------------------------------------------------------
def test_merge_deduplicates_by_chunk_id_and_by_span_and_keeps_existing_evidence():
    a = chunk("alpha", path="a.py", start=1, rank=1)
    b = chunk("beta", path="b.py", start=5, rank=2)
    same_id = chunk("alpha", path="a.py", start=1, rank=3)  # identical content -> same chunk_id
    same_span = chunk("different text", path="b.py", start=5, rank=4, chunk_id="f" * 16)
    fresh = chunk("gamma", path="c.py", start=9, rank=5)
    result = merge_evidence([a, b], [same_id, same_span, fresh], max_chunks=10, per_file_cap=None)
    assert result.evidence == (a, b, fresh) and result.accepted_count == 1
    assert result.rejected_duplicate == 2


def test_merge_never_changes_or_drops_what_is_already_known():
    existing = [chunk(f"t{i}", path=f"f{i}.py", start=i + 1, rank=i + 1) for i in range(3)]
    result = merge_evidence(existing, [], max_chunks=3, per_file_cap=None)
    assert result.evidence == tuple(existing) and result.accepted_count == 0
    result = merge_evidence(existing, [chunk("new", path="n.py")], max_chunks=3, per_file_cap=None)
    assert result.evidence == tuple(existing) and result.accepted_count == 0
    assert result.rejected_capacity == 1  # full: the new one is refused, not swapped in


def test_merge_respects_the_total_cap_and_the_per_file_cap():
    new = [chunk(f"x{i}", path="same.py", start=i * 10 + 1, rank=i + 1) for i in range(5)]
    result = merge_evidence([], new, max_chunks=10, per_file_cap=2)
    assert len(result.evidence) == 2 and result.accepted_count == 2
    assert result.rejected_per_file_cap == 3 and result.rejected_capacity == 0
    result = merge_evidence([], new, max_chunks=3, per_file_cap=None)
    assert len(result.evidence) == 3 and result.accepted_count == 3
    assert result.rejected_capacity == 2 and result.rejected_per_file_cap == 0


def test_merge_preserves_metadata_exactly():
    original = chunk("alpha", path="src/a.py", start=7, rank=4, score=0.31)
    merged = merge_evidence([], [original], max_chunks=5, per_file_cap=None).evidence
    assert merged[0] == original and merged[0].score == 0.31 and merged[0].rank == 4


# --- arrange ----------------------------------------------------------------------------------
def make_plan(question):
    decision = classify_intent(question)
    return decision, build_plan(decision, CONFIG)


def test_symbol_lookup_puts_the_defining_chunk_first_without_rewriting_ranks():
    decision, plan = make_plan("Where is make_chunk_id defined?")
    docs = chunk("usage notes", path="docs/readme.md", start=1, rank=1, score=0.9)
    definer = chunk("def make_chunk_id(): ...", path="src/ids.py", start=3, rank=2, score=0.4)
    other = chunk("unrelated", path="src/other.py", start=1, rank=3, score=0.3)
    # symbol_name is part of the metadata that is matched
    definer = definer.model_copy(update={"symbol_name": "make_chunk_id"})
    out = arrange_evidence([docs, other, definer], plan, symbols=decision.symbols, terms=())
    assert out[0].file_path == "src/ids.py"
    assert out[0].rank == 2 and out[0] == definer  # provenance untouched; only the order changed
    assert sorted(r.chunk_id for r in out) == sorted(r.chunk_id for r in (docs, other, definer))
    assert out[0].score == 0.4 and out[0].start_line == 3 and out[0].chunk_id == definer.chunk_id


def test_architecture_shows_one_chunk_per_file_before_a_second_from_the_same_file():
    decision, plan = make_plan("Explain the architecture of the indexer")
    a1 = chunk("a1", path="a.py", start=1)
    a2 = chunk("a2", path="a.py", start=20)
    b1 = chunk("b1", path="b.py", start=1)
    out = arrange_evidence([a1, a2, b1], plan, symbols=(), terms=())
    assert [r.file_path for r in out] == ["a.py", "b.py", "a.py"]


def test_other_intents_keep_discovery_order():
    decision, plan = make_plan("Explain how chunking works")
    items = [chunk(f"t{i}", path=f"f{i}.py", rank=i + 1) for i in range(4)]
    out = arrange_evidence(items, plan, symbols=(), terms=())
    assert [r.chunk_id for r in out] == [r.chunk_id for r in items]


# --- assessment -------------------------------------------------------------------------------
def assess(question, evidence, *, refinable=True, config=CONFIG, stop=None):
    return assess_evidence(
        evidence,
        classify_intent(question),
        key_terms(question),
        config,
        refinable=refinable,
        stop_reason=stop,
    )


def test_empty_evidence():
    a = assess("Explain how authentication works", [])
    assert a.outcome is AssessmentOutcome.EMPTY and a.stop_reason is None
    a = assess("Explain how authentication works", [], refinable=False, stop="max_rounds_reached")
    assert a.outcome is AssessmentOutcome.EMPTY and a.stop_reason == "max_rounds_reached"


def test_potentially_sufficient_insufficient_and_exhausted():
    q = "Explain how authentication and sessions work"
    good = [chunk("authentication of a session", path="src/a.py")]
    poor = [chunk("unrelated logging helper", path="src/log.py")]
    assert assess(q, good).outcome is AssessmentOutcome.POTENTIALLY_SUFFICIENT
    refinable = assess(q, poor)
    assert refinable.outcome is AssessmentOutcome.INSUFFICIENT_REFINABLE
    assert refinable.reason == "term_coverage_below_threshold"
    assert set(refinable.missing_terms) == {"authentication", "sessions"}
    exhausted = assess(q, poor, refinable=False, stop="max_rounds_reached")
    assert exhausted.outcome is AssessmentOutcome.EXHAUSTED
    assert exhausted.stop_reason == "max_rounds_reached"


def test_terms_match_through_the_path_symbol_names_and_crude_stemming():
    q = "Explain how authentication works"
    assert assess(q, [chunk("x = 1", path="src/authenticate.py")]).outcome is (
        AssessmentOutcome.POTENTIALLY_SUFFICIENT
    )


def test_the_coverage_threshold_is_configuration():
    q = "Explain how authentication and sessions work"
    half = [chunk("authentication only", path="src/x.py")]  # 1 of 2 terms
    assert assess(q, half).outcome is AssessmentOutcome.POTENTIALLY_SUFFICIENT  # 0.5 >= 0.5
    strict = AgentConfig(min_term_coverage=1.0)
    assert assess(q, half, config=strict).outcome is AssessmentOutcome.INSUFFICIENT_REFINABLE
    lax = AgentConfig(min_term_coverage=0.0)
    poor = [chunk("nothing relevant", path="src/y.py")]
    assert assess(q, poor, config=lax).outcome is AssessmentOutcome.POTENTIALLY_SUFFICIENT


def test_similarity_scores_play_no_role():
    q = "Explain how authentication works"
    high = [chunk("unrelated text", path="a.py", score=0.99)]
    low = [chunk("authentication here", path="a.py", score=-0.2)]
    assert assess(q, high).outcome is AssessmentOutcome.INSUFFICIENT_REFINABLE
    assert assess(q, low).outcome is AssessmentOutcome.POTENTIALLY_SUFFICIENT


def test_symbol_lookup_requires_the_named_symbol_in_the_evidence():
    q = "Where is make_chunk_id defined?"
    # the term check alone is satisfied by the path, but the identifier itself is absent
    without = assess(q, [chunk("def other(): ...", path="src/chunk_ids.py")])
    assert without.outcome is AssessmentOutcome.INSUFFICIENT_REFINABLE
    assert "symbol_not_in_evidence" in without.reason
    with_symbol = assess(q, [chunk("def make_chunk_id(): ...", path="src/ids.py")])
    assert with_symbol.outcome is AssessmentOutcome.POTENTIALLY_SUFFICIENT


def test_architecture_requires_several_files():
    q = "Explain the architecture of the indexer"
    one = [
        chunk("architecture notes", path="docs/a.md"),
        chunk("architecture two", path="docs/a.md", start=9),
    ]
    assert "too_few_files_for_architecture" in assess(q, one).reason
    two = [*one, chunk("architecture three", path="docs/b.md")]
    assert assess(q, two).outcome is AssessmentOutcome.POTENTIALLY_SUFFICIENT
    assert assess(q, one, config=AgentConfig(architecture_min_files=1)).outcome is (
        AssessmentOutcome.POTENTIALLY_SUFFICIENT
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"max_rounds": 0},
        {"max_rounds": 6},
        {"min_term_coverage": 1.5},
        {"max_evidence_chunks": 0},
        {"top_k": 51},
    ],
)
def test_invalid_limits_are_rejected(bad):
    with pytest.raises(ValueError):
        AgentConfig(**bad)
