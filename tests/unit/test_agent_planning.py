"""Intent-aware planning: the intent must change what retrieval does."""

import re

import pytest

from copilot.agents import AgentConfig, Intent, classify_intent
from copilot.agents.planning import build_plan, next_query
from copilot.agents.state import QueryKind
from copilot.agents.terms import key_terms, normalize_query
from copilot.retrieval.retriever import MAX_QUERY_CHARS

CONFIG = AgentConfig(top_k=5)


def plan_for(question):
    decision = classify_intent(question)
    return decision, build_plan(decision, CONFIG), key_terms(question)


def test_the_plan_differs_by_intent():
    plans = {
        i: build_plan(classify_intent(q), CONFIG)
        for i, q in {
            Intent.ARCHITECTURE: "Explain the architecture of the indexer",
            Intent.SYMBOL_LOOKUP: "Where is make_chunk_id defined?",
            Intent.CODE_EXPLAIN: "Explain how chunking works",
            Intent.DEBUGGING: "Why does the indexer crash?",
            Intent.GENERAL: "tell me something about indexing",
        }.items()
    }
    assert plans[Intent.ARCHITECTURE].top_k == 10 and plans[Intent.ARCHITECTURE].per_file_cap == 2
    assert plans[Intent.ARCHITECTURE].diversify_files
    assert plans[Intent.SYMBOL_LOOKUP].prefers_symbol_matches
    assert plans[Intent.CODE_EXPLAIN].top_k == 5 and plans[Intent.CODE_EXPLAIN].per_file_cap is None
    assert len({p.hint for p in plans.values()}) == 5  # every intent has its own hint (None once)


def test_architecture_top_k_is_capped_at_the_retriever_maximum():
    decision = classify_intent("Explain the architecture of the indexer")
    assert build_plan(decision, AgentConfig(top_k=40)).top_k == 50


@pytest.mark.parametrize(
    ("question", "first_kind"),
    [
        ("Where is `make_chunk_id` defined and which files call it?", QueryKind.SYMBOL),
        ("Explain how `make_chunk_id` builds identifiers for chunks", QueryKind.MISSING_TERMS),
    ],
)
def test_the_first_refinement_depends_on_the_intent(question, first_kind):
    decision, plan, terms = plan_for(question)
    kind, query = next_query(plan, decision, terms, terms, [normalize_query(question)])
    assert kind is first_kind
    assert "make_chunk_id" in query


def test_refined_queries_use_only_words_from_the_question_plus_the_fixed_hint():
    question = "Explain how authentication, session tokens and password hashing work"
    decision, plan, terms = plan_for(question)
    allowed = set(re.findall(r"\w+", question.lower())) | set((plan.hint or "").split())
    asked = [normalize_query(question)]
    for missing in (terms, terms[1:], terms[2:]):
        found = next_query(plan, decision, terms, missing, asked)
        if found is None:
            break
        assert set(re.findall(r"\w+", found[1].lower())) <= allowed
        assert len(found[1]) <= MAX_QUERY_CHARS
        asked.append(normalize_query(found[1]))


def test_a_query_is_never_repeated_and_none_is_returned_when_all_would_repeat():
    question = "authentication"  # one term, no hint (general): every candidate equals the original
    decision, plan, terms = plan_for(question)
    assert plan.hint is None
    assert next_query(plan, decision, terms, terms, [normalize_query(question)]) is None

    question = "Explain how authentication works"
    decision, plan, terms = plan_for(question)
    asked = [normalize_query(question)]
    seen = set(asked)
    for _ in range(6):
        found = next_query(plan, decision, terms, terms, asked)
        if found is None:
            break
        assert normalize_query(found[1]) not in seen
        seen.add(normalize_query(found[1]))
        asked.append(normalize_query(found[1]))
    assert found is None  # a finite, small candidate set is exhausted
