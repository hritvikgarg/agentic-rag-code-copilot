"""The deterministic intent router and the question-term helpers."""

import pytest

from copilot.agents import Intent, classify_intent
from copilot.agents.intent import PRECEDENCE, extract_symbols
from copilot.agents.terms import MAX_TERMS, coverage, key_terms, normalize_query, stem, term_in

EXPLAIN, LOOKUP, ARCH, DEBUG, GENERAL = (
    Intent.CODE_EXPLAIN,
    Intent.SYMBOL_LOOKUP,
    Intent.ARCHITECTURE,
    Intent.DEBUGGING,
    Intent.GENERAL,
)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Explain how authentication works.", EXPLAIN),
        ("How are chunk IDs generated?", EXPLAIN),
        ("What is the purpose of the retry module?", EXPLAIN),
        ("Where is the database connection configured?", LOOKUP),
        ("Which files implement user registration?", LOOKUP),
        ("Where is repository ingestion implemented?", LOOKUP),
        ("Who calls make_chunk_id?", LOOKUP),
        ("Explain the architecture of this repository.", ARCH),
        ("Explain the flow from the API endpoint to the database.", ARCH),
        ("Give me a high-level overview of the indexing pipeline.", ARCH),
        ("Why might this endpoint return HTTP 500?", DEBUG),
        ("Which files should I investigate for this bug?", DEBUG),
        ("Why does RagService.answer raise a KeyError?", DEBUG),
        ("The indexer crashes on empty files", DEBUG),
    ],
)
def test_each_intent_is_recognised(question, expected):
    decision = classify_intent(question)
    assert decision.intent is expected
    assert decision.rule_id.startswith(expected.value + ".")


@pytest.mark.parametrize(
    ("question", "winner", "also"),
    [
        ("Explain why the upload endpoint crashes", DEBUG, EXPLAIN),
        ("Explain where the session is configured", LOOKUP, EXPLAIN),
        ("Explain the architecture of the retrieval layer", ARCH, EXPLAIN),
        ("Where is the architecture documented?", LOOKUP, ARCH),
    ],
)
def test_overlapping_questions_follow_the_documented_precedence(question, winner, also):
    decision = classify_intent(question)
    assert decision.intent is winner
    assert also in decision.matched  # the overlap is visible, not silent
    assert PRECEDENCE.index(winner) < PRECEDENCE.index(also)
    assert list(decision.matched) == sorted(decision.matched, key=PRECEDENCE.index)


@pytest.mark.parametrize(
    ("question", "rule"),
    [
        ("auth?", "general.fallback"),  # one word, no recognised shape
        ("tell me something about the project", "general.fallback"),
        ("What does this function do?", "general.too-vague"),  # no subject at all
        ("Where is the repo", "general.too-vague"),
    ],
)
def test_ambiguous_questions_fall_back_to_general(question, rule):
    decision = classify_intent(question)
    assert decision.intent is GENERAL
    assert decision.rule_id == rule


def test_routing_is_deterministic():
    question = "Why does `load_index` return None for a missing manifest?"
    assert classify_intent(question) == classify_intent(question)


def test_symbols_come_from_the_question_only():
    q = "Why does `RagService.answer` fail when make_chunk_id gets src/ids.py and HashEmbedder?"
    symbols = extract_symbols(q)
    assert "RagService.answer" in symbols and "make_chunk_id" in symbols
    assert all(s.lower() in q.lower() for s in symbols)  # nothing invented
    assert len(symbols) <= 5
    assert extract_symbols("explain the retry behaviour") == ()


def test_key_terms_drop_boilerplate_and_are_bounded():
    assert key_terms("Where is the database connection configured?") == (
        "database",
        "connection",
        "configured",
    )
    assert key_terms("auth auth AUTH") == ("auth",)
    many = " ".join(f"term{i}x" for i in range(40))
    assert len(key_terms(many)) == MAX_TERMS


def test_matching_helpers():
    assert stem("authentication") == "authentic"
    assert stem("make_chunk_id") == "make_chunk_id"  # identifiers are not stemmed
    assert term_in("authentication", "def authenticate(user)")
    fraction, missing = coverage(("alpha", "beta"), "alpha only")
    assert fraction == 0.5 and missing == ("beta",)
    assert coverage((), "anything") == (None, ())
    assert normalize_query("Beta  alpha!") == normalize_query("alpha, BETA")
