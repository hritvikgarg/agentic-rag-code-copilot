"""Question terms and lexical matching, shared by the router, the planner and the assessor.

Everything here is a deterministic function of the user's question and of retrieved text. No LLM,
no embedding, no I/O. The terms are the question's own words (never words taken from retrieved
content), which is what keeps retrieved text from steering the workflow: repository content is
data for the final prompt, not input to query refinement.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

MAX_TERMS = 12  # bounds the refined queries and the work done per assessment
MIN_TERM_LENGTH = 3
MAX_TERM_LENGTH = 40

# Words that carry no repository-specific meaning: English function words plus the boilerplate
# of questions about code ("where", "implemented", "function", "file", ...). Deliberately small
# and plain so the behaviour is easy to read and predict.
_STOPWORD_TEXT = (
    "a an and are as at be been being but by can could did do does doing done for "
    "from had has have how i if in into is it its may might more most my no not of "
    "on or our should so some such than that the their them then there these they "
    "this those to was we were what when where which who whom why will with would "
    "you your about all also any each every explain describe tell show give please "
    "find locate list code codebase repo repository project app application system "
    "program source implemented implement implements implementation defined define "
    "defines used use uses using work works working handle handles handled file "
    "files function functions method methods class classes module modules line "
    "lines thing things way ways "
)
STOPWORDS = frozenset(_STOPWORD_TEXT.split())

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Longest first so "ations" wins over "s". A stem must stay at least 4 characters long.
_SUFFIXES = ("ations", "ation", "ings", "ing", "ers", "er", "ed", "es", "ion", "s")


def key_terms(question: str) -> tuple[str, ...]:
    """The question's content words, lowercase, in order of appearance, unique, at most 12.

    Identifiers (``make_chunk_id``, ``RagService``) are kept whole. Stop words, very short and
    very long tokens (a 40+ character token is more likely a pasted blob than a name) are dropped.
    """
    seen: dict[str, None] = {}
    for token in _WORD.findall(question):
        term = token.lower().strip("_")
        if not MIN_TERM_LENGTH <= len(term) <= MAX_TERM_LENGTH or term in STOPWORDS:
            continue
        seen.setdefault(term, None)
        if len(seen) >= MAX_TERMS:
            break
    return tuple(seen)


def stem(term: str) -> str:
    """Crude suffix-stripping stem ("authentication" -> "authentic"); identifiers stay unchanged."""
    if "_" in term or any(ch.isdigit() for ch in term):
        return term
    for suffix in _SUFFIXES:
        if term.endswith(suffix) and len(term) - len(suffix) >= 4:
            return term[: -len(suffix)]
    return term


def term_in(term: str, haystack_lower: str) -> bool:
    """Does ``term`` (or its stem) occur in the already-lowercased ``haystack_lower``?"""
    return term in haystack_lower or stem(term) in haystack_lower


def coverage(terms: Sequence[str], haystack_lower: str) -> tuple[float | None, tuple[str, ...]]:
    """``(fraction of terms found, terms not found)``; ``(None, ())`` when there are no terms."""
    if not terms:
        return None, ()
    missing = tuple(t for t in terms if not term_in(t, haystack_lower))
    return (len(terms) - len(missing)) / len(terms), missing


def normalize_query(query: str) -> str:
    """Canonical form used to decide "have we asked this before?": the set of words, sorted.

    Two queries made of the same words in any order, case or punctuation count as the same query.
    """
    return " ".join(sorted(set(re.findall(r"\w+", query.lower()))))


def join_terms(terms: Iterable[str]) -> str:
    return " ".join(terms)
