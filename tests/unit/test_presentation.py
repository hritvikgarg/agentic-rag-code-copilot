"""Pure helpers that decide what the user is told next to an answer."""

import html
import re

import pytest

from copilot.llm import LLMRateLimitError
from copilot.rag.citations import Citation
from copilot.rag.models import AnswerStatus, GenerationInfo, RagAnswer
from copilot.services.presentation import (
    USER_FACING_ERRORS,
    answer_warnings,
    escape_markdown,
    literal,
    parse_ignore_directories,
    source_title,
)


def cite(n=1):
    return Citation(
        source_number=n, file_path="a.py", start_line=1, end_line=2,
        chunk_id="0123456789abcdef", score=0.5, rank=n,
    )  # fmt: skip


def answered(**kw) -> RagAnswer:
    base = dict(
        status=AnswerStatus.ANSWERED, question="q", answer="a", sources=(cite(),),
        cited_source_numbers=(1,), retrieved_count=1, context_estimated_tokens=10,
        prompt_version="rag-v1", retrieval_seconds=0.1,
    )  # fmt: skip
    return RagAnswer(**{**base, **kw})


def test_parse_ignore_directories():
    assert parse_ignore_directories(" data, tests/ ,\nsubmission,, data ") == [
        "data", "tests", "submission",
    ]  # fmt: skip
    assert parse_ignore_directories("") == []


def test_a_clean_answer_has_no_warnings():
    assert answer_warnings(answered()) == []


def test_insufficient_evidence_says_the_model_was_not_called():
    a = answered(
        status=AnswerStatus.INSUFFICIENT_EVIDENCE, answer=None, sources=(),
        cited_source_numbers=(), insufficient_reason="no_results",
    )  # fmt: skip
    [msg] = answer_warnings(a)
    assert "not called" in msg and "nothing" in msg


def test_unknown_uncited_truncated_and_dropped_are_all_reported():
    gen = GenerationInfo(provider="p", model="m", finish_reason="MAX_TOKENS")
    a = answered(cited_source_numbers=(), unknown_source_numbers=(7,), generation=gen)
    text = " ".join(answer_warnings(a))
    assert "7" in text and "does not cite" in text and "output limit" in text


def test_provider_errors_are_user_facing_but_arbitrary_exceptions_are_not():
    assert isinstance(LLMRateLimitError("x"), USER_FACING_ERRORS)
    assert not isinstance(KeyError("path"), USER_FACING_ERRORS)


# ---- literal rendering of file names (Milestone 7.1) ---------------------------------------

NASTY_LOCATIONS = [
    "src/copilot/ingestion/__init__.py:1-16",
    "pkg/__main__.py:3-9",
    "a_b_c_d.py:1-2",
    "**bold**.py:1-2",
    "file[1](x).py:1-2",
    "weird`tick.py:1-2",
    "``double``.py:1-2",
    "$x$.py:1-2",
    ":smile:.py:1-2",
    "dir with spaces/__init__.py:1-2",
]


def test_literal_wraps_text_in_a_code_span():
    assert literal("src/pkg/__init__.py:1-16") == "`src/pkg/__init__.py:1-16`"
    assert literal("") == "" and literal("   ") == ""
    assert literal("a`b") == "``a`b``"  # fence longer than the longest backtick run
    assert literal("`a") == "`` `a ``"  # leading backtick is padded (CommonMark strips the pad)
    assert "\n" not in literal("two\nlines")


def test_escape_markdown_only_adds_backslashes():
    text = "failed on src/pkg/__init__.py and :smile: *x* [a](b) $x$ `t` \\ done"
    escaped = escape_markdown(text)
    assert re.sub(r"\\(.)", r"\1", escaped) == text  # un-escaping gives the original back
    assert "__init__" not in escaped  # every underscore now carries a backslash
    assert escape_markdown("plain words 123") == "plain words 123"


def test_source_title_keeps_the_location_literal_and_the_rest_readable():
    title = source_title("Source 2", "src/copilot/ingestion/__init__.py:1-16", 0.576, cited=True)
    assert "`src/copilot/ingestion/__init__.py:1-16`" in title
    assert title.endswith("score 0.576 (cited in the answer)")
    assert "(cited" not in source_title("Source 3", "a.py:1-2", 0.1, cited=False)


@pytest.mark.parametrize("location", NASTY_LOCATIONS)
def test_every_awkward_path_renders_verbatim_as_commonmark(location):
    """Independent check with a CommonMark renderer: the path comes back byte for byte."""
    markdown_it = pytest.importorskip("markdown_it")
    md = markdown_it.MarkdownIt()
    rendered = md.render(source_title("Source 1", location, 0.5, cited=False))
    assert "<strong>" not in rendered and "<em>" not in rendered
    codes = re.findall(r"<code>(.*?)</code>", rendered, flags=re.DOTALL)
    assert [html.unescape(c).strip() for c in codes] == [location]


def test_an_error_message_with_markdown_characters_renders_verbatim():
    markdown_it = pytest.importorskip("markdown_it")
    message = "cannot read src/pkg/__init__.py: *oops* [x](y) :smile: $z$ `q`"
    rendered = markdown_it.MarkdownIt().render(escape_markdown(message))
    text = html.unescape(re.sub(r"<[^>]+>", "", rendered)).strip()
    assert text == message
    assert "<strong>" not in rendered and "<em>" not in rendered and "<code>" not in rendered
