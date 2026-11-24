"""Pure helpers that decide what the user is told next to an answer."""

from copilot.llm import LLMRateLimitError
from copilot.rag.citations import Citation
from copilot.rag.models import AnswerStatus, GenerationInfo, RagAnswer
from copilot.services.presentation import (
    USER_FACING_ERRORS,
    answer_warnings,
    parse_ignore_directories,
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
