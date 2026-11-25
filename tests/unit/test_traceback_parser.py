"""Golden tests of the traceback parser (M12 S1): real CPython output and its variants."""

from __future__ import annotations

import pytest

from copilot.parsing import (
    ChainLink,
    DiagnosticCode,
    ParseStatus,
    PathStyle,
    parse_traceback,
)
from tests import traceback_helpers as tb


def _frames(result):
    return [f for seg in result.segments for f in seg.frames]


def test_single_traceback_with_two_frames_and_caret_lines():
    result = parse_traceback(tb.REAL_ATTRIBUTE_NONE)
    assert result.status is ParseStatus.COMPLETE
    assert result.diagnostics == ()
    assert len(result.segments) == 1
    seg = result.segments[0]
    assert seg.has_header and seg.link_to_next is ChainLink.NONE
    first, second = seg.frames
    assert (first.raw_path, first.path_style) == ("/tmp/tbgen/nm.py", PathStyle.POSIX_ABSOLUTE)
    assert (first.line, first.function, first.source_line) == (5, "<module>", "Svc().go(None)")
    assert (second.line, second.function, second.source_line) == (3, "go", "return x.value + 1")
    assert (first.index, second.index) == (0, 1)  # "most recent call last": input order kept
    assert seg.exception is not None
    assert seg.exception.type_name == "AttributeError"
    assert seg.exception.message == "'NoneType' object has no attribute 'value'"
    assert not seg.exception.is_syntax_error and seg.exception.module is None


def test_chained_exceptions_keep_order_and_link_kinds():
    result = parse_traceback(tb.REAL_CHAINED)
    assert result.status is ParseStatus.COMPLETE
    assert [s.index for s in result.segments] == [0, 1, 2]
    assert [s.link_to_next for s in result.segments] == [
        ChainLink.CAUSE,
        ChainLink.CONTEXT,
        ChainLink.NONE,
    ]
    assert [s.exception.type_name for s in result.segments] == [
        "KeyError",
        "RuntimeError",
        "ValueError",
    ]
    assert result.segments[0].exception.message == "'db'"
    assert [len(s.frames) for s in result.segments] == [2, 2, 2]
    final = result.final_segment
    assert final is result.segments[2]  # the exception that actually propagated
    assert [f.function for f in final.frames] == ["<module>", "run"]


def test_recursion_elided_output_records_the_repeat_count():
    result = parse_traceback(tb.REAL_RECURSION)
    assert result.status is ParseStatus.COMPLETE  # informational diagnostics only
    frames = result.segments[0].frames
    assert [f.repeat_count for f in frames] == [0, 0, 996]
    assert result.has_code(DiagnosticCode.RECURSION_ELIDED)
    assert result.segments[0].exception.type_name == "RecursionError"


def test_headerless_syntax_error_with_caret_detail():
    result = parse_traceback(tb.REAL_SYNTAX_HEADERLESS)
    assert result.status is ParseStatus.COMPLETE
    seg = result.segments[0]
    assert not seg.has_header
    frame = seg.frames[0]
    assert (frame.line, frame.function, frame.source_line) == (1, None, "def broken(:")
    exc = seg.exception
    assert exc.type_name == "SyntaxError" and exc.is_syntax_error
    assert exc.syntax is not None
    assert (exc.syntax.text, exc.syntax.caret_offset, exc.syntax.caret_length) == (
        "def broken(:",
        11,
        1,
    )
    assert not result.has_code(DiagnosticCode.MISSING_HEADER)


def test_indentation_error_is_a_syntax_family_error():
    result = parse_traceback(tb.REAL_INDENTATION_HEADERLESS)
    exc = result.segments[0].exception
    assert exc.type_name == "IndentationError" and exc.is_syntax_error
    assert exc.syntax.text == "x = 1" and exc.syntax.caret_offset == 0
    assert exc.message.startswith("expected an indented block")
    assert result.status is ParseStatus.COMPLETE


def test_syntax_error_raised_while_importing_has_a_header_and_a_function_less_last_frame():
    result = parse_traceback(tb.REAL_SYNTAX_IN_IMPORT)
    seg = result.segments[0]
    assert seg.has_header and len(seg.frames) == 2
    assert seg.frames[0].function == "<module>" and seg.frames[1].function is None
    assert seg.exception.is_syntax_error and seg.exception.syntax.text == "def broken(:"
    assert result.status is ParseStatus.COMPLETE


def test_circular_import_traceback():
    result = parse_traceback(tb.REAL_CIRCULAR_IMPORT)
    seg = result.segments[0]
    assert [f.raw_path for f in seg.frames] == ["/tmp/tbgen/ca.py", "/tmp/tbgen/cb.py"]
    assert seg.exception.type_name == "ImportError"
    assert seg.exception.message.startswith("cannot import name 'thing'")


def test_pseudo_file_frame_has_no_source_line():
    result = parse_traceback(tb.REAL_EXEC_STRING)
    assert result.status is ParseStatus.COMPLETE
    real, pseudo = result.segments[0].frames
    assert real.path_style is PathStyle.POSIX_ABSOLUTE
    assert (pseudo.raw_path, pseudo.path_style) == ("<string>", PathStyle.PSEUDO)
    assert pseudo.is_pseudo and pseudo.source_line is None and pseudo.line == 1
    assert result.has_code(DiagnosticCode.PSEUDO_FILE)


def test_exception_notes_and_hints_are_part_of_the_message():
    notes = parse_traceback(tb.REAL_NOTES).segments[0].exception
    assert notes.message == "bad value\nwhile loading config\nsecond note"
    hint = parse_traceback(tb.REAL_SUGGESTION).segments[0].exception
    assert hint.message.endswith("Did you mean: 'strip'?")
    assert hint.type_name == "AttributeError"


def test_third_party_and_stdlib_frames_are_ordinary_frames_for_the_parser():
    stdlib = parse_traceback(tb.REAL_STDLIB_FRAMES)
    assert stdlib.status is ParseStatus.COMPLETE
    assert len(stdlib.segments[0].frames) == 4
    exc = stdlib.segments[0].exception
    assert (exc.type_name, exc.module, exc.short_name) == (
        "json.decoder.JSONDecodeError",
        "json.decoder",
        "JSONDecodeError",
    )
    third = parse_traceback(tb.HAND_THIRD_PARTY).segments[0]
    assert "site-packages" in third.frames[1].raw_path  # classification is the resolver's job
    assert third.exception.type_name == "requests.exceptions.ConnectionError"


def test_missing_source_lines_and_message_less_exception():
    result = parse_traceback(tb.HAND_NO_SOURCE)
    seg = result.segments[0]
    assert [f.source_line for f in seg.frames] == [None, None]
    assert [f.line for f in seg.frames] == [10, 22]
    assert seg.exception.type_name == "KeyboardInterrupt" and seg.exception.message == ""
    assert result.status is ParseStatus.COMPLETE


def test_ansi_colour_codes_are_stripped_and_reported():
    result = parse_traceback(tb.HAND_ANSI)
    assert result.has_code(DiagnosticCode.ANSI_ESCAPES_STRIPPED)
    assert result.status is ParseStatus.COMPLETE
    assert result.segments[0].exception.type_name == "RuntimeError"
    assert result.segments[0].exception.message == "boom"
    assert "\x1b" not in result.segments[0].frames[0].raw_path


def test_windows_absolute_variant():
    result = parse_traceback(tb.windows_variant(tb.REAL_CHAINED))
    frames = _frames(result)
    assert {f.path_style for f in frames} == {PathStyle.WINDOWS_ABSOLUTE}
    assert frames[0].raw_path == "C:\\Users\\alice\\proj\\chain.py"
    assert [f.line for f in frames] == [6, 2, 12, 8, 16, 14]  # same structure as the original
    assert result.status is ParseStatus.COMPLETE


def test_relative_path_variant():
    result = parse_traceback(tb.relative_variant(tb.REAL_ATTRIBUTE_NONE))
    frames = _frames(result)
    assert {f.path_style for f in frames} == {PathStyle.RELATIVE}
    assert frames[0].raw_path == "app/nm.py"


@pytest.mark.parametrize(
    ("raw", "style"),
    [
        ("/srv/app/main.py", PathStyle.POSIX_ABSOLUTE),
        ("//srv/app/main.py", PathStyle.POSIX_ABSOLUTE),
        ("C:\\Users\\a\\main.py", PathStyle.WINDOWS_ABSOLUTE),
        ("c:/Users/a/main.py", PathStyle.WINDOWS_ABSOLUTE),
        ("\\\\server\\share\\main.py", PathStyle.UNC),
        ("app/main.py", PathStyle.RELATIVE),
        ("./main.py", PathStyle.RELATIVE),
        ("..\\main.py", PathStyle.RELATIVE),
        ("main.py", PathStyle.RELATIVE),
        ("/home/a b/c d.py", PathStyle.POSIX_ABSOLUTE),
        ("<stdin>", PathStyle.PSEUDO),
        ("<frozen importlib._bootstrap>", PathStyle.PSEUDO),
        ("<ipython-input-3-abc123>", PathStyle.PSEUDO),
        ("C:main.py", PathStyle.UNKNOWN),  # drive-relative: neither relative nor absolute
        ("\\Users\\a\\main.py", PathStyle.UNKNOWN),  # rooted without a drive
        ("file:///srv/app/main.py", PathStyle.UNKNOWN),
        ("~/main.py", PathStyle.RELATIVE),  # lexically relative; the resolver rejects it
    ],
)
def test_path_style_classification(raw, style):
    text = f'Traceback (most recent call last):\n  File "{raw}", line 3, in f\nValueError: x\n'
    frame = parse_traceback(text).segments[0].frames[0]
    assert frame.raw_path == raw and frame.path_style is style


def test_crlf_and_cr_line_endings_parse_like_lf():
    lf = parse_traceback(tb.REAL_CHAINED)
    crlf = parse_traceback(tb.REAL_CHAINED.replace("\n", "\r\n"))
    cr = parse_traceback(tb.REAL_CHAINED.replace("\n", "\r"))
    for other in (crlf, cr):
        assert other.segments == lf.segments
        assert other.status is ParseStatus.COMPLETE


def test_tab_indented_source_and_trailing_whitespace_are_normalised():
    text = (
        "Traceback (most recent call last):   \n"
        '  File "/a/b.py", line 7, in f  \n'
        "\tx = 1/0   \n"
        "ZeroDivisionError: division by zero   \n"
    )
    seg = parse_traceback(text).segments[0]
    assert seg.frames[0].source_line == "x = 1/0"
    assert seg.exception.message == "division by zero"


def test_leading_and_trailing_text_is_ignored_but_reported_as_information():
    text = "2026-01-01 ERROR something failed\n" + tb.REAL_ATTRIBUTE_NONE + "\nProcess finished\n"
    result = parse_traceback(text)
    assert result.status is ParseStatus.COMPLETE
    assert result.has_code(DiagnosticCode.LEADING_TEXT_IGNORED)
    assert result.has_code(DiagnosticCode.TRAILING_TEXT_IGNORED)
    assert result.segments[0].exception.type_name == "AttributeError"


def test_parsing_is_deterministic_and_pure():
    assert parse_traceback(tb.REAL_CHAINED) == parse_traceback(tb.REAL_CHAINED)
    first = parse_traceback(tb.REAL_STDLIB_FRAMES)
    assert first.input_digest == parse_traceback(tb.REAL_STDLIB_FRAMES).input_digest
    assert len(first.input_digest) == 12
    assert first.input_chars == len(tb.REAL_STDLIB_FRAMES)


def test_input_counters():
    result = parse_traceback(tb.REAL_ATTRIBUTE_NONE)
    assert result.input_lines == 8
    assert result.input_chars == len(tb.REAL_ATTRIBUTE_NONE)
