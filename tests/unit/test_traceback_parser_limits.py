"""Every bound of the traceback parser (M12 S1): reject, truncate visibly, never silently."""

from __future__ import annotations

import time

import pytest

from copilot.parsing import (
    ChainLink,
    DebugLimits,
    DiagnosticCode,
    ParseStatus,
    parse_traceback,
)
from tests import traceback_helpers as tb

BLOCK = 'Traceback (most recent call last):\n  File "/a.py", line 1, in f\nValueError: x\n'
CAUSE = "\nThe above exception was the direct cause of the following exception:\n\n"


def test_default_limits_are_the_documented_ones():
    lim = DebugLimits()
    assert (lim.max_input_chars, lim.max_lines, lim.max_line_chars) == (50_000, 3_000, 2_000)
    assert (lim.max_frames_per_segment, lim.keep_head_frames, lim.max_segments) == (200, 10, 20)
    assert (lim.max_path_chars, lim.max_function_chars, lim.max_source_line_chars) == (
        1_024,
        200,
        300,
    )
    assert (lim.max_type_name_chars, lim.max_message_chars, lim.max_message_lines) == (200, 500, 20)
    assert lim.max_diagnostics == 50


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_lines": 0},
        {"max_input_chars": -1},
        {"max_path_chars": True},
        {"max_message_chars": 1.5},
        {"keep_head_frames": 200},  # must be < max_frames_per_segment
    ],
)
def test_invalid_limits_are_rejected(kwargs):
    with pytest.raises(ValueError, match="DebugLimits"):
        DebugLimits(**kwargs)


def test_oversized_input_is_rejected_without_parsing():
    text = tb.REAL_ATTRIBUTE_NONE
    ok = parse_traceback(text, limits=DebugLimits(max_input_chars=len(text)))
    assert ok.status is ParseStatus.COMPLETE
    too_big = parse_traceback(text, limits=DebugLimits(max_input_chars=len(text) - 1))
    assert too_big.status is ParseStatus.FAILED
    assert too_big.segments == () and too_big.input_lines == 0
    assert too_big.input_chars == len(text)
    assert [d.code for d in too_big.diagnostics] == [DiagnosticCode.INPUT_TOO_LARGE]


def test_default_input_size_limit_is_enforced():
    result = parse_traceback("x" * 50_001)
    assert result.status is ParseStatus.FAILED
    assert result.has_code(DiagnosticCode.INPUT_TOO_LARGE)
    assert parse_traceback("x" * 50_000).has_code(DiagnosticCode.NO_TRACEBACK_FOUND)


def test_line_count_limit():
    text = tb.REAL_ATTRIBUTE_NONE  # 8 lines
    assert parse_traceback(text, limits=DebugLimits(max_lines=8)).status is ParseStatus.COMPLETE
    result = parse_traceback(text, limits=DebugLimits(max_lines=7))
    assert result.status is ParseStatus.FAILED and result.segments == ()
    assert result.has_code(DiagnosticCode.TOO_MANY_LINES) and result.input_lines == 8
    assert parse_traceback("x\n" * 3_001).has_code(DiagnosticCode.TOO_MANY_LINES)


def test_overlong_frame_line_marks_the_path_truncated_and_drops_the_line_number():
    long_path = "/srv/" + "d" * 2_500 + "/main.py"
    text = f'Traceback (most recent call last):\n  File "{long_path}", line 7, in f\nKeyError: k\n'
    result = parse_traceback(text)
    frame = result.segments[0].frames[0]
    assert frame.path_truncated is True
    assert frame.line is None  # the number may have been cut too: never trusted
    assert len(frame.raw_path) <= 1_024 and "main.py" not in frame.raw_path
    assert result.status is ParseStatus.PARTIAL
    assert {DiagnosticCode.LINE_TRUNCATED, DiagnosticCode.PATH_TRUNCATED} <= result.diagnostic_codes


def test_overlong_line_cut_inside_the_function_name_keeps_an_intact_path_and_number():
    text = (
        "Traceback (most recent call last):\n"
        f'  File "/srv/app/main.py", line 7, in {"f" * 2_500}\n'
        "KeyError: k\n"
    )
    result = parse_traceback(text)
    frame = result.segments[0].frames[0]
    assert (frame.raw_path, frame.path_truncated, frame.line) == ("/srv/app/main.py", False, 7)
    assert frame.function_truncated and len(frame.function) == 200
    assert {DiagnosticCode.LINE_TRUNCATED, DiagnosticCode.FUNCTION_TRUNCATED} <= (
        result.diagnostic_codes
    )
    assert result.status is ParseStatus.PARTIAL


def test_line_cut_inside_the_line_number_drops_the_number_but_not_the_path():
    header = '  File "/a.py", line 1'
    limits = DebugLimits(max_line_chars=len(header))
    text = f"Traceback (most recent call last):\n{header}2345, in f\nKeyError: k\n"
    result = parse_traceback(text, limits=limits)
    frame = result.segments[0].frames[0]
    assert (frame.raw_path, frame.path_truncated, frame.line) == ("/a.py", False, None)
    assert result.has_code(DiagnosticCode.LINE_TRUNCATED)
    assert result.status is ParseStatus.PARTIAL


def test_line_cut_inside_a_short_function_name_still_flags_the_function():
    header = '  File "/a.py", line 7, in handler'
    limits = DebugLimits(max_line_chars=len(header) - 3)  # cuts "handler" to "hand"
    text = f"Traceback (most recent call last):\n{header}\nKeyError: k\n"
    frame = parse_traceback(text, limits=limits).segments[0].frames[0]
    assert (frame.raw_path, frame.line, frame.function) == ("/a.py", 7, "hand")
    assert frame.function_truncated and not frame.path_truncated


def test_path_longer_than_the_path_limit_is_truncated_visibly():
    long_path = "/srv/" + "d" * 1_200 + "/main.py"  # shorter than a line, longer than a path
    text = f'Traceback (most recent call last):\n  File "{long_path}", line 7, in f\nKeyError: k\n'
    result = parse_traceback(text)
    frame = result.segments[0].frames[0]
    assert frame.path_truncated is True and len(frame.raw_path) == 1_024
    assert frame.line == 7  # the rest of the line was intact
    assert result.has_code(DiagnosticCode.PATH_TRUNCATED)
    assert result.status is ParseStatus.PARTIAL


def test_unterminated_path_is_treated_as_truncated():
    text = (
        'Traceback (most recent call last):\n  File "/srv/app/main.py, line 7, in f\nKeyError: k\n'
    )
    result = parse_traceback(text)
    frame = result.segments[0].frames[0]
    assert frame.path_truncated is True and frame.line is None
    codes = result.diagnostic_codes
    assert {DiagnosticCode.MALFORMED_FRAME, DiagnosticCode.PATH_TRUNCATED} <= codes


def test_function_source_and_message_limits():
    text = (
        "Traceback (most recent call last):\n"
        f'  File "/a.py", line 3, in {"fn" * 150}\n'
        f"    {'s' * 450}\n"
        f"KeyError: {'m' * 700}\n"
    )
    result = parse_traceback(text)
    frame = result.segments[0].frames[0]
    assert frame.function_truncated and len(frame.function) == 200
    assert frame.source_line_truncated and len(frame.source_line) == 300
    exc = result.segments[0].exception
    assert exc.message_truncated and len(exc.message) == 500
    assert {
        DiagnosticCode.FUNCTION_TRUNCATED,
        DiagnosticCode.SOURCE_LINE_TRUNCATED,
        DiagnosticCode.MESSAGE_TRUNCATED,
    } <= result.diagnostic_codes
    assert result.status is ParseStatus.PARTIAL


def test_message_line_limit():
    notes = "\n".join(f"note {n}" for n in range(30))
    head = 'Traceback (most recent call last):\n  File "/a.py", line 3, in f\n'
    text = f"{head}KeyError: first\n{notes}\n"
    exc = parse_traceback(text).segments[0].exception
    assert exc.message_truncated and exc.message.count("\n") == 19  # 20 lines kept
    assert exc.message.startswith("first\nnote 0") and "note 30" not in exc.message


def test_type_name_limit_makes_the_exception_line_malformed():
    text = f'Traceback (most recent call last):\n  File "/a.py", line 3, in f\n{"E" * 201}: boom\n'
    result = parse_traceback(text)
    assert result.segments[0].exception is None
    assert {
        DiagnosticCode.MALFORMED_EXCEPTION_LINE,
        DiagnosticCode.MISSING_EXCEPTION_LINE,
    } <= result.diagnostic_codes
    ok = f'Traceback (most recent call last):\n  File "/a.py", line 3, in f\n{"E" * 200}: boom\n'
    assert parse_traceback(ok).segments[0].exception.type_name == "E" * 200


def test_frame_limit_keeps_the_first_ten_and_the_last_hundred_ninety():
    result = parse_traceback(tb.many_frames(250))
    seg = result.segments[0]
    assert len(seg.frames) == 200 and seg.frames_omitted == 50
    indexes = [f.index for f in seg.frames]
    assert indexes[:10] == list(range(10)) and indexes[10:] == list(range(60, 250))
    assert seg.frames[-1].function == "f249"  # the innermost frame is always kept
    assert result.has_code(DiagnosticCode.FRAMES_TRUNCATED)
    assert result.status is ParseStatus.PARTIAL


def test_frame_limit_boundary():
    exact = parse_traceback(tb.many_frames(200))
    assert exact.segments[0].frames_omitted == 0 and exact.status is ParseStatus.COMPLETE
    over = parse_traceback(tb.many_frames(201))
    assert over.segments[0].frames_omitted == 1 and len(over.segments[0].frames) == 200


def test_segment_limit_keeps_the_last_twenty_segments():
    text = CAUSE.join([BLOCK] * 25)
    result = parse_traceback(text)
    assert len(result.segments) == 20 and result.segments_omitted == 5
    assert result.segments[0].index == 5 and result.segments[-1].index == 24
    assert result.segments[-1].link_to_next is ChainLink.NONE
    assert result.has_code(DiagnosticCode.SEGMENTS_TRUNCATED)
    assert result.status is ParseStatus.PARTIAL


def test_diagnostic_limit_is_bounded_and_still_makes_the_result_partial():
    lines = ["Traceback (most recent call last):"]
    lines += ['  File "<string>", line 1, in <module>'] * 120  # one info diagnostic each
    lines.append("ValueError: x")
    result = parse_traceback("\n".join(lines) + "\n")
    assert len(result.diagnostics) == 50 and result.diagnostics_truncated
    assert result.diagnostics[-1].code is DiagnosticCode.DIAGNOSTICS_TRUNCATED
    assert result.status is ParseStatus.PARTIAL  # the marker is a warning


def test_custom_limits_apply():
    limits = DebugLimits(max_message_chars=5, max_function_chars=3, max_path_chars=8)
    text = (
        "Traceback (most recent call last):\n"
        '  File "/srv/app/main.py", line 3, in handler\n'
        "KeyError: abcdefgh\n"
    )
    seg = parse_traceback(text, limits=limits).segments[0]
    assert seg.frames[0].raw_path == "/srv/app" and seg.frames[0].path_truncated
    assert seg.frames[0].function == "han" and seg.exception.message == "abcde"


@pytest.mark.parametrize(
    "text",
    [
        'File "' * 8_000,
        ('File "' + "a" * 1_990 + "\n") * 24,
        ": " * 24_000,
        ("Traceback (most recent call last):\n" * 1_400),
        ('  File "/a.py", line 1, in f\n    x\n' * 1_400),
        "\x1b[" * 20_000,
        "^~ " * 16_000,
        "A" * 49_999 + ":",
    ],
)
def test_pathological_inputs_finish_quickly(text):
    started = time.perf_counter()
    result = parse_traceback(text)
    assert time.perf_counter() - started < 5.0  # generous: a linear parser takes milliseconds
    assert result.status in set(ParseStatus)
