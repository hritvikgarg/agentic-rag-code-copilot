"""Deterministic fuzzing of the traceback parser (M12 S1).

Seeded ``random.Random`` instances only: a failure is reproducible from the seed in the test id.
The properties are the contract of the parser: it never raises on content, it is deterministic,
and every bound holds whatever the input looks like.
"""

from __future__ import annotations

import random

import pytest

from copilot.parsing import (
    ChainLink,
    DebugLimits,
    DiagnosticSeverity,
    ParseResult,
    ParseStatus,
    parse_traceback,
)
from tests import traceback_helpers as tb

CORPUS = [
    tb.REAL_ATTRIBUTE_NONE,
    tb.REAL_CHAINED,
    tb.REAL_RECURSION,
    tb.REAL_SYNTAX_HEADERLESS,
    tb.REAL_INDENTATION_HEADERLESS,
    tb.REAL_SYNTAX_IN_IMPORT,
    tb.REAL_CIRCULAR_IMPORT,
    tb.REAL_EXEC_STRING,
    tb.REAL_NOTES,
    tb.REAL_SUGGESTION,
    tb.REAL_STDLIB_FRAMES,
    tb.HAND_THIRD_PARTY,
    tb.HAND_NO_SOURCE,
    tb.HAND_ANSI,
    tb.windows_variant(tb.REAL_CHAINED),
    tb.relative_variant(tb.REAL_ATTRIBUTE_NONE),
    tb.many_frames(30),
]
TINY = DebugLimits(
    max_input_chars=2_000,
    max_lines=60,
    max_line_chars=80,
    max_frames_per_segment=4,
    keep_head_frames=1,
    max_segments=2,
    max_path_chars=20,
    max_function_chars=10,
    max_source_line_chars=15,
    max_type_name_chars=12,
    max_message_chars=20,
    max_message_lines=2,
    max_diagnostics=4,
)
JUNK = [
    '  File "',
    '", line ',
    "Traceback (most recent call last):",
    "The above exception was the direct cause of the following exception:",
    "During handling of the above exception, another exception occurred:",
    "  [Previous line repeated 99999999999 more times]",
    "    ^^^~~~",
    "Error: ",
    ": : :",
    "\x00\x1b[31m\u202e\u2028",
    "  | ExceptionGroup: x",
    "A" * 200,
    "\\\\server\\share",
    "   ",
    "",
]


def mutate(rng: random.Random, text: str) -> str:
    lines = text.split("\n")
    for _ in range(rng.randint(1, 4)):
        op = rng.choice(["drop", "dup", "swap", "junk", "cut", "char", "join"])
        if op == "cut":
            text = "\n".join(lines)
            lines = text[: rng.randint(0, len(text))].split("\n")
        elif not lines:
            continue
        elif op == "drop":
            del lines[rng.randrange(len(lines))]
        elif op == "dup":
            i = rng.randrange(len(lines))
            lines.insert(i, lines[i])
        elif op == "swap" and len(lines) > 1:
            i, j = rng.randrange(len(lines)), rng.randrange(len(lines))
            lines[i], lines[j] = lines[j], lines[i]
        elif op == "junk":
            lines.insert(rng.randrange(len(lines) + 1), rng.choice(JUNK))
        elif op == "char":
            i = rng.randrange(len(lines))
            if lines[i]:
                k = rng.randrange(len(lines[i]))
                lines[i] = lines[i][:k] + rng.choice('"\\:,\t\x00 ^~') + lines[i][k + 1 :]
        elif op == "join" and len(lines) > 1:
            i = rng.randrange(len(lines) - 1)
            lines[i : i + 2] = [lines[i] + lines[i + 1]]
    return "\n".join(lines)


def check_invariants(result: ParseResult, limits: DebugLimits) -> None:
    assert isinstance(result, ParseResult)
    assert len(result.diagnostics) <= limits.max_diagnostics
    if result.diagnostics_truncated:
        assert len(result.diagnostics) == limits.max_diagnostics
    assert len(result.segments) <= limits.max_segments
    assert (result.status is ParseStatus.FAILED) == (not result.segments)
    severities = {d.severity for d in result.diagnostics}
    if result.status is ParseStatus.COMPLETE:
        assert not severities & {DiagnosticSeverity.WARNING, DiagnosticSeverity.ERROR}
        assert not result.diagnostics_truncated
    if result.status is ParseStatus.FAILED:
        assert DiagnosticSeverity.ERROR in severities or result.diagnostics_truncated
    seg_indexes = [s.index for s in result.segments]
    assert seg_indexes == sorted(set(seg_indexes))
    if result.segments:
        assert result.segments[-1].link_to_next is ChainLink.NONE
    for seg in result.segments:
        assert len(seg.frames) <= limits.max_frames_per_segment
        indexes = [f.index for f in seg.frames]
        assert indexes == sorted(set(indexes))
        for frame in seg.frames:
            assert frame.line is None or frame.line >= 1
            assert len(frame.raw_path) <= limits.max_path_chars
            assert frame.function is None or len(frame.function) <= limits.max_function_chars
            assert frame.source_line is None or len(frame.source_line) <= (
                limits.max_source_line_chars
            )
            assert "\x00" not in frame.raw_path and "\u202e" not in frame.raw_path
        if seg.exception is not None:
            assert len(seg.exception.type_name) <= limits.max_type_name_chars
            assert len(seg.exception.message) <= limits.max_message_chars
            assert seg.exception.message.count("\n") < limits.max_message_lines


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("limits", [DebugLimits(), TINY], ids=["default", "tiny"])
def test_mutated_tracebacks_never_break_the_contract(seed, limits):
    rng = random.Random(seed)
    for text in CORPUS:
        for _ in range(25):
            mutated = mutate(rng, text)
            first = parse_traceback(mutated, limits=limits)
            check_invariants(first, limits)
            assert parse_traceback(mutated, limits=limits) == first  # deterministic


@pytest.mark.parametrize("seed", range(5))
def test_random_text_never_breaks_the_contract(seed):
    rng = random.Random(1_000 + seed)
    alphabet = ['File "', '", line ', "in ", "Traceback", ":", " ", "\n", "  ", "^", "~", "x", "1"]
    alphabet += ["Error", "\\", "/", "C:", "\x1b[", "\t", "\u202e", "<", ">", "|", "+-"]
    for _ in range(150):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 400)))
        for limits in (DebugLimits(), TINY):
            check_invariants(parse_traceback(text, limits=limits), limits)


def test_corpus_parses_cleanly_before_mutation():
    """Guards the fuzz corpus itself: every seed input is a recognised traceback."""
    for text in CORPUS:
        assert parse_traceback(text).status in {ParseStatus.COMPLETE, ParseStatus.PARTIAL}
