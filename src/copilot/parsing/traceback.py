"""Deterministic, bounded parser for Python tracebacks (Milestone 12, slice S1).

What it is: a pure function from untrusted text to a typed :class:`ParseResult`. What it is not: a
validator of Python behaviour, a path resolver (slice S2) or an interpreter of the traceback.

Security properties (each is covered by tests):

* **The text is data.** It is only ever split, compared and sliced with string operations and a
  few simple anchored regular expressions. Nothing is evaluated, compiled, imported or executed,
  no file is opened and no path in the text is looked at.
* **Linear time.** Input size, line count and line length are bounded before any matching; every
  rule is a string operation or a regular expression without nested quantifiers.
* **No silent truncation.** A cut field carries a ``*_truncated`` flag and a diagnostic. A
  truncated or unterminated path is flagged so it can never become a mapping candidate.
* **No raw text in diagnostics, logs or ``repr``.** The only thing logged is a DEBUG record with
  counts, the status and a short SHA-256 digest of the input.

Recognised input (see ``docs/m12-debugging-plan.md`` for the exact scope): standard Python
tracebacks with any number of frames, chained exceptions, Windows/POSIX/relative paths,
SyntaxError-family blocks (with or without the ``Traceback`` header), frames without source lines,
Python 3.11+ caret lines, recursion-elided output and pseudo-files. Not recognised (reported, never
guessed): ExceptionGroup output, pytest/IPython/JavaScript/Java traces and log-prefixed lines.
A bare word without ``:`` is accepted as an exception only if it ends like an exception name.
Non-blank lines that directly follow the exception line are treated as continuation of its message
(this is how Python prints ``add_note`` notes and "Did you mean" hints) until a blank line.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass, field

from copilot.parsing.models import (
    DIAGNOSTIC_SEVERITY,
    ChainLink,
    DebugLimits,
    DiagnosticCode,
    DiagnosticSeverity,
    ExceptionInfo,
    ParseDiagnostic,
    ParseResult,
    ParseStatus,
    PathStyle,
    StackFrame,
    SyntaxDetail,
    TracebackSegment,
)

logger = logging.getLogger(__name__)

TRACEBACK_HEADER = "Traceback (most recent call last):"
_CAUSE_LINE = "The above exception was the direct cause of the following exception:"
_CONTEXT_LINE = "During handling of the above exception, another exception occurred:"
_FRAME_PREFIX = 'File "'
_FRAME_MID = '", line '
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")  # disjoint character classes: linear
_TYPE_NAME = re.compile(r"[^\W\d]\w*(?:\.[^\W\d]\w*)*")  # '.' is not in \w: no ambiguity
_DIGITS = re.compile(r"\d+")
_REPEAT = re.compile(r"\[Previous line repeated (\d{1,9}) more times?\]")
_SYNTAX_ERRORS = frozenset({"SyntaxError", "IndentationError", "TabError"})
# A line with no ':' is accepted as an exception line only if it ends like an exception class
# name (KeyboardInterrupt, StopIteration, SystemExit, ...). Any other bare word is not trusted.
_BARE_SUFFIXES = ("Error", "Exception", "Warning", "Exit", "Interrupt", "Iteration")
_MAX_LINE_NUMBER = 100_000_000
_REMOVED = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})  # replaced by U+FFFD (tab is kept)
_RANK = {
    DiagnosticSeverity.INFO: 0,
    DiagnosticSeverity.WARNING: 1,
    DiagnosticSeverity.ERROR: 2,
}


# -- small pure helpers ---------------------------------------------------------------------


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:12]


def _clean_control(line: str) -> tuple[str, bool]:
    """Replace control/format/separator characters (not tab) by U+FFFD; report if any changed."""
    if line.isprintable():
        return line, False
    changed = False
    out: list[str] = []
    for ch in line:
        if ch != "\t" and unicodedata.category(ch) in _REMOVED:
            out.append("�")
            changed = True
        else:
            out.append(ch)
    return "".join(out), changed


def _classify_path(path: str) -> PathStyle:
    """Lexical classification only; the path is never touched on any filesystem."""
    if not path or "://" in path:
        return PathStyle.UNKNOWN
    if len(path) >= 2 and path[0] == "<" and path[-1] == ">":
        return PathStyle.PSEUDO
    if path.startswith("\\\\"):
        return PathStyle.UNC
    if len(path) >= 2 and path[1] == ":" and path[0].isascii() and path[0].isalpha():
        return PathStyle.WINDOWS_ABSOLUTE if path[2:3] in ("\\", "/") else PathStyle.UNKNOWN
    if path[0] == "/":
        return PathStyle.POSIX_ABSOLUTE
    if path[0] == "\\":
        return PathStyle.UNKNOWN  # rooted but drive-less: neither relative nor absolute
    return PathStyle.RELATIVE


def _is_caret_line(stripped: str) -> bool:
    """Python's error-location markers: only '^', '~' and blanks, with at least one '^'."""
    return "^" in stripped and not stripped.strip("^~ ")


def _frame_shape(stripped: str) -> bool:
    """True if ``stripped`` looks like ``File "...", line N...`` (cheap, used by the hint)."""
    if not stripped.startswith(_FRAME_PREFIX):
        return False
    close = stripped.rfind(_FRAME_MID)
    if close < len(_FRAME_PREFIX):
        return False
    return _DIGITS.match(stripped, close + len(_FRAME_MID)) is not None


def _is_group_line(stripped: str) -> bool:
    """ExceptionGroup output is drawn with '|' and '+-' prefixes (not parsed)."""
    return stripped.startswith(("+ Exception Group", "+-", "| ")) or stripped == "|"


def _looks_foreign(text: str, stripped: str) -> bool:
    """Markers of pytest, IPython, JavaScript or Java output (never of a plain Python trace)."""
    if text.startswith("E   "):
        return True  # pytest assertion detail
    if len(stripped) >= 7 and stripped[:3] in ("___", "===") and stripped[-3:] == stripped[:3]:
        return " " in stripped  # pytest section banner
    if "Traceback (most recent call last)" in stripped:
        return True  # IPython puts the header after the exception name
    if len(stripped) >= 20 and not stripped.strip("-"):
        return True  # IPython rule line
    if stripped.startswith("at ") and stripped.endswith(")") and ":" in stripped:
        return True  # JavaScript / Java stack frame
    return stripped.startswith(("Exception in thread ", "Caused by: "))


# -- parser state ---------------------------------------------------------------------------


@dataclass
class _Frame:
    index: int
    raw_path: str
    path_truncated: bool
    line: int | None
    function: str | None
    function_truncated: bool
    source_line: str | None = None
    source_truncated: bool = False
    source_indent: int = 0
    caret_line: str | None = None
    repeat_count: int = 0
    extra_source_noted: bool = False


@dataclass
class _Exception:
    type_name: str
    parts: list[str] = field(default_factory=list)
    chars: int = 0
    lines: int = 0
    truncated: bool = False
    full: bool = False


@dataclass
class _Segment:
    index: int
    has_header: bool
    frames: list[_Frame] = field(default_factory=list)
    exception: _Exception | None = None
    link: ChainLink = ChainLink.NONE


class _Diagnostics:
    """Bounded diagnostic list. ``worst`` is tracked even for diagnostics that were not stored."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self.items: list[ParseDiagnostic] = []
        self.truncated = False
        self.worst: DiagnosticSeverity | None = None
        self._once: set[DiagnosticCode] = set()

    def add(
        self,
        code: DiagnosticCode,
        *,
        severity: DiagnosticSeverity | None = None,
        segment: int | None = None,
        frame: int | None = None,
        line: int | None = None,
    ) -> None:
        sev = severity or DIAGNOSTIC_SEVERITY[code]
        if self.worst is None or _RANK[sev] > _RANK[self.worst]:
            self.worst = sev
        if self.truncated:
            return
        if len(self.items) >= self._limit - 1:
            self.truncated = True
            marker = DiagnosticCode.DIAGNOSTICS_TRUNCATED
            self.items.append(ParseDiagnostic(code=marker, severity=DIAGNOSTIC_SEVERITY[marker]))
            if _RANK[DIAGNOSTIC_SEVERITY[marker]] > _RANK[self.worst]:
                self.worst = DIAGNOSTIC_SEVERITY[marker]
            return
        self.items.append(
            ParseDiagnostic(
                code=code, severity=sev, segment_index=segment, frame_index=frame, line_number=line
            )
        )

    def add_once(self, code: DiagnosticCode, **kwargs: int | None) -> None:
        if code not in self._once:
            self._once.add(code)
            self.add(code, **kwargs)  # type: ignore[arg-type]


class _Parser:
    def __init__(self, limits: DebugLimits) -> None:
        self.lim = limits
        self.diag = _Diagnostics(limits.max_diagnostics)
        self.segments: list[_Segment] = []
        self.cur: _Segment | None = None  # open: a block whose exception line was not seen yet
        self.frame: _Frame | None = None
        self.awaiting: str | None = None  # None | "source" | "after_source"
        self.exc: _Exception | None = None  # exception whose message may still continue
        self.in_message = False
        self.foreign_seen = False
        self.line_no = 0
        self.line_truncated = False

    # -- segment lifecycle -------------------------------------------------------------
    def _new_segment(self, *, has_header: bool) -> _Segment:
        seg = _Segment(index=len(self.segments), has_header=has_header)
        self.segments.append(seg)
        self.cur = seg
        self.frame = None
        self.awaiting = None
        self.exc = None
        self.in_message = False
        return seg

    def _close(self) -> None:
        self.cur = None
        self.frame = None
        self.awaiting = None
        self.in_message = False

    # -- line dispatch -----------------------------------------------------------------
    def feed(self, text: str) -> None:
        stripped = text.strip()
        if not stripped:
            self.in_message = False
            self.awaiting = None
            return
        if stripped == TRACEBACK_HEADER:
            self._new_segment(has_header=True)
            return
        if stripped in (_CAUSE_LINE, _CONTEXT_LINE):
            self._chain(ChainLink.CAUSE if stripped == _CAUSE_LINE else ChainLink.CONTEXT)
            return
        if stripped.startswith(_FRAME_PREFIX):
            self._frame_line(stripped)
            return
        indented = text[0] in " \t"
        if indented and self.frame is not None and self.awaiting is not None:
            self._frame_detail(text, stripped)
            return
        if _is_group_line(stripped):
            self.diag.add_once(DiagnosticCode.EXCEPTION_GROUP_UNSUPPORTED, line=self.line_no)
            return
        if self.in_message and self.exc is not None:
            self._add_message(self.exc, stripped)
            return
        if self.cur is not None:
            self._exception_line(text, stripped, indented)
            return
        if _looks_foreign(text, stripped):
            self.foreign_seen = True
        code = (
            DiagnosticCode.TRAILING_TEXT_IGNORED
            if self.segments
            else DiagnosticCode.LEADING_TEXT_IGNORED
        )
        self.diag.add_once(code, line=self.line_no)

    def _chain(self, link: ChainLink) -> None:
        if not self.segments:
            self.diag.add(DiagnosticCode.CHAIN_LINK_WITHOUT_SEGMENT, line=self.line_no)
        else:
            self.segments[-1].link = link
        self._close()

    # -- frames ------------------------------------------------------------------------
    def _frame_line(self, stripped: str) -> None:
        lim = self.lim
        seg = self.cur if self.cur is not None else self._new_segment(has_header=False)
        self.in_message = False
        frame_index = len(seg.frames)
        body = stripped[len(_FRAME_PREFIX) :]
        close = body.rfind(_FRAME_MID)
        line: int | None = None
        function: str | None = None
        path_truncated = False
        function_truncated = False
        malformed = False
        if close == -1:
            path, malformed, path_truncated = body, True, True  # unterminated: incomplete path
        else:
            path = body[:close]
            rest = body[close + len(_FRAME_MID) :]
            digits = _DIGITS.match(rest)
            if digits is None:
                malformed = True
            else:
                line = self._line_number(digits.group(), seg.index, frame_index)
                tail = rest[digits.end() :]
                if tail.startswith(", in "):
                    function = tail[5:].strip() or None
                elif tail.strip():
                    malformed = True
                if self.line_truncated:
                    if not tail:
                        line = None  # cut inside or right after the number: not trustworthy
                    elif function is not None:
                        function_truncated = True  # the cut fell inside the function name
        if len(path) > lim.max_path_chars:
            path = path[: lim.max_path_chars]
            path_truncated = True
        if path_truncated:
            self.diag.add(
                DiagnosticCode.PATH_TRUNCATED,
                segment=seg.index,
                frame=frame_index,
                line=self.line_no,
            )
        if function is not None and len(function) > lim.max_function_chars:
            function = function[: lim.max_function_chars]
            function_truncated = True
        if function_truncated:
            self.diag.add(
                DiagnosticCode.FUNCTION_TRUNCATED,
                segment=seg.index,
                frame=frame_index,
                line=self.line_no,
            )
        if not path:
            malformed = True
        if malformed:
            self.diag.add(
                DiagnosticCode.MALFORMED_FRAME,
                segment=seg.index,
                frame=frame_index,
                line=self.line_no,
            )
        if _classify_path(path) is PathStyle.PSEUDO:
            self.diag.add(
                DiagnosticCode.PSEUDO_FILE, segment=seg.index, frame=frame_index, line=self.line_no
            )
        self.frame = _Frame(
            index=frame_index,
            raw_path=path,
            path_truncated=path_truncated,
            line=line,
            function=function,
            function_truncated=function_truncated,
        )
        seg.frames.append(self.frame)
        self.awaiting = "source"

    def _line_number(self, digits: str, segment: int, frame: int) -> int | None:
        value = int(digits) if len(digits) <= 9 else 0
        if value < 1 or value > _MAX_LINE_NUMBER:
            self.diag.add(
                DiagnosticCode.INVALID_LINE_NUMBER,
                segment=segment,
                frame=frame,
                line=self.line_no,
            )
            return None
        return value

    def _frame_detail(self, text: str, stripped: str) -> None:
        frame = self.frame
        assert frame is not None and self.cur is not None  # guaranteed by the caller
        seg = self.cur.index
        repeat = _REPEAT.fullmatch(stripped)
        if repeat is not None:
            frame.repeat_count = int(repeat.group(1))
            self.diag.add(
                DiagnosticCode.RECURSION_ELIDED, segment=seg, frame=frame.index, line=self.line_no
            )
            return
        if _is_caret_line(stripped):
            if frame.caret_line is None:
                frame.caret_line = text
            return
        if self.awaiting == "source":
            source = stripped
            if len(source) > self.lim.max_source_line_chars:
                source = source[: self.lim.max_source_line_chars]
                frame.source_truncated = True
                self.diag.add(
                    DiagnosticCode.SOURCE_LINE_TRUNCATED,
                    segment=seg,
                    frame=frame.index,
                    line=self.line_no,
                )
            frame.source_line = source
            frame.source_indent = len(text) - len(text.lstrip(" \t"))
            self.awaiting = "after_source"
            return
        if not frame.extra_source_noted:
            frame.extra_source_noted = True
            self.diag.add(
                DiagnosticCode.EXTRA_SOURCE_LINES_IGNORED,
                segment=seg,
                frame=frame.index,
                line=self.line_no,
            )

    # -- exception line and message ----------------------------------------------------
    def _exception_line(self, text: str, stripped: str, indented: bool) -> None:
        seg = self.cur
        assert seg is not None
        if indented:
            self.diag.add(DiagnosticCode.UNEXPECTED_LINE, segment=seg.index, line=self.line_no)
            return
        head, sep, rest = text.partition(":")
        type_name = head if sep else stripped
        if (
            len(type_name) > self.lim.max_type_name_chars
            or not _TYPE_NAME.fullmatch(type_name)
            or (not sep and not type_name.rsplit(".", 1)[-1].endswith(_BARE_SUFFIXES))
        ):
            self.diag.add(
                DiagnosticCode.MALFORMED_EXCEPTION_LINE, segment=seg.index, line=self.line_no
            )
            self._close()
            return
        exc = _Exception(type_name=type_name)
        seg.exception = exc
        self.exc = exc
        self._close()
        self.in_message = True
        first = rest[1:] if rest.startswith(" ") else rest
        if first.strip():
            self._add_message(exc, first.strip())

    def _add_message(self, exc: _Exception, piece: str) -> None:
        sep = 1 if exc.parts else 0
        room = self.lim.max_message_chars - exc.chars - sep
        if exc.full or exc.lines >= self.lim.max_message_lines or room <= 0:
            exc.full = True
            exc.truncated = True
            return
        if len(piece) > room:
            piece = piece[:room]
            exc.truncated = True
            exc.full = True
        exc.parts.append(piece)
        exc.chars += sep + len(piece)
        exc.lines += 1

    # -- result ------------------------------------------------------------------------
    def build(self) -> tuple[tuple[TracebackSegment, ...], int]:
        lim = self.lim
        drafts = self.segments
        omitted = max(0, len(drafts) - lim.max_segments)
        if omitted:
            self.diag.add(DiagnosticCode.SEGMENTS_TRUNCATED)
        last_index = len(drafts) - 1
        built: list[TracebackSegment] = []
        for seg in drafts[omitted:]:
            built.append(self._build_segment(seg, is_last=seg.index == last_index))
        return tuple(built), omitted

    def _build_segment(self, seg: _Segment, *, is_last: bool) -> TracebackSegment:
        lim = self.lim
        frames = seg.frames
        omitted_frames = 0
        if len(frames) > lim.max_frames_per_segment:
            tail = lim.max_frames_per_segment - lim.keep_head_frames
            omitted_frames = len(frames) - lim.max_frames_per_segment
            frames = [*frames[: lim.keep_head_frames], *frames[-tail:]]
            self.diag.add(DiagnosticCode.FRAMES_TRUNCATED, segment=seg.index)
        exception = self._build_exception(seg)
        if exception is None:
            self.diag.add(DiagnosticCode.MISSING_EXCEPTION_LINE, segment=seg.index)
        if not seg.has_header and not (exception is not None and exception.is_syntax_error):
            self.diag.add(DiagnosticCode.MISSING_HEADER, segment=seg.index)
        link = seg.link
        if is_last and link is not ChainLink.NONE:
            self.diag.add(DiagnosticCode.CHAIN_LINK_DANGLING, segment=seg.index)
            link = ChainLink.NONE
        elif not is_last and link is ChainLink.NONE:
            self.diag.add_once(DiagnosticCode.MULTIPLE_TRACEBACKS, segment=seg.index)
        return TracebackSegment(
            index=seg.index,
            has_header=seg.has_header,
            frames=tuple(self._build_frame(f) for f in frames),
            frames_omitted=omitted_frames,
            exception=exception,
            link_to_next=link,
        )

    def _build_exception(self, seg: _Segment) -> ExceptionInfo | None:
        raw = seg.exception
        if raw is None:
            return None
        if raw.truncated:
            self.diag.add(DiagnosticCode.MESSAGE_TRUNCATED, segment=seg.index)
        is_syntax = raw.type_name.rsplit(".", 1)[-1] in _SYNTAX_ERRORS
        syntax = None
        if is_syntax and seg.frames:
            last = seg.frames[-1]
            if last.source_line is not None:
                offset = length = None
                if last.caret_line is not None:
                    first = last.caret_line.index("^")
                    offset = max(0, first - last.source_indent)
                    length = last.caret_line.rindex("^") - first + 1
                syntax = SyntaxDetail(
                    text=last.source_line,
                    text_truncated=last.source_truncated,
                    caret_offset=offset,
                    caret_length=length,
                )
        return ExceptionInfo(
            type_name=raw.type_name,
            message="\n".join(raw.parts),
            message_truncated=raw.truncated,
            is_syntax_error=is_syntax,
            syntax=syntax,
        )

    @staticmethod
    def _build_frame(frame: _Frame) -> StackFrame:
        return StackFrame(
            index=frame.index,
            raw_path=frame.raw_path,
            path_style=_classify_path(frame.raw_path),
            path_truncated=frame.path_truncated,
            line=frame.line,
            function=frame.function,
            function_truncated=frame.function_truncated,
            source_line=frame.source_line,
            source_line_truncated=frame.source_truncated,
            repeat_count=frame.repeat_count,
        )


# -- public interface -----------------------------------------------------------------------


def _failed(text: str, code: DiagnosticCode, *, digest: str, lines: int = 0) -> ParseResult:
    return ParseResult(
        status=ParseStatus.FAILED,
        diagnostics=(ParseDiagnostic(code=code, severity=DIAGNOSTIC_SEVERITY[code]),),
        input_chars=len(text),
        input_lines=lines,
        input_digest=digest,
    )


def parse_traceback(text: str, *, limits: DebugLimits = DebugLimits()) -> ParseResult:  # noqa: B008
    """Parse ``text`` into segments, frames and exception information. Never raises on content.

    The result is bounded by ``limits``. Oversized input (characters or lines) is rejected with a
    ``FAILED`` result and nothing is parsed; for oversized input ``input_digest`` covers only the
    first ``max_input_chars`` characters and ``input_lines`` is 0 (the input is not scanned).

    Raises:
        TypeError: if ``text`` is not a ``str`` (a programming error, not a content problem).
    """
    if not isinstance(text, str):
        raise TypeError("parse_traceback expects str")
    if len(text) > limits.max_input_chars:
        return _failed(
            text,
            DiagnosticCode.INPUT_TOO_LARGE,
            digest=_digest(text[: limits.max_input_chars]),
        )
    digest = _digest(text)
    if not text.strip():
        return _failed(text, DiagnosticCode.INPUT_EMPTY, digest=digest)
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    line_count = normalised.count("\n") + (0 if normalised.endswith("\n") else 1)
    if line_count > limits.max_lines:
        return _failed(text, DiagnosticCode.TOO_MANY_LINES, digest=digest, lines=line_count)

    parser = _Parser(limits)
    flagged_ansi = flagged_controls = False
    for number, raw in enumerate(normalised.split("\n"), start=1):
        parser.line_no = number
        if "\x1b" in raw:
            cleaned = _ANSI.sub("", raw)
            if cleaned != raw and not flagged_ansi:
                flagged_ansi = True
                parser.diag.add(DiagnosticCode.ANSI_ESCAPES_STRIPPED, line=number)
            raw = cleaned
        parser.line_truncated = len(raw) > limits.max_line_chars
        if parser.line_truncated:
            raw = raw[: limits.max_line_chars]
            parser.diag.add(DiagnosticCode.LINE_TRUNCATED, line=number)
        line, changed = _clean_control(raw.rstrip())
        if changed and not flagged_controls:
            flagged_controls = True
            parser.diag.add(DiagnosticCode.CONTROL_CHARACTERS_REPLACED, line=number)
        parser.feed(line)

    segments, omitted = parser.build()
    if not segments:
        code = (
            DiagnosticCode.UNSUPPORTED_FORMAT
            if parser.foreign_seen
            else DiagnosticCode.NO_TRACEBACK_FOUND
        )
        parser.diag.add(code)
        status = ParseStatus.FAILED
    else:
        if parser.foreign_seen:
            parser.diag.add(DiagnosticCode.UNSUPPORTED_FORMAT, severity=DiagnosticSeverity.WARNING)
        worst = parser.diag.worst
        status = (
            ParseStatus.COMPLETE
            if worst is None or worst is DiagnosticSeverity.INFO
            else ParseStatus.PARTIAL
        )
    result = ParseResult(
        status=status,
        segments=segments,
        segments_omitted=omitted,
        diagnostics=tuple(parser.diag.items),
        diagnostics_truncated=parser.diag.truncated,
        input_chars=len(text),
        input_lines=line_count,
        input_digest=digest,
    )
    logger.debug(
        "traceback parsed: status=%s segments=%d frames=%d diagnostics=%d chars=%d digest=%s",
        result.status.value,
        len(result.segments),
        sum(len(s.frames) for s in result.segments),
        len(result.diagnostics),
        result.input_chars,
        result.input_digest,
    )
    return result


def looks_like_traceback(text: str, *, limits: DebugLimits = DebugLimits()) -> bool:  # noqa: B008
    """Cheap hint: does ``text`` contain a traceback header or a ``File "...", line N`` line?

    Only the first ``max_input_chars`` characters and ``max_lines`` lines are examined. A ``True``
    answer is a hint for routing (for example telling a user to use debug mode); it is not a
    validation. Use :func:`parse_traceback` for that.
    """
    if not isinstance(text, str) or not text:
        return False
    window = text[: limits.max_input_chars]
    if TRACEBACK_HEADER not in window and _FRAME_PREFIX not in window:
        return False
    for raw in window.split("\n", limits.max_lines)[: limits.max_lines]:
        stripped = raw.strip()
        if stripped == TRACEBACK_HEADER or _frame_shape(stripped):
            return True
    return False
