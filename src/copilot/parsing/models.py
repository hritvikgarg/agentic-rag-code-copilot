"""Typed output of the traceback parser (Milestone 12, slice S1).

Everything here is plain data. Two rules shape the models:

* **Raw traceback text never leaks.** Every field that holds text copied from the traceback
  (paths, function names, source lines, exception type and message) is excluded from ``repr``, so
  logging or printing a model cannot dump user text. :class:`ParseDiagnostic` carries only a fixed
  code, a severity and positions; its human-readable message comes from a constant table and never
  contains input text.
* **Truncation is never silent.** A field cut at a limit carries an explicit ``*_truncated`` flag
  *and* a diagnostic. In particular a frame whose ``path_truncated`` is true must never be used as
  a candidate for repository mapping (slice S2 enforces this).

Frame and segment order are exactly the input order: segments in the order Python printed them
(for chained exceptions the cause comes first and the final exception last) and frames oldest call
first ("most recent call last"). ``index`` fields keep the original position even when frames or
segments were omitted by a limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class DebugLimits:
    """Hard bounds for parsing untrusted traceback text.

    Input beyond ``max_input_chars`` or ``max_lines`` is rejected outright (nothing is parsed).
    Everything else is truncated *visibly*: a flag on the field and a diagnostic. Later slices may
    add fields (all with defaults); the parser-related ones are fixed here.
    """

    max_input_chars: int = 50_000
    max_lines: int = 3_000
    max_line_chars: int = 2_000
    max_frames_per_segment: int = 200
    keep_head_frames: int = 10  # of max_frames_per_segment, the rest is kept from the tail
    max_segments: int = 20  # the *last* max_segments are kept (the final exception matters most)
    max_path_chars: int = 1_024
    max_function_chars: int = 200
    max_source_line_chars: int = 300
    max_type_name_chars: int = 200
    max_message_chars: int = 500
    max_message_lines: int = 20
    max_diagnostics: int = 50

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"DebugLimits.{name} must be a positive integer")
        if self.keep_head_frames >= self.max_frames_per_segment:
            raise ValueError("DebugLimits.keep_head_frames must be < max_frames_per_segment")


class ParseStatus(StrEnum):
    """How much of the input could be parsed cleanly."""

    COMPLETE = "complete"  # parsed; nothing worse than informational diagnostics
    PARTIAL = "partial"  # parsed, but something was missing, malformed, truncated or ignored
    FAILED = "failed"  # nothing usable (empty, oversized, not a Python traceback)


class DiagnosticSeverity(StrEnum):
    INFO = "info"  # does not make the result partial
    WARNING = "warning"  # makes a successful parse ``partial``
    ERROR = "error"  # the input could not be parsed at all


class DiagnosticCode(StrEnum):
    """Fixed diagnostic codes. The set is closed: new situations add a code, never free text."""

    # whole-input problems (the result is FAILED)
    INPUT_EMPTY = "input_empty"
    INPUT_TOO_LARGE = "input_too_large"
    TOO_MANY_LINES = "too_many_lines"
    NO_TRACEBACK_FOUND = "no_traceback_found"
    UNSUPPORTED_FORMAT = "unsupported_format"  # pytest / IPython / JS / Java style text
    # structure
    MISSING_HEADER = "missing_header"
    MISSING_EXCEPTION_LINE = "missing_exception_line"
    MALFORMED_FRAME = "malformed_frame"
    MALFORMED_EXCEPTION_LINE = "malformed_exception_line"
    UNEXPECTED_LINE = "unexpected_line"
    INVALID_LINE_NUMBER = "invalid_line_number"
    EXCEPTION_GROUP_UNSUPPORTED = "exception_group_unsupported"
    CHAIN_LINK_WITHOUT_SEGMENT = "chain_link_without_segment"
    CHAIN_LINK_DANGLING = "chain_link_dangling"
    MULTIPLE_TRACEBACKS = "multiple_tracebacks"
    # truncation (never silent)
    LINE_TRUNCATED = "line_truncated"
    PATH_TRUNCATED = "path_truncated"
    FUNCTION_TRUNCATED = "function_truncated"
    SOURCE_LINE_TRUNCATED = "source_line_truncated"
    MESSAGE_TRUNCATED = "message_truncated"
    FRAMES_TRUNCATED = "frames_truncated"
    SEGMENTS_TRUNCATED = "segments_truncated"
    DIAGNOSTICS_TRUNCATED = "diagnostics_truncated"
    # sanitising / informational
    CONTROL_CHARACTERS_REPLACED = "control_characters_replaced"
    ANSI_ESCAPES_STRIPPED = "ansi_escapes_stripped"
    PSEUDO_FILE = "pseudo_file"
    RECURSION_ELIDED = "recursion_elided"
    EXTRA_SOURCE_LINES_IGNORED = "extra_source_lines_ignored"
    LEADING_TEXT_IGNORED = "leading_text_ignored"
    TRAILING_TEXT_IGNORED = "trailing_text_ignored"


DIAGNOSTIC_SEVERITY: dict[DiagnosticCode, DiagnosticSeverity] = {
    DiagnosticCode.INPUT_EMPTY: DiagnosticSeverity.ERROR,
    DiagnosticCode.INPUT_TOO_LARGE: DiagnosticSeverity.ERROR,
    DiagnosticCode.TOO_MANY_LINES: DiagnosticSeverity.ERROR,
    DiagnosticCode.NO_TRACEBACK_FOUND: DiagnosticSeverity.ERROR,
    DiagnosticCode.UNSUPPORTED_FORMAT: DiagnosticSeverity.ERROR,
    DiagnosticCode.MISSING_HEADER: DiagnosticSeverity.WARNING,
    DiagnosticCode.MISSING_EXCEPTION_LINE: DiagnosticSeverity.WARNING,
    DiagnosticCode.MALFORMED_FRAME: DiagnosticSeverity.WARNING,
    DiagnosticCode.MALFORMED_EXCEPTION_LINE: DiagnosticSeverity.WARNING,
    DiagnosticCode.UNEXPECTED_LINE: DiagnosticSeverity.WARNING,
    DiagnosticCode.INVALID_LINE_NUMBER: DiagnosticSeverity.WARNING,
    DiagnosticCode.EXCEPTION_GROUP_UNSUPPORTED: DiagnosticSeverity.WARNING,
    DiagnosticCode.CHAIN_LINK_WITHOUT_SEGMENT: DiagnosticSeverity.WARNING,
    DiagnosticCode.CHAIN_LINK_DANGLING: DiagnosticSeverity.WARNING,
    DiagnosticCode.MULTIPLE_TRACEBACKS: DiagnosticSeverity.WARNING,
    DiagnosticCode.LINE_TRUNCATED: DiagnosticSeverity.WARNING,
    DiagnosticCode.PATH_TRUNCATED: DiagnosticSeverity.WARNING,
    DiagnosticCode.FUNCTION_TRUNCATED: DiagnosticSeverity.WARNING,
    DiagnosticCode.SOURCE_LINE_TRUNCATED: DiagnosticSeverity.WARNING,
    DiagnosticCode.MESSAGE_TRUNCATED: DiagnosticSeverity.WARNING,
    DiagnosticCode.FRAMES_TRUNCATED: DiagnosticSeverity.WARNING,
    DiagnosticCode.SEGMENTS_TRUNCATED: DiagnosticSeverity.WARNING,
    DiagnosticCode.DIAGNOSTICS_TRUNCATED: DiagnosticSeverity.WARNING,
    DiagnosticCode.CONTROL_CHARACTERS_REPLACED: DiagnosticSeverity.WARNING,
    DiagnosticCode.ANSI_ESCAPES_STRIPPED: DiagnosticSeverity.INFO,
    DiagnosticCode.PSEUDO_FILE: DiagnosticSeverity.INFO,
    DiagnosticCode.RECURSION_ELIDED: DiagnosticSeverity.INFO,
    DiagnosticCode.EXTRA_SOURCE_LINES_IGNORED: DiagnosticSeverity.INFO,
    DiagnosticCode.LEADING_TEXT_IGNORED: DiagnosticSeverity.INFO,
    DiagnosticCode.TRAILING_TEXT_IGNORED: DiagnosticSeverity.INFO,
}

# Fixed, input-free explanations (safe to show, log and persist).
DIAGNOSTIC_MESSAGES: dict[DiagnosticCode, str] = {
    DiagnosticCode.INPUT_EMPTY: "the input is empty",
    DiagnosticCode.INPUT_TOO_LARGE: "the input exceeds the size limit and was not parsed",
    DiagnosticCode.TOO_MANY_LINES: "the input exceeds the line limit and was not parsed",
    DiagnosticCode.NO_TRACEBACK_FOUND: "no Python traceback was found in the input",
    DiagnosticCode.UNSUPPORTED_FORMAT: "the input looks like a non-Python or unsupported format",
    DiagnosticCode.MISSING_HEADER: "a traceback block has no 'Traceback (most recent call last):'",
    DiagnosticCode.MISSING_EXCEPTION_LINE: "a traceback block ends without an exception line",
    DiagnosticCode.MALFORMED_FRAME: "a stack frame line could not be fully parsed",
    DiagnosticCode.MALFORMED_EXCEPTION_LINE: "a line that should be the exception line is invalid",
    DiagnosticCode.UNEXPECTED_LINE: "a line inside a traceback block was not recognised",
    DiagnosticCode.INVALID_LINE_NUMBER: "a frame has a missing, zero or implausible line number",
    DiagnosticCode.EXCEPTION_GROUP_UNSUPPORTED: "ExceptionGroup tracebacks are not parsed",
    DiagnosticCode.CHAIN_LINK_WITHOUT_SEGMENT: "a chained-exception marker has no preceding block",
    DiagnosticCode.CHAIN_LINK_DANGLING: "a chained-exception marker has no following block",
    DiagnosticCode.MULTIPLE_TRACEBACKS: "several unrelated tracebacks were found in the input",
    DiagnosticCode.LINE_TRUNCATED: "a line exceeded the line-length limit and was cut",
    DiagnosticCode.PATH_TRUNCATED: "a file path is truncated and is not a valid path",
    DiagnosticCode.FUNCTION_TRUNCATED: "a function name was cut at the length limit",
    DiagnosticCode.SOURCE_LINE_TRUNCATED: "a source line was cut at the length limit",
    DiagnosticCode.MESSAGE_TRUNCATED: "an exception message was cut at the length limit",
    DiagnosticCode.FRAMES_TRUNCATED: "frames were omitted because of the frame limit",
    DiagnosticCode.SEGMENTS_TRUNCATED: "segments were omitted because of the segment limit",
    DiagnosticCode.DIAGNOSTICS_TRUNCATED: "further diagnostics were not recorded",
    DiagnosticCode.CONTROL_CHARACTERS_REPLACED: "control characters in the input were replaced",
    DiagnosticCode.ANSI_ESCAPES_STRIPPED: "terminal colour escape sequences were removed",
    DiagnosticCode.PSEUDO_FILE: "a frame refers to a pseudo-file such as '<string>'",
    DiagnosticCode.RECURSION_ELIDED: "repeated frames were elided by Python ('previous line')",
    DiagnosticCode.EXTRA_SOURCE_LINES_IGNORED: "extra source lines of a frame were ignored",
    DiagnosticCode.LEADING_TEXT_IGNORED: "text before the traceback was ignored",
    DiagnosticCode.TRAILING_TEXT_IGNORED: "text after the traceback was ignored",
}


class PathStyle(StrEnum):
    """Purely lexical classification of the path as written in the traceback."""

    POSIX_ABSOLUTE = "posix_absolute"  # /srv/app/main.py
    WINDOWS_ABSOLUTE = "windows_absolute"  # C:\Users\me\app\main.py (or C:/...)
    UNC = "unc"  # \\server\share\main.py
    RELATIVE = "relative"  # app/main.py, ./main.py, ..\main.py
    PSEUDO = "pseudo"  # <string>, <stdin>, <frozen importlib._bootstrap>
    UNKNOWN = "unknown"  # empty, URL-like or otherwise not a plain path


class ChainLink(StrEnum):
    """How a segment relates to the *next* segment in the input."""

    CAUSE = "cause"  # "The above exception was the direct cause of the following exception"
    CONTEXT = "context"  # "During handling of the above exception, another exception occurred"
    NONE = "none"  # no link (the last segment always has NONE)


_FROZEN = ConfigDict(frozen=True, extra="forbid")


class SyntaxDetail(BaseModel):
    """What Python printed for a SyntaxError-family exception (from its last frame)."""

    model_config = _FROZEN

    text: str | None = Field(default=None, repr=False)  # the source text shown (stripped)
    text_truncated: bool = False
    # Position of the first caret inside ``text`` (0-based, as displayed) and the caret run
    # length. It is *not* the column in the original file: Python strips leading blanks.
    caret_offset: int | None = Field(default=None, ge=0)
    caret_length: int | None = Field(default=None, ge=1)


class ExceptionInfo(BaseModel):
    """The final ``Type: message`` line of one traceback block (untrusted text)."""

    model_config = _FROZEN

    type_name: str = Field(repr=False)  # as printed, e.g. "KeyError", "requests.exceptions.Timeout"
    message: str = Field(default="", repr=False)
    message_truncated: bool = False
    is_syntax_error: bool = False  # SyntaxError, IndentationError or TabError
    syntax: SyntaxDetail | None = Field(default=None, repr=False)

    @property
    def short_name(self) -> str:
        """The class name without its module prefix."""
        return self.type_name.rsplit(".", 1)[-1]

    @property
    def module(self) -> str | None:
        """The module prefix as printed (``None`` for builtins such as ``KeyError``)."""
        head, dot, _ = self.type_name.rpartition(".")
        return head if dot else None


class StackFrame(BaseModel):
    """One ``File "...", line N, in func`` entry. Every text field is untrusted user data.

    ``raw_path`` is the path exactly as written (after control-character replacement), cut at
    ``DebugLimits.max_path_chars``. It is **not** a repository path; the resolver (slice S2)
    decides whether it maps to anything. ``path_truncated`` is true when the path may be
    incomplete: cut by a limit, or an unterminated quote. Such a path is incomplete and must never
    be used as a mapping candidate.
    """

    model_config = _FROZEN

    index: int = Field(ge=0)  # original position within the segment (gaps = omitted frames)
    raw_path: str = Field(repr=False)
    path_style: PathStyle
    path_truncated: bool = False
    line: int | None = Field(default=None, ge=1)  # None: missing, zero, implausible or truncated
    function: str | None = Field(default=None, repr=False)
    function_truncated: bool = False
    source_line: str | None = Field(
        default=None, repr=False
    )  # claimed by the traceback, unverified
    source_line_truncated: bool = False
    # Extra repetitions Python elided after this frame ("[Previous line repeated N more times]").
    repeat_count: int = Field(default=0, ge=0)

    @property
    def is_pseudo(self) -> bool:
        return self.path_style is PathStyle.PSEUDO


class TracebackSegment(BaseModel):
    """One ``Traceback (most recent call last):`` block (or a header-less SyntaxError block)."""

    model_config = _FROZEN

    index: int = Field(ge=0)  # original position among the segments of the input
    has_header: bool
    frames: tuple[StackFrame, ...] = ()
    frames_omitted: int = Field(default=0, ge=0)  # frames dropped by the frame limit
    exception: ExceptionInfo | None = None
    link_to_next: ChainLink = ChainLink.NONE


class ParseDiagnostic(BaseModel):
    """One structured observation about the input. Never contains input text."""

    model_config = _FROZEN

    code: DiagnosticCode
    severity: DiagnosticSeverity
    segment_index: int | None = Field(default=None, ge=0)
    frame_index: int | None = Field(default=None, ge=0)
    line_number: int | None = Field(default=None, ge=1)  # 1-based line of the *input*

    @property
    def message(self) -> str:
        """The fixed explanation for ``code`` (constant text, safe to display and log)."""
        return DIAGNOSTIC_MESSAGES[self.code]


class ParseResult(BaseModel):
    """The complete, bounded result of :func:`copilot.parsing.parse_traceback`."""

    model_config = _FROZEN

    status: ParseStatus
    segments: tuple[TracebackSegment, ...] = ()
    segments_omitted: int = Field(default=0, ge=0)
    diagnostics: tuple[ParseDiagnostic, ...] = ()
    diagnostics_truncated: bool = False
    input_chars: int = Field(ge=0)
    input_lines: int = Field(ge=0)
    input_digest: str  # first 12 hex characters of SHA-256 over the input (for traces/logs)

    @property
    def final_segment(self) -> TracebackSegment | None:
        """The last segment: for chained exceptions, the exception that actually propagated."""
        return self.segments[-1] if self.segments else None

    @property
    def diagnostic_codes(self) -> frozenset[DiagnosticCode]:
        return frozenset(d.code for d in self.diagnostics)

    def has_code(self, code: DiagnosticCode) -> bool:
        return code in self.diagnostic_codes
