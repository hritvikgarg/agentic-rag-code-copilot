"""Parsers for untrusted developer input (currently: Python tracebacks, Milestone 12).

The Python AST logic used for chunking lives in ``copilot.chunking.ast_chunker``; this package
holds deterministic text parsers whose output is plain typed data.
"""

from copilot.parsing.models import (
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
from copilot.parsing.traceback import looks_like_traceback, parse_traceback

__all__ = [
    "ChainLink",
    "DebugLimits",
    "DiagnosticCode",
    "DiagnosticSeverity",
    "ExceptionInfo",
    "ParseDiagnostic",
    "ParseResult",
    "ParseStatus",
    "PathStyle",
    "StackFrame",
    "SyntaxDetail",
    "TracebackSegment",
    "looks_like_traceback",
    "parse_traceback",
]
