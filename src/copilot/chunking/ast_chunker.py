"""Structure-aware chunking strategy "ast" (Strategy B), Python only.

The baseline cuts every file into line windows and knows nothing about code. This strategy parses
Python with the standard-library ``ast`` module (parse only: indexed code is never imported or
executed) and cuts along definition boundaries, so a function or class is not split in the middle
unless it is larger than the token budget. Everything it cannot segment (other languages, Markdown,
config files, Python that does not parse) goes through the *same* line splitter the baseline uses,
so a mixed-language repository stays indexable under one strategy identity.

Segmentation policy (the full description and examples are in ``docs/chunking.md``):

* **Tiling.** Structural chunks do not overlap and do not nest. Each source line belongs to
  exactly one *region*; a definition is chunked once, never both as a parent and as its children.
* **Units.** Top-level ``def`` / ``async def`` / ``class`` statements are units. Consecutive
  other top-level statements (imports, docstring, constants, ``if __name__`` blocks, ...) are
  packed together into ``module`` chunks.
* **Gaps.** Comment lines between statements attach *forward* to the next statement, so decorators,
  banner comments and the comment above a function travel with it. Blank lines may be left out
  at chunk edges. Comments after the last statement of the file form a trailing module chunk.
* **Fits the budget -> stays whole.** A unit whose estimated tokens are within ``max_tokens``
  is one chunk, nested definitions included. Only an oversized unit is split: a class at member
  boundaries (the class header, docstring and class-level statements form the first chunk), a
  function at statement boundaries (signature + docstring + leading statements first). Neighbouring
  pieces are packed greedily up to the budget.
* **One statement larger than the budget** (a huge literal, a giant ``if`` block) is cut by the
  baseline ``split_into_windows`` / ``split_long_line``; those pieces keep the enclosing symbol and
  are marked ``segmentation="ast_split"``.
* **Coverage.** Every non-blank line of the file is in at least one chunk; only blank lines may be
  left out. This is verified for every file before its chunks are returned; a violation (a bug)
  makes the file fall back to the line splitter instead of silently losing source.

The token budget uses the heuristic ``estimate_tokens`` (see ``copilot.utils.tokens``): a safety
budget, not an exact count for any real tokenizer.
"""

from __future__ import annotations

import ast
import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

from copilot.chunking import line_chunker
from copilot.chunking.base import SymbolInfo, build_chunk
from copilot.chunking.windows import Window, split_into_windows, split_lines
from copilot.models.chunk import Chunk, ChunkType, Segmentation
from copilot.models.ingestion import SourceFile
from copilot.utils.tokens import estimate_tokens

logger = logging.getLogger(__name__)

STRATEGY_NAME = "ast"
# Bump when segmentation, symbol naming, packing or fallback behaviour changes output for the same
# input and parameters. It is part of every chunk id. (A change of the shared line splitter is
# picked up automatically through the ``fallback`` parameter below; a change of the token
# estimator is not versioned there and must bump this too.)
STRATEGY_VERSION = 1

#: Languages that get AST segmentation; every other language falls back to line windows.
SUPPORTED_LANGUAGES = frozenset({"python"})

# Why a file was chunked by the fallback instead of by AST segmentation. Reason codes only: the raw
# parser message is never stored or logged (it can quote source text).
REASON_UNSUPPORTED_LANGUAGE = "unsupported_language"
REASON_SYNTAX_ERROR = "syntax_error"
REASON_INVALID_SOURCE = "invalid_source"
REASON_RECURSION_LIMIT = "recursion_limit"
REASON_MEMORY = "memory_error"
REASON_NEWLINES = "non_normalised_newlines"
REASON_INTERNAL_CHECK = "internal_check_failed"

_DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass(frozen=True)
class AstFileReport:
    """What the chunker did with one file (diagnostics; never persisted in an index)."""

    segmentation: str  # "ast" (file was structurally segmented) or "fallback"
    fallback_reason: str | None  # a REASON_* code when segmentation == "fallback"
    oversized_definitions: tuple[str, ...] = ()  # "class Foo" / "function Foo.bar" that were split
    split_statements: int = 0  # statements larger than the budget cut by the line splitter


@dataclass(frozen=True)
class _Region:
    """A contiguous, inclusive line range (1-based) holding whole statements."""

    start: int
    end: int
    node: ast.stmt | None = None  # the def/class statement if this region is exactly one


@dataclass(frozen=True)
class _Owner:
    """Who a chunk belongs to: its chunk type and symbol metadata."""

    chunk_type: ChunkType
    symbol: SymbolInfo = field(default_factory=SymbolInfo)


@dataclass(frozen=True)
class _Scope:
    """The scope statements are being walked in: module level, or inside one class."""

    owner: _Owner  # owner of non-definition statements in this scope
    class_qname: str | None = None  # qualified name of the enclosing class (None at module level)


@dataclass(frozen=True)
class _Draft:
    window: Window
    owner: _Owner
    segmentation: Segmentation


class _Unusable(Exception):
    """Internal: the AST result cannot be trusted; carries a REASON_* code."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def statement_first_line(node: ast.stmt) -> int:
    """First line of a statement; for decorated definitions that is the first decorator."""
    decorators = getattr(node, "decorator_list", ())
    return min([node.lineno, *(d.lineno for d in decorators)])


class _Segmenter:
    """Cuts one parsed Python file into drafts (see the module docstring for the policy)."""

    def __init__(
        self, lines: Sequence[str], tree: ast.Module, *, max_tokens: int, size: int, overlap: int
    ) -> None:
        self.lines = lines
        self.tree = tree
        self.max_tokens = max_tokens
        self.size = size
        self.overlap = overlap
        self.n = len(lines)
        self.prefix = [0]
        for line in lines:
            self.prefix.append(self.prefix[-1] + estimate_tokens(line))
        self.drafts: list[_Draft] = []
        self.oversized: list[str] = []
        self.split_statements = 0

    # ---- helpers -------------------------------------------------------------------------

    def _cost(self, start: int, end: int) -> int:
        """Estimated tokens of lines ``start..end`` (additive over lines)."""
        return self.prefix[end] - self.prefix[start - 1]

    def _first_nonblank(self, start: int, stop: int) -> int:
        for number in range(start, stop + 1):
            if self.lines[number - 1].strip():
                return number
        return stop

    def _regions(
        self, stmts: Sequence[ast.stmt], prev_end: int, *, attach_first: bool
    ) -> list[_Region]:
        """Regions of ``stmts`` in order.

        A region starts at the first non-blank line after the previous region (so decorators and
        comments travel with the next statement) and ends at the statement's last line. For the
        first statement of a class/function body (``attach_first=False``) the region starts at the
        statement itself: the lines before it belong to the header region.
        """
        regions: list[_Region] = []
        for i, stmt in enumerate(stmts):
            first, last = statement_first_line(stmt), stmt.end_lineno
            if last is None:
                raise _Unusable(REASON_INTERNAL_CHECK)
            if regions and first <= regions[-1].end:  # shares a line: ``a = 1; b = 2``
                previous = regions[-1]
                regions[-1] = _Region(previous.start, max(previous.end, last))
                continue
            if i == 0 and not attach_first:
                start = first
            else:
                start = self._first_nonblank(prev_end + 1, first)
            regions.append(_Region(start, last, stmt if isinstance(stmt, _DEF_NODES) else None))
            prev_end = last
        return regions

    # ---- emission -------------------------------------------------------------------------

    def _emit(self, start: int, end: int, owner: _Owner) -> None:
        """One chunk of whole lines ``start..end`` (blank edges trimmed; blank-only is dropped)."""
        while start <= end and not self.lines[start - 1].strip():
            start += 1
        while end >= start and not self.lines[end - 1].strip():
            end -= 1
        if start > end:
            return
        content = "\n".join(self.lines[start - 1 : end])
        self.drafts.append(_Draft(Window(start, end, content), owner, "ast"))

    def _emit_split(self, start: int, end: int, owner: _Owner) -> None:
        """A statement larger than the budget: the baseline line/fragment splitter cuts it."""
        self.split_statements += 1
        windows = split_into_windows(
            self.lines[start - 1 : end],
            size=self.size,
            overlap=self.overlap,
            max_tokens=self.max_tokens,
            first_line_number=start,
        )
        self.drafts.extend(_Draft(w, owner, "ast_split") for w in windows)

    def _flush_group(self, group: Sequence[_Region], owner: _Owner) -> None:
        """Pack consecutive regions greedily into chunks within the budget."""
        current: tuple[int, int] | None = None
        for region in group:
            if self._cost(region.start, region.end) > self.max_tokens:
                if current is not None:
                    self._emit(*current, owner)
                    current = None
                self._emit_split(region.start, region.end, owner)
            elif current is None:
                current = (region.start, region.end)
            elif self._cost(current[0], region.end) <= self.max_tokens:
                current = (current[0], region.end)
            else:
                self._emit(*current, owner)
                current = (region.start, region.end)
        if current is not None:
            self._emit(*current, owner)

    # ---- walking --------------------------------------------------------------------------

    def run(self) -> list[_Draft]:
        body = self.tree.body
        regions = self._regions(body, 0, attach_first=True)
        last_end = regions[-1].end if regions else 0
        if last_end < self.n and any(line.strip() for line in self.lines[last_end:]):
            regions.append(_Region(self._first_nonblank(last_end + 1, self.n), self.n))  # tail
        module = _Scope(_Owner(ChunkType.MODULE))
        self._walk(regions, module, lead=None)
        return self.drafts

    def _walk(self, regions: Sequence[_Region], scope: _Scope, lead: _Region | None) -> None:
        """Walk a module/class body: pack plain statements, emit each definition as a unit."""
        group: list[_Region] = [] if lead is None else [lead]
        for region in regions:
            if region.node is None:
                group.append(region)
                continue
            self._flush_group(group, scope.owner)
            group = []
            self._definition(region, region.node, scope)
        self._flush_group(group, scope.owner)

    def _definition(self, region: _Region, node: ast.stmt, scope: _Scope) -> None:
        assert isinstance(node, _DEF_NODES)
        qualified = node.name if scope.class_qname is None else f"{scope.class_qname}.{node.name}"
        symbol = SymbolInfo(node.name, qualified, scope.class_qname)
        if isinstance(node, ast.ClassDef):
            owner = _Owner(ChunkType.CLASS, symbol)
        else:
            owner = _Owner(
                ChunkType.FUNCTION if scope.class_qname is None else ChunkType.METHOD, symbol
            )
        if self._cost(region.start, region.end) <= self.max_tokens:
            self._emit(region.start, region.end, owner)
            return

        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        self.oversized.append(f"{kind} {qualified}")
        header_end = statement_first_line(node.body[0]) - 1
        header = _Region(region.start, header_end) if header_end >= region.start else None
        prev_end = header_end if header is not None else region.start - 1
        members = self._regions(node.body, prev_end, attach_first=False)
        if isinstance(node, ast.ClassDef):
            self._walk(members, _Scope(owner, qualified), lead=header)
        else:  # a function splits at statement boundaries; nested defs are plain statements
            plain = [_Region(m.start, m.end) for m in members]
            self._flush_group(([header] if header is not None else []) + plain, owner)


def _verify(lines: Sequence[str], drafts: Sequence[_Draft], max_tokens: int) -> str | None:
    """Check the guarantees for one file; return the name of a failed check, else ``None``."""
    n = len(lines)
    covered: set[int] = set()
    previous_key: tuple[int, int, int] | None = None
    for draft in drafts:
        w = draft.window
        key = (w.start_line, w.end_line, -1 if w.fragment_index is None else w.fragment_index)
        if previous_key is not None and key <= previous_key:
            return "chunk order or uniqueness"
        previous_key = key
        if not 1 <= w.start_line <= w.end_line <= n or not w.content.strip():
            return "range"
        if estimate_tokens(w.content) > max_tokens:
            return "token budget"
        if w.fragment_index is None:
            if w.content != "\n".join(lines[w.start_line - 1 : w.end_line]):
                return "content equals source lines"
        elif w.start_line != w.end_line or w.content not in lines[w.start_line - 1]:
            return "fragment"
        covered.update(range(w.start_line, w.end_line + 1))
    if any(line.strip() and number not in covered for number, line in enumerate(lines, 1)):
        return "source coverage"
    return None


class AstChunker:
    """Python structure-aware chunker; other files use the baseline line splitter."""

    def __init__(
        self, *, max_tokens: int, fallback_size_lines: int, fallback_overlap_lines: int
    ) -> None:
        if (
            fallback_size_lines < 1
            or not 0 <= fallback_overlap_lines < fallback_size_lines
            or max_tokens < 1
        ):
            raise ValueError(
                "require fallback_size_lines >= 1, "
                "0 <= fallback_overlap_lines < fallback_size_lines, max_tokens >= 1"
            )
        self.max_tokens = max_tokens
        self.fallback_size_lines = fallback_size_lines
        self.fallback_overlap_lines = fallback_overlap_lines

    @property
    def name(self) -> str:
        return STRATEGY_NAME

    @property
    def version(self) -> int:
        return STRATEGY_VERSION

    def params(self) -> dict[str, int | str]:
        return {
            "max_tokens": self.max_tokens,
            "fallback_size_lines": self.fallback_size_lines,
            "fallback_overlap_lines": self.fallback_overlap_lines,
            # The line splitter and its version are an input of this strategy's output.
            "fallback": f"{line_chunker.STRATEGY_NAME}@{line_chunker.STRATEGY_VERSION}",
        }

    def chunk_file(self, file: SourceFile) -> list[Chunk]:
        return self.analyze(file)[0]

    def analyze(self, file: SourceFile) -> tuple[list[Chunk], AstFileReport]:
        """Chunks of ``file`` plus a report of how they were produced."""
        lines = split_lines(file.content)
        if not lines:
            return [], AstFileReport("ast", None)
        if file.language not in SUPPORTED_LANGUAGES:
            return self._fallback(file, lines, REASON_UNSUPPORTED_LANGUAGE)
        try:
            drafts, report = self._segment(file, lines)
        except _Unusable as exc:
            return self._fallback(file, lines, exc.reason)
        chunks = [
            build_chunk(
                file,
                self,
                index=i,
                window=d.window,
                chunk_type=d.owner.chunk_type,
                symbol=d.owner.symbol,
                segmentation=d.segmentation,
            )
            for i, d in enumerate(drafts)
        ]
        return chunks, report

    def _segment(self, file: SourceFile, lines: list[str]) -> tuple[list[_Draft], AstFileReport]:
        if "\r" in file.content:  # the parser would count lines differently from split_lines
            raise _Unusable(REASON_NEWLINES)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # e.g. SyntaxWarning for "\d" in a string
                tree = ast.parse(file.content, filename="<indexed>", mode="exec")
        except SyntaxError:
            raise _Unusable(REASON_SYNTAX_ERROR) from None
        except (ValueError, OverflowError):
            raise _Unusable(REASON_INVALID_SOURCE) from None
        except RecursionError:
            raise _Unusable(REASON_RECURSION_LIMIT) from None
        except MemoryError:
            raise _Unusable(REASON_MEMORY) from None

        segmenter = _Segmenter(
            lines,
            tree,
            max_tokens=self.max_tokens,
            size=self.fallback_size_lines,
            overlap=self.fallback_overlap_lines,
        )
        try:
            drafts = segmenter.run()
        except RecursionError:
            raise _Unusable(REASON_RECURSION_LIMIT) from None
        failed = _verify(lines, drafts, self.max_tokens)
        if failed is not None:
            logger.warning("ast chunker self-check failed (%s) for %s", failed, file.relative_path)
            raise _Unusable(REASON_INTERNAL_CHECK)
        return drafts, AstFileReport(
            "ast", None, tuple(segmenter.oversized), segmenter.split_statements
        )

    def _fallback(
        self, file: SourceFile, lines: list[str], reason: str
    ) -> tuple[list[Chunk], AstFileReport]:
        """Baseline line windows, stamped with this strategy's identity (segmentation=fallback)."""
        if reason != REASON_UNSUPPORTED_LANGUAGE:
            logger.info("ast chunker: line fallback for %s (%s)", file.relative_path, reason)
        windows = split_into_windows(
            lines,
            size=self.fallback_size_lines,
            overlap=self.fallback_overlap_lines,
            max_tokens=self.max_tokens,
        )
        chunks = [
            build_chunk(
                file,
                self,
                index=i,
                window=w,
                chunk_type=ChunkType.LINE_WINDOW,
                segmentation="fallback",
            )
            for i, w in enumerate(windows)
        ]
        return chunks, AstFileReport("fallback", reason)
