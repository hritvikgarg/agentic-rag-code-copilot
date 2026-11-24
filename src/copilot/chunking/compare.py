"""Descriptive structural comparison of the baseline and the structure-aware chunkers.

``python -m copilot.chunking.compare PATH`` chunks a repository with both strategies (same
ingestion, same size/overlap/token settings) and prints *structural* statistics. It runs no
embedding model, no retrieval and no LLM.

**These numbers describe chunk shapes. They are not evidence about retrieval accuracy**: whether
structure-aware chunks retrieve better is the Milestone 10 experiment. Every statistic is defined
in :data:`DEFINITIONS` (also printed with ``--definitions`` and written into ``--json`` output).

The "definitions" ground truth comes from parsing each Python file with ``ast`` independently of
either chunker: every ``def`` / ``async def`` / ``class`` that is a direct member of the module or
of a class (nested classes included), i.e. exactly the units the structure-aware chunker targets.
Functions nested inside functions are not counted.
"""

from __future__ import annotations

import argparse
import ast
import json
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from copilot.chunking.ast_chunker import AstChunker, AstFileReport, statement_first_line
from copilot.chunking.base import Chunker, chunk_repository
from copilot.chunking.errors import ChunkingError
from copilot.chunking.registry import create_chunker
from copilot.chunking.windows import split_lines
from copilot.config import Settings, get_settings, setup_logging
from copilot.ingestion.errors import IngestionError
from copilot.ingestion.loader import ingest_repository
from copilot.ingestion.policy import IngestionPolicy
from copilot.models.chunk import Chunk
from copilot.models.ingestion import IngestionResult, SourceFile
from copilot.utils.tokens import estimate_tokens

DEFINITIONS: dict[str, str] = {
    "chunks_all / chunks_python": "chunks of all ingested files / of Python (.py) files only",
    "tokens_*_min|median|max": (
        "min / median / max of the heuristic token estimate (copilot.utils.tokens) per chunk; "
        "an estimate for a safety budget, not a real tokenizer's count"
    ),
    "python_files_unparseable": "Python files that ast.parse rejects (no definitions counted)",
    "definitions_total": (
        "def / async def / class statements that are direct members of a module or of a class "
        "in parseable Python files (functions nested inside functions are not counted)"
    ),
    "definitions_oversized": (
        "definitions whose own lines (decorators included) exceed the token budget; no strategy "
        "can keep these in one chunk"
    ),
    "definitions_kept_whole": (
        "definitions not larger than the budget whose complete line range lies inside ONE "
        "non-fragment chunk (a class chunk that contains a method counts for that method too)"
    ),
    "definitions_cut": (
        "definitions not larger than the budget that no single chunk contains completely"
    ),
    "kept_whole_pct": "definitions_kept_whole / (definitions_total - definitions_oversized)",
    "symbol_chunks_pct": "Python chunks that carry a qualified_name / chunks_python",
    "symbol_lines_pct": (
        "non-blank lines of parseable Python files lying in at least one chunk that carries a "
        "qualified_name / all such lines"
    ),
    "line_coverage_pct": (
        "non-blank lines of ALL ingested files lying in at least one chunk / all such lines "
        "(a fragmented over-long line counts as covered)"
    ),
    "overlap_duplicated_lines": (
        "extra copies of lines created by overlapping chunks (from ChunkStats)"
    ),
    "oversized_definitions_split": (
        "definitions the structure-aware chunker had to cut because they exceed the budget "
        "(0 for the baseline, which does not know definitions)"
    ),
    "statements_line_split": (
        "single statements larger than the budget that the structure-aware chunker handed to the "
        "baseline line/fragment splitter"
    ),
    "fallback_files": (
        "files chunked by the line splitter inside the structure-aware strategy, by reason "
        "(unsupported_language = not Python; others = Python that could not be parsed/segmented)"
    ),
}


@dataclass(frozen=True)
class Definition:
    kind: str  # "class" | "function"
    qualified_name: str
    start_line: int
    end_line: int


def python_definitions(content: str) -> list[Definition] | None:
    """Ground-truth definitions of a Python file, or ``None`` if it does not parse."""
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError):
        return None
    found: list[Definition] = []

    def walk(body: Sequence[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                qualified = f"{prefix}{node.name}"
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                if node.end_lineno is not None:
                    found.append(
                        Definition(kind, qualified, statement_first_line(node), node.end_lineno)
                    )
                if isinstance(node, ast.ClassDef):
                    walk(node.body, qualified + ".")

    walk(tree.body, "")
    return found


class BoundaryExample(BaseModel):
    """A definition the structure-aware chunker kept whole and the baseline cut."""

    model_config = ConfigDict(frozen=True)

    file_path: str
    qualified_name: str
    kind: str
    lines: str
    baseline_chunks_touching_it: list[str]
    ast_chunk: str


class StrategyProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: str
    version: int
    params: dict[str, int | str]
    chunks_all: int
    chunks_python: int
    tokens_all_min: int
    tokens_all_median: float
    tokens_all_max: int
    tokens_python_min: int
    tokens_python_median: float
    tokens_python_max: int
    definitions_total: int
    definitions_oversized: int
    definitions_kept_whole: int
    definitions_cut: int
    kept_whole_pct: float
    symbol_chunks_pct: float
    symbol_lines_pct: float
    line_coverage_pct: float
    overlap_duplicated_lines: int
    oversized_definitions_split: int
    statements_line_split: int
    fallback_files: dict[str, int]
    segmentation: dict[str, int]
    by_chunk_type: dict[str, int]


class Comparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    repository: str
    files_total: int
    files_python: int
    python_files_unparseable: int
    max_tokens: int
    baseline: StrategyProfile
    ast: StrategyProfile
    examples: list[BoundaryExample]
    definitions: dict[str, str] = DEFINITIONS
    note: str = (
        "Descriptive chunk-shape statistics only. They say nothing about retrieval accuracy; "
        "that is measured in Milestone 10."
    )


def _pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 1) if whole else 0.0


def _spans(chunks: Sequence[Chunk]) -> list[tuple[int, int, Chunk]]:
    return [(c.start_line, c.end_line, c) for c in chunks if not c.is_fragment]


def _containing(d: Definition, chunks: Sequence[Chunk]) -> Chunk | None:
    for start, end, chunk in _spans(chunks):
        if start <= d.start_line and d.end_line <= end:
            return chunk
    return None


def _nonblank_lines(file: SourceFile) -> set[int]:
    return {n for n, line in enumerate(split_lines(file.content), 1) if line.strip()}


def _covered(chunks: Sequence[Chunk]) -> set[int]:
    return {n for c in chunks for n in range(c.start_line, c.end_line + 1)}


def _profile(
    chunker: Chunker,
    ingestion: IngestionResult,
    by_file: dict[str, list[Chunk]],
    truth: dict[str, list[Definition]],
    max_tokens: int,
    all_chunks: Sequence[Chunk],
    overlap_duplicated: int,
    reports: Sequence[AstFileReport],
) -> StrategyProfile:
    python_chunks = [c for c in all_chunks if c.language == "python"]
    tokens_all = sorted(c.token_estimate for c in all_chunks)
    tokens_py = sorted(c.token_estimate for c in python_chunks)

    total = oversized = whole = 0
    symbol_lines = parseable_lines = 0
    covered_lines = nonblank_total = 0
    for file in ingestion.files:
        chunks = by_file.get(file.relative_path, [])
        nonblank = _nonblank_lines(file)
        nonblank_total += len(nonblank)
        covered_lines += len(nonblank & _covered(chunks))
        if file.relative_path not in truth:
            continue
        parseable_lines += len(nonblank)
        symbol_lines += len(
            nonblank & _covered([c for c in chunks if c.qualified_name is not None])
        )
        lines = split_lines(file.content)
        for d in truth[file.relative_path]:
            total += 1
            if estimate_tokens("\n".join(lines[d.start_line - 1 : d.end_line])) > max_tokens:
                oversized += 1
            elif _containing(d, chunks) is not None:
                whole += 1

    fitting = total - oversized
    fallback = Counter(r.fallback_reason for r in reports if r.fallback_reason)
    return StrategyProfile(
        strategy=chunker.name,
        version=chunker.version,
        params=chunker.params(),
        chunks_all=len(all_chunks),
        chunks_python=len(python_chunks),
        tokens_all_min=tokens_all[0] if tokens_all else 0,
        tokens_all_median=float(statistics.median(tokens_all)) if tokens_all else 0.0,
        tokens_all_max=tokens_all[-1] if tokens_all else 0,
        tokens_python_min=tokens_py[0] if tokens_py else 0,
        tokens_python_median=float(statistics.median(tokens_py)) if tokens_py else 0.0,
        tokens_python_max=tokens_py[-1] if tokens_py else 0,
        definitions_total=total,
        definitions_oversized=oversized,
        definitions_kept_whole=whole,
        definitions_cut=fitting - whole,
        kept_whole_pct=_pct(whole, fitting),
        symbol_chunks_pct=_pct(
            sum(c.qualified_name is not None for c in python_chunks), len(python_chunks)
        ),
        symbol_lines_pct=_pct(symbol_lines, parseable_lines),
        line_coverage_pct=_pct(covered_lines, nonblank_total),
        overlap_duplicated_lines=overlap_duplicated,
        oversized_definitions_split=sum(len(r.oversized_definitions) for r in reports),
        statements_line_split=sum(r.split_statements for r in reports),
        fallback_files=dict(sorted(fallback.items())),
        segmentation=dict(
            sorted(Counter(c.segmentation for c in all_chunks if c.segmentation).items())
        ),
        by_chunk_type=dict(sorted(Counter(c.chunk_type.value for c in all_chunks).items())),
    )


class _Recording:
    """Wraps an ``AstChunker`` so one chunking pass also yields its per-file reports."""

    def __init__(self, inner: AstChunker) -> None:
        self.inner = inner
        self.reports: list[AstFileReport] = []

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def version(self) -> int:
        return self.inner.version

    def params(self) -> dict[str, int | str]:
        return self.inner.params()

    def chunk_file(self, file: SourceFile) -> list[Chunk]:
        chunks, report = self.inner.analyze(file)
        self.reports.append(report)
        return chunks


def compare_chunkers(
    ingestion: IngestionResult,
    baseline: Chunker,
    structural: Chunker,
    *,
    max_tokens: int,
    max_examples: int = 5,
    path_prefixes: Sequence[str] = (),
) -> Comparison:
    """Chunk ``ingestion`` with both strategies and describe the chunk shapes."""
    if path_prefixes:
        kept = tuple(f for f in ingestion.files if f.relative_path.startswith(tuple(path_prefixes)))
        ingestion = ingestion.model_copy(update={"files": kept})

    truth: dict[str, list[Definition]] = {}
    unparseable = 0
    python_files = [f for f in ingestion.files if f.language == "python"]
    for file in python_files:
        defs = python_definitions(file.content)
        if defs is None:
            unparseable += 1
        else:
            truth[file.relative_path] = defs

    results = {}
    reports: dict[str, list[AstFileReport]] = {baseline.name: [], structural.name: []}
    for chunker in (baseline, structural):
        runner: Chunker = chunker
        if isinstance(chunker, AstChunker):
            recording = _Recording(chunker)
            reports[chunker.name] = recording.reports
            runner = recording
        result = chunk_repository(ingestion, runner)
        grouped: dict[str, list[Chunk]] = {}
        for chunk in result.chunks:
            grouped.setdefault(chunk.file_path, []).append(chunk)
        results[chunker.name] = (result, grouped)

    profiles = {
        chunker.name: _profile(
            chunker,
            ingestion,
            results[chunker.name][1],
            truth,
            max_tokens,
            results[chunker.name][0].chunks,
            results[chunker.name][0].stats.overlap_duplicated_lines,
            reports[chunker.name],
        )
        for chunker in (baseline, structural)
    }

    examples: list[BoundaryExample] = []
    base_by_file, ast_by_file = results[baseline.name][1], results[structural.name][1]
    lines_cache = {f.relative_path: split_lines(f.content) for f in python_files}
    for file in python_files:
        for d in truth.get(file.relative_path, []):
            if len(examples) >= max_examples:
                break
            lines = lines_cache[file.relative_path]
            if estimate_tokens("\n".join(lines[d.start_line - 1 : d.end_line])) > max_tokens:
                continue
            owner = _containing(d, ast_by_file.get(file.relative_path, []))
            if owner is None or _containing(d, base_by_file.get(file.relative_path, [])):
                continue
            touching = [
                f"{c.start_line}-{c.end_line}"
                for c in base_by_file.get(file.relative_path, [])
                if c.start_line <= d.end_line and c.end_line >= d.start_line
            ]
            examples.append(
                BoundaryExample(
                    file_path=file.relative_path,
                    qualified_name=d.qualified_name,
                    kind=d.kind,
                    lines=f"{d.start_line}-{d.end_line}",
                    baseline_chunks_touching_it=touching,
                    ast_chunk=f"{owner.start_line}-{owner.end_line} ({owner.chunk_type.value})",
                )
            )

    return Comparison(
        repository=ingestion.repository_name,
        files_total=len(ingestion.files),
        files_python=len(python_files),
        python_files_unparseable=unparseable,
        max_tokens=max_tokens,
        baseline=profiles[baseline.name],
        ast=profiles[structural.name],
        examples=examples,
    )


_ROWS: list[tuple[str, str]] = [
    ("chunks (all files)", "chunks_all"),
    ("chunks (Python files)", "chunks_python"),
    ("tokens/chunk, all: min", "tokens_all_min"),
    ("tokens/chunk, all: median", "tokens_all_median"),
    ("tokens/chunk, all: max", "tokens_all_max"),
    ("tokens/chunk, Python: min", "tokens_python_min"),
    ("tokens/chunk, Python: median", "tokens_python_median"),
    ("tokens/chunk, Python: max", "tokens_python_max"),
    ("definitions (ground truth)", "definitions_total"),
    ("  larger than the budget", "definitions_oversized"),
    ("  kept whole in one chunk", "definitions_kept_whole"),
    ("  cut across chunks", "definitions_cut"),
    ("  kept_whole_pct", "kept_whole_pct"),
    ("symbol_chunks_pct", "symbol_chunks_pct"),
    ("symbol_lines_pct", "symbol_lines_pct"),
    ("line_coverage_pct", "line_coverage_pct"),
    ("overlap_duplicated_lines", "overlap_duplicated_lines"),
    ("oversized definitions split", "oversized_definitions_split"),
    ("statements line-split", "statements_line_split"),
]


def format_comparison(c: Comparison, *, definitions: bool = False) -> str:
    out = [
        f"repository: {c.repository}   files: {c.files_total} "
        f"(Python: {c.files_python}, unparseable: {c.python_files_unparseable})   "
        f"token budget: {c.max_tokens}",
        "",
        f"{'statistic':<32}{'baseline (line)':>18}{'structure-aware (ast)':>24}",
    ]
    for label, key in _ROWS:
        left, right = getattr(c.baseline, key), getattr(c.ast, key)
        fmt = (lambda v: f"{v:.1f}") if isinstance(left, float) else str
        out.append(f"{label:<32}{fmt(left):>18}{fmt(right):>24}")
    fallback = f"{'fallback files by reason':<32}{c.baseline.fallback_files!s:>18}"
    out.append(f"{fallback}   {c.ast.fallback_files}")
    out.append(f"{'segmentation':<32}{c.baseline.segmentation!s:>18}   {c.ast.segmentation}")
    out.append(f"{'chunk types':<32}{c.baseline.by_chunk_type!s:>18}   {c.ast.by_chunk_type}")
    out.append("")
    out.append("definitions kept whole by 'ast' and cut by the baseline (examples):")
    out.extend(
        f"  {e.file_path}:{e.lines}  {e.kind} {e.qualified_name}  "
        f"ast chunk {e.ast_chunk}; baseline chunks touching it: "
        f"{', '.join(e.baseline_chunks_touching_it)}"
        for e in c.examples
    )
    if not c.examples:
        out.append("  (none)")
    out.append("")
    out.append(c.note)
    if definitions:
        out.append("")
        out.extend(f"{name}: {text}" for name, text in c.definitions.items())
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m copilot.chunking.compare",
        description="Descriptive baseline-vs-structure-aware chunk statistics (no model, no LLM).",
    )
    parser.add_argument("path", help="repository directory")
    parser.add_argument("--ignore-dir", action="append", default=[], help="directory name to prune")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="only analyse files whose repository-relative path starts with this prefix",
    )
    parser.add_argument("--examples", type=int, default=5, help="boundary examples to list")
    parser.add_argument(
        "--definitions", action="store_true", help="print every statistic's definition"
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    setup_logging()
    settings: Settings = get_settings()
    try:
        policy = IngestionPolicy.from_settings(settings).with_extra_ignored_directories(
            *args.ignore_dir
        )
        ingestion = ingest_repository(args.path, policy)
        comparison = compare_chunkers(
            ingestion,
            create_chunker("line", settings=settings),
            create_chunker("ast", settings=settings),
            max_tokens=settings.chunk_max_tokens,
            max_examples=max(args.examples, 0),
            path_prefixes=args.include,
        )
    except (IngestionError, ChunkingError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(comparison.model_dump(mode="json"), indent=2))
    else:
        print(format_comparison(comparison, definitions=args.definitions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
