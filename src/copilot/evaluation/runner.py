"""Run a benchmark against a ``Retriever`` and across a configuration matrix.

``evaluate_retriever`` scores one index. ``run_matrix`` builds one index per
``(chunk cap, embedding text style)`` pair, holding **everything else fixed** (repository tree,
embedding model, line size, overlap, index type, top-k depth, questions, matching rule), and scores
each. Nothing here decides a "winner": it reports measurements and the chunk-size confound.
"""

from __future__ import annotations

import logging
import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from copilot.config.settings import Settings
from copilot.embeddings.base import Embedder
from copilot.evaluation.benchmark import Benchmark, BenchmarkFormatError, verify_regions
from copilot.evaluation.matching import first_hit_rank
from copilot.evaluation.metrics import DEFAULT_KS, QuestionOutcome, Summary, summarize
from copilot.ingestion import IngestionPolicy, ingest_repository
from copilot.retrieval import Retriever
from copilot.vectorstore import build_repository_index
from copilot.vectorstore.faiss_backend import FAISS_VERSION

logger = logging.getLogger(__name__)

DEFAULT_CAPS = (512, 768, 1024)
DEFAULT_STYLES = ("prefixed", "raw")


def verify_benchmark(
    benchmark: Benchmark,
    repo: str | Path,
    *,
    settings: Settings,
    ignore_directories: Sequence[str] = (),
    repository_name: str | None = None,
) -> None:
    """Raise ``BenchmarkFormatError`` unless every region matches the repository's source text."""
    policy = IngestionPolicy.from_settings(settings).with_extra_ignored_directories(
        *ignore_directories
    )
    ingestion = ingest_repository(repo, policy, repository_name=repository_name)
    problems = verify_regions(benchmark.questions, ingestion.files)
    if problems:
        shown = "; ".join(str(p) for p in problems[:5])
        more = f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""
        raise BenchmarkFormatError(
            f"the benchmark does not match this repository ({len(problems)} problem(s)): "
            f"{shown}{more}. Evaluate against the commit the benchmark was written for."
        )


def evaluate_retriever(
    retriever: Retriever,
    benchmark: Benchmark,
    *,
    depth: int = max(DEFAULT_KS),
    ks: Sequence[int] = DEFAULT_KS,
) -> tuple[list[QuestionOutcome], Summary]:
    """Retrieve ``depth`` results per question and score them against the ground truth."""
    outcomes: list[QuestionOutcome] = []
    for question in benchmark.questions:
        results = retriever.retrieve(question.question, depth)
        rank = first_hit_rank(results, question.relevant)
        outcomes.append(
            QuestionOutcome(
                question_id=question.id,
                first_hit_rank=rank,
                first_hit_lines=results[rank - 1].line_count if rank is not None else None,
                retrieved_locations=tuple(r.location for r in results),
                retrieved_lines=tuple(r.line_count for r in results),
            )
        )
    return outcomes, summarize(outcomes, ks, depth=depth)


@dataclass(frozen=True)
class ConfigResult:
    """Measurements of one ``(cap, style)`` configuration."""

    chunk_cap: int
    text_style: str
    index_id: str
    chunks: int
    index_bytes: int  # index.faiss + chunks.jsonl + manifest.json
    mean_chunk_lines: float  # size profile of the whole index, not only of retrieved chunks
    seconds_embed: float
    seconds_build_total: float
    summary: Summary
    outcomes: tuple[QuestionOutcome, ...] = field(repr=False)


@dataclass(frozen=True)
class MatrixResult:
    """Every configuration plus the fixed conditions they share."""

    configs: tuple[ConfigResult, ...]
    conditions: dict[str, object]


def run_configuration(
    repo: str | Path,
    benchmark: Benchmark,
    *,
    embedder: Embedder,
    settings: Settings,
    chunk_cap: int,
    text_style: str,
    indexes_dir: str | Path,
    ignore_directories: Sequence[str] = (),
    repository_name: str | None = None,
    depth: int = max(DEFAULT_KS),
    ks: Sequence[int] = DEFAULT_KS,
) -> ConfigResult:
    """Build an index with ``chunk_cap``/``text_style``, then evaluate it."""
    configured = settings.model_copy(update={"chunk_max_tokens": chunk_cap})
    started = time.perf_counter()
    report = build_repository_index(
        repo,
        embedder=embedder,
        settings=configured,
        indexes_dir=indexes_dir,
        repository_name=repository_name,
        ignore_directories=ignore_directories,
        text_style=text_style,  # type: ignore[arg-type]
        overwrite=True,
    )
    build_seconds = time.perf_counter() - started
    retriever = Retriever.open(
        report.index_path,
        repo,
        embedder=embedder,
        settings=configured,
        ignore_directories=ignore_directories,
    )
    outcomes, summary = evaluate_retriever(retriever, benchmark, depth=depth, ks=ks)
    records = retriever.index.records
    return ConfigResult(
        chunk_cap=chunk_cap,
        text_style=text_style,
        index_id=report.index_id,
        chunks=report.chunks,
        index_bytes=sum(report.artifact_sizes.values()),
        mean_chunk_lines=sum(r.end_line - r.start_line + 1 for r in records) / len(records),
        seconds_embed=report.seconds_embed,
        seconds_build_total=build_seconds,
        summary=summary,
        outcomes=tuple(outcomes),
    )


def run_matrix(
    repo: str | Path,
    benchmark: Benchmark,
    *,
    embedder: Embedder,
    settings: Settings,
    indexes_dir: str | Path,
    caps: Sequence[int] = DEFAULT_CAPS,
    styles: Sequence[str] = DEFAULT_STYLES,
    ignore_directories: Sequence[str] = (),
    repository_name: str | None = None,
    depth: int = max(DEFAULT_KS),
    ks: Sequence[int] = DEFAULT_KS,
    progress: Callable[[str], None] | None = None,
) -> MatrixResult:
    """Evaluate every ``(cap, style)`` pair. The benchmark is verified against ``repo`` first."""
    verify_benchmark(
        benchmark,
        repo,
        settings=settings,
        ignore_directories=ignore_directories,
        repository_name=repository_name,
    )
    results: list[ConfigResult] = []
    for cap in caps:
        for style in styles:
            if progress:
                progress(f"cap={cap} style={style}: building index and evaluating ...")
            results.append(
                run_configuration(
                    repo,
                    benchmark,
                    embedder=embedder,
                    settings=settings,
                    chunk_cap=cap,
                    text_style=style,
                    indexes_dir=indexes_dir,
                    ignore_directories=ignore_directories,
                    repository_name=repository_name,
                    depth=depth,
                    ks=ks,
                )
            )
    info = embedder.info
    meta = benchmark.meta
    conditions: dict[str, object] = {
        "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark_name": meta.name if meta else None,
        "benchmark_commit": meta.commit if meta else None,
        "benchmark_independent": meta.independent if meta else None,
        "questions": len(benchmark.questions),
        "embedding_model": info.model_id,
        "embedding_dimension": info.dimension,
        "embedding_runtime": f"{info.runtime} {info.runtime_version or ''}".strip(),
        "index_type": "IndexFlatIP",
        "faiss_version": FAISS_VERSION,
        "chunking_strategy": settings.chunking_strategy,
        "chunk_size_lines": settings.chunk_size_lines,
        "chunk_overlap_lines": settings.chunk_overlap_lines,
        "caps": list(caps),
        "styles": list(styles),
        "retrieval_depth": depth,
        "ignore_directories": list(ignore_directories),
        "python": platform.python_version(),
        "platform": platform.system(),
    }
    return MatrixResult(configs=tuple(results), conditions=conditions)


def _fmt_pct(x: float) -> str:
    return f"{100 * x:5.1f}"


def format_table(result: MatrixResult) -> str:
    """A fixed-width text table of the matrix (no ranking, no "best" marker)."""
    ks = sorted(result.configs[0].summary.hit_at)
    header = (
        ["cap", "style", "chunks", "idx KB", "embed s", "chunk ln"]
        + [f"Hit@{k}" for k in ks]
        + ["MRR", "ln/hit", "ln/ret"]
    )
    rows = []
    for c in result.configs:
        s = c.summary
        rows.append(
            [
                str(c.chunk_cap),
                c.text_style,
                str(c.chunks),
                f"{c.index_bytes / 1024:.0f}",
                f"{c.seconds_embed:.1f}",
                f"{c.mean_chunk_lines:.1f}",
                *[_fmt_pct(s.hit_at[k]) for k in ks],
                f"{s.mrr:.3f}",
                "-" if s.mean_lines_per_hit is None else f"{s.mean_lines_per_hit:.1f}",
                "-" if s.mean_retrieved_lines is None else f"{s.mean_retrieved_lines:.1f}",
            ]
        )
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]

    def line(cells: list[str]) -> str:
        return "  ".join(cell.rjust(widths[i]) for i, cell in enumerate(cells))

    n = result.configs[0].summary.n_questions
    return "\n".join(
        [
            f"{n} questions; Hit@k in %; ln/hit = mean lines of the first matching chunk; "
            "ln/ret = mean lines of all retrieved chunks",
            line(header),
            *(line(r) for r in rows),
        ]
    )


def to_jsonable(result: MatrixResult) -> dict[str, object]:
    """A JSON-serialisable record of the matrix, including per-question ranks."""
    return {
        "conditions": result.conditions,
        "configs": [
            {
                "chunk_cap": c.chunk_cap,
                "text_style": c.text_style,
                "index_id": c.index_id,
                "chunks": c.chunks,
                "index_bytes": c.index_bytes,
                "mean_chunk_lines": c.mean_chunk_lines,
                "seconds_embed": c.seconds_embed,
                "seconds_build_total": c.seconds_build_total,
                "summary": {
                    "n_questions": c.summary.n_questions,
                    "depth": c.summary.depth,
                    "hit_at": {str(k): v for k, v in c.summary.hit_at.items()},
                    "hit_counts": {str(k): v for k, v in c.summary.hit_counts.items()},
                    "mrr": c.summary.mrr,
                    "mean_lines_per_hit": c.summary.mean_lines_per_hit,
                    "mean_retrieved_lines": c.summary.mean_retrieved_lines,
                },
                "questions": [
                    {
                        "id": o.question_id,
                        "first_hit_rank": o.first_hit_rank,
                        "first_hit_lines": o.first_hit_lines,
                        "retrieved": list(o.retrieved_locations),
                    }
                    for o in c.outcomes
                ],
            }
            for c in result.configs
        ],
    }
