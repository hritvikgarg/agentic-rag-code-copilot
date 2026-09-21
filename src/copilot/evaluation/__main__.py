"""Evaluation commands: ``python -m copilot.evaluation {verify,seal,run,matrix}``.

* ``verify BENCH --repo PATH``: check every ground-truth region against the repository's text.
* ``seal BENCH --repo PATH``: (maintainers) write each region's ``sha256`` into the file after a
  human has checked the regions.
* ``run BENCH --repo PATH --index INDEX_DIR``: score an existing index.
* ``matrix BENCH --repo PATH``: build one index per (chunk cap x text style) and score each.
* ``compare BENCH --repo PATH --index INDEX_DIR --output FILE``: (Milestone 6) ask a subset of
  questions through the plain LLM and through RAG and write the paired answers for manual rating.
  This calls the hosted model (after the secret gate) and needs COPILOT_LLM_MODEL + GEMINI_API_KEY.
* ``summarize FILE``: aggregate the human ratings entered in a comparison file.

Output contains repository-relative paths and numbers only. The embedding model is local.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from copilot.chunking.errors import ChunkingError
from copilot.config import Settings, get_settings, setup_logging
from copilot.embeddings.errors import EmbeddingError
from copilot.embeddings.factory import create_embedder
from copilot.evaluation.benchmark import (
    Benchmark,
    BenchmarkFormatError,
    dump_benchmark,
    load_benchmark,
    seal_regions,
)
from copilot.evaluation.comparison import (
    format_summary,
    load_comparison,
    run_comparison,
    select_questions,
    summarize_ratings,
    write_comparison,
)
from copilot.evaluation.metrics import DEFAULT_KS
from copilot.evaluation.runner import (
    DEFAULT_CAPS,
    DEFAULT_STYLES,
    evaluate_retriever,
    format_table,
    run_matrix,
    to_jsonable,
    verify_benchmark,
)
from copilot.ingestion import IngestionPolicy, ingest_repository
from copilot.ingestion.errors import IngestionError
from copilot.llm import LLMError, create_llm_client
from copilot.retrieval import Retriever
from copilot.retrieval.errors import RetrievalError
from copilot.security import SecurityError
from copilot.vectorstore.errors import VectorStoreError


def _ignore_dirs(args: argparse.Namespace, benchmark: Benchmark) -> list[str]:
    if args.ignore_dir:
        return list(args.ignore_dir)
    return list(benchmark.meta.ignore_directories) if benchmark.meta else []


def _repo_name(args: argparse.Namespace, benchmark: Benchmark) -> str | None:
    return args.name or (benchmark.meta.repository_name if benchmark.meta else None)


def _cmd_verify(args: argparse.Namespace, settings: Settings) -> int:
    benchmark = load_benchmark(args.benchmark)
    verify_benchmark(
        benchmark,
        args.repo,
        settings=settings,
        ignore_directories=_ignore_dirs(args, benchmark),
        repository_name=_repo_name(args, benchmark),
    )
    regions = sum(len(q.relevant) for q in benchmark.questions)
    print(f"OK: {len(benchmark.questions)} questions, {regions} regions match the repository text")
    return 0


def _cmd_seal(args: argparse.Namespace, settings: Settings) -> int:
    benchmark = load_benchmark(args.benchmark)
    policy = IngestionPolicy.from_settings(settings).with_extra_ignored_directories(
        *_ignore_dirs(args, benchmark)
    )
    ingestion = ingest_repository(args.repo, policy, repository_name=_repo_name(args, benchmark))
    sealed = seal_regions(benchmark.questions, ingestion.files)
    Path(args.benchmark).write_text(dump_benchmark(sealed), encoding="utf-8", newline="\n")
    print(f"sealed {sum(len(q.relevant) for q in sealed)} regions in {len(sealed)} questions")
    return 0


def _cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    benchmark = load_benchmark(args.benchmark)
    ignore = _ignore_dirs(args, benchmark)
    verify_benchmark(
        benchmark,
        args.repo,
        settings=settings,
        ignore_directories=ignore,
        repository_name=_repo_name(args, benchmark),
    )
    retriever = Retriever.open(
        args.index,
        args.repo,
        embedder=create_embedder(settings),
        settings=settings,
        ignore_directories=ignore,
    )
    outcomes, summary = evaluate_retriever(retriever, benchmark, depth=args.depth)
    print(f"index {retriever.index.index_id}: {summary.n_questions} questions, depth {args.depth}")
    for o in outcomes:
        rank = "miss" if o.first_hit_rank is None else f"rank {o.first_hit_rank}"
        top = o.retrieved_locations[0] if o.retrieved_locations else "-"
        print(f"  {o.question_id:<32} {rank:<8} top1: {top}")
    print()
    for k in sorted(summary.hit_at):
        share = f"{summary.hit_counts[k]}/{summary.n_questions}"
        print(f"Hit@{k}: {100 * summary.hit_at[k]:.1f}%  ({share})")
    print(f"MRR: {summary.mrr:.3f}")
    if summary.mean_lines_per_hit is not None:
        print(f"mean lines per hit: {summary.mean_lines_per_hit:.1f}")
    return 0


def _cmd_matrix(args: argparse.Namespace, settings: Settings) -> int:
    benchmark = load_benchmark(args.benchmark)
    result = run_matrix(
        args.repo,
        benchmark,
        embedder=create_embedder(settings),
        settings=settings,
        indexes_dir=args.work_dir,
        caps=args.caps,
        styles=args.styles,
        ignore_directories=_ignore_dirs(args, benchmark),
        repository_name=_repo_name(args, benchmark),
        depth=args.depth,
        progress=lambda message: print(message, file=sys.stderr),
    )
    print(format_table(result))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(to_jsonable(result), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"\nresults written to {Path(args.output).name}")
    print(
        "\nNo winner is declared: compare Hit@k together with the line columns (bigger chunks "
        "overlap targets more easily) and read docs/evaluation.md before drawing conclusions."
    )
    return 0


def _cmd_compare(args: argparse.Namespace, settings: Settings) -> int:
    benchmark = load_benchmark(args.benchmark)
    ignore = _ignore_dirs(args, benchmark)
    verify_benchmark(
        benchmark,
        args.repo,
        settings=settings,
        ignore_directories=ignore,
        repository_name=_repo_name(args, benchmark),
    )
    questions = select_questions(benchmark, args.question_id, args.limit)
    llm = create_llm_client(settings)
    retriever = Retriever.open(
        args.index,
        args.repo,
        embedder=create_embedder(settings),
        settings=settings,
        ignore_directories=ignore,
    )
    print(
        f"note: {len(questions)} questions x 2 requests are sent to {llm.provider}/{llm.model} "
        "after the secret scan; retrieved repository chunks are part of the RAG requests.",
        file=sys.stderr,
    )
    run = run_comparison(
        questions,
        retriever,
        llm,
        top_k=args.top_k,
        settings=settings,
        benchmark_name=benchmark.meta.name if benchmark.meta else None,
        repository_name=retriever.index.spec.repository_name,
        index_id=retriever.index.index_id,
        delay_seconds=args.delay_seconds,
        progress=lambda message: print(message, file=sys.stderr),
    )
    write_comparison(run, args.output)
    errors = sum(1 for e in run.entries if e.plain.error or e.rag.error)
    print(f"wrote {len(run.entries)} paired answers to {Path(args.output).name}")
    if errors:
        print(f"{errors} question(s) had an error; see the 'error' fields.", file=sys.stderr)
    print(
        "Nothing is rated yet. Fill in plain_rating / rag_rating per the rubric in the file, "
        "then run: python -m copilot.evaluation summarize FILE"
    )
    return 0


def _cmd_summarize(args: argparse.Namespace, settings: Settings) -> int:
    del settings
    print(format_summary(summarize_ratings(load_comparison(args.file))))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m copilot.evaluation", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("benchmark", help="benchmark JSONL file")
        p.add_argument("--repo", required=True, help="the repository tree the benchmark targets")
        p.add_argument(
            "--ignore-dir",
            action="append",
            default=[],
            help="directory to prune (default: the benchmark's meta file)",
        )
        p.add_argument("--name", help="repository name (default: the benchmark's meta file)")

    verify = sub.add_parser("verify", help="check the regions against the repository text")
    common(verify)
    seal = sub.add_parser("seal", help="write region sha256 values (after human verification)")
    common(seal)
    run = sub.add_parser("run", help="score an existing index")
    common(run)
    run.add_argument("--index", required=True, help="index directory (data/indexes/<id>)")
    run.add_argument("--depth", type=int, default=max(DEFAULT_KS), help="results per question")
    matrix = sub.add_parser("matrix", help="build and score every cap x style configuration")
    common(matrix)
    matrix.add_argument("--work-dir", required=True, help="where the matrix indexes are written")
    matrix.add_argument("--output", help="write per-question results as JSON here")
    matrix.add_argument("--caps", type=int, nargs="+", default=list(DEFAULT_CAPS))
    matrix.add_argument(
        "--styles", nargs="+", choices=["prefixed", "raw"], default=list(DEFAULT_STYLES)
    )
    matrix.add_argument("--depth", type=int, default=max(DEFAULT_KS), help="results per question")

    compare = sub.add_parser("compare", help="plain LLM vs RAG on a question subset (hosted LLM)")
    common(compare)
    compare.add_argument("--index", required=True, help="index directory (data/indexes/<id>)")
    compare.add_argument("--output", required=True, help="write the paired answers as JSON here")
    compare.add_argument("--limit", type=int, help="use only the first N questions")
    compare.add_argument(
        "--question-id", action="append", default=[], help="use this question id (repeatable)"
    )
    compare.add_argument("--top-k", type=int, default=None, help="chunks per RAG question")
    compare.add_argument(
        "--delay-seconds", type=float, default=0.0, help="pause after each request (rate limits)"
    )
    summarize = sub.add_parser("summarize", help="aggregate the manual ratings in a compare file")
    summarize.add_argument("file", help="a file written by 'compare'")

    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
    setup_logging()
    settings = get_settings()
    commands = {
        "verify": _cmd_verify,
        "seal": _cmd_seal,
        "run": _cmd_run,
        "matrix": _cmd_matrix,
        "compare": _cmd_compare,
        "summarize": _cmd_summarize,
    }
    try:
        return commands[args.command](args, settings)
    except (
        SecurityError,
        LLMError,
        ValueError,
        BenchmarkFormatError,
        RetrievalError,
        VectorStoreError,
        EmbeddingError,
        IngestionError,
        ChunkingError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
