"""Evaluation commands: ``python -m copilot.evaluation {verify,seal,run,matrix,compare,summarize}``.

* ``verify BENCH --repo PATH``: check every ground-truth region against the repository's text.
* ``seal BENCH --repo PATH``: (maintainers) write each region's ``sha256`` into the file after a
  human has checked the regions.
* ``run BENCH --repo PATH --index INDEX_DIR``: score an existing index.
* ``matrix BENCH --repo PATH``: build one index per (chunk cap x text style) and score each.
* ``compare BENCH --repo PATH --index INDEX_DIR --output FILE``: (Milestone 6) ask a subset of
  questions through the plain LLM and through RAG and write the paired answers for manual rating.
  This calls the hosted model (after the secret gate) and needs COPILOT_LLM_MODEL + GEMINI_API_KEY.
  ``--resume`` continues an existing ``--output`` file: sides that already succeeded are reused
  unchanged and only failed sides are retried; see ``docs/evaluation.md`` for the resume/retry
  design and why it exists. Transient provider failures (HTTP 429/503/504) are retried
  automatically under a bounded backoff (``--retry-*`` flags) independently of ``--delay-seconds``
  (which paces *every* request, successful or not). If several hosted requests in a row still end
  in HTTP 429 after their retries (``--max-consecutive-rate-limit-failures``, default 3) the run
  stops with exit code 3 and a saved checkpoint instead of retrying for hours; resume later.
* ``summarize FILE``: aggregate the human ratings entered in a comparison file.
* ``snapshot create --commit REV [--index INDEX_DIR]`` / ``snapshot verify --snapshot TREE --index
  INDEX_DIR``: freeze a commit under ``data/snapshots/`` and check it against an index, so the
  repository being evaluated stays fixed while the application keeps changing (Milestone 7.1;
  see ``copilot.evaluation.snapshot`` and ``docs/evaluation.md``).

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
    DEFAULT_MAX_CONSECUTIVE_RATE_LIMIT_FAILURES,
    ComparisonEntry,
    ComparisonRun,
    format_summary,
    load_comparison,
    run_comparison,
    select_questions,
    side_state,
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
from copilot.evaluation.snapshot import (
    SnapshotError,
    create_snapshot,
    resolve_commit,
    verify_tree_against_index,
)
from copilot.ingestion import IngestionPolicy, ingest_repository
from copilot.ingestion.errors import IngestionError
from copilot.llm import LLMError, create_llm_client
from copilot.llm.retry import (
    DEFAULT_INITIAL_DELAY_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_DELAY_SECONDS,
    DEFAULT_MULTIPLIER,
    RetryPolicy,
)
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


def _side_counts(
    run: ComparisonRun, prior_by_id: dict[str, ComparisonEntry], side: str
) -> tuple[int, int, int]:
    """(successful, reused-from-existing-file, failed) for ``side`` ("plain" or "rag").

    ``failed`` counts only sides whose call was made and ended in an error; sides never attempted
    are counted separately by ``_not_attempted_count``.
    """
    successful = sum(1 for e in run.entries if side_state(getattr(e, side)) == "successful")
    reused = sum(
        1
        for e in run.entries
        if (prior := prior_by_id.get(e.question_id)) is not None
        and side_state(getattr(prior, side)) == "successful"
    )
    failed = sum(1 for e in run.entries if side_state(getattr(e, side)) == "failed")
    return successful, reused, failed


def _not_attempted_count(run: ComparisonRun, side: str) -> int:
    return sum(1 for e in run.entries if side_state(getattr(e, side)) == "not_attempted")


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

    output_path = Path(args.output)
    resume_from = None
    if args.resume:
        if output_path.exists():
            resume_from = load_comparison(output_path)
            print(
                f"--resume: found {len(resume_from.entries)} existing entries in "
                f"{output_path.name}; only failed sides will be re-asked",
                file=sys.stderr,
            )
        else:
            print(
                f"--resume: {output_path.name} does not exist yet; starting a new run",
                file=sys.stderr,
            )

    retry_policy = RetryPolicy(
        initial_delay_seconds=args.retry_initial_delay,
        multiplier=args.retry_multiplier,
        max_delay_seconds=args.retry_max_delay,
        max_attempts=args.retry_max_attempts,
    )

    threshold = args.max_consecutive_rate_limit_failures or None  # 0 disables the breaker
    breaker_text = (
        f"{threshold} consecutive sides end in HTTP 429 after their retries"
        if threshold
        else "never for rate limits (circuit breaker disabled)"
    )
    print(
        f"note: {len(questions)} questions x 2 requests are sent to {llm.provider}/{llm.model} "
        "after the secret scan; retrieved repository chunks are part of the RAG requests. "
        f"transient failures (HTTP 429/503/504) retry up to {retry_policy.max_attempts} attempts "
        f"({retry_policy.initial_delay_seconds:g}s.."
        f"{retry_policy.max_delay_seconds:g}s backoff); the run stops after "
        f"{breaker_text}.",
        file=sys.stderr,
    )
    try:
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
            resume_from=resume_from,
            retry_policy=retry_policy,
            checkpoint_path=output_path,
            max_consecutive_rate_limit_failures=threshold,
        )
    except KeyboardInterrupt:
        print(
            f"\ninterrupted: every side finished so far is saved in {output_path.name} (the "
            "request in flight is lost). Re-run with --resume to continue.",
            file=sys.stderr,
        )
        return 130

    write_comparison(run, output_path)  # final write; checkpoint_path already wrote every side

    prior_by_id = {e.question_id: e for e in (resume_from.entries if resume_from else [])}
    plain_ok, plain_reused, plain_failed = _side_counts(run, prior_by_id, "plain")
    rag_ok, rag_reused, rag_failed = _side_counts(run, prior_by_id, "rag")
    plain_todo = _not_attempted_count(run, "plain")
    rag_todo = _not_attempted_count(run, "rag")
    complete = plain_failed == rag_failed == plain_todo == rag_todo == 0

    print(f"wrote {len(run.entries)} questions to {output_path.name}")
    print(f"total questions: {len(run.entries)}")
    print(
        f"plain: {plain_ok} successful ({plain_reused} reused from the existing file), "
        f"{plain_failed} failed, {plain_todo} not attempted"
    )
    print(
        f"rag:   {rag_ok} successful ({rag_reused} reused from the existing file), "
        f"{rag_failed} failed, {rag_todo} not attempted"
    )
    if run.stop_reason is not None:
        print(f"run STOPPED EARLY (circuit breaker): {run.stop_reason}", file=sys.stderr)
        print("run INCOMPLETE: stopped early; completed answers and ratings are saved")
    elif complete:
        print("run complete: every question has both sides successful")
    else:
        print("run INCOMPLETE: some side(s) failed after retries or were never attempted")
    print(f"output: {output_path}")
    if not complete:
        print(
            "Re-run the same command with --resume to ask only the failed and not-attempted "
            "sides; successful answers and any ratings you entered are kept.",
            file=sys.stderr,
        )
    if run.stop_reason is not None:
        return 3  # distinct from 0 (finished), 2 (error) and 130 (Ctrl+C)
    if not complete:
        return 0
    print(
        "Fill in plain_rating / rag_rating per the rubric in the file, "
        "then run: python -m copilot.evaluation summarize FILE"
    )
    return 0


def _print_check(check, tree: Path) -> None:
    status = "MATCH" if check.matches else "MISMATCH"
    print(
        f"{status}: {tree} has {check.file_count} accepted files; fingerprint "
        f"{check.actual_fingerprint[:16]} vs index {check.index_id} "
        f"({check.expected_fingerprint[:16]})"
    )


def _cmd_snapshot(args: argparse.Namespace, settings: Settings) -> int:
    ignore = list(args.ignore_dir)
    if args.snapshot_command == "verify":
        check = verify_tree_against_index(
            args.snapshot, args.index, ignore_directories=ignore, settings=settings
        )
        _print_check(check, Path(args.snapshot))
        return 0 if check.matches else 2

    commit = resolve_commit(args.repo_root, args.commit)
    tree, check = create_snapshot(
        args.repo_root,
        commit,
        args.dest,
        index_dir=args.index,
        ignore_directories=ignore,
        settings=settings,
    )
    print(f"snapshot of commit {commit[:12]} is at: {tree}")
    if check is not None:
        _print_check(check, tree)
        print(f"use it as: compare ... --repo {tree} --index {args.index}")
    else:
        print(
            "not verified against an index (no --index given); run "
            "'snapshot verify --snapshot TREE --index INDEX_DIR' before relying on it",
            file=sys.stderr,
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
        "--delay-seconds",
        type=float,
        default=0.0,
        help="pause after every request, success or failure (request pacing, not retry backoff)",
    )
    compare.add_argument(
        "--resume",
        action="store_true",
        help="continue an existing --output file: reuse successful sides, retry only failed ones",
    )
    compare.add_argument(
        "--retry-initial-delay",
        type=float,
        default=DEFAULT_INITIAL_DELAY_SECONDS,
        help="backoff wait before the first retry of a transient failure, in seconds "
        f"(default: {DEFAULT_INITIAL_DELAY_SECONDS:g})",
    )
    compare.add_argument(
        "--retry-multiplier",
        type=float,
        default=DEFAULT_MULTIPLIER,
        help=f"backoff growth factor per retry (default: {DEFAULT_MULTIPLIER:g})",
    )
    compare.add_argument(
        "--retry-max-delay",
        type=float,
        default=DEFAULT_MAX_DELAY_SECONDS,
        help=f"backoff wait cap, in seconds (default: {DEFAULT_MAX_DELAY_SECONDS:g})",
    )
    compare.add_argument(
        "--retry-max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="attempts per side before giving up (1 = no retries; "
        f"default: {DEFAULT_MAX_ATTEMPTS})",
    )
    compare.add_argument(
        "--max-consecutive-rate-limit-failures",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_RATE_LIMIT_FAILURES,
        help="circuit breaker: stop after this many hosted sides in a row end in HTTP 429 after "
        "their retries (0 disables; "
        f"default: {DEFAULT_MAX_CONSECUTIVE_RATE_LIMIT_FAILURES})",
    )
    snapshot = sub.add_parser(
        "snapshot", help="freeze a commit as the repository to evaluate; verify it against an index"
    )
    snap_sub = snapshot.add_subparsers(dest="snapshot_command", required=True)
    snap_create = snap_sub.add_parser("create", help="git-archive a commit into data/snapshots/")
    snap_create.add_argument("--commit", required=True, help="commit, branch or tag to freeze")
    snap_create.add_argument("--repo-root", default=".", help="the git checkout (default: .)")
    snap_create.add_argument("--dest", default="data/snapshots", help="snapshots directory")
    snap_create.add_argument(
        "--index", help="verify against this index before installing (recommended)"
    )
    snap_create.add_argument(
        "--ignore-dir", action="append", default=[], help="same set the index was built with"
    )
    snap_verify = snap_sub.add_parser("verify", help="check a snapshot tree against an index")
    snap_verify.add_argument("--snapshot", required=True, help="the snapshot's tree directory")
    snap_verify.add_argument("--index", required=True, help="index directory")
    snap_verify.add_argument(
        "--ignore-dir", action="append", default=[], help="same set the index was built with"
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
        "snapshot": _cmd_snapshot,
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
        SnapshotError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
