"""Command line for the Milestone 10 chunking experiment.

``match``   choose the size-matched Arms C and D from chunk statistics only (no retrieval).
``freeze``  validate the design without retrieval and write ``frozen-config.json``.
``build``   build the four experimental indexes (offline; separate directories).
``run``     fixed-k retrieval, fixed-budget simulation, latency and paired statistics (offline).
``report``  tables, plots and the human-readable report from the saved results.

``build`` and ``run`` force the Hugging Face libraries offline: they never download anything and
fail with a clear message if the embedding model is not in the local cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from copilot.chunking.base import chunk_repository
from copilot.config.settings import get_settings
from copilot.evaluation.chunking_experiment import builds, design, execute, matching, report
from copilot.evaluation.chunking_experiment.artifacts import ArtifactError, write_text_atomic
from copilot.evaluation.chunking_experiment.config import (
    ExperimentConfigError,
    canonical_json,
    file_sha256,
    ingest_snapshot,
    production_arm,
)
from copilot.evaluation.chunking_experiment.embedding_cache import TimeBudgetExceeded
from copilot.evaluation.chunking_experiment.profile import SizeProfile, profile_by_scope

DEFAULT_TREE = "data/snapshots/a282ec471ebf1387cb4feafcfe732ad24a431c7a/tree"
DEFAULT_REPOSITORY = "agentic-rag-code-copilot"
DEFAULT_IGNORES = ("data", "tests", "submission")


def _row(label: str, p: SizeProfile) -> str:
    return (
        f"{label:<22}{p.chunk_count:>7}{p.tokens_min:>6}{p.tokens_p25:>7.0f}{p.tokens_median:>8.1f}"
        f"{p.tokens_mean:>8.1f}{p.tokens_p75:>7.0f}{p.tokens_max:>6}{p.redundancy:>8.3f}"
        f"{p.non_blank_coverage:>8.3f}"
    )


_HEADER = (
    f"{'':<22}{'chunks':>7}{'min':>6}{'p25':>7}{'median':>8}{'mean':>8}{'p75':>7}{'max':>6}"
    f"{'redund':>8}{'nb-cov':>8}"
)


def _cmd_match(args: argparse.Namespace) -> int:
    settings = get_settings()
    ingestion = ingest_snapshot(
        args.tree,
        settings,
        repository_name=args.repository_name,
        ignore_directories=tuple(args.ignore_dir),
    )
    arm_a = production_arm("A", "baseline", "line", settings)
    arm_b = production_arm("B", "ast", "ast", settings)
    profiles = {}
    for arm in (arm_a, arm_b):
        chunks = chunk_repository(ingestion, arm.chunker()).chunks
        profiles[arm.arm_id] = profile_by_scope(chunks, ingestion.files)
    base = {
        "base_size": settings.chunk_size_lines,
        "base_overlap": settings.chunk_overlap_lines,
        "max_tokens": settings.chunk_max_tokens,
    }
    match_c = matching.match_arm_c(ingestion, profiles["B"]["python"], **base)
    match_d = matching.match_arm_d(ingestion, profiles["A"]["python"], **base)

    print(
        f"files accepted: {len(ingestion.files)}  python files: "
        f"{sum(f.language == 'python' for f in ingestion.files)}"
    )
    print("\nPython chunks (estimated tokens)")
    print(_HEADER)
    print(_row("A baseline", profiles["A"]["python"]))
    print(_row("B ast", profiles["B"]["python"]))
    for result in (match_c, match_d):
        print(
            f"\nArm {result.arm} -> target Arm {result.target_arm}: "
            f"{result.parameter_name}={result.selected.parameter} "
            f"median error {result.selected.median_error:+.1%}, mean error "
            f"{result.selected.mean_error:+.1%}, adequate={result.adequate} "
            f"(tolerance {result.tolerance:.0%})"
        )
        print(_HEADER)
        print(_row(f"{result.arm} selected", result.selected.python))
    payload = {
        "schema": "chunking-experiment-matching/1",
        "tree": args.tree,
        "files_accepted": len(ingestion.files),
        "arm_a": arm_a.to_json(),
        "arm_b": arm_b.to_json(),
        "profiles": {
            arm: {scope: p.to_json() for scope, p in scopes.items()}
            for arm, scopes in profiles.items()
        },
        "arm_c": match_c.to_json(),
        "arm_d": match_d.to_json(),
    }
    try:
        out = write_text_atomic(args.out, canonical_json(payload), overwrite=args.overwrite)
    except ArtifactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"\nwrote {out}")
    return 0


def _cmd_freeze(args: argparse.Namespace) -> int:
    settings = get_settings()
    matching_payload = json.loads(Path(args.matching).read_text(encoding="utf-8"))
    arms = design.arms_from_matching(matching_payload, settings)
    snapshot_meta = json.loads(
        (Path(args.tree).parent / "snapshot.json").read_text(encoding="utf-8")
    )
    ignores = tuple(args.ignore_dir)
    try:
        check = design.design_check(
            arms=arms,
            settings=settings,
            tree=args.tree,
            repository_name=args.repository_name,
            ignore_directories=ignores,
            benchmark_path=args.benchmark,
            reference_index=args.reference_index,
        )
    except ExperimentConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    config = design.frozen_config(
        arms=arms,
        settings=settings,
        tree=args.tree,
        snapshot_commit=snapshot_meta["commit"],
        repository_name=args.repository_name,
        ignore_directories=ignores,
        benchmark_path=args.benchmark,
        matching_path=args.matching,
        matching=matching_payload,
        check=check,
    )
    out_dir = Path(args.out_dir)
    try:
        write_text_atomic(
            out_dir / "frozen-config.json", canonical_json(config), overwrite=args.overwrite
        )
        write_text_atomic(
            out_dir / "design-check.json", canonical_json(check), overwrite=args.overwrite
        )
    except ArtifactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"questions={check['questions']} regions={check['regions']} validated against snapshot")
    for arm_id, info in check["arms"].items():
        py = info["profiles"]["python"]
        print(
            f"arm {arm_id}: chunks={info['chunk_count']} python median={py['tokens_median']} "
            f"mean={py['tokens_mean']:.1f} redundancy={py['redundancy']:.3f}"
        )
    print(f"wrote {out_dir}/frozen-config.json and design-check.json")
    return 0


def _offline() -> None:
    """Never touch the network (model hub libraries honour these variables)."""
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"


def _embedder(settings: object) -> object:
    from copilot.embeddings.factory import create_embedder

    return create_embedder(settings)  # type: ignore[arg-type]


def _load_design(args: argparse.Namespace) -> dict[str, object]:
    design_dir = Path(args.design_dir)
    return execute.verify_design(
        design_dir / "frozen-config.json", design_dir / "frozen-config.sha256"
    )


def _cmd_build(args: argparse.Namespace) -> int:
    _offline()
    settings = get_settings()
    try:
        config = _load_design(args)
        _, arms, ignores = execute.prepare(config, settings)
        only = args.only.split(",") if args.only else None
        embedder = _embedder(settings)
        records = builds.build_all(
            arms,
            tree=config["snapshot"]["tree"],
            repository_name=config["repository_name"],
            ignore_directories=ignores,
            settings=settings,
            embedder=embedder,
            index_root=Path(args.index_root),
            results_dir=Path(args.results_dir),
            protected=[Path("data/indexes"), Path("data/snapshots")],
            only=only,
            overwrite=args.overwrite,
            progress=lambda m: print(m, flush=True),
            cache_root=Path(args.embedding_cache_root),
            time_budget_seconds=args.time_budget,
        )
    except TimeBudgetExceeded:
        return 3  # paused, not failed: run the same command again
    except (ExperimentConfigError, ArtifactError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"built {len(records)} index(es)")
    return 0


def _read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"cannot read {path}: run the earlier phase first") from exc


def _cmd_run(args: argparse.Namespace) -> int:
    _offline()
    settings = get_settings()
    results_dir = Path(args.results_dir)
    partial = results_dir / "partial"
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    say = lambda m: print(m, flush=True)  # noqa: E731
    try:
        config = _load_design(args)
        repetitions = int(config["latency"]["repetitions"])
        final_files = [results_dir / name for name in execute.RESULT_FILES]
        if args.phase in ("all", "finalize") and not args.overwrite:
            for target in final_files:  # fail before any work if a result would be replaced
                if target.exists():
                    raise ArtifactError(f"{target} exists; pass --overwrite")
        benchmark, arms, ignores = execute.prepare(config, settings)
        records = builds.load_build_records(results_dir, [a.arm_id for a in arms])

        def retrievers() -> dict[str, object]:
            return execute.open_retrievers(
                config,
                records,
                index_root=Path(args.index_root),
                embedder=_embedder(settings),
                settings=settings,
                ignores=ignores,
            )

        if args.phase == "retrieval":
            target = partial / "retrieval.json"
            data = execute.retrieval_phase(config, benchmark, retrievers(), progress=say)
            write_text_atomic(target, canonical_json(data), overwrite=args.overwrite)
            say(f"wrote {target}")
            return 0
        if args.phase == "latency":
            per_question = _read_json(partial / "retrieval.json")
            if not 0 <= args.repetition < repetitions:
                raise ArtifactError(f"--repetition must be in 0..{repetitions - 1}")
            part = execute.latency_repetition(
                retrievers(),
                benchmark,
                execute.top_chunk_ids(per_question),
                repetition=args.repetition,
                depth=config["latency"]["depth"],
                warmup=config["latency"]["warmup_queries"],
            )
            target = partial / f"latency-{args.repetition}.json"
            write_text_atomic(target, canonical_json(part), overwrite=args.overwrite)
            say(f"wrote {target}")
            return 0
        if args.phase == "finalize":
            per_question = _read_json(partial / "retrieval.json")
            parts = [_read_json(partial / f"latency-{r}.json") for r in range(repetitions)]
            results = execute.assemble_results(config, records, per_question, parts)
        else:
            results = execute.run_experiment(config, records, benchmark, retrievers(), progress=say)
        finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        execute.write_results(
            results_dir,
            results,
            config_sha256=file_sha256(Path(args.design_dir) / "frozen-config.json"),
            started_utc=started,
            finished_utc=finished,
            overwrite=args.overwrite,
        )
    except (ExperimentConfigError, ArtifactError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for arm, agg in results["aggregates"]["arms"].items():
        hits = {k: v["count"] for k, v in agg["binary"].items() if k.startswith("hit@")}
        print(f"arm {arm}: {hits} mrr@10={agg['continuous']['mrr@10']:.3f}")
    print(f"wrote results to {results_dir}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    try:
        written = report.generate_report(
            Path(args.results_dir),
            Path(args.design_dir),
            Path(args.out_dir),
            overwrite=args.overwrite,
        )
    except (ExperimentConfigError, ArtifactError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {len(written)} files below {args.out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m copilot.evaluation.chunking_experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("match", help="size-match Arms C and D from chunk statistics only")
    m.add_argument("--tree", default=DEFAULT_TREE)
    m.add_argument("--repository-name", default=DEFAULT_REPOSITORY)
    m.add_argument("--ignore-dir", action="append", default=list(DEFAULT_IGNORES))
    m.add_argument(
        "--out", default="experiments/chunking-comparison/design/matching.json", type=Path
    )
    m.add_argument("--overwrite", action="store_true")
    m.set_defaults(func=_cmd_match)
    f = sub.add_parser("freeze", help="validate the design and write frozen-config.json")
    f.add_argument("--tree", default=DEFAULT_TREE)
    f.add_argument("--repository-name", default=DEFAULT_REPOSITORY)
    f.add_argument("--ignore-dir", action="append", default=list(DEFAULT_IGNORES))
    f.add_argument("--benchmark", default="benchmarks/copilot_self_055a8d5.jsonl")
    f.add_argument("--reference-index", default="data/indexes/4aaeab9c650ab76c")
    f.add_argument("--matching", default="experiments/chunking-comparison/design/matching.json")
    f.add_argument("--out-dir", default="experiments/chunking-comparison/design", type=Path)
    f.add_argument("--overwrite", action="store_true")
    f.set_defaults(func=_cmd_freeze)
    for name, helptext, handler in (
        ("build", "build the four indexes (offline)", _cmd_build),
        ("run", "run the retrieval experiment (offline)", _cmd_run),
    ):
        c = sub.add_parser(name, help=helptext)
        c.add_argument("--design-dir", default="experiments/chunking-comparison/design")
        c.add_argument("--results-dir", default="experiments/chunking-comparison/results")
        c.add_argument("--index-root", default="data/experiments/chunking-comparison/indexes")
        c.add_argument("--overwrite", action="store_true")
        if name == "build":
            c.add_argument("--only", help="comma-separated arm ids, e.g. A,B")
            c.add_argument(
                "--time-budget",
                type=float,
                default=None,
                help="pause embedding after this many seconds (exit code 3); rerun to continue",
            )
            c.add_argument(
                "--embedding-cache-root",
                default="data/experiments/chunking-comparison/embedding-cache",
            )
        else:
            c.add_argument(
                "--phase",
                choices=("all", "retrieval", "latency", "finalize"),
                default="all",
                help="'all' runs everything in one process; the others split it into short steps",
            )
            c.add_argument(
                "--repetition", type=int, default=0, help="latency repetition (phase latency)"
            )
        c.set_defaults(func=handler)
    r = sub.add_parser("report", help="tables, plots and the generated report from saved results")
    r.add_argument("--design-dir", default="experiments/chunking-comparison/design")
    r.add_argument("--results-dir", default="experiments/chunking-comparison/results")
    r.add_argument("--out-dir", default="experiments/chunking-comparison/report", type=Path)
    r.add_argument("--overwrite", action="store_true")
    r.set_defaults(func=_cmd_report)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
