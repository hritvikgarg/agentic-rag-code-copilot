"""Command line for the Milestone 10 chunking experiment.

``match``   choose the size-matched Arms C and D from chunk statistics only (no retrieval).
``freeze``  validate the design without retrieval and write ``frozen-config.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from copilot.chunking.base import chunk_repository
from copilot.config.settings import get_settings
from copilot.evaluation.chunking_experiment import design, matching
from copilot.evaluation.chunking_experiment.artifacts import ArtifactError, write_text_atomic
from copilot.evaluation.chunking_experiment.config import (
    ExperimentConfigError,
    canonical_json,
    ingest_snapshot,
    production_arm,
)
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
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
