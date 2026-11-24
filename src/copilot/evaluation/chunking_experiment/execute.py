"""Run the retrieval experiment against the four built indexes (offline).

Phases: (1) one ranked retrieval pool of ``pool_depth`` chunks per question per arm, scored for the
fixed-k metrics and the fixed-budget simulation; (2) a latency measurement with a rotating arm
order; (3) paired statistics. Results are returned, never written here; ``write_results`` stores
them atomically and refuses to overwrite a completed run.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from copilot.chunking.stats import percentile
from copilot.config.settings import Settings
from copilot.embeddings.base import Embedder
from copilot.evaluation.benchmark import (
    Benchmark,
    BenchmarkFormatError,
    load_benchmark,
    verify_regions,
)
from copilot.evaluation.chunking_experiment import analysis
from copilot.evaluation.chunking_experiment.artifacts import (
    ArtifactError,
    safe_artifact_path,
    write_text_atomic,
)
from copilot.evaluation.chunking_experiment.config import (
    ArmConfig,
    ExperimentConfigError,
    arms_from_config,
    canonical_json,
    file_sha256,
    ingest_snapshot,
    load_frozen_config,
)
from copilot.evaluation.chunking_experiment.environment import environment_info
from copilot.evaluation.chunking_experiment.scoring import QuestionScore, score_question
from copilot.retrieval import Retriever

RESULT_FILES = (
    "per-question.json",
    "aggregates.json",
    "paired-statistics.json",
    "latency.json",
    "run-manifest.json",
)


def verify_design(config_path: Path, sha_path: Path) -> dict[str, Any]:
    """Load the frozen configuration and check it against the recorded hash."""
    try:
        expected = sha_path.read_text(encoding="utf-8").split()[0]
    except (OSError, IndexError) as exc:
        raise ExperimentConfigError(
            f"cannot read the frozen configuration hash {sha_path}; was the design frozen?"
        ) from exc
    return load_frozen_config(config_path, expected_sha256=expected)


def prepare(
    config: Mapping[str, Any], settings: Settings
) -> tuple[Benchmark, list[ArmConfig], tuple[str, ...]]:
    """Validate the benchmark against the frozen snapshot; return it with the arms and ignores."""
    bench_info = config["benchmark"]
    if file_sha256(bench_info["path"]) != bench_info["sha256"]:
        raise ExperimentConfigError("the benchmark file changed after the design was frozen")
    benchmark = load_benchmark(bench_info["path"])
    if len(benchmark.questions) != bench_info["questions"]:
        raise ExperimentConfigError("the benchmark question count changed")
    ignores = tuple(config["ignore_directories"])
    ingestion = ingest_snapshot(
        config["snapshot"]["tree"],
        settings,
        repository_name=config["repository_name"],
        ignore_directories=ignores,
    )
    problems = verify_regions(benchmark.questions, ingestion.files)
    if problems:
        raise BenchmarkFormatError(f"{len(problems)} benchmark region(s) do not match the snapshot")
    return benchmark, arms_from_config(config), ignores


def open_retrievers(
    config: Mapping[str, Any],
    builds: Mapping[str, Mapping[str, Any]],
    *,
    index_root: Path,
    embedder: Embedder,
    settings: Settings,
    ignores: Sequence[str],
) -> dict[str, Retriever]:
    """One ``Retriever`` per arm (each re-ingests the snapshot and verifies its index)."""
    retrievers = {}
    for arm_id, record in builds.items():
        path = safe_artifact_path(index_root, record["index_path"])
        retrievers[arm_id] = Retriever.open(
            path,
            config["snapshot"]["tree"],
            embedder=embedder,
            settings=settings,
            ignore_directories=ignores,
        )
    return retrievers


def retrieval_pass(
    retrievers: Mapping[str, Retriever],
    benchmark: Benchmark,
    *,
    pool_depth: int,
    budgets: Sequence[int],
    progress: Callable[[str], None] = print,
) -> tuple[dict[str, list[QuestionScore]], dict[str, dict[str, list[str]]]]:
    """Score every question under every arm. Also returns each arm's top-10 chunk ids."""
    scores: dict[str, list[QuestionScore]] = {arm: [] for arm in retrievers}
    top_ids: dict[str, dict[str, list[str]]] = {arm: {} for arm in retrievers}
    total = len(benchmark.questions)
    for number, question in enumerate(benchmark.questions, start=1):
        for arm, retriever in retrievers.items():
            pool = retriever.retrieve(question.question, pool_depth)
            scores[arm].append(score_question(pool, question, budgets=budgets))
            top_ids[arm][question.id] = [r.chunk_id for r in pool[:10]]
        if number % 5 == 0 or number == total:
            progress(f"retrieved {number}/{total} questions")
    return scores, top_ids


def latency_repetition(
    retrievers: Mapping[str, Retriever],
    benchmark: Benchmark,
    top_ids: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    repetition: int,
    depth: int,
    warmup: int,
) -> dict[str, Any]:
    """One latency repetition: warm-up, then every question under every arm, arm order rotated.

    The timed region is exactly one ``retrieve`` call (query embedding, exact search, chunk
    materialisation). Warm-up calls are not recorded. The top-``depth`` chunk ids of every timed
    call are compared with the scoring pass: deterministic exact search must reproduce them.
    """
    arms = list(retrievers)
    shift = repetition % len(arms)
    order = arms[shift:] + arms[:shift]
    questions = benchmark.questions
    for question in questions[:warmup]:
        for arm in arms:
            retrievers[arm].retrieve(question.question, depth)
    samples: dict[str, list[float]] = {arm: [] for arm in arms}
    mismatches = 0
    for question in questions:
        for arm in order:
            started = time.perf_counter()
            results = retrievers[arm].retrieve(question.question, depth)
            samples[arm].append(time.perf_counter() - started)
            if [r.chunk_id for r in results] != list(top_ids[arm][question.id]):
                mismatches += 1
    return {
        "repetition": repetition,
        "arm_order": order,
        "warmup_queries": warmup,
        "ranking_mismatches": mismatches,
        "samples_seconds": samples,
    }


def summarize_latency(parts: Sequence[Mapping[str, Any]], depth: int) -> dict[str, Any]:
    """Median / p95 per arm over all repetitions."""
    arms = list(parts[0]["samples_seconds"])
    samples = {arm: [v for part in parts for v in part["samples_seconds"][arm]] for arm in arms}
    summary = {}
    for arm, values in samples.items():
        ordered = sorted(values)
        summary[arm] = {
            "samples": len(values),
            "median_ms": 1000 * statistics.median(values),
            "p95_ms": 1000 * percentile(ordered, 95),
            "mean_ms": 1000 * statistics.fmean(values),
            "min_ms": 1000 * ordered[0],
            "max_ms": 1000 * ordered[-1],
        }
    return {
        "depth": depth,
        "repetitions": len(parts),
        "warmup_queries_per_repetition": parts[0]["warmup_queries"],
        "arm_order_per_repetition": [p["arm_order"] for p in parts],
        "ranking_mismatches_vs_scoring_pass": sum(p["ranking_mismatches"] for p in parts),
        "summary": summary,
        "samples_seconds": samples,
    }


def efficiency(builds: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Index cost per arm from the build records."""
    return {
        arm: {
            "chunk_count": b["chunk_count"],
            "vector_count": b["vector_count"],
            "index_bytes": b["bytes"]["index_total"],
            "seconds_chunk": b["seconds"]["chunk"],
            "seconds_embed": b["seconds"]["embed"],
            "seconds_build_total": b["seconds"]["total"],
        }
        for arm, b in builds.items()
    }


def top_chunk_ids(per_question: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """``{arm: {question id: [top-10 chunk ids]}}`` from scored per-question JSON."""
    return {
        arm: {row["question_id"]: [r["chunk_id"] for r in row["retrieved"]] for row in rows}
        for arm, rows in per_question.items()
    }


def assemble_results(
    config: Mapping[str, Any],
    builds: Mapping[str, Mapping[str, Any]],
    per_question: Mapping[str, Sequence[Mapping[str, Any]]],
    latency_parts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregates, paired statistics and latency summary from the raw phase outputs."""
    budgets = list(config["budget"]["budgets"])
    stat = config["statistics"]
    threshold = float(config["coverage"]["threshold"])
    scores = {
        arm: [QuestionScore.from_json(dict(row)) for row in rows]
        for arm, rows in per_question.items()
    }
    aggregates = {
        arm: analysis.aggregate_arm(arm_scores, budgets, threshold)
        for arm, arm_scores in scores.items()
    }
    paired = analysis.compare_arms(
        scores,
        comparisons=stat["comparisons"],
        budgets=budgets,
        coverage_threshold=threshold,
        primary=stat["primary"],
        seed=stat["seed"],
        resamples=stat["bootstrap_resamples"],
        confidence=stat["confidence"],
    )
    return {
        "per_question": {arm: list(rows) for arm, rows in per_question.items()},
        "aggregates": {"arms": aggregates, "efficiency": efficiency(builds)},
        "paired": paired,
        "latency": summarize_latency(latency_parts, config["latency"]["depth"]),
    }


def retrieval_phase(
    config: Mapping[str, Any],
    benchmark: Benchmark,
    retrievers: Mapping[str, Retriever],
    *,
    progress: Callable[[str], None] = print,
) -> dict[str, list[dict[str, Any]]]:
    """Scoring pass: per arm, one JSON row per question (fixed-k, coverage and budgets)."""
    scores, _ = retrieval_pass(
        retrievers,
        benchmark,
        pool_depth=config["budget"]["pool_depth"],
        budgets=list(config["budget"]["budgets"]),
        progress=progress,
    )
    return {arm: [s.to_json() for s in rows] for arm, rows in scores.items()}


def run_experiment(
    config: Mapping[str, Any],
    builds: Mapping[str, Mapping[str, Any]],
    benchmark: Benchmark,
    retrievers: Mapping[str, Retriever],
    *,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Everything the experiment measures in one process (retrieval, latency, statistics)."""
    per_question = retrieval_phase(config, benchmark, retrievers, progress=progress)
    top_ids = top_chunk_ids(per_question)
    parts = []
    for repetition in range(config["latency"]["repetitions"]):
        parts.append(
            latency_repetition(
                retrievers,
                benchmark,
                top_ids,
                repetition=repetition,
                depth=config["latency"]["depth"],
                warmup=config["latency"]["warmup_queries"],
            )
        )
        progress(f"latency repetition {repetition + 1}/{config['latency']['repetitions']} done")
    return assemble_results(config, builds, per_question, parts)


def write_results(
    results_dir: Path,
    results: Mapping[str, Any],
    *,
    config_sha256: str,
    started_utc: str,
    finished_utc: str,
    overwrite: bool = False,
) -> list[Path]:
    """Write the result files atomically; refuse to replace a completed run."""
    targets = {name: safe_artifact_path(results_dir, name) for name in RESULT_FILES}
    existing = [name for name, path in targets.items() if path.exists()]
    if existing and not overwrite:
        raise ArtifactError(f"results already exist ({', '.join(existing)}); pass --overwrite")
    manifest = {
        "schema": "chunking-experiment-run/1",
        "config_sha256": config_sha256,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "environment": environment_info(),
        "files": [n for n in RESULT_FILES if n != "run-manifest.json"],
    }
    payloads = {
        "per-question.json": {
            "schema": "chunking-experiment-per-question/1",
            "arms": results["per_question"],
        },
        "aggregates.json": {"schema": "chunking-experiment-aggregates/1", **results["aggregates"]},
        "paired-statistics.json": {"schema": "chunking-experiment-paired/1", **results["paired"]},
        "latency.json": {"schema": "chunking-experiment-latency/1", **results["latency"]},
        "run-manifest.json": manifest,
    }
    return [
        write_text_atomic(targets[name], canonical_json(payloads[name]), overwrite=True)
        for name in RESULT_FILES
    ]
