"""Design freeze: validate everything that does not need retrieval, then write the frozen config.

``freeze`` reads the descriptive size-matching result, re-validates the benchmark against the
frozen snapshot, checks that Arm A reproduces the existing index's chunk ids, checks that the four
arms cut non-Python files identically, and writes ``frozen-config.json`` plus ``design-check.json``.
No embedding model is loaded and no query is run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from copilot.chunking.base import chunk_repository
from copilot.config.settings import Settings
from copilot.evaluation.benchmark import load_benchmark, verify_regions
from copilot.evaluation.chunking_experiment.config import (
    ARM_IDS,
    SCHEMA,
    ArmConfig,
    ExperimentConfigError,
    file_sha256,
    ingest_snapshot,
)
from copilot.evaluation.chunking_experiment.profile import profile_by_scope
from copilot.evaluation.snapshot import verify_tree_against_index
from copilot.models.chunk import Chunk
from copilot.vectorstore.records import chunk_ids_digest
from copilot.vectorstore.validation import sort_chunks_canonically

SEED = 20261027
BOOTSTRAP_RESAMPLES = 20000
CONFIDENCE = 0.95
KS = (1, 3, 5, 10)
POOL_DEPTH = 30
BUDGETS = (512, 1024, 2048)
COVERAGE_THRESHOLD = 0.5
LATENCY_WARMUP = 3
LATENCY_REPETITIONS = 5
COMPARISONS = (("B", "A"), ("B", "C"), ("C", "A"), ("D", "A"), ("D", "B"))
PRIMARY = {"comparison": ["B", "A"], "endpoint": "hit@5", "alpha": 0.05}


def arms_from_matching(matching: Mapping[str, Any], settings: Settings) -> list[ArmConfig]:
    """The four arms, from the production defaults and the selected Arm C/D parameters."""
    base = {
        "size_lines": settings.chunk_size_lines,
        "overlap_lines": settings.chunk_overlap_lines,
        "max_tokens": settings.chunk_max_tokens,
    }
    arm_a = matching["arm_a"]
    arm_b = matching["arm_b"]
    ast_params = dict(arm_b["params"])
    return [
        ArmConfig(
            "A",
            "baseline line windows (production)",
            arm_a["strategy"],
            arm_a["strategy_version"],
            arm_a["params"],
        ),
        ArmConfig(
            "B",
            "AST structure-aware (production, unchanged)",
            arm_b["strategy"],
            arm_b["strategy_version"],
            ast_params,
        ),
        ArmConfig(
            "C",
            "size-matched line windows (Python window re-sized, no overlap)",
            "line_sized",
            1,
            {
                "python_size_lines": int(matching["arm_c"]["selected_parameter"]),
                "python_overlap_lines": 0,
                **base,
            },
        ),
        ArmConfig(
            "D",
            "AST with adjacent structural chunks merged",
            "ast_merged",
            1,
            {
                **ast_params,
                "merge_max_tokens": int(matching["arm_d"]["selected_parameter"]),
                "base": f"ast@{arm_b['strategy_version']}",
            },
        ),
    ]


def _non_python_signature(chunks: tuple[Chunk, ...]) -> list[tuple[str, int, int, str]]:
    return [
        (c.file_path, c.start_line, c.end_line, c.content_sha256)
        for c in chunks
        if c.language != "python"
    ]


def design_check(
    *,
    arms: list[ArmConfig],
    settings: Settings,
    tree: str | Path,
    repository_name: str,
    ignore_directories: tuple[str, ...],
    benchmark_path: str | Path,
    reference_index: str | Path,
) -> dict[str, Any]:
    """Everything verifiable without retrieval (raises ``ExperimentConfigError`` on a failure)."""
    ingestion = ingest_snapshot(
        tree, settings, repository_name=repository_name, ignore_directories=ignore_directories
    )
    bench = load_benchmark(benchmark_path)
    problems = verify_regions(bench.questions, ingestion.files)
    if problems:
        raise ExperimentConfigError(
            f"{len(problems)} benchmark region(s) do not match the snapshot: {problems[0]}"
        )
    snapshot = verify_tree_against_index(
        tree, reference_index, ignore_directories=ignore_directories, settings=settings
    )
    if not snapshot.matches:
        raise ExperimentConfigError("the snapshot does not match the reference index fingerprint")

    per_arm: dict[str, Any] = {}
    signatures = {}
    for arm in arms:
        chunks = chunk_repository(ingestion, arm.chunker()).chunks
        signatures[arm.arm_id] = _non_python_signature(chunks)
        profiles = profile_by_scope(chunks, ingestion.files)
        per_arm[arm.arm_id] = {
            "chunk_count": len(chunks),
            "profiles": {scope: p.to_json() for scope, p in profiles.items()},
            "chunk_ids_sha256": chunk_ids_digest(
                [c.chunk_id for c in sort_chunks_canonically(chunks)]
            ),
        }
    reference = json.loads((Path(reference_index) / "manifest.json").read_text(encoding="utf-8"))
    identical = all(signatures[a] == signatures["A"] for a in ARM_IDS)
    if not identical:
        raise ExperimentConfigError("the arms do not chunk non-Python files identically")
    questions = len(bench.questions)
    regions = sum(len(q.relevant) for q in bench.questions)
    return {
        "questions": questions,
        "regions": regions,
        "regions_validated": True,
        "benchmark_sha256": file_sha256(benchmark_path),
        "snapshot_matches_reference_index": True,
        "reference_index_id": reference["index_id"],
        "reference_chunk_count": reference["chunk_count"],
        "arm_a_chunk_count_equals_reference": per_arm["A"]["chunk_count"]
        == reference["chunk_count"],
        "non_python_chunking_identical_across_arms": identical,
        "non_python_chunks": len(signatures["A"]),
        "files_accepted": len(ingestion.files),
        "arms": per_arm,
    }


def frozen_config(
    *,
    arms: list[ArmConfig],
    settings: Settings,
    tree: str,
    snapshot_commit: str,
    repository_name: str,
    ignore_directories: tuple[str, ...],
    benchmark_path: str,
    matching_path: str,
    matching: Mapping[str, Any],
    check: Mapping[str, Any],
) -> dict[str, Any]:
    """The configuration written to ``frozen-config.json``."""
    return {
        "schema": SCHEMA,
        "experiment": "chunking-comparison",
        "snapshot": {"tree": tree, "commit": snapshot_commit},
        "repository_name": repository_name,
        "ignore_directories": list(ignore_directories),
        "benchmark": {
            "path": benchmark_path,
            "sha256": check["benchmark_sha256"],
            "questions": check["questions"],
            "regions": check["regions"],
        },
        "embedding": {
            "model": settings.embedding_model,
            "text_style": "prefixed",
            "representation_version": 1,
            "dimension": 768,
        },
        "arms": [a.to_json() for a in arms],
        "matching": {
            "source": matching_path,
            "source_sha256": file_sha256(matching_path),
            "tolerance": matching["arm_c"]["tolerance"],
            "arm_c_adequate": matching["arm_c"]["adequate"],
            "arm_d_adequate": matching["arm_d"]["adequate"],
            "arm_c_median_error": matching["arm_c"]["selected_median_error"],
            "arm_d_median_error": matching["arm_d"]["selected_median_error"],
        },
        "fixed_k": {"ks": list(KS), "depth": max(KS), "mrr_depth": max(KS)},
        "budget": {
            "pool_depth": POOL_DEPTH,
            "budgets": list(BUDGETS),
            "builder": "copilot.rag.context.build_context(preserve_order=False)",
        },
        "coverage": {"threshold": COVERAGE_THRESHOLD},
        "latency": {
            "depth": max(KS),
            "warmup_queries": LATENCY_WARMUP,
            "repetitions": LATENCY_REPETITIONS,
            "arm_order": "rotated by one position each repetition",
        },
        "statistics": {
            "seed": SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "confidence": CONFIDENCE,
            "primary": PRIMARY,
            "comparisons": [list(c) for c in COMPARISONS],
            "multiple_comparisons": "Holm over all non-primary McNemar tests",
        },
    }
