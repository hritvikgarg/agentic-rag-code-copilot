"""Build the four experimental indexes (offline) and record how each was built.

Each arm gets its own directory below the experiment's index root; the existing indexes in
``data/indexes`` are never touched (the root is checked against them). Everything except the chunker
is held constant by construction: the same snapshot, ingestion rules, embedder, representation and
index type are passed to ``build_repository_index`` for every arm.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from copilot.config.settings import Settings
from copilot.embeddings.base import Embedder
from copilot.evaluation.chunking_experiment.artifacts import (
    ArtifactError,
    safe_artifact_path,
    write_text_atomic,
)
from copilot.evaluation.chunking_experiment.config import ArmConfig, canonical_json
from copilot.evaluation.chunking_experiment.embedding_cache import (
    ResumableEmbedder,
    TimeBudgetExceeded,
)
from copilot.evaluation.chunking_experiment.environment import environment_info
from copilot.evaluation.snapshot import read_index_manifest
from copilot.vectorstore import build_repository_index


def directory_bytes(path: Path) -> int:
    """Total size of the regular files below ``path``."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def check_index_root(root: Path, protected: Sequence[Path]) -> Path:
    """The resolved index root; refuse a root inside or equal to a protected directory."""
    resolved = root.resolve()
    for guarded in protected:
        g = guarded.resolve()
        if resolved == g or g in resolved.parents or resolved in g.parents:
            raise ArtifactError(f"the experiment index root {resolved} overlaps protected {g}")
    return resolved


def build_arm(
    arm: ArmConfig,
    *,
    tree: str,
    repository_name: str,
    ignore_directories: Sequence[str],
    settings: Settings,
    embedder: Embedder,
    index_root: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build one arm's index and return its build record.

    If ``embedder`` is a ``ResumableEmbedder`` the record's embedding time is the sum of the
    measured per-batch times (``embedding_mode`` says so); the wall time of this call is kept too.
    """
    target = safe_artifact_path(index_root, arm.arm_id)
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise ArtifactError(f"{target} already holds an index; pass overwrite=True to rebuild")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report = build_repository_index(
        tree,
        embedder=embedder,
        settings=settings,
        indexes_dir=target,
        repository_name=repository_name,
        ignore_directories=ignore_directories,
        text_style="prefixed",
        overwrite=overwrite,
        chunker=arm.chunker(),
    )
    manifest = read_index_manifest(report.index_path)
    resumable = isinstance(embedder, ResumableEmbedder)
    embed_seconds = embedder.recorded_seconds if resumable else report.seconds_embed
    total_seconds = (
        report.seconds_ingest + report.seconds_chunk + embed_seconds + report.seconds_index_and_save
        if resumable
        else report.seconds_total
    )
    return {
        "arm_id": arm.arm_id,
        "label": arm.label,
        "strategy": arm.strategy,
        "strategy_version": arm.strategy_version,
        "chunker_params": dict(arm.params),
        "built_at_utc": started,
        "index_id": report.index_id,
        "index_path": f"{arm.arm_id}/{report.index_id}",
        "repository_fingerprint": manifest.spec.repository_fingerprint,
        "chunk_ids_sha256": manifest.chunk_ids_sha256,
        "accepted_files": report.files,
        "chunk_count": report.chunks,
        "vector_count": report.vectors,
        "dimension": report.dimension,
        "embedding_model": report.embedding_model_id,
        "text_style": report.text_style,
        "representation_version": report.representation_version,
        "index_type": report.index_type,
        "embedding_mode": "resumable-cache" if resumable else "single-call",
        "seconds": {
            "ingest": report.seconds_ingest,
            "chunk": report.seconds_chunk,
            "embed": embed_seconds,
            "index_and_save": report.seconds_index_and_save,
            "total": total_seconds,
            "embed_wall_final_call": report.seconds_embed,
        },
        "bytes": {
            "artifacts": dict(report.artifact_sizes),
            "index_total": sum(report.artifact_sizes.values()),
            "directory_total": directory_bytes(report.index_path),
        },
    }


def _partial(root: Path, arm: ArmConfig) -> bool:
    """True when the arm's index directory exists but holds no saved index (nothing to protect)."""
    target = root / arm.arm_id
    return target.exists() and not any(target.rglob("manifest.json"))


def build_all(
    arms: Sequence[ArmConfig],
    *,
    tree: str,
    repository_name: str,
    ignore_directories: Sequence[str],
    settings: Settings,
    embedder: Embedder,
    index_root: Path,
    results_dir: Path,
    protected: Sequence[Path],
    only: Sequence[str] | None = None,
    overwrite: bool = False,
    progress: Callable[[str], None] = print,
    cache_root: Path | None = None,
    time_budget_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Build the selected arms in order, writing one build record per arm.

    With ``time_budget_seconds`` and ``cache_root`` the embedding step is resumable per arm; when
    the budget runs out ``TimeBudgetExceeded`` propagates and no record is written for that arm
    (run the same command again to continue).
    """
    root = check_index_root(index_root, protected)
    chosen = [a for a in arms if only is None or a.arm_id in only]
    if not chosen:
        raise ArtifactError("no arm selected")
    builds_dir = safe_artifact_path(results_dir, "builds")
    for arm in chosen:  # fail before spending any time if a record would be overwritten
        record = builds_dir / f"{arm.arm_id}.json"
        if record.exists() and not overwrite:
            raise ArtifactError(f"{record} exists; pass --overwrite to rebuild this arm")

    cold_started = time.perf_counter()
    info = embedder.info  # loads the model once: cold start is timed separately from indexing
    cold_seconds = time.perf_counter() - cold_started
    progress(f"embedding model {info.model_id} loaded in {cold_seconds:.1f}s (not part of builds)")
    records = []
    for arm in chosen:
        progress(f"building arm {arm.arm_id} ({arm.strategy} v{arm.strategy_version}) ...")
        arm_embedder: Embedder = embedder
        if time_budget_seconds is not None:
            if cache_root is None:
                raise ArtifactError("a time budget needs an embedding cache directory")
            arm_embedder = ResumableEmbedder(
                embedder,
                safe_artifact_path(cache_root, arm.arm_id),
                batch_size=settings.embedding_batch_size,
                time_budget_seconds=time_budget_seconds,
            )
        try:
            record = build_arm(
                arm,
                tree=tree,
                repository_name=repository_name,
                ignore_directories=ignore_directories,
                settings=settings,
                embedder=arm_embedder,
                index_root=root,
                overwrite=overwrite or (time_budget_seconds is not None and _partial(root, arm)),
            )
        except TimeBudgetExceeded as exc:
            progress(
                f"arm {arm.arm_id}: {exc}; run the same command again to continue "
                f"(finished batches are cached)"
            )
            raise
        record["model_cold_start_seconds"] = cold_seconds
        record["environment"] = environment_info()
        write_text_atomic(builds_dir / f"{arm.arm_id}.json", canonical_json(record), overwrite=True)
        progress(
            f"arm {arm.arm_id}: index {record['index_id']} chunks={record['chunk_count']} "
            f"embed={record['seconds']['embed']:.1f}s total={record['seconds']['total']:.1f}s"
        )
        records.append(record)
    return records


def load_build_records(results_dir: Path, arm_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """The build record of every arm (``ArtifactError`` naming the missing ones)."""
    out = {}
    missing = []
    for arm_id in arm_ids:
        path = results_dir / "builds" / f"{arm_id}.json"
        if path.is_file():
            out[arm_id] = json.loads(path.read_text(encoding="utf-8"))
        else:
            missing.append(arm_id)
    if missing:
        raise ArtifactError(
            f"no build record for arm(s) {', '.join(missing)}; run the 'build' command first"
        )
    return out
