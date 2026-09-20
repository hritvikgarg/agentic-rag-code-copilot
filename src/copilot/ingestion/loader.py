"""Repository ingestion: turn a directory into typed, safe ``SourceFile`` objects.

Pipeline for one run::

    resolve root -> walk (prune, classify by name/type/size) -> bounded read -> decode
                 -> apply repository limits -> SourceFile / SkippedFile + statistics

Logging: one INFO summary per run, WARNINGs for security-relevant or limit events, DEBUG for
per-file skips (relative path and reason only). File contents and absolute paths are never logged.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections import Counter
from pathlib import Path

from copilot.config.settings import get_settings
from copilot.ingestion.discovery import (
    Candidate,
    Pruned,
    Rejected,
    UnreadableDirectory,
    walk_repository,
)
from copilot.ingestion.errors import InvalidRepositoryError
from copilot.ingestion.policy import IngestionPolicy
from copilot.ingestion.reader import decode_source, read_bytes_safely
from copilot.models.ingestion import (
    IngestionResult,
    IngestionStats,
    SkippedFile,
    SkipReason,
    SourceFile,
)
from copilot.utils.paths import is_within_root

logger = logging.getLogger(__name__)


def resolve_repository_root(root: str | os.PathLike[str]) -> Path:
    """Validate and fully resolve the user-chosen repository root.

    The root itself may be reached through a symlink (the user chose it deliberately); symlinks
    *inside* the repository are never followed.

    Raises:
        InvalidRepositoryError: if the path is empty, missing, not a directory, or the filesystem
            root (scanning ``/`` or ``C:\\`` is never what a user means).
    """
    raw = os.fspath(root)
    if not raw or "\x00" in raw:
        raise InvalidRepositoryError("repository path is empty or invalid")
    try:
        resolved = Path(os.path.realpath(Path(raw).expanduser()))
    except (OSError, ValueError, RuntimeError) as exc:
        raise InvalidRepositoryError("repository path could not be resolved") from exc
    if not resolved.is_dir():
        raise InvalidRepositoryError("repository path does not exist or is not a directory")
    if resolved == resolved.parent:
        raise InvalidRepositoryError("refusing to scan a filesystem root")
    return resolved


def ingest_repository(
    root: str | os.PathLike[str],
    policy: IngestionPolicy | None = None,
    *,
    repository_name: str | None = None,
) -> IngestionResult:
    """Discover and read the useful, safe files of a repository.

    Args:
        root: Repository directory.
        policy: Rules and limits; defaults to ``IngestionPolicy.from_settings(get_settings())``.
        repository_name: Identifier stored on every file; defaults to the directory name.

    Returns:
        ``IngestionResult`` with accepted files (sorted by relative path), skipped files and
        statistics. If a repository-wide limit is hit the scan stops early and
        ``stats.truncated`` is ``True`` - callers must surface that to the user.

    Raises:
        InvalidRepositoryError: if ``root`` is not a scannable directory.
    """
    started = time.perf_counter()
    policy = policy or IngestionPolicy.from_settings(get_settings())
    real_root = resolve_repository_root(root)
    name = (repository_name or real_root.name).strip()
    if not name:
        raise InvalidRepositoryError("repository name is empty")

    files: list[SourceFile] = []
    skipped: list[SkippedFile] = []
    reasons: Counter[str] = Counter()
    pruned: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    discovered = 0
    unreadable_directories = 0
    total_bytes = 0
    truncation_reason: str | None = None

    def skip(relative_path: str, reason: SkipReason) -> None:
        skipped.append(SkippedFile(relative_path=relative_path, reason=reason))
        reasons[reason.value] += 1
        if reason in (SkipReason.UNSAFE_PATH, SkipReason.SYMLINK):
            logger.warning("Skipped %s (%s)", relative_path, reason.value)
        else:
            logger.debug("Skipped %s (%s)", relative_path, reason.value)

    for event in walk_repository(real_root, policy):
        if isinstance(event, Pruned):
            pruned[event.name] += 1
            continue
        if isinstance(event, UnreadableDirectory):
            unreadable_directories += 1
            logger.warning("Could not list directory %s", event.relative_path)
            continue

        discovered += 1
        if isinstance(event, Rejected):
            skip(event.relative_path, event.reason)
            continue

        assert isinstance(event, Candidate)
        if len(files) >= policy.max_files:
            truncation_reason = f"max_files={policy.max_files}"
            skip(event.relative_path, SkipReason.LIMIT_EXCEEDED)
            break

        outcome = _load_candidate(event, real_root, name, policy)
        if isinstance(outcome, SkipReason):
            skip(event.relative_path, outcome)
            continue
        if total_bytes + outcome.size_bytes > policy.max_total_bytes:
            truncation_reason = f"max_total_bytes={policy.max_total_bytes}"
            skip(event.relative_path, SkipReason.LIMIT_EXCEEDED)
            break

        files.append(outcome)
        total_bytes += outcome.size_bytes
        languages[outcome.language] += 1

    if truncation_reason:
        logger.warning(
            "Repository scan stopped early (%s); results are incomplete", truncation_reason
        )

    files.sort(key=lambda f: f.relative_path)
    skipped.sort(key=lambda s: s.relative_path)
    stats = IngestionStats(
        files_discovered=discovered,
        files_accepted=len(files),
        files_skipped=len(skipped),
        skip_reasons=dict(sorted(reasons.items())),
        directories_pruned=sum(pruned.values()),
        pruned_directory_names=dict(sorted(pruned.items())),
        directories_unreadable=unreadable_directories,
        total_bytes_accepted=total_bytes,
        languages=dict(sorted(languages.items())),
        truncated=truncation_reason is not None,
        truncation_reason=truncation_reason,
        elapsed_seconds=time.perf_counter() - started,
    )
    logger.info("Ingested repository %r: %s", name, stats.summary())
    return IngestionResult(
        repository_name=name, files=tuple(files), skipped=tuple(skipped), stats=stats
    )


def _load_candidate(
    candidate: Candidate, real_root: Path, repository_name: str, policy: IngestionPolicy
) -> SourceFile | SkipReason:
    """Read and decode one candidate, or return the reason it was rejected."""
    # Defence in depth: discovery never follows links, but re-check right before opening.
    if not is_within_root(candidate.path, real_root):
        return SkipReason.UNSAFE_PATH
    data = read_bytes_safely(candidate.path, policy.max_file_size_bytes)
    if isinstance(data, SkipReason):
        return data
    text = decode_source(data)
    if isinstance(text, SkipReason):
        return text
    return SourceFile(
        repository_name=repository_name,
        relative_path=candidate.relative_path,
        extension=candidate.extension,
        language=candidate.language,
        size_bytes=len(data),
        content=text,
        sha256=hashlib.sha256(data).hexdigest(),
    )
