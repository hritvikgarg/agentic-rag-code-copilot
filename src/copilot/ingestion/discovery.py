"""Safe, deterministic repository traversal.

``walk_repository`` yields *events* instead of returning a list, so the caller can stream, count
and stop early (repository limits) without the traversal knowing about limits.

Safety properties (each covered by tests):

* **Pruning, not filtering.** Ignored and sensitive directories are never entered; their contents
  are never enumerated.
* **Symlinks are never followed** (files or directories); Windows junctions are treated the same.
  A link whose target resolves outside the root is reported as ``UNSAFE_PATH``, any other link as
  ``SYMLINK``.
* **Only regular files are candidates**; FIFOs, sockets and devices are rejected before any open.
* **Cheap name checks run before any I/O**: sensitive name, generated file, unsupported extension,
  then size (from ``stat``, not by reading).
* **Deterministic order**: directory entries are sorted by name at every level.
* Traversal is iterative (no recursion limit) and needs no file contents.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import NamedTuple

from copilot.ingestion.policy import IngestionPolicy
from copilot.models.ingestion import SkipReason
from copilot.utils.paths import is_within_root


class Candidate(NamedTuple):
    """A file that passed every name/type/size check and may now be read."""

    path: Path  # absolute, runtime only - never stored in models
    relative_path: str
    extension: str
    language: str


class Rejected(NamedTuple):
    """A file-level entry that was seen and rejected."""

    relative_path: str
    reason: SkipReason


class Pruned(NamedTuple):
    """A directory that was not entered."""

    name: str
    relative_path: str


class UnreadableDirectory(NamedTuple):
    """A directory that could not be listed (permissions, vanished, ...)."""

    relative_path: str


Event = Candidate | Rejected | Pruned | UnreadableDirectory


def _is_link(entry: os.DirEntry[str]) -> bool:
    """Symlink or Windows junction (``is_junction`` is always False on other platforms)."""
    return entry.is_symlink() or entry.is_junction()


def _classify_link(entry: os.DirEntry[str], real_root: Path) -> SkipReason:
    """A link is never followed; decide only how to *report* it."""
    return (
        SkipReason.SYMLINK
        if is_within_root(Path(entry.path), real_root)
        else SkipReason.UNSAFE_PATH
    )


def _classify_file(
    entry: os.DirEntry[str], relative: str, policy: IngestionPolicy
) -> Candidate | Rejected:
    name = entry.name
    if "\\" in name:  # legal on Linux, but would be ambiguous with Windows separators
        return Rejected(relative, SkipReason.UNSAFE_PATH)
    if policy.is_sensitive_file(name):
        return Rejected(relative, SkipReason.SENSITIVE)
    if policy.is_generated_file(name):
        return Rejected(relative, SkipReason.IGNORED_FILE)
    language = policy.language_for(name)
    if language is None:
        return Rejected(relative, SkipReason.UNSUPPORTED_EXTENSION)
    try:
        size = entry.stat(follow_symlinks=False).st_size
    except OSError:
        return Rejected(relative, SkipReason.UNREADABLE)
    if size > policy.max_file_size_bytes:
        return Rejected(relative, SkipReason.OVERSIZED)
    extension = os.path.splitext(name)[1].lower()
    return Candidate(Path(entry.path), relative, extension, language)


def walk_repository(real_root: Path, policy: IngestionPolicy) -> Iterator[Event]:
    """Traverse ``real_root`` (must already be resolved) and yield events in a stable order."""
    stack: list[tuple[Path, tuple[str, ...]]] = [(real_root, ())]
    while stack:
        directory, parts = stack.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda e: e.name)
        except OSError:
            yield UnreadableDirectory(PurePosixPath(*parts).as_posix())
            continue

        subdirectories: list[tuple[Path, tuple[str, ...]]] = []
        for entry in entries:
            entry_parts = (*parts, entry.name)
            relative = PurePosixPath(*entry_parts).as_posix()

            if _is_link(entry):
                yield Rejected(relative, _classify_link(entry, real_root))
            elif entry.is_dir(follow_symlinks=False):
                if policy.is_ignored_directory(entry.name) or policy.is_sensitive_directory(
                    entry.name
                ):
                    yield Pruned(entry.name, relative)
                else:
                    subdirectories.append((Path(entry.path), entry_parts))
            elif entry.is_file(follow_symlinks=False):
                yield _classify_file(entry, relative, policy)
            else:
                yield Rejected(relative, SkipReason.SPECIAL_FILE)

        # Reverse so the alphabetically first subdirectory is popped (visited) first.
        stack.extend(reversed(subdirectories))
