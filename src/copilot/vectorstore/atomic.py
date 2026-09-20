"""Crash-safe directory writes: stage, sync, then rename into place.

A vector index is several files (index, mapping, manifest) that are only meaningful together.
Writing them straight into the final directory would leave a half-written directory that looks like
an index if the process dies midway. Instead:

1. all files are written into a sibling *staging* directory named ``.staging-<name>-<random>``
   (never a valid index location; readers ignore names starting with ``.``);
2. every file is flushed and ``fsync``-ed, the manifest last;
3. one ``os.replace`` renames the staging directory to the final name. A directory rename is atomic
   on the same filesystem, so the final path either does not exist or holds a *complete* index;
4. on any error the staging directory is removed.

Replacing an existing index (``overwrite=True``): the old directory is first renamed aside, the new
one renamed in, and only then is the old one deleted. If the second rename fails the old index is
put back. There is a brief moment where the final path does not exist, but never one where it holds
a partial index. Random suffixes here name temporary directories only; they never identify an index.

This is intentionally modest (a local, single-user tool): no locking against two concurrent
writers, and it does not defend against power loss on filesystems that reorder renames.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path

from copilot.vectorstore.errors import IndexExistsError, IndexStorageError

STAGING_PREFIX = ".staging-"
REPLACED_PREFIX = ".replaced-"


def _write_file(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` in binary mode (no newline translation) and fsync it."""
    with open(path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _sync_directory(path: Path) -> None:
    """Best-effort fsync of a directory entry (not possible on Windows)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_directory_atomically(
    target: Path, files: Mapping[str, bytes], *, overwrite: bool = False
) -> Path:
    """Create ``target`` containing ``files`` (written in mapping order) atomically."""
    for name in files:
        if name != Path(name).name or name in ("", ".", ".."):
            raise IndexStorageError(f"invalid artifact file name {name!r}")
    if target.exists() and not overwrite:
        raise IndexExistsError(f"an index already exists at {target.name!r}; pass overwrite=True")

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{STAGING_PREFIX}{target.name}-", dir=parent))
    try:
        for name, data in files.items():
            _write_file(staging / name, data)
        _sync_directory(staging)
        _swap_into_place(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    _sync_directory(parent)
    return target


def _swap_into_place(staging: Path, target: Path) -> None:
    backup: Path | None = None
    try:
        if target.exists():
            backup = target.parent / f"{REPLACED_PREFIX}{target.name}-{uuid.uuid4().hex[:8]}"
            os.replace(target, backup)
        os.replace(staging, target)
    except OSError as exc:
        if backup is not None and not target.exists():
            os.replace(backup, target)  # put the previous index back
        raise IndexStorageError(f"could not move the new index into place: {exc}") from exc
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)
