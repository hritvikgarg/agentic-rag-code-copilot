"""Frozen repository snapshots for evaluation (Milestone 7.1).

Problem. The benchmark indexes this very repository, and an index is only valid for the exact
accepted source content it was built from (``repository_fingerprint``). Every implementation
commit therefore makes the *live* working tree "stale" for the existing index, and a comparison
run that is resumed against a different tree could silently mix answers from two code states.

Separation of concerns (four things, four places)::

    the application under development   the repository checkout you keep committing to
    the repository being evaluated       data/snapshots/<commit>/tree/   (frozen, read-only use)
    the persistent vector index          data/indexes/<index_id>/        (never edited)
    the comparison results               data/evaluation/*.json          (one file per experiment)

A snapshot is ``git archive <commit>`` extracted into ``data/snapshots/<full commit sha>/tree/``
(``data/`` is git-ignored, so it never reaches a commit). ``git archive`` is used instead of a
``git worktree`` because it adds nothing to ``.git``, leaves no branch/worktree state to clean up,
produces exactly the committed bytes (the repository's ``.gitattributes`` forces LF), and cannot
be accidentally edited into a different state by ``git checkout``/``git commit``.

Nothing is trusted on faith. A snapshot is only *accepted* when its fingerprint - computed with
the same ingestion policy, ignore directories and repository name the index was built with -
equals the fingerprint recorded in the index manifest. ``create_snapshot(..., index_dir=...)``
extracts into a temporary directory, verifies, and only then moves the tree into place; a
mismatch leaves nothing behind. Index manifests and comparison files are never modified.

The metadata file ``snapshot.json`` lives *next to* ``tree/`` (not inside it) because ``.json`` is
an accepted source extension: a file inside the tree would change the fingerprint.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from copilot.config import Settings, get_settings
from copilot.ingestion import IngestionPolicy, ingest_repository
from copilot.utils.atomic_file import write_text_atomically
from copilot.vectorstore import IndexManifest, repository_fingerprint

SNAPSHOT_SCHEMA = "evaluation-snapshot/1"
TREE_DIRNAME = "tree"
METADATA_FILENAME = "snapshot.json"
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_GIT_TIMEOUT_SECONDS = 120


class SnapshotError(ValueError):
    """A snapshot could not be created or verified. Messages are fixed text (no raw git output)."""


@dataclass(frozen=True)
class SnapshotCheck:
    """The result of comparing a tree's fingerprint with an index's."""

    matches: bool
    expected_fingerprint: str  # from the index manifest
    actual_fingerprint: str  # computed from the tree
    file_count: int
    index_id: str
    repository_name: str


def read_index_manifest(index_dir: str | os.PathLike[str]) -> IndexManifest:
    """The validated ``manifest.json`` of an index directory (no vectors are loaded)."""
    path = Path(index_dir) / "manifest.json"
    try:
        return IndexManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise SnapshotError(
            f"cannot read the index manifest at {path.name} ({type(exc).__name__})"
        ) from None


def verify_tree_against_index(
    tree: str | os.PathLike[str],
    index_dir: str | os.PathLike[str],
    *,
    ignore_directories: Sequence[str] = (),
    settings: Settings | None = None,
) -> SnapshotCheck:
    """Does ``tree`` have the fingerprint the index was built from?

    Uses exactly the ingestion the retriever uses (``IngestionPolicy`` from the settings plus the
    same extra ignored directories and the index's repository name), so "matches" here means
    ``Retriever.open`` will accept the tree. No embedding model is loaded.
    """
    settings = settings or get_settings()
    spec = read_index_manifest(index_dir)
    policy = IngestionPolicy.from_settings(settings).with_extra_ignored_directories(
        *ignore_directories
    )
    ingestion = ingest_repository(tree, policy, repository_name=spec.spec.repository_name)
    if ingestion.stats.truncated:
        raise SnapshotError(
            f"ingestion stopped early ({ingestion.stats.truncation_reason}); the tree cannot be "
            "compared with the index"
        )
    actual = repository_fingerprint(ingestion.files)
    return SnapshotCheck(
        matches=actual == spec.spec.repository_fingerprint,
        expected_fingerprint=spec.spec.repository_fingerprint,
        actual_fingerprint=actual,
        file_count=len(ingestion.files),
        index_id=spec.index_id,
        repository_name=spec.spec.repository_name,
    )


def _git(repo_root: Path, *args: str, binary: bool = False) -> bytes | str:
    """Run ``git -C repo_root ARGS`` (argument list, no shell). Raises ``SnapshotError``."""
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "-C", str(repo_root), *args],  # noqa: S607 - git is resolved from PATH
            capture_output=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        raise SnapshotError("git was not found on PATH") from None
    except subprocess.TimeoutExpired:
        raise SnapshotError(f"git {args[0]} timed out") from None
    if result.returncode != 0:
        raise SnapshotError(f"git {args[0]} failed (exit code {result.returncode})")
    return result.stdout if binary else result.stdout.decode("utf-8", errors="replace").strip()


def resolve_commit(repo_root: str | os.PathLike[str], revision: str) -> str:
    """The full 40-hex commit id for ``revision`` (a sha, short sha, branch or tag)."""
    if not revision or revision.startswith("-"):
        raise SnapshotError("the revision must be a commit, branch or tag name")
    commit = str(
        _git(Path(repo_root), "rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}")
    )
    if not _FULL_SHA.match(commit):
        raise SnapshotError(f"{revision!r} did not resolve to a commit")
    return commit


def export_commit(repo_root: str | os.PathLike[str], commit: str, destination: Path) -> None:
    """Extract the tree of ``commit`` into the (new, empty) ``destination`` directory.

    ``tarfile``'s ``data`` filter refuses absolute paths, ``..`` escapes, device files and links
    that point outside ``destination`` - so a hostile archive cannot write elsewhere.
    """
    if not _FULL_SHA.match(commit):
        raise SnapshotError("export_commit needs a full 40-hex commit id")
    archive = _git(Path(repo_root), "archive", "--format=tar", commit, binary=True)
    assert isinstance(archive, bytes)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            tar.extractall(destination, filter="data")  # noqa: S202 - the "data" filter is set
    except (tarfile.TarError, OSError) as exc:
        raise SnapshotError(f"cannot extract the archive ({type(exc).__name__})") from None


def snapshot_paths(snapshots_dir: str | os.PathLike[str], commit: str) -> tuple[Path, Path]:
    """(``<dir>/<commit>/tree``, ``<dir>/<commit>/snapshot.json``) for a full commit id."""
    base = Path(snapshots_dir) / commit
    return base / TREE_DIRNAME, base / METADATA_FILENAME


def create_snapshot(
    repo_root: str | os.PathLike[str],
    revision: str,
    snapshots_dir: str | os.PathLike[str],
    *,
    index_dir: str | os.PathLike[str] | None = None,
    ignore_directories: Sequence[str] = (),
    settings: Settings | None = None,
) -> tuple[Path, SnapshotCheck | None]:
    """Freeze ``revision`` under ``snapshots_dir`` and return ``(tree_path, check)``.

    With ``index_dir`` the snapshot is verified against that index *before* it is installed:
    a fingerprint mismatch raises ``SnapshotError`` and leaves nothing under ``snapshots_dir``.
    Without it the snapshot is installed unverified (``check`` is ``None``); verify it later with
    ``verify_tree_against_index``. An existing snapshot is never overwritten or modified - it is
    returned (and re-verified when ``index_dir`` is given).
    """
    repo = Path(repo_root)
    commit = resolve_commit(repo, revision)
    tree_path, meta_path = snapshot_paths(snapshots_dir, commit)

    if tree_path.exists():
        check = (
            _verified(tree_path, index_dir, ignore_directories, settings)
            if index_dir is not None
            else None
        )
        return tree_path, check

    snapshots = Path(snapshots_dir)
    snapshots.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{commit[:12]}-", dir=snapshots))
    try:
        staged_tree = staging / TREE_DIRNAME
        export_commit(repo, commit, staged_tree)
        check = (
            _verified(staged_tree, index_dir, ignore_directories, settings)
            if index_dir is not None
            else None
        )
        metadata = {
            "schema_version": SNAPSHOT_SCHEMA,
            "commit": commit,
            "tree_sha": str(_git(repo, "rev-parse", "--verify", f"{commit}^{{tree}}")),
            "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "verified_against_index": check.index_id if check else None,
            "repository_fingerprint": check.actual_fingerprint if check else None,
            "ignore_directories": list(ignore_directories),
        }
        write_text_atomically(
            staging / METADATA_FILENAME, json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        final_dir = tree_path.parent
        os.replace(staging, final_dir)  # atomic on one filesystem; staging is inside snapshots/
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return tree_path, check


def _verified(
    tree: Path,
    index_dir: str | os.PathLike[str] | None,
    ignore_directories: Sequence[str],
    settings: Settings | None,
) -> SnapshotCheck:
    assert index_dir is not None
    check = verify_tree_against_index(
        tree, index_dir, ignore_directories=ignore_directories, settings=settings
    )
    if not check.matches:
        raise SnapshotError(
            "the snapshot does not match the index: fingerprint "
            f"{check.actual_fingerprint[:16]} (snapshot) != {check.expected_fingerprint[:16]} "
            f"(index {check.index_id}). Another commit, a different --ignore-dir set or a "
            "different repository name was used. Nothing was installed or changed."
        )
    return check
