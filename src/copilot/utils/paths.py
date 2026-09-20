"""Portable, defensive path helpers.

Two path spellings are used in this project:

* **Stored / canonical form** - a repository-relative POSIX string such as ``src/app/main.py``.
  It is what ``SourceFile.relative_path`` holds and what citations will show. It never contains
  backslashes, ``.``/``..`` components, drive letters or a leading separator, so the same value is
  valid on Windows and Linux and never leaks a host path.
* **Runtime form** - a ``pathlib.Path`` on the current machine, used only while reading files.

Everything here is pure string/``pathlib`` logic; nothing reads file contents.
"""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePath, PurePosixPath

_SEPARATORS = re.compile(r"[\\/]+")
_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")


class UnsafePathError(ValueError):
    """A path is absolute, escapes its root, or is otherwise not a safe repository path."""


def relative_posix(path: PurePath, root: PurePath) -> str:
    """Return ``path`` relative to ``root`` as a POSIX string (``.`` for the root itself).

    Works on any ``PurePath`` flavour, so Windows-style paths can be handled on Linux in tests.

    Raises:
        UnsafePathError: if ``path`` is not inside ``root``.
    """
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise UnsafePathError("path is not inside the repository root") from exc


def validate_relative_posix(value: str) -> str:
    """Validate the canonical stored form of a repository-relative path and return it unchanged.

    Raises:
        UnsafePathError: for empty, absolute, backslash-containing, drive-prefixed values, values
            with NUL bytes, or values containing empty, ``.`` or ``..`` components.
    """
    if not value:
        raise UnsafePathError("relative path is empty")
    if "\x00" in value:
        raise UnsafePathError("relative path contains a NUL byte")
    if "\\" in value:
        raise UnsafePathError("relative path must use '/' separators only")
    if value.startswith("/") or _DRIVE_PREFIX.match(value):
        raise UnsafePathError("relative path must not be absolute")
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise UnsafePathError("relative path contains empty, '.' or '..' components")
    return value


def is_within_root(candidate: Path, real_root: Path) -> bool:
    """True if ``candidate`` (after resolving symlinks) lies inside ``real_root``.

    ``real_root`` must already be fully resolved (``os.path.realpath``). Comparison is done with
    ``pathlib`` so Windows' case-insensitive semantics are respected on Windows.
    """
    try:
        resolved = Path(os.path.realpath(candidate))
    except (OSError, ValueError, RuntimeError):
        return False
    return resolved == real_root or resolved.is_relative_to(real_root)


def safe_join(real_root: Path, relative: str) -> Path:
    """Join untrusted user input to ``real_root`` without ever leaving it.

    Both ``/`` and ``\\`` are treated as separators regardless of the host OS, so input such as
    ``..\\..\\secret`` is rejected on Linux exactly as it would be on Windows. The result is
    checked again after symlink resolution.

    Raises:
        UnsafePathError: on empty/absolute/drive-prefixed input, ``..`` components, NUL bytes, or
            if the resolved location is outside ``real_root``.
    """
    if not relative or "\x00" in relative:
        raise UnsafePathError("empty path or NUL byte")
    if relative.startswith(("/", "\\")) or _DRIVE_PREFIX.match(relative):
        raise UnsafePathError("absolute paths are not allowed")
    parts = [p for p in _SEPARATORS.split(relative) if p not in ("", ".")]
    if not parts:
        raise UnsafePathError("path has no components")
    if any(p == ".." for p in parts):
        raise UnsafePathError("'..' components are not allowed")
    candidate = real_root.joinpath(*PurePosixPath(*parts).parts)
    if not is_within_root(candidate, real_root):
        raise UnsafePathError("path resolves outside the repository root")
    return candidate
