"""Filesystem helpers shared by tests that need real symbolic links."""

from __future__ import annotations

import errno
from pathlib import Path

import pytest

# Windows: "A required privilege is not held by the client" (creating symlinks needs Developer
# Mode or an elevated process).
_WINERROR_PRIVILEGE_NOT_HELD = 1314
_NOT_PERMITTED_ERRNOS = {errno.EPERM, errno.EACCES, errno.ENOSYS, errno.ENOTSUP}


def symlink_not_permitted(exc: BaseException) -> bool:
    """True only when ``exc`` means "this environment cannot create symbolic links".

    Anything else (a missing parent directory, an existing path, a full disk, ...) is a real
    problem and must not be turned into a skip.
    """
    if isinstance(exc, NotImplementedError):
        return True
    if isinstance(exc, OSError):
        return getattr(exc, "winerror", None) == _WINERROR_PRIVILEGE_NOT_HELD or (
            exc.errno in _NOT_PERMITTED_ERRNOS
        )
    return False


def make_symlink_or_skip(link: Path, target: Path, *, is_dir: bool = False) -> None:
    """Create ``link -> target``; skip the calling test only if symlinks are unavailable here.

    The test's own assertions still run unchanged wherever symlinks can be created (Linux, macOS,
    and Windows with Developer Mode or elevation).
    """
    try:
        link.symlink_to(target, target_is_directory=is_dir)
    except (OSError, NotImplementedError) as exc:
        if symlink_not_permitted(exc):
            pytest.skip(f"symbolic links cannot be created in this environment ({exc})")
        raise
