"""Software, hardware and git facts recorded next to every experiment artifact."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib import metadata
from typing import Any

_ENV_ALLOWLIST = (
    "COPILOT_EMBEDDING_BATCH_SIZE",
    "COPILOT_EMBEDDING_THREADS",
    "COPILOT_EMBEDDING_MODEL",
    "HF_HUB_OFFLINE",
)
_PACKAGES = ("numpy", "faiss-cpu", "fastembed", "onnxruntime", "pydantic", "tokenizers")


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def memory_bytes() -> int | None:
    """Total physical memory where the platform exposes it (``None`` otherwise)."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def git_head(cwd: str = ".") -> str | None:
    """The current commit hash (``None`` outside a git checkout)."""
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def environment_info() -> dict[str, Any]:
    """Hardware and software versions relevant to reproducing a measurement."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes(),
        "packages": {name: _version(name) for name in _PACKAGES},
        "git_head": git_head(),
        "environment_variables": {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ},
    }
