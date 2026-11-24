"""Safe, atomic, non-clobbering artifact writes for the experiment.

* Outputs are only written below an explicit base directory (traversal and symlink-escape safe).
* A completed result is never overwritten unless ``overwrite=True`` is passed explicitly.
* Content is written to a temporary sibling and moved into place with ``os.replace``, so a crash
  can never leave a half-written result under the final name.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


class ArtifactError(RuntimeError):
    """An artifact path is unsafe or would overwrite a completed result."""


def safe_artifact_path(base: str | Path, relative: str | Path) -> Path:
    """``base / relative`` if it stays inside ``base`` once resolved, else ``ArtifactError``."""
    base_path = Path(base).resolve()
    candidate = (base_path / relative).resolve()
    if base_path != candidate and base_path not in candidate.parents:
        raise ArtifactError(f"{relative} escapes the artifact directory {base_path}")
    return candidate


def write_text_atomic(path: str | Path, text: str, *, overwrite: bool = False) -> Path:
    """Write ``text`` (UTF-8, LF) to ``path`` atomically; refuse to replace unless ``overwrite``."""
    target = Path(path)
    if target.exists() and not overwrite:
        raise ArtifactError(f"{target} already exists; pass overwrite=True to replace a result")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target
