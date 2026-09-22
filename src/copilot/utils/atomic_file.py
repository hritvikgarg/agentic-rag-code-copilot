"""Crash-safe single-file writes: write beside the target, fsync, then atomically rename.

Reuses the same safety pattern as ``copilot.vectorstore.atomic`` (write into a staging location,
fsync, then one ``os.replace``) but for a single file rather than a directory of files. The two
are kept as separate, small functions rather than one shared implementation because the artifacts
they protect are shaped differently: a vector index is several files that are only meaningful
together (so the unit that must appear-or-not-appear atomically is a whole directory), while an
evaluation comparison file is one JSON document (so the unit is one file). Forcing both through a
single-file-oriented or a single-directory-oriented function would make one of the two callers
work around a shape it doesn't have; this module exists so that a checkpoint write and the final
write of a comparison run go through the *same* function, rather than each evaluation write path
inventing its own variant of "write then rename".

A reader of ``path`` only ever sees the previous complete file or the new complete file - never a
half-written one - even if the process is killed (Ctrl+C, a crash, the machine losing power)
between the write and the rename. On any failure the temporary file is removed and ``path`` is
left exactly as it was before the call.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_text_atomically(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` so a concurrent reader never observes a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
