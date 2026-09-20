"""Bounded, race-resistant file reading and text decoding.

Reading policy:

* The file is opened with ``O_NOFOLLOW`` (where the OS has it) so a path that was swapped for a
  symlink after discovery is refused, with ``O_NONBLOCK`` so a swapped-in FIFO cannot hang the
  scan, and with ``O_BINARY`` on Windows so the C runtime does not translate newlines or stop at
  Ctrl-Z. The opened descriptor is ``fstat``-ed and must be a regular file.
* At most ``max_bytes + 1`` bytes are read, so a file that grew after ``stat`` is still bounded.
* Decoding: UTF-16 only when a BOM is present (Windows PowerShell 5 writes it), otherwise strict
  UTF-8 with an optional BOM. NUL bytes mean *binary*. There is deliberately **no** latin-1
  fallback: it never fails, so it would happily turn binary garbage into "text".
* Newlines are normalised to ``\\n`` so line numbers mean the same thing on every platform.
  (Downstream code should split on ``"\\n"``, not ``str.splitlines()``, which also splits on
  form-feed, vertical-tab and Unicode line separators that editors do not count as line breaks.)

Nothing here logs file contents.
"""

from __future__ import annotations

import codecs
import errno
import os
import stat
from pathlib import Path

from copilot.models.ingestion import SkipReason

_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
    | getattr(os, "O_BINARY", 0)
)
_NUL_SNIFF_BYTES = 8192
_UTF16_BOMS = (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)


def read_bytes_safely(path: Path, max_bytes: int) -> bytes | SkipReason:
    """Read a regular file of at most ``max_bytes`` bytes, or say why it was rejected."""
    try:
        fd = os.open(path, _OPEN_FLAGS)
    except OSError as exc:
        # ELOOP is what O_NOFOLLOW reports for a symlink; treat it as an escape attempt.
        return SkipReason.UNSAFE_PATH if exc.errno == errno.ELOOP else SkipReason.UNREADABLE
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return SkipReason.SPECIAL_FILE
        if info.st_size > max_bytes:
            return SkipReason.OVERSIZED
        with os.fdopen(fd, "rb", closefd=False) as handle:
            data = handle.read(max_bytes + 1)
    except OSError:
        return SkipReason.UNREADABLE
    finally:
        os.close(fd)
    return SkipReason.OVERSIZED if len(data) > max_bytes else data


def normalize_newlines(text: str) -> str:
    """Convert CRLF and lone CR to LF (line *count* is preserved)."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def decode_source(data: bytes) -> str | SkipReason:
    """Decode raw bytes to normalised text, or return ``BINARY`` / ``ENCODING_ERROR``."""
    try:
        if data.startswith(_UTF16_BOMS):
            text = data.decode("utf-16")  # the BOM selects endianness and is consumed
        else:
            if b"\x00" in data[:_NUL_SNIFF_BYTES]:
                return SkipReason.BINARY
            text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return SkipReason.ENCODING_ERROR
    if "\x00" in text:  # NUL later in the file, or UTF-32 mis-read as UTF-16
        return SkipReason.BINARY
    return normalize_newlines(text)
