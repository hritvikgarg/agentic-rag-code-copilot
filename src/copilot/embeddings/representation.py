"""How a ``Chunk`` becomes the text that is embedded (the "embedding representation").

Raw source (``Chunk.content``) is what we cite and show; it is never modified. The text that is
*embedded* is a separate string built here, deterministically, from the chunk:

``prefixed`` (default)::

    File: src/auth/service.py
    Language: python
    Lines: 10-40
    <blank line>
    <chunk content>

``raw``: just ``chunk.content``.

**Why a prefix at all.** A chunk cut from the middle of a file lacks context; a file path such as
``src/auth/service.py`` carries strong hints (auth, service) that the code text may not repeat.
**Why it is configurable.** Whether the prefix *helps retrieval* is unmeasured (retrieval does not
exist yet); Milestone 5 indexes one style and can A/B the other. The prefix costs a small, fixed-ish
number of tokens (measured by ``python -m copilot.embeddings representations``).

Bump ``REPRESENTATION_VERSION`` when the prefix format changes: it belongs in index manifests, since
vectors built from different representations are not comparable.
"""

from __future__ import annotations

from typing import Literal

from copilot.models.chunk import Chunk

REPRESENTATION_VERSION = 1
TextStyle = Literal["prefixed", "raw"]


def _line_label(chunk: Chunk) -> str:
    if chunk.is_fragment:
        return f"{chunk.start_line} (part {chunk.fragment_index + 1} of {chunk.fragment_count})"  # type: ignore[operator]
    return f"{chunk.start_line}-{chunk.end_line}"


def prefixed_text(chunk: Chunk) -> str:
    """Metadata header (file, language, lines) followed by the chunk's raw content."""
    header = f"File: {chunk.file_path}\nLanguage: {chunk.language}\nLines: {_line_label(chunk)}\n\n"
    return header + chunk.content


def embedding_text(chunk: Chunk, style: TextStyle = "prefixed") -> str:
    """The text to embed for ``chunk`` in the given style."""
    if style == "raw":
        return chunk.content
    if style == "prefixed":
        return prefixed_text(chunk)
    raise ValueError(f"unknown embedding text style {style!r}; use 'prefixed' or 'raw'")
