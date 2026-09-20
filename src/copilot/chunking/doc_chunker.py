"""Markdown section splitting for documentation files.

Documentation is prose organised by headings, so the natural unit is a *section*, not a fixed
line window. Sections larger than the configured window/token limits are split with the same
window algorithm as code, so no chunk ever exceeds the limits.

Supported: ATX headings (``# Title`` .. ``###### Title``); headings inside fenced code blocks are
ignored. Not supported: setext headings (``Title`` underlined with ``===``), HTML headings.
A heading with no body of its own (e.g. ``# Title`` directly followed by ``## Section``) is merged
into the following section so it does not become a useless one-line chunk.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_CLOSING_HASHES_RE = re.compile(r"[ \t]+#+$")


@dataclass(frozen=True)
class Section:
    """Lines ``[start, end)`` (0-based, end exclusive) and the heading path that applies."""

    start: int
    end: int
    heading: str | None


def _headings(lines: Sequence[str]) -> list[tuple[int, int, str]]:
    """(line index, level, title) of every real heading (outside fenced code blocks)."""
    found: list[tuple[int, int, str]] = []
    fence: str | None = None  # the opening fence string while inside a code block
    for i, line in enumerate(lines):
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            continue
        if fence is not None:
            continue
        match = _HEADING_RE.match(line)
        if match:
            title = _CLOSING_HASHES_RE.sub("", match.group(2) or "").strip()
            found.append((i, len(match.group(1)), title))
    return found


def split_markdown_sections(lines: Sequence[str]) -> list[Section]:
    """Split Markdown lines into sections at headings (see module docstring)."""
    headings = _headings(lines)
    n = len(lines)
    if not headings:
        return [Section(0, n, None)] if n else []

    # (section, body_has_text): the body excludes the heading line itself.
    raw: list[tuple[Section, bool]] = []
    if headings[0][0] > 0:  # preamble before the first heading
        preamble = Section(0, headings[0][0], None)
        raw.append((preamble, any(line.strip() for line in lines[: headings[0][0]])))
    stack: list[tuple[int, str]] = []
    for pos, (index, level, title) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        end = headings[pos + 1][0] if pos + 1 < len(headings) else n
        path = " > ".join(t for _, t in stack if t) or None
        has_body = any(line.strip() for line in lines[index + 1 : end])
        raw.append((Section(index, end, path), has_body))

    # A blank preamble is dropped; a heading-only section is merged into the one that follows.
    merged: list[Section] = []
    carry_start: int | None = None
    for pos, (section, has_body) in enumerate(raw):
        is_last = pos == len(raw) - 1
        if not has_body and not is_last:
            if section.heading is not None or section.start > 0:
                carry_start = section.start if carry_start is None else carry_start
            continue
        start = section.start if carry_start is None else carry_start
        merged.append(Section(start, section.end, section.heading))
        carry_start = None
    return merged
