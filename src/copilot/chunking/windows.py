"""Line-window splitting: the core algorithm of the baseline chunking strategy (Strategy A).

Applies identically to every supported text file (source code, Markdown, JSON, YAML). It never
looks at language, syntax or document structure.

**Primary boundary:** a configurable window of ``size`` lines, advancing so that consecutive
windows share ``overlap`` lines.
**Safety boundary:** the estimated token count (``copilot.utils.tokens.estimate_tokens``) may
*shorten* a window so it stays within ``max_tokens`` before embedding. It only counts tokens; it
does not inspect structure. The estimate is provisional: it is not checked against the real
embedding tokenizer until Milestone 4.

Rules:

1. A window starts at line ``start`` and takes up to ``size`` lines; if that exceeds
   ``max_tokens`` it is shortened to the longest run of whole lines that fits.
2. The next window starts ``overlap`` lines before the previous end, but always after the previous
   start, and only windows that add at least one not-yet-covered line are emitted (so a cap-shrunk
   window never produces a cascade of near-duplicates).
3. A single physical line longer than ``max_tokens`` cannot be shortened by removing lines, so it
   is split deterministically into contiguous *fragments* on token boundaries (concatenating them
   reproduces the line exactly). Every fragment keeps that line's own number; a fragment is not a
   new source line.
4. Whitespace-only windows are dropped (nothing to embed or retrieve).

The baseline caveat is the point of the experiment: windows know nothing about program structure,
so a function can be cut in half. Structure-aware chunking (Milestone 9) is the comparison.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from copilot.utils.tokens import estimate_tokens, token_starts


@dataclass(frozen=True)
class Window:
    """A contiguous piece of a file. Line numbers are 1-based and inclusive."""

    start_line: int
    end_line: int
    content: str
    # Both None for a window of whole lines. For a piece of one over-long physical line both are
    # set: this is piece ``fragment_index`` (0-based) of ``fragment_count``, and
    # ``start_line == end_line`` is that line's own number (a fragment is not a new source line).
    fragment_index: int | None = None
    fragment_count: int | None = None


def split_lines(text: str) -> list[str]:
    """Split normalised text into lines on ``"\\n"`` only.

    A trailing newline terminates the last line rather than starting an empty one, so
    ``"a\\nb\\n"`` is two lines. (``str.splitlines`` is deliberately not used: it also splits on
    ``\\x0b``, ``\\x0c``, ``\\u2028`` and others, which would make line numbers disagree with
    editors and with ingestion's newline normalisation.)
    """
    if text == "":
        return []
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines


def split_long_line(line: str, max_tokens: int) -> list[str]:
    """Split ``line`` into pieces of at most ``max_tokens`` estimated tokens.

    Cuts happen at token starts, so no estimated token straddles two pieces and
    ``"".join(pieces) == line``.
    """
    starts = list(token_starts(line))
    cuts = starts[::max_tokens]
    cuts[0] = 0  # leading whitespace belongs to the first piece
    cuts.append(len(line))
    return [line[a:b] for a, b in zip(cuts, cuts[1:], strict=False)]


def split_into_windows(
    lines: Sequence[str],
    *,
    size: int,
    overlap: int,
    max_tokens: int,
    first_line_number: int = 1,
) -> list[Window]:
    """Split ``lines`` into windows (see module docstring for the rules)."""
    if size < 1 or not 0 <= overlap < size or max_tokens < 1:
        raise ValueError("require size >= 1, 0 <= overlap < size and max_tokens >= 1")

    n = len(lines)
    counts = [estimate_tokens(line) for line in lines]
    windows: list[Window] = []
    covered = 0  # lines [0, covered) are already part of an emitted (or dropped) window
    start = 0

    while start < n:
        if counts[start] > max_tokens:
            line_no = first_line_number + start
            pieces = split_long_line(lines[start], max_tokens)
            windows.extend(
                Window(line_no, line_no, piece, fragment_index=k, fragment_count=len(pieces))
                for k, piece in enumerate(pieces)
            )
            covered = max(covered, start + 1)
            start += 1
            continue

        end = min(start + size, n)
        tokens = sum(counts[start:end])
        while tokens > max_tokens:  # terminates with end > start: counts[start] <= max_tokens
            end -= 1
            tokens -= counts[end]

        if end <= covered:  # adds no new line: slide right instead of emitting a duplicate
            start += 1
            continue

        content = "\n".join(lines[start:end])
        if content.strip():
            windows.append(Window(first_line_number + start, first_line_number + end - 1, content))
        covered = end
        if end >= n:
            break
        start = max(start + 1, end - overlap)

    return windows
