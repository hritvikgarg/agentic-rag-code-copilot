"""The hit rule: same file and at least one physical line in common.

A retrieved chunk matches a ground-truth region when ``file_path`` is equal **and** the
inclusive line ranges share at least one line. Adjacent ranges (one ends on line 10, the
other starts on 11) do not match. Chunk ids are never compared. See ``docs/evaluation.md``
for why this rule favours large chunks (a big chunk overlaps a small region more easily) and
how that confound is reported.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

from copilot.evaluation.benchmark import Region


class Located(Protocol):
    """Anything with a repository-relative path and an inclusive line range."""

    file_path: str
    start_line: int
    end_line: int


def region_overlaps(item: Located, region: Region) -> bool:
    """True if ``item`` is in ``region``'s file and shares at least one line with it."""
    return (
        item.file_path == region.file_path
        and item.start_line <= region.end_line
        and region.start_line <= item.end_line
    )


def first_hit_rank(results: Sequence[Located], regions: Iterable[Region]) -> int | None:
    """1-based rank of the first result that matches any region, or ``None`` if none does."""
    regions = tuple(regions)
    for rank, result in enumerate(results, start=1):
        if any(region_overlaps(result, region) for region in regions):
            return rank
    return None
