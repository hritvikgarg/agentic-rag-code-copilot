"""Supplementary, coverage-based relevance (reported separately from the primary hit rule).

*Region coverage* of a set of retrieved chunks for one question is the fraction of the question's
ground-truth lines (all its regions together) that lie inside at least one of the chunks (same
file). A coverage hit is ``coverage >= threshold``. Coverage rewards chunks that contain the whole
region, so like the primary any-overlap rule it does not remove the preference for large chunks.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from copilot.evaluation.benchmark import Region
from copilot.evaluation.matching import Located


def region_coverage(chunks: Iterable[Located], regions: Sequence[Region]) -> float:
    """Fraction of the regions' lines covered by the union of ``chunks`` (0.0 if no lines)."""
    chunks = tuple(chunks)
    total = 0
    covered = 0
    for region in regions:
        lines = range(region.start_line, region.end_line + 1)
        total += len(lines)
        spans = [(c.start_line, c.end_line) for c in chunks if c.file_path == region.file_path]
        covered += sum(1 for n in lines if any(a <= n <= b for a, b in spans))
    return covered / total if total else 0.0
