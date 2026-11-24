"""Paired statistics for comparing two retrieval configurations on the same questions.

Everything here is deterministic and dependency-light (``numpy`` only; no SciPy):

* ``paired_table``     the 2x2 contingency table of two binary outcomes per question;
* ``exact_mcnemar``    the exact two-sided McNemar test (binomial on the discordant pairs);
* ``holm_adjust``      Holm step-down adjustment of a family of p-values;
* ``bootstrap_indices`` / ``bootstrap_ci``  a *paired* percentile bootstrap: questions are resampled
  with replacement, the same draw is applied to every metric and comparison.

Interpretation limits (see ``docs/experiments/chunking-comparison.md``): the questions are a small,
non-random benchmark, so intervals describe this benchmark, not a population; a non-significant
result is not evidence of equivalence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedTable:
    """Paired binary outcomes of configuration X versus Y over the same questions."""

    both: int  # X and Y succeeded
    only_x: int  # X succeeded, Y did not
    only_y: int  # Y succeeded, X did not
    neither: int

    @property
    def n(self) -> int:
        return self.both + self.only_x + self.only_y + self.neither

    @property
    def discordant(self) -> int:
        return self.only_x + self.only_y

    @property
    def difference(self) -> float:
        """Success rate of X minus success rate of Y (a fraction; multiply by 100 for points)."""
        return (self.only_x - self.only_y) / self.n if self.n else 0.0

    def to_json(self) -> dict[str, int]:
        return {
            "both": self.both,
            "only_x": self.only_x,
            "only_y": self.only_y,
            "neither": self.neither,
        }


def paired_table(x: Sequence[bool], y: Sequence[bool]) -> PairedTable:
    """The paired table of two equally long binary outcome sequences."""
    if len(x) != len(y):
        raise ValueError("paired outcomes must have the same length")
    both = sum(1 for a, b in zip(x, y, strict=True) if a and b)
    only_x = sum(1 for a, b in zip(x, y, strict=True) if a and not b)
    only_y = sum(1 for a, b in zip(x, y, strict=True) if b and not a)
    return PairedTable(both, only_x, only_y, len(x) - both - only_x - only_y)


def exact_mcnemar(only_x: int, only_y: int) -> float:
    """Exact two-sided McNemar p-value: ``min(1, 2 * P(Binomial(n, 1/2) <= min(b, c)))``.

    ``n = b + c`` is the number of discordant pairs. With no discordant pair the p-value is 1.
    Integer arithmetic only, so it is exact for any realistic ``n``.
    """
    if only_x < 0 or only_y < 0:
        raise ValueError("counts must be non-negative")
    n = only_x + only_y
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_x, only_y) + 1))
    return min(1.0, 2 * tail / 2**n)


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (same keys). Controls the family-wise error rate."""
    ordered = sorted(p_values.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[key] = running
    return {key: adjusted[key] for key in p_values}


def bootstrap_indices(n_items: int, resamples: int, seed: int) -> np.ndarray:
    """``(resamples, n_items)`` question indices drawn with replacement (deterministic per seed)."""
    if n_items < 1 or resamples < 1:
        raise ValueError("need at least one item and one resample")
    return np.random.default_rng(seed).integers(0, n_items, size=(resamples, n_items))


def bootstrap_ci(
    values: Sequence[float], indices: np.ndarray, confidence: float = 0.95
) -> tuple[float, float]:
    """Percentile CI of the mean of ``values`` over the shared resampled ``indices``."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    data = np.asarray(values, dtype=float)
    if data.shape[0] != indices.shape[1]:
        raise ValueError("values and bootstrap indices disagree on the number of questions")
    means = data[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)
