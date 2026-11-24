"""Size matching of the control arms (Task 4), driven by chunk statistics only.

Policy (fixed before any retrieval was run; see ``docs/experiments/chunking-comparison.md``):

* **Arm C** (``line_sized``): baseline line windows with *no overlap* on Python files, window size
  searched over the bounded integer grid ``C_SIZE_GRID``. Target: the Python chunk-size
  distribution of **Arm B** (original AST). Overlap 0 is fixed because Arm B has none; matching
  on redundancy therefore needs no search dimension.
* **Arm D** (``ast_merged``): the AST chunker with merging, ``merge_max_tokens`` searched over the
  bounded grid ``D_MERGE_GRID`` (never above the 512 cap). Target: the Python chunk-size
  distribution of **Arm A** (original baseline).
* Selection is lexicographic on (|relative error of the median|, |relative error of the mean|,
  smaller parameter). The match is called *adequate* when the median error is within
  ``TOLERANCE``; otherwise the closest feasible configuration is reported as such.
* Only Python chunks are matched. Non-Python files are cut identically (baseline 60/10/512) in all
  four arms, so all differences stay confined to Python chunking.

No retrieval result is read or produced here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from copilot.chunking.base import Chunker, chunk_repository
from copilot.chunking.experimental import MergedAstChunker, SizedLineChunker
from copilot.evaluation.chunking_experiment.profile import SizeProfile, profile_by_scope
from copilot.models.ingestion import IngestionResult

TOLERANCE = 0.10
C_SIZE_GRID = tuple(range(4, 61))  # window sizes in lines (60 = the baseline's own size)
D_MERGE_GRID = tuple(range(128, 513, 16))  # merge budgets in estimated tokens (512 = the cap)


@dataclass(frozen=True)
class Candidate:
    """One evaluated grid point."""

    parameter: int
    python: SizeProfile
    median_error: float  # (candidate - target) / target, signed
    mean_error: float


@dataclass(frozen=True)
class MatchResult:
    """Outcome of one bounded search."""

    arm: str
    target_arm: str
    parameter_name: str
    grid: tuple[int, ...]
    target: SizeProfile
    candidates: tuple[Candidate, ...]
    selected: Candidate
    adequate: bool
    tolerance: float

    def to_json(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "target_arm": self.target_arm,
            "parameter_name": self.parameter_name,
            "grid": list(self.grid),
            "tolerance": self.tolerance,
            "adequate": self.adequate,
            "target_python": self.target.to_json(),
            "selected_parameter": self.selected.parameter,
            "selected_median_error": self.selected.median_error,
            "selected_mean_error": self.selected.mean_error,
            "selected_python": self.selected.python.to_json(),
            "candidates": [
                {
                    "parameter": c.parameter,
                    "median_error": c.median_error,
                    "mean_error": c.mean_error,
                    "tokens_median": c.python.tokens_median,
                    "tokens_mean": c.python.tokens_mean,
                    "chunk_count": c.python.chunk_count,
                    "redundancy": c.python.redundancy,
                }
                for c in self.candidates
            ],
        }


def _relative(value: float, target: float) -> float:
    return (value - target) / target if target else 0.0


def select(candidates: Sequence[Candidate]) -> Candidate:
    """Lexicographic best: smallest |median error|, then |mean error|, then smallest parameter."""
    return min(candidates, key=lambda c: (abs(c.median_error), abs(c.mean_error), c.parameter))


def _search(
    *,
    arm: str,
    target_arm: str,
    parameter_name: str,
    grid: Sequence[int],
    target: SizeProfile,
    ingestion: IngestionResult,
    make: Callable[[int], Chunker],
) -> MatchResult:
    candidates = []
    for parameter in grid:
        chunks = chunk_repository(ingestion, make(parameter)).chunks
        python = profile_by_scope(chunks, ingestion.files)["python"]
        candidates.append(
            Candidate(
                parameter,
                python,
                _relative(python.tokens_median, target.tokens_median),
                _relative(python.tokens_mean, target.tokens_mean),
            )
        )
    best = select(candidates)
    return MatchResult(
        arm=arm,
        target_arm=target_arm,
        parameter_name=parameter_name,
        grid=tuple(grid),
        target=target,
        candidates=tuple(candidates),
        selected=best,
        adequate=abs(best.median_error) <= TOLERANCE,
        tolerance=TOLERANCE,
    )


def match_arm_c(
    ingestion: IngestionResult,
    target: SizeProfile,
    *,
    base_size: int,
    base_overlap: int,
    max_tokens: int,
) -> MatchResult:
    """Search the window size of Arm C against Arm B's Python profile."""
    return _search(
        arm="C",
        target_arm="B",
        parameter_name="python_size_lines",
        grid=C_SIZE_GRID,
        target=target,
        ingestion=ingestion,
        make=lambda size: SizedLineChunker(
            python_size_lines=size,
            python_overlap_lines=0,
            size_lines=base_size,
            overlap_lines=base_overlap,
            max_tokens=max_tokens,
        ),
    )


def match_arm_d(
    ingestion: IngestionResult,
    target: SizeProfile,
    *,
    base_size: int,
    base_overlap: int,
    max_tokens: int,
) -> MatchResult:
    """Search the merge budget of Arm D against Arm A's Python profile."""
    return _search(
        arm="D",
        target_arm="A",
        parameter_name="merge_max_tokens",
        grid=[g for g in D_MERGE_GRID if g <= max_tokens],
        target=target,
        ingestion=ingestion,
        make=lambda budget: MergedAstChunker(
            max_tokens=max_tokens,
            fallback_size_lines=base_size,
            fallback_overlap_lines=base_overlap,
            merge_max_tokens=budget,
        ),
    )
