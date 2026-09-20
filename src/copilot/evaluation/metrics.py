"""Hit@k, MRR and the chunk-size confound metrics.

Definitions (all computed over the ranked results retrieved to ``depth``, default 10):

* ``Hit@k``: the fraction of questions whose first matching result has rank <= k.
* ``MRR``: the mean over questions of ``1 / rank_of_first_hit`` (0 when there is no hit within
  ``depth``). Truncated at ``depth``: a hit at rank 11 counts as a miss.
* ``mean_lines_per_hit``: the mean, over questions that have a hit, of the number of physical source
  lines in the **first matching chunk**. One chunk per question, so overlapping duplicates are not
  counted twice.
* ``mean_retrieved_lines``: the mean line count of *all* retrieved chunks (all questions, ranks
  1..depth): a size profile of what the configuration returns.

Why the last two exist: with an overlap-based hit rule, a larger chunk covers more lines and so
overlaps a small ground-truth region more easily. Higher Hit@k with much larger chunks is not
evidence of better retrieval. No significance is claimed anywhere; with a few dozen questions a
one-question difference moves Hit@k by several points.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_KS = (1, 3, 5, 10)


@dataclass(frozen=True)
class QuestionOutcome:
    """What one question retrieved and where (if anywhere) it first hit."""

    question_id: str
    first_hit_rank: int | None  # None: no hit within the retrieved depth
    first_hit_lines: int | None  # physical lines of the first matching chunk
    retrieved_locations: tuple[str, ...]  # "path:start-end" per rank
    retrieved_lines: tuple[int, ...]  # physical lines per rank


@dataclass(frozen=True)
class Summary:
    """Aggregate metrics over a set of question outcomes."""

    n_questions: int
    depth: int
    hit_at: dict[int, float]  # k -> fraction of questions
    hit_counts: dict[int, int]  # k -> number of questions
    mrr: float
    mean_lines_per_hit: float | None  # None when nothing hit
    mean_retrieved_lines: float | None  # None when nothing was retrieved


def summarize(
    outcomes: Sequence[QuestionOutcome], ks: Sequence[int] = DEFAULT_KS, *, depth: int | None = None
) -> Summary:
    """Aggregate ``outcomes``. Raises ``ValueError`` for no outcomes or an invalid ``k``."""
    if not outcomes:
        raise ValueError("cannot summarise zero questions")
    if any(k < 1 for k in ks):
        raise ValueError("every k must be >= 1")
    depth = depth if depth is not None else max(ks)
    n = len(outcomes)
    ranks = [o.first_hit_rank for o in outcomes]
    if any(r is not None and r < 1 for r in ranks):
        raise ValueError("ranks are 1-based")
    counts = {k: sum(1 for r in ranks if r is not None and r <= k) for k in ks}
    reciprocal = [1.0 / r if r is not None and r <= depth else 0.0 for r in ranks]
    hit_lines = [o.first_hit_lines for o in outcomes if o.first_hit_lines is not None]
    all_lines = [count for o in outcomes for count in o.retrieved_lines]
    return Summary(
        n_questions=n,
        depth=depth,
        hit_at={k: counts[k] / n for k in ks},
        hit_counts=counts,
        mrr=sum(reciprocal) / n,
        mean_lines_per_hit=sum(hit_lines) / len(hit_lines) if hit_lines else None,
        mean_retrieved_lines=sum(all_lines) / len(all_lines) if all_lines else None,
    )
