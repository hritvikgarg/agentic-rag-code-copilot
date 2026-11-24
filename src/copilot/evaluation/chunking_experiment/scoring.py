"""Score one question for one arm: fixed-k metrics and the fixed-context-budget simulation.

Input is the ranked retrieval *pool* (top ``POOL_DEPTH`` chunks, deterministic exact search). The
fixed-k metrics use its first ``depth`` (10) results; the budget simulation uses the whole pool
through the production ``build_context`` (retrieval only, no LLM).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from copilot.evaluation.benchmark import BenchmarkQuestion
from copilot.evaluation.chunking_experiment.relevance import region_coverage
from copilot.evaluation.matching import first_hit_rank, region_overlaps
from copilot.rag.context import build_context
from copilot.retrieval import RetrievalResult

DEPTH = 10


@dataclass(frozen=True)
class BudgetScore:
    """What a context of at most ``budget`` estimated tokens contains for one question."""

    budget: int
    hit: bool  # some included chunk overlaps a ground-truth region
    coverage: float
    chunks: int
    distinct_files: int
    tokens: int
    utilization: float
    included_chunk_ids: tuple[str, ...]
    dropped_for_budget: int

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> BudgetScore:
        return cls(
            budget=int(data["budget"]),
            hit=bool(data["hit"]),
            coverage=float(data["coverage"]),
            chunks=int(data["chunks"]),
            distinct_files=int(data["distinct_files"]),
            tokens=int(data["tokens"]),
            utilization=float(data["utilization"]),
            included_chunk_ids=tuple(data["included_chunk_ids"]),
            dropped_for_budget=int(data["dropped_for_budget"]),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "budget": self.budget,
            "hit": self.hit,
            "coverage": self.coverage,
            "chunks": self.chunks,
            "distinct_files": self.distinct_files,
            "tokens": self.tokens,
            "utilization": self.utilization,
            "included_chunk_ids": list(self.included_chunk_ids),
            "dropped_for_budget": self.dropped_for_budget,
        }


@dataclass(frozen=True)
class QuestionScore:
    """All measurements of one question under one arm."""

    question_id: str
    first_hit_rank: int | None  # within the top DEPTH, primary any-overlap rule
    first_hit_lines: int | None
    coverage_at: dict[int, float]
    retrieved: tuple[dict[str, object], ...]  # top DEPTH, for inspection
    retrieved_tokens: int  # estimated tokens of the top-DEPTH chunks
    retrieved_lines: tuple[int, ...]
    distinct_files: int
    budgets: dict[int, BudgetScore]

    def hit_at(self, k: int) -> bool:
        return self.first_hit_rank is not None and self.first_hit_rank <= k

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.first_hit_rank if self.first_hit_rank is not None else 0.0

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> QuestionScore:
        return cls(
            question_id=str(data["question_id"]),
            first_hit_rank=data["first_hit_rank"],
            first_hit_lines=data["first_hit_lines"],
            coverage_at={int(k): float(v) for k, v in data["coverage_at"].items()},
            retrieved=tuple(data["retrieved"]),
            retrieved_tokens=int(data["retrieved_tokens"]),
            retrieved_lines=tuple(data["retrieved_lines"]),
            distinct_files=int(data["distinct_files"]),
            budgets={int(k): BudgetScore.from_json(v) for k, v in data["budgets"].items()},
        )

    def to_json(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "first_hit_rank": self.first_hit_rank,
            "first_hit_lines": self.first_hit_lines,
            "reciprocal_rank": self.reciprocal_rank,
            "coverage_at": {str(k): v for k, v in self.coverage_at.items()},
            "retrieved_tokens": self.retrieved_tokens,
            "retrieved_lines": list(self.retrieved_lines),
            "distinct_files": self.distinct_files,
            "retrieved": list(self.retrieved),
            "budgets": {str(b): s.to_json() for b, s in self.budgets.items()},
        }


def _retrieved_row(result: RetrievalResult, question: BenchmarkQuestion) -> dict[str, object]:
    return {
        "rank": result.rank,
        "chunk_id": result.chunk_id,
        "file_path": result.file_path,
        "start_line": result.start_line,
        "end_line": result.end_line,
        "score": result.score,
        "token_estimate": result.token_estimate,
        "chunk_type": result.chunk_type.value,
        "qualified_name": result.qualified_name,
        "relevant": any(region_overlaps(result, r) for r in question.relevant),
    }


def score_budget(
    pool: Sequence[RetrievalResult], question: BenchmarkQuestion, budget: int
) -> BudgetScore:
    """Select evidence with the production context builder and score it."""
    built = build_context(pool, budget)
    included = built.included
    return BudgetScore(
        budget=budget,
        hit=any(region_overlaps(r, region) for r in included for region in question.relevant),
        coverage=region_coverage(included, question.relevant),
        chunks=len(included),
        distinct_files=len({r.file_path for r in included}),
        tokens=built.estimated_tokens,
        utilization=built.estimated_tokens / budget,
        included_chunk_ids=tuple(r.chunk_id for r in included),
        dropped_for_budget=sum(1 for d in built.dropped if d.reason == "budget"),
    )


def score_question(
    pool: Sequence[RetrievalResult],
    question: BenchmarkQuestion,
    *,
    budgets: Sequence[int],
    coverage_ks: Sequence[int] = (5, 10),
) -> QuestionScore:
    """Score the ranked ``pool`` (rank 1 first) for ``question``."""
    top = list(pool[:DEPTH])
    rank = first_hit_rank(top, question.relevant)
    return QuestionScore(
        question_id=question.id,
        first_hit_rank=rank,
        first_hit_lines=top[rank - 1].line_count if rank is not None else None,
        coverage_at={k: region_coverage(top[:k], question.relevant) for k in coverage_ks},
        retrieved=tuple(_retrieved_row(r, question) for r in top),
        retrieved_tokens=sum(r.token_estimate for r in top),
        retrieved_lines=tuple(r.line_count for r in top),
        distinct_files=len({r.file_path for r in top}),
        budgets={b: score_budget(pool, question, b) for b in budgets},
    )
