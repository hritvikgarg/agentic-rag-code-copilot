"""Aggregate per-question scores per arm and compare arms with paired statistics."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from copilot.evaluation.chunking_experiment.scoring import QuestionScore
from copilot.evaluation.stats import (
    bootstrap_ci,
    bootstrap_indices,
    exact_mcnemar,
    holm_adjust,
    paired_table,
)

HIT_KS = (1, 3, 5, 10)
COVERAGE_KS = (5, 10)


def binary_endpoints(budgets: Sequence[int], coverage_threshold: float) -> dict[str, Any]:
    """``{endpoint name: function(QuestionScore) -> bool}`` for every binary endpoint."""
    endpoints: dict[str, Any] = {f"hit@{k}": (lambda s, k=k: s.hit_at(k)) for k in HIT_KS}
    for k in COVERAGE_KS:
        endpoints[f"coverage_hit@{k}"] = lambda s, k=k: s.coverage_at[k] >= coverage_threshold
    for b in budgets:
        endpoints[f"budget_hit@{b}"] = lambda s, b=b: s.budgets[b].hit
    return endpoints


def continuous_endpoints(budgets: Sequence[int]) -> dict[str, Any]:
    """``{endpoint name: function(QuestionScore) -> float}`` for every continuous endpoint."""
    endpoints: dict[str, Any] = {"mrr@10": lambda s: s.reciprocal_rank}
    for k in COVERAGE_KS:
        endpoints[f"coverage@{k}"] = lambda s, k=k: s.coverage_at[k]
    for b in budgets:
        endpoints[f"budget_coverage@{b}"] = lambda s, b=b: s.budgets[b].coverage
    return endpoints


def aggregate_arm(
    scores: Sequence[QuestionScore], budgets: Sequence[int], coverage_threshold: float
) -> dict[str, Any]:
    """Per-arm summary: counts, rates, MRR, coverage, retrieved size, budget results."""
    n = len(scores)
    ranks = [s.first_hit_rank for s in scores if s.first_hit_rank is not None]
    result: dict[str, Any] = {"n_questions": n, "binary": {}, "continuous": {}}
    for name, fn in binary_endpoints(budgets, coverage_threshold).items():
        count = sum(1 for s in scores if fn(s))
        result["binary"][name] = {"count": count, "rate": count / n}
    for name, fn in continuous_endpoints(budgets).items():
        result["continuous"][name] = statistics.fmean(fn(s) for s in scores)
    result["first_rank"] = {
        "questions_with_hit_in_top10": len(ranks),
        "mean": statistics.fmean(ranks) if ranks else None,
        "median": statistics.median(ranks) if ranks else None,
    }
    lines_per_hit = [s.first_hit_lines for s in scores if s.first_hit_lines is not None]
    all_lines = [x for s in scores for x in s.retrieved_lines]
    result["size"] = {
        "mean_lines_per_hit": statistics.fmean(lines_per_hit) if lines_per_hit else None,
        "mean_retrieved_lines": statistics.fmean(all_lines) if all_lines else None,
        "mean_retrieved_tokens_top10": statistics.fmean(s.retrieved_tokens for s in scores),
        "mean_distinct_files_top10": statistics.fmean(s.distinct_files for s in scores),
    }
    result["budgets"] = {
        str(b): {
            "mean_chunks": statistics.fmean(s.budgets[b].chunks for s in scores),
            "mean_distinct_files": statistics.fmean(s.budgets[b].distinct_files for s in scores),
            "mean_tokens": statistics.fmean(s.budgets[b].tokens for s in scores),
            "mean_utilization": statistics.fmean(s.budgets[b].utilization for s in scores),
            "mean_dropped_for_budget": statistics.fmean(
                s.budgets[b].dropped_for_budget for s in scores
            ),
        }
        for b in budgets
    }
    return result


def compare_arms(
    scores: Mapping[str, Sequence[QuestionScore]],
    *,
    comparisons: Sequence[Sequence[str]],
    budgets: Sequence[int],
    coverage_threshold: float,
    primary: Mapping[str, Any],
    seed: int,
    resamples: int,
    confidence: float,
) -> dict[str, Any]:
    """Paired comparison of every ``(X, Y)`` in ``comparisons`` (difference is X minus Y)."""
    arms = list(scores)
    n = len(scores[arms[0]])
    ids = [s.question_id for s in scores[arms[0]]]
    for arm in arms:
        if [s.question_id for s in scores[arm]] != ids:
            raise ValueError("every arm must be scored on the same questions in the same order")
    indices = bootstrap_indices(n, resamples, seed)
    binary = binary_endpoints(budgets, coverage_threshold)
    continuous = continuous_endpoints(budgets)

    out: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    primary_key = f"{primary['comparison'][0]}-{primary['comparison'][1]}/{primary['endpoint']}"
    for x_arm, y_arm in comparisons:
        key = f"{x_arm}-{y_arm}"
        entry: dict[str, Any] = {"x": x_arm, "y": y_arm, "binary": {}, "continuous": {}}
        for name, fn in binary.items():
            x = [bool(fn(s)) for s in scores[x_arm]]
            y = [bool(fn(s)) for s in scores[y_arm]]
            table = paired_table(x, y)
            diffs = [float(a) - float(b) for a, b in zip(x, y, strict=True)]
            lo, hi = bootstrap_ci(diffs, indices, confidence)
            p = exact_mcnemar(table.only_x, table.only_y)
            full = f"{key}/{name}"
            raw_p[full] = p
            entry["binary"][name] = {
                "count_x": sum(x),
                "count_y": sum(y),
                "difference_pp": 100 * table.difference,
                "ci_pp": [100 * lo, 100 * hi],
                "table": table.to_json(),
                "improved": table.only_x,
                "regressed": table.only_y,
                "mcnemar_p": p,
            }
        for name, fn in continuous.items():
            diffs = [
                float(fn(a)) - float(fn(b))
                for a, b in zip(scores[x_arm], scores[y_arm], strict=True)
            ]
            lo, hi = bootstrap_ci(diffs, indices, confidence)
            entry["continuous"][name] = {
                "mean_x": statistics.fmean(fn(s) for s in scores[x_arm]),
                "mean_y": statistics.fmean(fn(s) for s in scores[y_arm]),
                "mean_difference": statistics.fmean(diffs),
                "ci": [lo, hi],
            }
        out[key] = entry
    family = {k: p for k, p in raw_p.items() if k != primary_key}
    adjusted = holm_adjust(family)
    for key, entry in out.items():
        for name, row in entry["binary"].items():
            full = f"{key}/{name}"
            row["is_primary"] = full == primary_key
            row["holm_p"] = None if full == primary_key else adjusted[full]
    return {
        "primary": dict(primary),
        "primary_key": primary_key,
        "bootstrap": {"seed": seed, "resamples": resamples, "confidence": confidence},
        "holm_family_size": len(family),
        "comparisons": out,
    }
