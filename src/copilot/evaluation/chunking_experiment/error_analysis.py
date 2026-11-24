"""Per-question error analysis tables derived from the saved results (no retrieval is run).

Categories (primary hit rule, Hit@5 unless stated): questions B improved / regressed against A;
questions every arm missed in the top 10; for every question/arm the share of top-5 chunks that come
from documentation (Markdown) files, the number of top-10 chunks that repeat a symbol (fragments of
one definition), and the redundancy of the retrieved set (sum of chunk lines over distinct lines).
The narrative interpretation is written by hand in the results document, from these records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

DOC_SUFFIXES = (".md", ".rst", ".txt")
REQUIRED_ARMS = ("A", "B", "C", "D")


def _hit5(row: Mapping[str, Any]) -> bool:
    rank = row["first_hit_rank"]
    return rank is not None and rank <= 5


def _redundancy(retrieved: Sequence[Mapping[str, Any]]) -> float:
    slots = 0
    distinct: set[tuple[str, int]] = set()
    for r in retrieved:
        slots += r["end_line"] - r["start_line"] + 1
        distinct.update((r["file_path"], n) for n in range(r["start_line"], r["end_line"] + 1))
    return slots / len(distinct) if distinct else 1.0


def question_profile(row: Mapping[str, Any]) -> dict[str, Any]:
    """Diagnostics of one question under one arm."""
    top5 = row["retrieved"][:5]
    docs = sum(1 for r in top5 if r["file_path"].lower().endswith(DOC_SUFFIXES))
    names = [r["qualified_name"] for r in row["retrieved"] if r["qualified_name"]]
    repeated = sum(1 for n in names if names.count(n) > 1)
    return {
        "first_hit_rank": row["first_hit_rank"],
        "hit@5": _hit5(row),
        "docs_in_top5": docs,
        "repeated_symbol_chunks_top10": repeated,
        "retrieved_redundancy_top10": _redundancy(row["retrieved"]),
        "first_hit_lines": row["first_hit_lines"],
        "top5": [
            {
                "location": f"{r['file_path']}:{r['start_line']}-{r['end_line']}",
                "tokens": r["token_estimate"],
                "type": r["chunk_type"],
                "symbol": r["qualified_name"],
                "relevant": r["relevant"],
            }
            for r in top5
        ],
    }


def analyse(
    per_question: Mapping[str, Sequence[Mapping[str, Any]]],
    regions: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Group questions by outcome pattern and attach per-arm diagnostics.

    ``regions`` maps a question id to its ground-truth locations (``path:start-end`` strings).
    """
    missing = [arm for arm in REQUIRED_ARMS if arm not in per_question]
    if missing:
        raise ValueError(f"error analysis needs arms {REQUIRED_ARMS}; missing {missing}")
    arms = list(per_question)
    ids = [r["question_id"] for r in per_question[arms[0]]]
    by_arm = {arm: {r["question_id"]: r for r in rows} for arm, rows in per_question.items()}
    questions: dict[str, Any] = {}
    for qid in ids:
        questions[qid] = {
            "expected": list(regions.get(qid, ())),
            "arms": {arm: question_profile(by_arm[arm][qid]) for arm in arms},
        }

    def hit5(arm: str, qid: str) -> bool:
        return questions[qid]["arms"][arm]["hit@5"]

    groups: dict[str, list[str]] = {
        "b_improved_over_a": [q for q in ids if hit5("B", q) and not hit5("A", q)],
        "b_regressed_vs_a": [q for q in ids if hit5("A", q) and not hit5("B", q)],
        "d_improved_over_a": [q for q in ids if hit5("D", q) and not hit5("A", q)],
        "d_regressed_vs_a": [q for q in ids if hit5("A", q) and not hit5("D", q)],
        "b_improved_over_c": [q for q in ids if hit5("B", q) and not hit5("C", q)],
        "b_regressed_vs_c": [q for q in ids if hit5("C", q) and not hit5("B", q)],
        "all_arms_miss_top10": [
            q for q in ids if all(questions[q]["arms"][a]["first_hit_rank"] is None for a in arms)
        ],
        "all_arms_hit_top5": [q for q in ids if all(hit5(a, q) for a in arms)],
    }
    summary = {}
    for arm in arms:
        profiles = [questions[q]["arms"][arm] for q in ids]
        misses = [p for p in profiles if not p["hit@5"]]
        summary[arm] = {
            "mean_docs_in_top5": sum(p["docs_in_top5"] for p in profiles) / len(profiles),
            "misses_with_3plus_docs_in_top5": sum(1 for p in misses if p["docs_in_top5"] >= 3),
            "mean_repeated_symbol_chunks_top10": sum(
                p["repeated_symbol_chunks_top10"] for p in profiles
            )
            / len(profiles),
            "mean_retrieved_redundancy_top10": sum(
                p["retrieved_redundancy_top10"] for p in profiles
            )
            / len(profiles),
        }
    return {"groups": groups, "arm_summary": summary, "questions": questions}
