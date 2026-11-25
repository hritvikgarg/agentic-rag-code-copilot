"""Judge-versus-human agreement on a filled audit worksheet, with full row accounting.

Every worksheet row ends in exactly one bucket - eligible, unrated, invalid, duplicate, unexpected
or excluded-with-a-reason - and every bucket is reported with identifiers and counts; nothing is
skipped silently. Statistics use eligible rows only (a valid human rating *and* a valid judge
rating for the same item). Rule-scored system abstentions are **not** judge ratings: they are
verified by ``audit_rules`` and never enter these statistics.

Undefined chance-corrected statistics are reported as ``null`` with the reason next to them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from copilot.evaluation.generation.audit import agreement, parse_human_row
from copilot.evaluation.generation.judging import JudgedItem
from copilot.evaluation.generation.rubric import (
    EvaluationError,
    score_assessment,
    validate_assessment,
)
from copilot.evaluation.generation.schema import GenerationQuestion

ARMS = ("plain", "rag")


def evaluate_agreement(
    rows: Sequence[Mapping[str, str]],
    key: Sequence[Mapping[str, str]],
    questions: Mapping[str, Mapping[str, GenerationQuestion]],
    judged: Mapping[str, Mapping[tuple[str, str], JudgedItem]],
    *,
    rater: str,
    pilot: bool = False,
) -> dict[str, object]:
    """``questions``/``judged`` map a repository id to its questions / judgement items.

    ``pilot=True`` (a handful of answers) reports raw counts and per-item pairs and leaves every
    chance-corrected statistic undefined (``null`` with a reason): a kappa on eight answers is
    noise, so none is offered.
    """
    key_by_id: dict[str, Mapping[str, str]] = {}
    key_problems: list[dict[str, str]] = []
    for entry in key:
        item_id = entry.get("item_id", "")
        if item_id in key_by_id:
            key_problems.append({"item_id": item_id, "reason": "duplicate key entry"})
            continue
        key_by_id[item_id] = entry

    counts: dict[str, int] = {}
    for row in rows:
        item_id = row.get("item_id", "")
        counts[item_id] = counts.get(item_id, 0) + 1
    duplicates = sorted(i for i, n in counts.items() if n > 1)
    unexpected = sorted(i for i in counts if i not in key_by_id)
    missing = sorted(set(key_by_id) - set(counts))

    unrated: list[str] = []
    invalid: list[dict[str, str]] = []
    excluded: dict[str, list[str]] = {}
    judge_missing: list[str] = []
    human_rated = 0
    valid_judge_rows = 0
    eligible: list[tuple[GenerationQuestion, str, object, object]] = []
    eligible_ids: list[str] = []

    def exclude(reason: str, item_id: str) -> None:
        excluded.setdefault(reason, []).append(item_id)

    for row in rows:
        item_id = row.get("item_id", "")
        if item_id in duplicates or item_id not in key_by_id:
            continue
        entry = key_by_id[item_id]
        repo, qid, arm = (
            entry.get("repository_id", ""),
            entry.get("question_id", ""),
            entry.get("arm", ""),
        )
        if repo not in questions:
            exclude("repository_not_selected", item_id)
            continue
        question = questions[repo].get(qid)
        if question is None:
            exclude("unknown_question_id", item_id)
            continue
        if arm not in ARMS:
            exclude("invalid_arm_in_key", item_id)
            continue
        machine = judged.get(repo, {}).get((qid, arm))
        if machine is not None and machine.state == "rule":
            exclude("rule_scored_item_not_a_judge_rating", item_id)
            continue
        judge_ok = (
            machine is not None and machine.state == "judged" and machine.assessment is not None
        )
        if judge_ok:
            valid_judge_rows += 1
        try:
            human = parse_human_row(row, [f.id for f in question.key_facts], rater)
            if human is not None:
                validate_assessment(question, human)
        except EvaluationError as exc:
            invalid.append({"item_id": item_id, "reason": str(exc)[:200]})
            continue
        if human is None:
            unrated.append(item_id)
            continue
        human_rated += 1
        if not judge_ok:
            judge_missing.append(item_id)
            continue
        assert machine is not None and machine.assessment is not None
        eligible.append((question, arm, human, machine.assessment))
        eligible_ids.append(item_id)

    human_scores = [score_assessment(q, h).score for q, _, h, _ in eligible]  # type: ignore[arg-type]
    judge_scores = [score_assessment(q, j).score for q, _, _, j in eligible]  # type: ignore[arg-type]
    human_abs = [h.abstained for _, _, h, _ in eligible]  # type: ignore[attr-defined]
    judge_abs = [j.abstained for _, _, _, j in eligible]  # type: ignore[attr-defined]
    human_facts: list[bool] = []
    judge_facts: list[bool] = []
    for _, _, h, j in eligible:
        for fid, verdict in h.fact_verdicts.items():  # type: ignore[attr-defined]
            human_facts.append(verdict.value == "supported")
            judge_facts.append(j.fact_verdicts[fid].value == "supported")  # type: ignore[attr-defined]

    pairs = list(zip(human_scores, judge_scores, strict=True))
    exact = sum(1 for h, j in pairs if h == j)
    two_point = sum(1 for h, j in pairs if abs(h - j) >= 2)
    within_one = sum(1 for h, j in pairs if abs(h - j) <= 1)
    # Coverage: of the audited answers a model had to judge (rule-scored items are excluded), the
    # share that has a valid judgement - missing or failed judgements count against the judge.
    judge_expected = len(key_by_id) - len(excluded.get("rule_scored_item_not_a_judge_rating", []))
    per_item = [
        {
            "item_id": iid,
            "human_score": h,
            "judge_score": j,
            "human_abstained": ha,
            "judge_abstained": ja,
        }
        for iid, h, j, ha, ja in zip(
            eligible_ids, human_scores, judge_scores, human_abs, judge_abs, strict=True
        )
    ]

    problems = bool(invalid or duplicates or unexpected or missing or key_problems)
    blocks = {
        "correctness_score_0_2": agreement(human_scores, judge_scores, ordinal=True).to_json(),
        "correctness_score_0_2_unweighted": agreement(human_scores, judge_scores).to_json(),
        "abstained": agreement(human_abs, judge_abs).to_json(),
        "key_fact_supported": agreement(human_facts, judge_facts).to_json(),
    }
    if pilot:
        for block in blocks.values():
            block["kappa"] = None
            block["kappa_undefined_reason"] = (
                f"pilot sample (n={block['n']}) is too small for a chance-corrected statistic; "
                "only raw counts are reported"
            )
    return {
        "kind": "judge-vs-human agreement (rule-scored system abstentions are NOT included)",
        "rated_answers": len(eligible),
        "completeness": {
            "worksheet_rows": len(rows),
            "key_entries": len(key_by_id),
            "human_rated": human_rated,
            "unrated_rows": len(unrated),
            "rows_with_valid_judge_rating": valid_judge_rows,
            "eligible_rows": len(eligible),
            "failed_or_missing_judge_ratings": sorted(judge_missing),
            "invalid_ratings": invalid,
            "duplicate_item_ids": duplicates,
            "unexpected_item_ids": unexpected,
            "key_items_missing_from_worksheet": missing,
            "key_problems": key_problems,
            "other_exclusions": {k: sorted(v) for k, v in sorted(excluded.items())},
            "unrated_item_ids": sorted(unrated),
        },
        "rows_accounted_for": not problems,
        "fully_rated": not unrated and not problems,
        **blocks,
        "correctness_exact_agreement": {
            "n": len(pairs),
            "exact": exact,
            "fraction": exact / len(pairs) if pairs else None,
        },  # fmt: skip
        "correctness_within_one_point": {
            "n": len(pairs),
            "count": within_one,
            "fraction": within_one / len(pairs) if pairs else None,
        },  # fmt: skip
        "correctness_two_point_disagreements": {
            "n": len(pairs),
            "count": two_point,
            "fraction": two_point / len(pairs) if pairs else None,
        },  # fmt: skip
        "valid_judgement_coverage": {
            "answers_requiring_a_judgement": judge_expected,
            "with_a_valid_judgement": valid_judge_rows,
            "fraction": valid_judge_rows / judge_expected if judge_expected > 0 else None,
        },
        "pilot": pilot,
        "per_item": per_item,
        "note": "statistics use eligible rows only; kappa is null with a reason when undefined",
    }
