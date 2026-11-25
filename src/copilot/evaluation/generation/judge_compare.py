"""Compare two judgement sets of the same generated answers (Milestone 11B).

A pilot has a handful of answers, so this is a *per-item* comparison, not a statistical test: every
answer is listed with both judges' scores, and a chance-corrected statistic (kappa) is offered only
from ``MIN_N_FOR_KAPPA`` answers upward - never for the eight-answer pilot.

Before any number is computed the two sets are verified to describe the same experiment: same
repository, protocol hash (which covers the benchmark digests and the generation model), the same
answered items with the same blind ids, and - for every judged answer - the same judge input hash
(a hash of the question, the reference and the answer text), so a difference between the judges can
only come from the judges. Any difference is reported as an identity problem and the comparison is
marked invalid.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from copilot.evaluation.generation.audit import agreement
from copilot.evaluation.generation.judging import JudgedItem, JudgementFile
from copilot.evaluation.generation.rubric import EvaluationError, score_assessment
from copilot.evaluation.generation.schema import GenerationQuestion

MIN_N_FOR_KAPPA = 30


def judge_label(file: JudgementFile) -> str:
    base = f"{file.judge_provider}/{file.judge_model}"
    if file.judge_runtime_fingerprint:
        return f"{base}@{file.judge_runtime_fingerprint[:12]}"
    return base


def identity_problems(a: JudgementFile, b: JudgementFile) -> list[str]:
    """Why the two files do not describe the same generated answers (empty when they do)."""
    problems: list[str] = []
    for name in (
        "repository_id",
        "protocol_hash",
        "generation_provider",
        "generation_model",
        "rubric_version",
        "judge_prompt_version",
    ):
        if getattr(a, name) != getattr(b, name):
            problems.append(f"{name} differs ({getattr(a, name)!r} vs {getattr(b, name)!r})")
    keys_a, keys_b = a.by_key(), b.by_key()
    for key in sorted(set(keys_a) ^ set(keys_b)):
        side = "first" if key in keys_a else "second"
        problems.append(f"{key[0]}/{key[1]} exists only in the {side} judgement set")
    for key in sorted(set(keys_a) & set(keys_b)):
        ia, ib = keys_a[key], keys_b[key]
        if ia.blind_id != ib.blind_id:
            problems.append(f"{key[0]}/{key[1]}: blind ids differ")
        if (ia.state == "rule") != (ib.state == "rule") or (ia.state == "skipped") != (
            ib.state == "skipped"
        ):
            problems.append(f"{key[0]}/{key[1]}: the answer was handled differently (rule/skipped)")
        ha = ia.record.input_sha256 if ia.record else None
        hb = ib.record.input_sha256 if ib.record else None
        if ha is not None and hb is not None and ha != hb:
            problems.append(f"{key[0]}/{key[1]}: the judge inputs differ (not the same answer)")
    return problems


def _valid(question: GenerationQuestion, item: JudgedItem) -> tuple[int | None, str | None]:
    """``(score, None)`` for a valid model judgement; ``(None, why)`` otherwise."""
    if item.state == "rule":
        return None, "rule-scored system abstention (not a judge rating)"
    if item.state != "judged" or item.assessment is None:
        return None, f"no valid judgement (state {item.state})"
    try:
        return score_assessment(question, item.assessment).score, None
    except EvaluationError as exc:
        return None, f"invalid assessment: {str(exc)[:80]}"


def compare_judgements(
    a: JudgementFile,
    b: JudgementFile,
    questions: Mapping[str, GenerationQuestion],
) -> dict[str, Any]:
    problems = identity_problems(a, b)
    keys_a, keys_b = a.by_key(), b.by_key()
    rows: list[dict[str, Any]] = []
    for key in sorted(set(keys_a) & set(keys_b)):
        question = questions.get(key[0])
        ia, ib = keys_a[key], keys_b[key]
        if question is None:
            problems.append(f"{key[0]}: not in the benchmark")
            continue
        score_a, why_a = _valid(question, ia)
        score_b, why_b = _valid(question, ib)
        rows.append(
            {
                "question_id": key[0],
                "arm": key[1],
                "answerable": question.is_answerable,
                "state_a": ia.state,
                "state_b": ib.state,
                "score_a": score_a,
                "score_b": score_b,
                "abstained_a": ia.assessment.abstained if ia.assessment else None,
                "abstained_b": ib.assessment.abstained if ib.assessment else None,
                "response_type_a": ia.assessment.response_type if ia.assessment else None,
                "response_type_b": ib.assessment.response_type if ib.assessment else None,
                "missing_a": why_a if score_a is None and ia.state != "rule" else None,
                "missing_b": why_b if score_b is None and ib.state != "rule" else None,
                "exact": None if score_a is None or score_b is None else score_a == score_b,
                "difference_b_minus_a": None
                if score_a is None or score_b is None
                else score_b - score_a,
            }
        )
    both = [r for r in rows if r["score_a"] is not None and r["score_b"] is not None]
    n = len(both)
    exact = sum(1 for r in both if r["exact"])
    two_point = sum(1 for r in both if abs(r["difference_b_minus_a"]) >= 2)
    matrix = [[0, 0, 0] for _ in range(3)]
    for r in both:
        matrix[r["score_a"]][r["score_b"]] += 1
    kappa: dict[str, Any]
    if n >= MIN_N_FOR_KAPPA:
        kappa = agreement(
            [r["score_a"] for r in both], [r["score_b"] for r in both], ordinal=True
        ).to_json()
    else:
        kappa = {
            "n": n,
            "kappa": None,
            "kappa_undefined_reason": (
                f"only {n} answer(s) were scored by both judges (< {MIN_N_FOR_KAPPA}); a "
                "chance-corrected statistic would be noise, so none is reported"
            ),
        }
    return {
        "kind": "judge-vs-judge comparison (not a validation of either judge)",
        "judge_a": judge_label(a),
        "judge_b": judge_label(b),
        "same_judge": judge_label(a) == judge_label(b),
        "identity_verified": not problems,
        "identity_problems": problems,
        "answers_in_both": len(rows),
        "scored_by_both": n,
        "missing_in_a": sorted(f"{r['question_id']}/{r['arm']}" for r in rows if r["missing_a"]),
        "missing_in_b": sorted(f"{r['question_id']}/{r['arm']}" for r in rows if r["missing_b"]),
        "rule_scored": sorted(
            f"{r['question_id']}/{r['arm']}" for r in rows if r["state_a"] == "rule"
        ),
        "exact_agreement": {"count": exact, "n": n, "fraction": exact / n if n else None},
        "two_point_disagreements": {"count": two_point, "n": n},
        "confusion_a_rows_b_columns_scores_0_1_2": matrix,
        "chance_corrected": kappa,
        "items": rows,
    }


def render_text(result: Mapping[str, Any]) -> str:
    lines = [
        f"A: {result['judge_a']}",
        f"B: {result['judge_b']}",
        f"identity verified: {result['identity_verified']}",
    ]
    lines += [f"  problem: {p}" for p in result["identity_problems"]]
    lines.append(f"{'question/arm':<34} {'A':>3} {'B':>3}  note")
    for r in result["items"]:
        note = r["missing_a"] or r["missing_b"] or ("" if r["exact"] else "DIFFERENT")
        sa = "-" if r["score_a"] is None else r["score_a"]
        sb = "-" if r["score_b"] is None else r["score_b"]
        lines.append(f"{r['question_id'] + '/' + r['arm']:<34} {sa!s:>3} {sb!s:>3}  {note}")
    e = result["exact_agreement"]
    lines.append(
        f"exact agreement {e['count']}/{e['n']}; two-point disagreements "
        f"{result['two_point_disagreements']['count']}; missing in A {len(result['missing_in_a'])}"
        f", missing in B {len(result['missing_in_b'])}"
    )
    lines.append(
        f"chance-corrected: {result['chance_corrected'].get('kappa_undefined_reason', '')}"
    )
    return "\n".join(lines)
