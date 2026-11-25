"""Who can be audited, who cannot, and why - the population behind a manual-audit sample.

The audit worksheet is a *sample of paired answers*. Only a question whose two arms both have
auditable text and a valid judge rating can enter it, so the sampled share must always be reported
against the eligible population **and** the full benchmark; otherwise a "20 % audit" can quietly be
a much smaller share of the experiment. This module classifies every benchmark question, counts
every exclusion with its reason, and reports whether the experiment is finalized.

Exclusion reasons (a question can have several; its *primary* reason is the first that applies, in
this fixed order, so the primary counts sum to the number of excluded questions):

* ``generation_failed``       an arm's generation call failed after its retries;
* ``not_attempted``           an arm was never attempted (a stopped or partial run);
* ``no_text``                 an arm succeeded but returned no text;
* ``rule_scored_system_abstention``  the RAG system abstained without calling the model; there is
  no model text to audit - these items are checked by a separate human verification worksheet
  (``audit_rules``), never by the judge-vs-human worksheet;
* ``judge_rating_missing``    an auditable answer has no valid judge rating (failed or not judged).

Finalization: an *official* sample may be drawn only when no side is ``not_attempted`` and every
auditable answer has a judge rating. Otherwise a sample is allowed only as ``provisional`` (for
development) and every artifact says so.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from copilot.evaluation.generation.analysis import RepoInputs
from copilot.evaluation.generation.audit import AuditUnit
from copilot.evaluation.generation.execution import SideRecord
from copilot.evaluation.generation.judging import JudgedItem
from copilot.evaluation.generation.rubric import EvaluationError, score_assessment
from copilot.evaluation.generation.schema import GenerationQuestion

PRIMARY_ORDER = (
    "generation_failed",
    "not_attempted",
    "no_text",
    "rule_scored_system_abstention",
    "judge_rating_missing",
)
ARMS = ("plain", "rag")


@dataclass(frozen=True)
class QuestionStatus:
    repository_id: str
    question_id: str
    answerable: bool
    reasons: tuple[str, ...]  # every applicable reason, in PRIMARY_ORDER

    @property
    def eligible(self) -> bool:
        return not self.reasons

    @property
    def primary_reason(self) -> str | None:
        return self.reasons[0] if self.reasons else None

    def unit(self) -> AuditUnit:
        return AuditUnit(self.repository_id, self.question_id, self.answerable)


def _valid_rating(question: GenerationQuestion, item: JudgedItem | None) -> bool:
    if item is None or item.state != "judged" or item.assessment is None:
        return False
    try:
        score_assessment(question, item.assessment)
    except EvaluationError:
        return False
    return True


def _side_reasons(side: SideRecord | None) -> list[str]:
    if side is None or side.state == "not_attempted":
        return ["not_attempted"]
    if side.state == "failed":
        return ["generation_failed"]
    if side.system_insufficient:
        return ["rule_scored_system_abstention"]
    if not side.has_text:
        return ["no_text"]
    return []


def classify_questions(
    inputs: Sequence[RepoInputs], *, require_judge_rating: bool = True
) -> list[QuestionStatus]:
    """One status per benchmark question of every repository in ``inputs``.

    ``require_judge_rating=False`` (the blinded pilot worksheet, drawn *before* any judging) makes
    eligibility depend on the generated answers only: the human rates every answer with text.
    """
    statuses: list[QuestionStatus] = []
    for item in inputs:
        entries = {e.question_id: e for e in item.run.entries}
        judged = item.judgements.by_key() if item.judgements else {}
        for question_id, question in sorted(item.questions.items()):
            entry = entries.get(question_id)
            reasons: set[str] = set()
            for arm in ARMS:
                side = getattr(entry, arm) if entry is not None else None
                side_reasons = _side_reasons(side)
                reasons.update(side_reasons)
                if (
                    require_judge_rating
                    and not side_reasons
                    and not _valid_rating(question, judged.get((question_id, arm)))
                ):
                    reasons.add("judge_rating_missing")
            statuses.append(
                QuestionStatus(
                    item.run.repository_id,
                    question_id,
                    question.is_answerable,
                    tuple(r for r in PRIMARY_ORDER if r in reasons),
                )  # fmt: skip
            )
    return statuses


def pilot_problems(inputs: Sequence[RepoInputs]) -> list[str]:
    """Why a pilot worksheet cannot be drawn yet: generation incomplete. Judge-independent."""
    problems: list[str] = []
    for item in inputs:
        repo = item.run.repository_id
        not_attempted = sum(
            1 for e in item.run.entries for arm in ARMS if getattr(e, arm).state == "not_attempted"
        )
        missing_entries = len(item.questions) - len(item.run.entries)
        if not_attempted or missing_entries:
            problems.append(f"{repo}: generation is incomplete (sides were never attempted)")
        if item.run.stop_reason:
            problems.append(f"{repo}: the run stopped early ({item.run.stop_reason})")
    return problems


def finalization_problems(inputs: Sequence[RepoInputs]) -> list[str]:
    """Why the experiment is not final (empty when an official audit sample may be drawn)."""
    problems: list[str] = []
    for item in inputs:
        repo = item.run.repository_id
        if item.judgements is None:
            problems.append(f"{repo}: no judgement file (the judging step has not run)")
        not_attempted = sum(
            1 for e in item.run.entries for arm in ARMS if getattr(e, arm).state == "not_attempted"
        )
        missing_entries = len(item.questions) - len(item.run.entries)
        if not_attempted or missing_entries:
            count = not_attempted + 2 * max(missing_entries, 0)
            problems.append(f"{repo}: {count} side(s) were never attempted")
        if item.run.stop_reason:
            problems.append(f"{repo}: the run stopped early ({item.run.stop_reason})")
    for status in classify_questions(inputs):
        if "judge_rating_missing" in status.reasons:
            problems.append(
                f"{status.repository_id}/{status.question_id}: an auditable answer has no valid "
                "judge rating"
            )
    return problems


def population_report(
    inputs: Sequence[RepoInputs],
    statuses: Sequence[QuestionStatus],
    sample: Sequence[AuditUnit],
    *,
    problems: Sequence[str],
    provisional: bool,
) -> dict[str, object]:
    """Counts of the population, exclusions and the selected sample (JSON-serialisable)."""
    sides = [(i, getattr(e, arm)) for i in inputs for e in i.run.entries for arm in ARMS]
    states = [side.state for _, side in sides]
    completed = sum(
        1 for i in inputs for e in i.run.entries
        if e.plain.state == "successful" and e.rag.state == "successful"
    )  # fmt: skip
    excluded = [s for s in statuses if not s.eligible]
    primary = {r: sum(1 for s in excluded if s.primary_reason == r) for r in PRIMARY_ORDER}
    incidence = {r: sum(1 for s in statuses if r in s.reasons) for r in PRIMARY_ORDER}
    eligible = [s for s in statuses if s.eligible]
    chosen = {(u.repository_id, u.question_id) for u in sample}

    strata: dict[str, dict[str, int]] = {}
    for s in statuses:
        key = f"{s.repository_id}/{'answerable' if s.answerable else 'unanswerable'}"
        cell = strata.setdefault(key, {"questions": 0, "eligible": 0, "selected": 0})
        cell["questions"] += 1
        cell["eligible"] += int(s.eligible)
        cell["selected"] += int((s.repository_id, s.question_id) in chosen)

    missing_judge = sum(1 for s in statuses if "judge_rating_missing" in s.reasons)
    return {
        "provisional": provisional,
        "provisional_reasons": list(problems),
        "total_benchmark_questions": len(statuses),
        "total_responses": 2 * len(statuses),
        "completed_pairs": completed,
        "eligible_pairs": len(eligible),
        "excluded_pairs": len(excluded),
        "exclusions_by_primary_reason": primary,
        "exclusion_incidence_by_reason": incidence,
        "rule_scored_system_abstentions": sum(1 for _, s in sides if s.system_insufficient),
        "failed_responses": sum(1 for st in states if st == "failed"),
        "not_attempted_responses": sum(1 for st in states if st == "not_attempted")
        + 2 * sum(max(len(i.questions) - len(i.run.entries), 0) for i in inputs),
        "questions_with_missing_or_failed_judge_ratings": missing_judge,
        "selected_sample_size": len(chosen),
        "sample_fraction_of_eligible": len(chosen) / len(eligible) if eligible else None,
        "sample_fraction_of_full_population": len(chosen) / len(statuses) if statuses else None,
        "strata": strata,
        "selected_questions": sorted(f"{r}/{q}" for r, q in chosen),
        "excluded_questions": {
            f"{s.repository_id}/{s.question_id}": list(s.reasons) for s in excluded
        },
        "note": "the sample is drawn from eligible pairs only; compare its size with BOTH the "
        "eligible population and the full population",
    }


def describe_empty_sample(report: Mapping[str, object]) -> str:
    """The diagnostic printed when no question is eligible."""
    reasons = report["exclusions_by_primary_reason"]
    detail = ", ".join(f"{k}={v}" for k, v in reasons.items() if v)  # type: ignore[union-attr]
    return (
        f"no eligible pair to audit: {report['total_benchmark_questions']} benchmark question(s), "
        f"{report['completed_pairs']} completed pair(s), {report['excluded_pairs']} excluded "
        f"({detail or 'no reason recorded'}); rule-scored system abstentions="
        f"{report['rule_scored_system_abstentions']}, failed={report['failed_responses']}, "
        f"not attempted={report['not_attempted_responses']}, questions with missing judge "
        f"ratings={report['questions_with_missing_or_failed_judge_ratings']}"
    )
