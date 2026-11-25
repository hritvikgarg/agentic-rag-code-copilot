"""Judging the answers of a generation run (one file per repository, resumable).

Every *successful* side with answer text is judged by an ``LLMJudge`` against the same reference
material, blind to the arm. A RAG system abstention (insufficient evidence, the model was never
called) has no text to judge; it is scored by a **rule** instead - every key fact is missing and the
response abstains - which is exactly what the rubric prescribes, and it is labelled
``evaluator_kind="rule"`` so it is never mistaken for a judgement. Failed and not-attempted sides
are recorded as ``skipped`` with the reason; nothing is defaulted.

A judge failure (provider error, blocked output, unparsable JSON, wrong fact ids) is recorded as a
``failed`` item and retried on the next resume. The file is checkpointed atomically after each item,
so an interruption loses at most one judgement. Judge identity (provider, model, prompt and rubric
versions, and for a local judge its runtime fingerprint - model digest, quantisation, server
version, context size, seed, ...) is stored; resuming with a different judge, runtime or rubric is
refused, so two judges can never be mixed in one file.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from copilot.evaluation.generation.execution import GenerationRun, SideRecord
from copilot.evaluation.generation.judge import (
    JUDGE_PROMPT_VERSION,
    JudgeRecord,
    LLMJudge,
)
from copilot.evaluation.generation.reference import ReferenceMaterial
from copilot.evaluation.generation.rubric import RUBRIC_VERSION, Assessment, FactVerdict
from copilot.evaluation.generation.schema import GenerationQuestion
from copilot.utils.atomic_file import write_text_atomically

logger = logging.getLogger(__name__)

JUDGEMENT_SCHEMA = "generation-judgements/1"
Arm = Literal["plain", "rag"]
ItemState = Literal["judged", "rule", "failed", "skipped"]
RULE_EVALUATOR_ID = f"system-abstention-rule/{RUBRIC_VERSION}"


class JudgementMismatchError(ValueError):
    """The existing judgement file belongs to a different run or judge."""


class JudgedItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    question_id: str
    arm: Arm
    blind_id: str
    state: ItemState
    assessment: Assessment | None = None
    record: JudgeRecord | None = None  # the judge call's record (judged / failed)
    reused: bool = False  # served from the judge cache or an earlier successful item
    skipped_reason: str | None = None


class JudgementFile(BaseModel):
    schema_version: str = JUDGEMENT_SCHEMA
    repository_id: str
    protocol_hash: str
    generation_provider: str
    generation_model: str
    judge_provider: str
    judge_model: str
    judge_prompt_version: str
    rubric_version: str
    same_model_as_generator: bool
    test_artifact: bool
    items: list[JudgedItem]
    # Local-judge runtime identity (None for hosted judges, so older files load unchanged).
    judge_runtime_fingerprint: str | None = None
    judge_runtime: dict[str, object] | None = None

    def context_failures(self) -> int:
        """Failed items whose failure was a context/truncation refusal (never valid judgements)."""
        return sum(
            1
            for i in self.items
            if i.state == "failed"
            and i.record is not None
            and (i.record.error or "").startswith("LLMContextLimitError")
        )

    def counts(self) -> dict[str, int]:
        out = {"judged": 0, "rule": 0, "failed": 0, "skipped": 0}
        for item in self.items:
            out[item.state] += 1
        return out

    def by_key(self) -> dict[tuple[str, str], JudgedItem]:
        return {(i.question_id, i.arm): i for i in self.items}


def blind_id(seed: int, repository_id: str, question_id: str, arm: str) -> str:
    """A stable, opaque id for one answer: it does not reveal the arm (the arm is hashed in)."""
    digest = hashlib.sha256(f"{seed}|{repository_id}|{question_id}|{arm}".encode()).hexdigest()
    return f"J-{digest[:12]}"


def judgement_path(output_dir: str | Path, repository_id: str) -> Path:
    return Path(output_dir) / f"judgements-{repository_id}.json"


def write_judgements(file: JudgementFile, path: str | Path) -> None:
    write_text_atomically(Path(path), file.model_dump_json(indent=2) + "\n")


def load_judgements(path: str | Path) -> JudgementFile:
    try:
        return JudgementFile.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValidationError, UnicodeDecodeError) as exc:
        raise JudgementMismatchError(f"cannot read judgement file: {type(exc).__name__}") from exc


def rule_assessment(question: GenerationQuestion) -> Assessment:
    """The assessment of a system abstention: nothing was said, so no key fact is conveyed."""
    verdicts = (
        {fact.id: FactVerdict.MISSING for fact in question.key_facts}
        if question.is_answerable
        else {}
    )
    return Assessment(
        evaluator_kind="rule", evaluator_id=RULE_EVALUATOR_ID, fact_verdicts=verdicts,
        material_unsupported_claims=(), groundedness=2, abstained=True,
        notes="system abstention: the model was not called",
    )  # fmt: skip


def _is_rate_limit(record: JudgeRecord) -> bool:
    return record.state == "failed" and "RateLimit" in (record.error or "")


def _item_for(
    question: GenerationQuestion,
    side: SideRecord,
    judge: LLMJudge,
    reference: ReferenceMaterial,
    blind: str,
    prior: JudgedItem | None,
) -> tuple[JudgedItem, bool]:
    """``(item, made_a_judge_call_that_was_not_cached)``."""
    base = {"question_id": question.id, "arm": side.arm, "blind_id": blind}
    if side.state != "successful":
        return JudgedItem(state="skipped", skipped_reason=f"side is {side.state}", **base), False
    if side.system_insufficient and not side.has_text:
        return JudgedItem(state="rule", assessment=rule_assessment(question), **base), False
    answer = side.answer or ""
    request = judge.request_for(question, answer, reference)
    input_sha = hashlib.sha256(request.outbound_text().encode("utf-8")).hexdigest()
    if (
        prior is not None
        and prior.state == "judged"
        and prior.record is not None
        and prior.record.input_sha256 == input_sha
    ):
        return prior.model_copy(update={"reused": True}), False
    record, reused = judge.judge(question, answer, reference, blind_id=blind)
    if record.state == "successful":
        return (
            JudgedItem(
                state="judged", assessment=record.assessment, record=record, reused=reused, **base
            ),
            not reused,
        )
    return JudgedItem(state="failed", record=record, **base), True


def judge_run(
    run: GenerationRun,
    questions: Mapping[str, GenerationQuestion],
    references: Mapping[str, ReferenceMaterial],
    judge: LLMJudge,
    *,
    checkpoint_path: str | Path,
    resume_from: JudgementFile | None = None,
    seed: int = 0,
    max_consecutive_rate_limit_failures: int | None = 3,
    progress: Callable[[str], None] | None = None,
) -> JudgementFile:
    """Judge every successful side of ``run``; reuse earlier successes; checkpoint every item."""
    judge_id = (judge.llm.provider, judge.llm.model)
    identity = judge.runtime_identity()  # None for hosted judges; LLMError if unidentifiable
    fingerprint = identity.fingerprint() if identity is not None else None
    if resume_from is not None:
        mismatches = [
            label
            for label, old, new in (
                ("protocol_hash", resume_from.protocol_hash, run.protocol_hash),
                ("repository_id", resume_from.repository_id, run.repository_id),
                ("judge", (resume_from.judge_provider, resume_from.judge_model), judge_id),
                (
                    "judge_runtime_fingerprint",
                    resume_from.judge_runtime_fingerprint,
                    fingerprint,
                ),
                ("judge_prompt_version", resume_from.judge_prompt_version, JUDGE_PROMPT_VERSION),
                ("rubric_version", resume_from.rubric_version, RUBRIC_VERSION),
                ("generation_model", resume_from.generation_model, run.llm_model),
            )
            if old != new
        ]
        if mismatches:
            raise JudgementMismatchError(
                "cannot resume judging: " + ", ".join(mismatches) + " differ from this run"
            )
    prior_items = resume_from.by_key() if resume_from else {}
    items: dict[tuple[str, str], JudgedItem] = dict(prior_items)

    def make_file() -> JudgementFile:
        ordered = [
            items[(e.question_id, arm)]
            for e in run.entries
            for arm in ("plain", "rag")
            if (e.question_id, arm) in items
        ]
        return JudgementFile(
            repository_id=run.repository_id, protocol_hash=run.protocol_hash,
            generation_provider=run.llm_provider, generation_model=run.llm_model,
            judge_provider=judge_id[0], judge_model=judge_id[1],
            judge_prompt_version=JUDGE_PROMPT_VERSION, rubric_version=RUBRIC_VERSION,
            same_model_as_generator=judge_id == (run.llm_provider, run.llm_model),
            test_artifact=run.test_artifact or judge.llm.provider == "fake", items=ordered,
            judge_runtime_fingerprint=fingerprint,
            judge_runtime=identity.model_dump(mode="json") if identity is not None else None,
        )  # fmt: skip

    streak = 0
    stopped = False
    for index, entry in enumerate(run.entries):
        question = questions[entry.question_id]
        if progress:
            progress(f"[judge {run.repository_id} {index + 1}/{len(run.entries)}] {question.id}")
        for arm in ("plain", "rag"):
            key = (question.id, arm)
            prior = items.get(key)
            if prior is not None and prior.state in ("judged", "rule"):
                side = getattr(entry, arm)
                fresh, _ = (
                    _item_for(
                        question,
                        side,
                        judge,
                        references[question.id],
                        blind_id(seed, run.repository_id, question.id, arm),
                        prior,
                    )  # fmt: skip
                    if prior.state == "judged"
                    else (prior, False)
                )
                items[key] = fresh
                continue
            if stopped:
                items[key] = JudgedItem(
                    question_id=question.id, arm=arm,
                    blind_id=blind_id(seed, run.repository_id, question.id, arm),
                    state="skipped", skipped_reason="judging stopped (rate-limit breaker)",
                )  # fmt: skip
                continue
            item, called = _item_for(
                question, getattr(entry, arm), judge, references[question.id],
                blind_id(seed, run.repository_id, question.id, arm), prior,
            )  # fmt: skip
            items[key] = item
            write_judgements(make_file(), checkpoint_path)
            if called:
                limited = item.record is not None and _is_rate_limit(item.record)
                streak = streak + 1 if limited else 0
                if (
                    max_consecutive_rate_limit_failures is not None
                    and streak >= max_consecutive_rate_limit_failures
                ):
                    stopped = True
    file = make_file()
    write_judgements(file, checkpoint_path)
    return file
