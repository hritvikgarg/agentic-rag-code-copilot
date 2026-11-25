"""Synthetic judge controls (Milestone 11B): do the judge's verdicts make sense?

A *control* is a hand-written answer to a real benchmark question together with the outcomes any
sensible judge must produce (``generation-eval/judge-controls/controls-v1.json``). The expected
outcomes were decided by the project team independently of every judge, and the benchmark's own
reference answers are not reused as controls, so a judge that merely echoes the reference text gains
nothing. The suite covers: correct substantive answer, partially correct, confident fabrication,
contradicted repository fact, correct insufficient-evidence abstention, incorrect refusal of an
answerable question, generic refusal, mixed answer-plus-abstention and an invented repository
path/symbol. Two further groups need no model: *parser controls* (malformed structured judgements
that the strict parser must reject, plus two valid ones it must accept) and one *security control*
(an answer carrying a synthetic planted secret that the secret gate must stop).

**Isolation.** Control results live in their own namespace (``synthetic-judge-controls``), in a
directory that may not be inside the primary or pilot directories. Nothing in this module is
imported by the analysis, audit or report code, the control ids are prefixed ``ctl-`` and none of
them is a benchmark question id used as a measurement: controls are never counted in any primary or
pilot metric (``tests/unit/test_judge_controls.py`` enforces both).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from copilot.evaluation.generation.judge import (
    JUDGE_PROMPT_VERSION,
    JudgeParseError,
    JudgeRecord,
    LLMJudge,
    parse_judge_output,
)
from copilot.evaluation.generation.reference import ReferenceMaterial
from copilot.evaluation.generation.rubric import RUBRIC_VERSION, EvaluationError, score_assessment
from copilot.evaluation.generation.schema import GenerationQuestion
from copilot.utils.atomic_file import write_text_atomically

CONTROLS_PATH = "generation-eval/judge-controls/controls-v1.json"
CONTROLS_SCHEMA = "judge-controls/1"
RESULTS_SCHEMA = "judge-controls-results/1"
NAMESPACE = "synthetic-judge-controls"

MODEL_KINDS = (
    "correct_substantive",
    "partially_correct",
    "confident_fabrication",
    "contradicted_fact",
    "correct_abstention",
    "incorrect_refusal",
    "generic_refusal",
    "mixed_answer_and_abstention",
    "invented_symbol",
)


class ControlsError(ValueError):
    """The controls specification or a control result file is invalid or does not match."""


class Expectation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    response_types: list[str] = Field(min_length=1)
    abstained: bool
    fact_verdicts: dict[str, list[str]]
    unsupported_claims: Literal["none", "at_least_one", "any"]
    correctness_scores: list[int] = Field(min_length=1)


class Control(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    control_id: str
    kind: str
    repository_id: str
    question_id: str
    answer: str = Field(min_length=1)
    expected: Expectation

    @model_validator(mode="after")
    def _known(self) -> Control:
        if self.kind not in MODEL_KINDS:
            raise ValueError(f"unknown control kind {self.kind!r}")
        if not self.control_id.startswith("ctl-"):
            raise ValueError("control ids start with 'ctl-'")
        return self


class ParserControl(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    control_id: str
    kind: str
    repository_id: str
    question_id: str
    raw_output: str
    expected: Literal["accept", "reject"]


class SecurityControl(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    control_id: str
    repository_id: str
    question_id: str
    answer_prefix: str
    note: str = ""


class ControlsSpec(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    schema_: str = Field(alias="schema")
    version: str
    namespace: str
    controls: list[Control]
    security_control: SecurityControl
    parser_controls: list[ParserControl]

    @model_validator(mode="after")
    def _consistent(self) -> ControlsSpec:
        if self.schema_ != CONTROLS_SCHEMA or self.namespace != NAMESPACE:
            raise ValueError("not a judge-controls/1 specification in the controls namespace")
        ids = [c.control_id for c in self.controls]
        ids += [c.control_id for c in self.parser_controls] + [self.security_control.control_id]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate control ids")
        missing = [k for k in MODEL_KINDS if not any(c.kind == k for c in self.controls)]
        if missing:
            raise ValueError(f"no control of kind: {', '.join(missing)}")
        return self


def controls_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_controls(path: str | Path) -> ControlsSpec:
    try:
        return ControlsSpec.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise ControlsError(f"cannot load the judge controls: {type(exc).__name__}") from exc


def check_against_benchmarks(
    spec: ControlsSpec,
    questions: Mapping[str, Mapping[str, GenerationQuestion]],
    excluded_question_ids: Mapping[str, tuple[str, ...]] | None = None,
) -> None:
    """Every control must name a real question, match its fact ids and avoid excluded questions
    (the pilot's)."""
    excluded = excluded_question_ids or {}
    everything: list[tuple[str, str, str, Expectation | None]] = [
        (c.control_id, c.repository_id, c.question_id, c.expected) for c in spec.controls
    ]
    everything += [
        (c.control_id, c.repository_id, c.question_id, None) for c in spec.parser_controls
    ]
    sc = spec.security_control
    everything.append((sc.control_id, sc.repository_id, sc.question_id, None))
    for control_id, repo, qid, expected in everything:
        question = questions.get(repo, {}).get(qid)
        if question is None:
            raise ControlsError(f"{control_id}: unknown question {repo}/{qid}")
        if qid in excluded.get(repo, ()) and not control_id.startswith("ctl-parse"):
            raise ControlsError(f"{control_id}: {repo}/{qid} is a pilot question")
        if expected is not None:
            facts = {f.id for f in question.key_facts}
            if set(expected.fact_verdicts) != facts:
                raise ControlsError(f"{control_id}: expected fact ids differ from the question's")


def security_token() -> str:
    """A synthetic, non-functional token-shaped string, assembled at run time (never stored)."""
    return "ghp_" + hashlib.sha256(b"synthetic-judge-control-secret").hexdigest()[:36]


# --------------------------------------------------------------------------- results


class ControlResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    control_id: str
    kind: str
    repository_id: str
    question_id: str
    state: Literal["judged", "failed"]
    score: int | None = None
    response_type: str | None = None
    abstained: bool | None = None
    fact_verdicts: dict[str, str] = Field(default_factory=dict)
    unsupported_claims: int | None = None
    checks: dict[str, bool] = Field(default_factory=dict)
    passed: bool = False
    reused: bool = False
    error: str | None = None
    record: JudgeRecord | None = None


class ParserResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    control_id: str
    kind: str
    expected: str
    outcome: Literal["accepted", "rejected"]
    passed: bool


class SecurityResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    control_id: str
    state: Literal["judged", "failed"]
    error: str | None
    gate_fired: bool
    secret_in_artifact: bool
    passed: bool


class ControlsRun(BaseModel):
    schema_version: str = RESULTS_SCHEMA
    namespace: str = NAMESPACE
    pass_label: Literal["pass1", "repeat"]
    controls_version: str
    controls_sha256: str
    validation_spec_sha256: str
    judge_provider: str
    judge_model: str
    judge_prompt_version: str = JUDGE_PROMPT_VERSION
    rubric_version: str = RUBRIC_VERSION
    runtime_fingerprint: str | None = None
    runtime: dict[str, Any] | None = None
    cache_used: bool
    cache_hits: int = 0
    test_artifact: bool = False
    results: list[ControlResult] = Field(default_factory=list)
    parser_results: list[ParserResult] = Field(default_factory=list)
    security_result: SecurityResult | None = None

    def summary(self) -> dict[str, Any]:
        judged = [r for r in self.results if r.state == "judged"]
        by_kind: dict[str, dict[str, int]] = {}
        for r in self.results:
            cell = by_kind.setdefault(r.kind, {"controls": 0, "judged": 0, "passed": 0})
            cell["controls"] += 1
            cell["judged"] += int(r.state == "judged")
            cell["passed"] += int(r.passed)
        context = sum(1 for r in self.results if (r.error or "").startswith("LLMContextLimitError"))
        return {
            "model_controls": len(self.results),
            "judged": len(judged),
            "parse_success_fraction": len(judged) / len(self.results) if self.results else None,
            "all_expectations_met": sum(r.passed for r in self.results),
            "context_limit_failures": context,
            "by_kind": by_kind,
            "parser_controls_passed": sum(r.passed for r in self.parser_results),
            "parser_controls": len(self.parser_results),
            "security_control_passed": None
            if self.security_result is None
            else self.security_result.passed,
        }


def results_path(directory: str | Path, pass_label: str) -> Path:
    return Path(directory) / f"judge-controls-{pass_label}.json"


def load_results(path: str | Path) -> ControlsRun:
    try:
        return ControlsRun.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise ControlsError(f"cannot read control results: {type(exc).__name__}") from exc


def evaluate_expectation(
    control: Control, question: GenerationQuestion, record: JudgeRecord
) -> ControlResult:
    """Compare a judge's assessment with the control's expected outcomes (no judge involved)."""
    base: dict[str, Any] = {
        "control_id": control.control_id,
        "kind": control.kind,
        "repository_id": control.repository_id,
        "question_id": control.question_id,
        "record": record,
    }
    if record.state != "successful" or record.assessment is None:
        return ControlResult(state="failed", error=record.error, **base)
    a = record.assessment
    try:
        score = score_assessment(question, a).score
    except EvaluationError as exc:
        return ControlResult(state="failed", error=f"EvaluationError: {exc}"[:300], **base)
    exp = control.expected
    verdicts = {fid: v.value for fid, v in a.fact_verdicts.items()}
    unsupported = len(a.material_unsupported_claims)
    checks = {
        "response_type": a.response_type in exp.response_types,
        "abstained": a.abstained == exp.abstained,
        "fact_verdicts": all(
            verdicts.get(f) in allowed for f, allowed in exp.fact_verdicts.items()
        ),
        "unsupported_claims": {
            "none": unsupported == 0,
            "at_least_one": unsupported >= 1,
            "any": True,
        }[exp.unsupported_claims],
        "correctness_score": score in exp.correctness_scores,
    }
    return ControlResult(
        state="judged", score=score, response_type=a.response_type, abstained=a.abstained,
        fact_verdicts=verdicts, unsupported_claims=unsupported, checks=checks,
        passed=all(checks.values()), **base,
    )  # fmt: skip


def run_parser_controls(
    spec: ControlsSpec, questions: Mapping[str, Mapping[str, GenerationQuestion]]
) -> list[ParserResult]:
    out: list[ParserResult] = []
    for c in spec.parser_controls:
        try:
            parse_judge_output(
                c.raw_output, questions[c.repository_id][c.question_id], evaluator_id="control"
            )
            outcome = "accepted"
        except JudgeParseError:
            outcome = "rejected"
        passed = (outcome == "accepted") == (c.expected == "accept")
        out.append(
            ParserResult(
                control_id=c.control_id,
                kind=c.kind,
                expected=c.expected,
                outcome=outcome,
                passed=passed,
            )  # fmt: skip
        )
    return out


class ControlsMismatchError(ControlsError):
    """An existing control result file belongs to a different judge, runtime or specification."""


def run_controls(
    spec: ControlsSpec,
    judge: LLMJudge,
    questions: Mapping[str, Mapping[str, GenerationQuestion]],
    references: Mapping[str, Mapping[str, ReferenceMaterial]],
    *,
    pass_label: Literal["pass1", "repeat"],
    spec_sha256: str,
    validation_spec_sha256: str,
    checkpoint_path: str | Path,
    prior: ControlsRun | None = None,
    progress: Callable[[str], None] | None = None,
) -> ControlsRun:
    """Judge every control (resumable, checkpointed after each one) and the security control."""
    identity = judge.runtime_identity()  # None for a hosted judge; LLMError if unidentifiable
    fingerprint = identity.fingerprint() if identity is not None else None
    cache_used = judge.cache is not None
    header: dict[str, Any] = {
        "pass_label": pass_label,
        "controls_version": spec.version,
        "controls_sha256": spec_sha256,
        "validation_spec_sha256": validation_spec_sha256,
        "judge_provider": judge.llm.provider,
        "judge_model": judge.llm.model,
        "runtime_fingerprint": fingerprint,
        "cache_used": cache_used,
    }
    if prior is not None:
        differing = [k for k, v in header.items() if getattr(prior, k) != v]
        if differing:
            raise ControlsMismatchError(
                "cannot resume the controls: " + ", ".join(differing) + " differ from this run"
            )
    done: dict[str, ControlResult] = {
        r.control_id: r for r in (prior.results if prior else []) if r.state == "judged"
    }
    hits = sum(1 for r in done.values() if r.reused)
    token = security_token()

    def snapshot(results: dict[str, ControlResult], **extra: Any) -> ControlsRun:
        ordered = [results[c.control_id] for c in spec.controls if c.control_id in results]
        return ControlsRun(
            runtime=identity.model_dump(mode="json") if identity is not None else None,
            cache_hits=hits, test_artifact=judge.llm.provider == "fake", results=ordered,
            **header, **extra,
        )  # fmt: skip

    for index, control in enumerate(spec.controls, 1):
        if control.control_id in done:
            continue
        if progress:
            progress(f"[controls {pass_label} {index}/{len(spec.controls)}] {control.control_id}")
        question = questions[control.repository_id][control.question_id]
        reference = references[control.repository_id][control.question_id]
        record, reused = judge.judge(
            question, control.answer, reference, blind_id=control.control_id
        )
        result = evaluate_expectation(control, question, record)
        if reused:
            result = result.model_copy(update={"reused": True})
            hits += 1
        done[control.control_id] = result
        write_text_atomically(
            Path(checkpoint_path), snapshot(done).model_dump_json(indent=2) + "\n"
        )

    sc = spec.security_control
    s_question = questions[sc.repository_id][sc.question_id]
    s_reference = references[sc.repository_id][sc.question_id]
    s_record, _ = judge.judge(
        s_question, f"{sc.answer_prefix} {token}", s_reference, blind_id=sc.control_id
    )
    gate_fired = s_record.state == "failed" and "RepositorySecretRiskError" in (
        s_record.error or ""
    )
    final = snapshot(
        done,
        parser_results=run_parser_controls(spec, questions),
        security_result=None,
    )
    leaked = token in final.model_dump_json() or token in s_record.model_dump_json()
    security = SecurityResult(
        control_id=sc.control_id, state="failed" if s_record.state == "failed" else "judged",
        error=s_record.error if s_record.state == "failed" else None, gate_fired=gate_fired,
        secret_in_artifact=leaked, passed=gate_fired and not leaked,
    )  # fmt: skip
    final = final.model_copy(update={"security_result": security})
    write_text_atomically(Path(checkpoint_path), final.model_dump_json(indent=2) + "\n")
    return final
