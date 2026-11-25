"""The predeclared validation plan for the local judge (Milestone 11B).

``generation-eval/judge-validation/judge-validation-v1.json`` was committed before any local or
hosted judge execution on the pilot or the synthetic controls. This module loads it, exposes its
SHA-256 (every control result and validation report records that hash, so a result can always be
tied to the exact thresholds it was judged against) and evaluates artifacts against it. The
thresholds are read from the file, never hard-coded here, and a test pins the file's hash: changing
a threshold after seeing results therefore shows up as a failing test and a new version.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from copilot.evaluation.generation.judge_controls import ControlsRun, security_token
from copilot.evaluation.generation.judging import JudgementFile
from copilot.evaluation.generation.rubric import EvaluationError, score_assessment
from copilot.evaluation.generation.schema import GenerationQuestion
from copilot.llm.runtime import RuntimeIdentity

VALIDATION_SPEC_PATH = "generation-eval/judge-validation/judge-validation-v1.json"
VALIDATION_SPEC_SCHEMA = "judge-validation/1"
HOSTED_PROFILE_PATH = "generation-eval/judge-validation/judge-validation-v1-hosted.json"
HOSTED_PROFILE_SCHEMA = "judge-validation-hosted/1"
HOSTED_PROVIDER = "gemini"
REQUIRED_SECTIONS = (
    "judge",
    "controls",
    "technical_acceptance",
    "semantic_pilot_acceptance",
    "primary_audit_acceptance",
)


class ValidationSpecError(ValueError):
    """The validation specification is missing, malformed or not the expected version."""


def spec_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_validation_spec(path: str | Path) -> dict[str, Any]:
    try:
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValidationSpecError(f"cannot read the validation specification: {path}") from exc
    if not isinstance(spec, dict) or spec.get("schema") != VALIDATION_SPEC_SCHEMA:
        raise ValidationSpecError("not a judge-validation/1 specification")
    missing = [name for name in REQUIRED_SECTIONS if name not in spec]
    if missing:
        raise ValidationSpecError(f"the validation specification lacks: {', '.join(missing)}")
    return spec


class HostedProfileError(ValueError):
    """The hosted-judge profile is missing, malformed, or not tied to the exact v1 thresholds."""


def load_hosted_profile(
    path: str | Path, base_spec_path: str | Path = VALIDATION_SPEC_PATH
) -> dict[str, Any]:
    """Load the hosted-judge profile and check it names the exact v1 specification it inherits.

    The profile changes no threshold: it pins the sha256 of ``judge-validation-v1.json`` and states
    which criteria are adapted (and why) because a hosted API has no local runtime to fingerprint.
    """
    try:
        profile = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HostedProfileError(f"cannot read the hosted-judge profile: {path}") from exc
    if not isinstance(profile, dict) or profile.get("schema") != HOSTED_PROFILE_SCHEMA:
        raise HostedProfileError("not a judge-validation-hosted/1 profile")
    for name in ("base_spec", "judge", "inherited_unchanged", "adapted", "not_applicable"):
        if name not in profile:
            raise HostedProfileError(f"the hosted-judge profile lacks: {name}")
    if profile["base_spec"].get("sha256") != spec_sha256(base_spec_path):
        raise HostedProfileError(
            "the hosted-judge profile was written for a different judge-validation-v1 file"
        )
    if profile["judge"].get("provider") != HOSTED_PROVIDER:
        raise HostedProfileError(f"the hosted profile is for provider {HOSTED_PROVIDER!r}")
    return profile


@dataclass(frozen=True)
class HostedContext:
    """What the hosted adaptations of T3/T5/T6 compare against."""

    expected_model: str
    max_output_tokens: int

    def __post_init__(self) -> None:
        if not self.expected_model.strip():
            raise HostedProfileError("the expected hosted judge model must be named explicitly")
        if self.max_output_tokens < 16:
            raise HostedProfileError("the hosted judge output cap is not valid")


# --------------------------------------------------------------------------- evaluation

PASS, FAIL, NOT_EVALUATED = "PASS", "FAIL", "NOT_EVALUATED"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass
class ValidationInputs:
    """The artifacts of one validation run. Any may be absent; its criteria are then NOT_EVALUATED
    (never silently passed)."""

    controls: ControlsRun | None = None
    controls_repeat: ControlsRun | None = None
    controls_dir: str | None = None
    controls_repeat_dir: str | None = None
    pilot: Mapping[str, JudgementFile] = field(default_factory=dict)
    pilot_repeat: Mapping[str, JudgementFile] = field(default_factory=dict)
    pilot_dir: str | None = None
    pilot_repeat_dir: str | None = None
    questions: Mapping[str, Mapping[str, GenerationQuestion]] = field(default_factory=dict)
    agreement: Mapping[str, Any] | None = None


def _crit(cid: str, status: str, detail: str, **values: Any) -> dict[str, Any]:
    return {"id": cid, "status": status, "detail": detail, **values}


def _roll_up(criteria: Sequence[Mapping[str, Any]]) -> str:
    statuses = {c["status"] for c in criteria}
    if FAIL in statuses:
        return FAIL
    return PASS if statuses == {PASS} else "INCOMPLETE"


def _artifact_runtimes(inputs: ValidationInputs) -> list[tuple[str, str | None, dict | None]]:
    out: list[tuple[str, str | None, dict | None]] = []
    for label, run in (("controls", inputs.controls), ("controls-repeat", inputs.controls_repeat)):
        if run is not None:
            out.append((label, run.runtime_fingerprint, run.runtime))
    for label, files in (("pilot", inputs.pilot), ("pilot-repeat", inputs.pilot_repeat)):
        for repo, file in files.items():
            out.append((f"{label}:{repo}", file.judge_runtime_fingerprint, file.judge_runtime))
    return out


def _pilot_records(files: Mapping[str, JudgementFile]):
    for repo, file in files.items():
        for item in file.items:
            yield repo, item


def _identities(inputs: ValidationInputs) -> list[tuple[str, str, str]]:
    """``(artifact, provider, model)`` for every judge artifact of a hosted validation run."""
    out: list[tuple[str, str, str]] = []
    for label, run in (("controls", inputs.controls), ("controls-repeat", inputs.controls_repeat)):
        if run is not None:
            out.append((label, run.judge_provider, run.judge_model))
    for label, files in (("pilot", inputs.pilot), ("pilot-repeat", inputs.pilot_repeat)):
        for repo, file in files.items():
            out.append((f"{label}:{repo}", file.judge_provider, file.judge_model))
    return out


def evaluate_technical(
    spec: Mapping[str, Any], inputs: ValidationInputs, hosted: HostedContext | None = None
) -> list[dict[str, Any]]:
    tech = spec["technical_acceptance"]
    judge = spec["judge"]
    out: list[dict[str, Any]] = []

    # T1 - every pilot answer with model text has a successful judgement
    if not inputs.pilot:
        out.append(_crit("T1_pilot_judgements_parse", NOT_EVALUATED, "no pilot judgements given"))
    else:
        items = [i for _, i in _pilot_records(inputs.pilot) if i.state in ("judged", "failed")]
        judged = sum(1 for i in items if i.state == "judged")
        ok = bool(items) and judged == len(items)
        out.append(
            _crit(
                "T1_pilot_judgements_parse",
                PASS if ok else FAIL,
                f"{judged}/{len(items)} pilot answers with model text have a valid judgement",
                judged=judged,
                answers=len(items),
            )  # fmt: skip
        )

    # T2 - first-pass control parse success
    rule = tech["T2_control_parse_success"]
    if inputs.controls is None:
        out.append(_crit("T2_control_parse_success", NOT_EVALUATED, "no control results given"))
    else:
        total = len(inputs.controls.results)
        judged = sum(1 for r in inputs.controls.results if r.state == "judged")
        expected = spec["controls"]["model_controls_expected"]
        fraction = judged / total if total else 0.0
        ok = total == expected and fraction >= rule["min_fraction"]
        out.append(
            _crit(
                "T2_control_parse_success",
                PASS if ok else FAIL,
                f"{judged}/{total} first-pass model controls parsed (expected {expected} controls,"
                f" minimum fraction {rule['min_fraction']})",
                judged=judged,
                controls=total,
                fraction=fraction,
            )  # fmt: skip
        )

    # T3 - no accepted judgement may show a context overflow / truncation
    violations: list[str] = []
    refused = 0
    sources: list[tuple[str, Any, dict | None]] = []
    for label, run in (("controls", inputs.controls), ("controls-repeat", inputs.controls_repeat)):
        if run is not None:
            sources.append((label, [r.record for r in run.results if r.record], run.runtime))
            refused += sum(
                1 for r in run.results if (r.error or "").startswith("LLMContextLimitError")
            )
    for label, files in (("pilot", inputs.pilot), ("pilot-repeat", inputs.pilot_repeat)):
        for repo, file in files.items():
            sources.append(
                (f"{label}:{repo}", [i.record for i in file.items if i.record], file.judge_runtime)
            )
            refused += file.context_failures()
    if not sources:
        out.append(
            _crit("T3_no_accepted_context_overflow_or_truncation", NOT_EVALUATED, "no artifacts")
        )
    else:
        for label, records, runtime in sources:
            if hosted is not None:
                # Hosted adaptation: there is no local window to overflow, so only the output cap
                # applies. Reasoning tokens count against the cap (docs/llm.md, "Thinking models").
                cap = hosted.max_output_tokens
                for rec in records:
                    if rec.state != "successful":
                        continue
                    c = rec.completion_tokens
                    if c is None:
                        violations.append(f"{label}/{rec.question_id}: token counts missing")
                    elif c + (rec.reasoning_tokens or 0) >= cap:
                        violations.append(
                            f"{label}/{rec.question_id}: completion {c} + reasoning "
                            f"{rec.reasoning_tokens or 0} reaches the {cap}-token cap"
                        )
                continue
            if not runtime:
                violations.append(f"{label}: no runtime description to check against")
                continue
            ctx, cap = runtime["context_size"], runtime["max_output_tokens"]
            for rec in records:
                if rec.state != "successful":
                    continue
                p, c = rec.prompt_tokens, rec.completion_tokens
                if p is None or c is None:
                    violations.append(f"{label}/{rec.question_id}: token counts missing")
                elif p >= ctx - cap or p + c >= ctx or c >= cap:
                    violations.append(f"{label}/{rec.question_id}: prompt {p} + completion {c}")
        maximum = tech["T3_no_accepted_context_overflow_or_truncation"]["max_violations"]
        out.append(
            _crit(
                "T3_no_accepted_context_overflow_or_truncation",
                PASS if len(violations) <= maximum else FAIL,
                f"{len(violations)} accepted judgement(s) show a possible overflow/truncation; "
                f"{refused} request(s) were correctly refused for context reasons",
                violations=violations[:20],
                refused_for_context=refused,
            )  # fmt: skip
        )

    # T4 - the security control stopped at the gate and no artifact contains the secret
    runs = [r for r in (inputs.controls, inputs.controls_repeat) if r is not None]
    if not runs:
        out.append(_crit("T4_no_security_bypass", NOT_EVALUATED, "no control results given"))
    else:
        problems = []
        token = security_token()
        for run in runs:
            sr = run.security_result
            if sr is None or not sr.passed:
                problems.append(f"{run.pass_label}: the security control did not pass")
            if token in run.model_dump_json():
                problems.append(f"{run.pass_label}: the planted secret appears in the artifact")
        for files in (inputs.pilot, inputs.pilot_repeat):
            for repo, file in files.items():
                if token in file.model_dump_json():
                    problems.append(f"{repo}: the planted secret appears in a judgement file")
        maximum = tech["T4_no_security_bypass"]["max_violations"]
        out.append(
            _crit(
                "T4_no_security_bypass",
                PASS if len(problems) <= maximum else FAIL,
                f"{len(problems)} security problem(s)",
                problems=problems,
            )  # fmt: skip
        )

    # T5 - fingerprints present, recomputable, equal to the declared judge, shared by all artifacts
    runtimes = _artifact_runtimes(inputs)
    if hosted is not None:
        identities = _identities(inputs)
        if not identities:
            out.append(_crit("T5H_hosted_judge_identity", NOT_EVALUATED, "no artifacts"))
        else:
            problems = [
                f"{label}: judge is {provider}/{model}, expected "
                f"{HOSTED_PROVIDER}/{hosted.expected_model}"
                for label, provider, model in identities
                if (provider, model) != (HOSTED_PROVIDER, hosted.expected_model)
            ]
            out.append(
                _crit(
                    "T5H_hosted_judge_identity",
                    PASS if not problems else FAIL,
                    f"{len(problems)} problem(s) across {len(identities)} artifact(s); the hosted "
                    "API exposes no digest, quantization or seed, so provider and model id are the "
                    "whole identity",
                    problems=problems[:20],
                    judge=f"{HOSTED_PROVIDER}/{hosted.expected_model}",
                )
            )
    elif not runtimes:
        out.append(_crit("T5_valid_runtime_fingerprint", NOT_EVALUATED, "no artifacts"))
    else:
        problems = []
        seen: set[str] = set()
        want = {
            "provider": judge["provider"], "model_tag": judge["model"],
            "quantization": judge["quantization"], "context_size": judge["context_size"],
            "seed": judge["seed"], "temperature": judge["temperature"],
            "max_output_tokens": judge["max_output_tokens"],
            "structured_output": judge["structured_output"],
        }  # fmt: skip
        for label, fingerprint, runtime in runtimes:
            if fingerprint is None or not _HEX64.match(fingerprint) or not runtime:
                problems.append(f"{label}: no valid fingerprint/runtime")
                continue
            seen.add(fingerprint)
            try:
                if RuntimeIdentity(**runtime).fingerprint() != fingerprint:
                    problems.append(f"{label}: the fingerprint does not match its runtime")
            except (TypeError, ValueError):
                problems.append(f"{label}: the runtime description is malformed")
            for key, expected in want.items():
                if runtime.get(key) != expected:
                    problems.append(
                        f"{label}: {key} is {runtime.get(key)!r}, expected {expected!r}"
                    )
        if len(seen) > 1:
            problems.append("the artifacts do not share one runtime fingerprint")
        out.append(
            _crit(
                "T5_valid_runtime_fingerprint",
                PASS if not problems else FAIL,
                f"{len(problems)} problem(s) across {len(runtimes)} artifact(s)",
                problems=problems[:20],
                fingerprints=sorted(seen),
            )  # fmt: skip
        )

    # T6 - the repeat pass really was independent
    if inputs.controls_repeat is None and not inputs.pilot_repeat:
        out.append(_crit("T6_cache_isolation_and_resume", NOT_EVALUATED, "no repeat pass given"))
    else:
        problems = []
        rep = inputs.controls_repeat
        if rep is not None:
            if rep.cache_used or rep.cache_hits:
                problems.append("the repeat control pass used the cache")
            if (
                inputs.controls is not None
                and rep.runtime_fingerprint != inputs.controls.runtime_fingerprint
            ):
                problems.append("the control passes have different runtime fingerprints")
            if (
                hosted is not None
                and inputs.controls is not None
                and (rep.judge_provider, rep.judge_model)
                != (inputs.controls.judge_provider, inputs.controls.judge_model)
            ):
                problems.append("the control passes were judged by different hosted models")
            if inputs.controls_dir and inputs.controls_dir == inputs.controls_repeat_dir:
                problems.append("both control passes share a directory")
        for repo, file in inputs.pilot_repeat.items():
            if any(i.reused for i in file.items):
                problems.append(f"{repo}: the repeat pilot pass reused earlier judgements")
            first = inputs.pilot.get(repo)
            if (
                first is not None
                and first.judge_runtime_fingerprint != file.judge_runtime_fingerprint
            ):
                problems.append(f"{repo}: the pilot passes have different runtime fingerprints")
            if (
                hosted is not None
                and first is not None
                and (first.judge_provider, first.judge_model)
                != (file.judge_provider, file.judge_model)
            ):
                problems.append(f"{repo}: the pilot passes were judged by different hosted models")
        if inputs.pilot_dir and inputs.pilot_dir == inputs.pilot_repeat_dir:
            problems.append("both pilot passes share a directory")
        out.append(
            _crit(
                "T6_cache_isolation_and_resume",
                PASS if not problems else FAIL,
                f"{len(problems)} problem(s)",
                problems=problems,
            )  # fmt: skip
        )
    return out


def _score(question: GenerationQuestion | None, assessment: Any) -> int | None:
    if question is None or assessment is None:
        return None
    try:
        return score_assessment(question, assessment).score
    except EvaluationError:
        return None


def evaluate_semantic(spec: Mapping[str, Any], inputs: ValidationInputs) -> list[dict[str, Any]]:
    sem = spec["semantic_pilot_acceptance"]
    out: list[dict[str, Any]] = []
    agreement = inputs.agreement

    # S1 / S2 - blinded human agreement on the pilot
    s1, s2 = sem["S1_human_exact_agreement"], sem["S2_no_two_point_disagreement"]
    if agreement is None:
        out += [
            _crit("S1_human_exact_agreement", NOT_EVALUATED, "no pilot human-agreement file given"),
            _crit(
                "S2_no_two_point_disagreement", NOT_EVALUATED, "no pilot human-agreement file given"
            ),
        ]
    else:
        exact = agreement["correctness_exact_agreement"]
        two = agreement["correctness_two_point_disagreements"]
        n = exact["n"]
        if n < 4 or not agreement.get("rows_accounted_for", False):
            why = f"only {n} rated pilot answer(s) (or unaccounted rows): inconclusive"
            out += [
                _crit("S1_human_exact_agreement", NOT_EVALUATED, why, n=n),
                _crit("S2_no_two_point_disagreement", NOT_EVALUATED, why, n=n),
            ]
        else:
            note = "" if n == s1["expected_answers"] else f" (expected {s1['expected_answers']})"
            ok1 = exact["fraction"] is not None and exact["fraction"] >= s1["min_fraction"]
            out.append(
                _crit(
                    "S1_human_exact_agreement",
                    PASS if ok1 else FAIL,
                    f"{exact['exact']}/{n} exact{note}; minimum {s1['min_fraction']}",
                    n=n,
                    exact=exact["exact"],
                    fraction=exact["fraction"],
                )  # fmt: skip
            )
            ok2 = two["count"] <= s2["max_count"]
            out.append(
                _crit(
                    "S2_no_two_point_disagreement",
                    PASS if ok2 else FAIL,
                    f"{two['count']} two-point disagreement(s) among {n}",
                    n=n,
                    count=two["count"],
                )  # fmt: skip
            )

    first = inputs.controls
    # S3 - fabricated negative controls
    s3 = sem["S3_fabricated_negative_controls"]
    if first is None:
        out.append(_crit("S3_fabricated_negative_controls", NOT_EVALUATED, "no control results"))
    else:
        members = [r for r in first.results if r.kind in s3["kinds"]]
        hit = [r for r in members if r.state == "judged" and r.score == 0]
        ok = bool(members) and len(hit) / len(members) >= s3["required_fraction"]
        out.append(
            _crit(
                "S3_fabricated_negative_controls",
                PASS if ok else FAIL,
                f"{len(hit)}/{len(members)} fabricated/contradicted/invented controls scored 0",
                missed=[r.control_id for r in members if r not in hit],
            )  # fmt: skip
        )

    # S4 - abstention discrimination
    if first is None:
        out.append(_crit("S4_abstention_discrimination", NOT_EVALUATED, "no control results"))
    else:
        good = [r for r in first.results if r.kind == "correct_abstention"]
        bad = [r for r in first.results if r.kind == "incorrect_refusal"]
        right = [r for r in good if r.state == "judged" and r.abstained is True and r.score == 2]
        right += [r for r in bad if r.state == "judged" and r.abstained is True and r.score == 0]
        total = len(good) + len(bad)
        ok = total > 0 and len(right) == total
        out.append(
            _crit(
                "S4_abstention_discrimination",
                PASS if ok else FAIL,
                f"{len(right)}/{total} abstention controls judged correctly "
                "(appropriate abstention = abstained & 2, incorrect refusal = abstained & 0)",
                wrong=[r.control_id for r in good + bad if r not in right],
            )  # fmt: skip
        )

    # S5 - repeat consistency over the pilot answers and the model controls
    s5 = sem["S5_repeat_consistency"]
    if (
        first is None
        or inputs.controls_repeat is None
        or not inputs.pilot
        or not inputs.pilot_repeat
    ):
        out.append(
            _crit(
                "S5_repeat_consistency",
                NOT_EVALUATED,
                "needs both control passes and both pilot passes",
            )  # fmt: skip
        )
    else:
        pairs: list[tuple[str, int | None, int | None]] = []
        repeat_controls = {r.control_id: r for r in inputs.controls_repeat.results}
        for r in first.results:
            other = repeat_controls.get(r.control_id)
            pairs.append((r.control_id, r.score, other.score if other else None))
        for repo, file in inputs.pilot.items():
            again = inputs.pilot_repeat[repo].by_key() if repo in inputs.pilot_repeat else {}
            questions = inputs.questions.get(repo, {})
            for key, item in file.by_key().items():
                if item.state not in ("judged", "failed"):
                    continue
                other = again.get(key)
                pairs.append(
                    (
                        f"{repo}/{key[0]}/{key[1]}",
                        _score(questions.get(key[0]), item.assessment),
                        _score(questions.get(key[0]), other.assessment) if other else None,
                    )  # fmt: skip
                )
        same = sum(1 for _, a, b in pairs if a is not None and a == b)
        fraction = same / len(pairs) if pairs else 0.0
        ok = bool(pairs) and fraction >= s5["min_fraction"]
        out.append(
            _crit(
                "S5_repeat_consistency",
                PASS if ok else FAIL,
                f"{same}/{len(pairs)} items scored identically in both passes "
                f"(minimum fraction {s5['min_fraction']})",
                items=len(pairs),
                identical=same,
                fraction=fraction,
                inconsistent=[i for i, a, b in pairs if a is None or a != b],
            )  # fmt: skip
        )
    return out


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion (``None`` when ``n`` is 0)."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def evaluate_primary_audit(spec: Mapping[str, Any], agreement: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the primary-audit criteria to an ``audit-agree`` result (not used for the pilot)."""
    rule = spec["primary_audit_acceptance"]
    exact = agreement["correctness_exact_agreement"]
    two = agreement["correctness_two_point_disagreements"]
    coverage = agreement["valid_judgement_coverage"]
    kappa_block = agreement["correctness_score_0_2"]
    kappa = kappa_block.get("kappa")
    n = exact["n"]
    interval = wilson_interval(exact["exact"], n)
    two_fraction = two["fraction"]
    cov = coverage["fraction"]
    criteria = [
        _crit(
            "P1_exact_agreement",
            PASS
            if exact["fraction"] is not None and exact["fraction"] >= rule["exact_agreement_min"]
            else FAIL,
            f"{exact['exact']}/{n} exact (minimum {rule['exact_agreement_min']})",
            value=exact["fraction"],
            wilson_95=list(interval) if interval else None,
        ),
        _crit(
            "P2_weighted_kappa",
            PASS if kappa is not None and kappa >= rule["weighted_kappa_min"] else FAIL,
            (
                f"quadratic-weighted kappa {kappa:.3f} (minimum {rule['weighted_kappa_min']})"
                if kappa is not None
                else "kappa is undefined (" + str(kappa_block.get("kappa_undefined_reason")) + "): "
                "not a pass"
            ),
            value=kappa,
        ),
        _crit(
            "P3_severe_two_point_disagreements",
            PASS
            if two_fraction is not None
            and two_fraction <= rule["severe_two_point_disagreement_max_fraction"]
            else FAIL,
            f"{two['count']}/{n} two-point disagreements (maximum fraction "
            f"{rule['severe_two_point_disagreement_max_fraction']})",
            value=two_fraction,
        ),
        _crit(
            "P4_valid_judgement_coverage",
            PASS if cov is not None and cov >= rule["valid_judgement_coverage_min"] else FAIL,
            f"{coverage['with_a_valid_judgement']}/{coverage['answers_requiring_a_judgement']} "
            f"answers have a valid judgement (minimum {rule['valid_judgement_coverage_min']})",
            value=cov,
        ),
    ]
    status = PASS if all(c["status"] == PASS for c in criteria) else FAIL
    completeness = agreement.get("completeness", {})
    return {
        "criteria": criteria,
        "status": status,
        "sample_size": n,
        "denominators": {
            "rated_answers": agreement.get("rated_answers"),
            "key_entries": completeness.get("key_entries"),
            "answers_requiring_a_judgement": coverage["answers_requiring_a_judgement"],
        },
        "exclusions": completeness.get("other_exclusions", {}),
        "failed_or_missing_judge_ratings": completeness.get("failed_or_missing_judge_ratings", []),
        "conclusion": (
            "The local judge met every predeclared primary-audit criterion on this audit sample."
            if status == PASS
            else rule["on_failure"]
        ),
        "label": "validated-on-this-sample" if status == PASS else "EXPLORATORY",
    }


def _self_preference(inputs: ValidationInputs) -> dict[str, Any]:
    """Documented, never scored: does the judge also generate the answers it grades?"""
    flags = sorted(
        {file.same_model_as_generator for file in inputs.pilot.values()}
        | {file.same_model_as_generator for file in inputs.pilot_repeat.values()}
    )
    if not flags:
        return {
            "same_model_as_generator": None,
            "note": "no pilot judgements were given, so the judge/generator relation is unknown",
        }
    same = True in flags
    return {
        "same_model_as_generator": same,
        "note": (
            "the judge model IS the generation model: LLM judges can favour their own model's "
            "text (self-preference bias); conclusions that rest on this judge must say so"
            if same
            else "the judge model differs from the generation model, which reduces but does not "
            "remove self-preference when both come from the same vendor family"
        ),
    }


def evaluate_validation(
    spec: Mapping[str, Any],
    inputs: ValidationInputs,
    spec_hash: str,
    *,
    hosted: HostedContext | None = None,
    profile: Mapping[str, Any] | None = None,
    profile_hash: str | None = None,
) -> dict[str, Any]:
    """Evaluate the artifacts against ``spec``; with ``hosted`` the hosted adaptations apply.

    A hosted run needs both the profile (so its hash is recorded) and the context (the model id
    the operator declared). Thresholds always come from ``spec``; the profile adapts only how
    T3/T5/T6 identify the judge and never relaxes a threshold.
    """
    if (hosted is None) != (profile is None):
        raise HostedProfileError("a hosted evaluation needs both the profile and the context")
    technical = evaluate_technical(spec, inputs, hosted)
    semantic = evaluate_semantic(spec, inputs)
    t, s = _roll_up(technical), _roll_up(semantic)
    report: dict[str, Any] = {
        "validation_spec": {"version": spec["version"], "sha256": spec_hash},
        "technical_acceptance": {"status": t, "criteria": technical},
        "semantic_pilot_acceptance": {"status": s, "criteria": semantic},
        "overall": "PILOT_CRITERIA_MET"
        if t == s == PASS
        else (FAIL if FAIL in (t, s) else "INCOMPLETE"),
        "interpretation": spec["interpretation"],
    }
    if hosted is not None and profile is not None:
        report["hosted_profile"] = {
            "version": profile["version"],
            "sha256": profile_hash,
            "expected_judge": f"{HOSTED_PROVIDER}/{hosted.expected_model}",
            "output_cap_tokens": hosted.max_output_tokens,
            "not_applicable": sorted(profile["not_applicable"]),
        }
        report["self_preference"] = _self_preference(inputs)
    return report
