# ruff: noqa: F811  (imported pytest fixtures are re-declared as test arguments)
"""Milestone 11B: evaluating artifacts against the predeclared validation specification.

Each criterion is exercised at and around its threshold with synthetic artifacts built by the real
runners (an oracle judge standing in for the local model), so the tests show both that a good run
passes and that every individual failure mode is caught - and that missing evidence is never read
as a pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from copilot.evaluation.generation.judge import JudgeCache, LLMJudge
from copilot.evaluation.generation.judge_controls import ControlsRun, security_token
from copilot.evaluation.generation.judge_validation import (
    FAIL,
    NOT_EVALUATED,
    PASS,
    VALIDATION_SPEC_PATH,
    ValidationInputs,
    evaluate_primary_audit,
    evaluate_technical,
    evaluate_validation,
    load_validation_spec,
    spec_sha256,
    wilson_interval,
)
from copilot.evaluation.generation.judging import judge_run
from tests.unit.test_generation_judging import _generation, _judge_reply, _refs
from tests.unit.test_judge_controls import (
    _run,
    careless_reply,
    inputs,  # noqa: F401  (module fixture)
    oracle_reply,
    spec,  # noqa: F401  (module fixture)
)
from tests.unit.test_judge_runtime_identity import LocalFake, identity

ROOT = Path(__file__).resolve().parents[2]
SPEC = load_validation_spec(ROOT / VALIDATION_SPEC_PATH)
HASH = spec_sha256(ROOT / VALIDATION_SPEC_PATH)


def _ident(**kw):
    return identity(provider="ollama", **kw)


def _statuses(criteria) -> dict[str, str]:
    return {c["id"].split("_")[0]: c["status"] for c in criteria}


@pytest.fixture(scope="module")
def good(spec, inputs, tmp_path_factory):  # noqa: F811
    """Two control passes and two pilot passes from a well-behaved local judge."""
    base = tmp_path_factory.mktemp("validation")
    first = _run(spec, inputs, oracle_reply(spec), base / "c1")
    repeat = _run(spec, inputs, oracle_reply(spec), base / "c2", label="repeat", cache=False)

    run, ctx = _generation(base / "gen")
    questions = {"demo": {q.id: q for q in ctx.questions}}

    def pilot(directory: Path, *, cache: bool):
        llm = LocalFake(_judge_reply, ident=_ident())
        judge = LLMJudge(
            llm, max_output_tokens=1024, retry_sleep=lambda s: None,
            cache=JudgeCache(directory / "cache") if cache else None,
        )  # fmt: skip
        return judge_run(
            run, {q.id: q for q in ctx.questions}, _refs(ctx), judge,
            checkpoint_path=directory / "judgements-demo.json",
        )  # fmt: skip

    return {
        "controls": first,
        "repeat": repeat,
        "pilot": {"demo": pilot(base / "p1", cache=True)},
        "pilot_repeat": {"demo": pilot(base / "p2", cache=False)},
        "questions": questions,
    }


def agreement(n=8, exact=8, two=0, rows_ok=True):
    return {
        "rows_accounted_for": rows_ok,
        "correctness_exact_agreement": {"n": n, "exact": exact, "fraction": exact / n},
        "correctness_two_point_disagreements": {"n": n, "count": two, "fraction": two / n},
    }


def build(good, **over) -> ValidationInputs:
    values = {
        "controls": good["controls"], "controls_repeat": good["repeat"],
        "controls_dir": "c1", "controls_repeat_dir": "c2",
        "pilot": good["pilot"], "pilot_repeat": good["pilot_repeat"],
        "pilot_dir": "p1", "pilot_repeat_dir": "p2",
        "questions": good["questions"], "agreement": agreement(),
    }  # fmt: skip
    values.update(over)
    return ValidationInputs(**values)


# ---------------------------------------------------------------- the good run


def test_a_well_behaved_run_meets_every_pilot_criterion(good):
    report = evaluate_validation(SPEC, build(good), HASH)
    assert report["technical_acceptance"]["status"] == PASS
    assert report["semantic_pilot_acceptance"]["status"] == PASS
    assert report["overall"] == "PILOT_CRITERIA_MET"
    assert report["validation_spec"] == {"version": "judge-validation-v1", "sha256": HASH}
    assert "does not validate the judge" in report["interpretation"]
    ids = [c["id"].split("_")[0] for c in report["technical_acceptance"]["criteria"]]
    assert ids == ["T1", "T2", "T3", "T4", "T5", "T6"]


def test_missing_evidence_is_never_a_pass(good):
    report = evaluate_validation(SPEC, ValidationInputs(), HASH)
    assert report["overall"] == "INCOMPLETE"
    every = (
        report["technical_acceptance"]["criteria"] + report["semantic_pilot_acceptance"]["criteria"]
    )
    assert {c["status"] for c in every} == {NOT_EVALUATED}
    partial = evaluate_validation(SPEC, build(good, controls_repeat=None, pilot_repeat={}), HASH)
    assert _statuses(partial["semantic_pilot_acceptance"]["criteria"])["S5"] == NOT_EVALUATED
    assert partial["overall"] == "INCOMPLETE"


# ---------------------------------------------------------------- technical criteria


def test_t1_a_failed_pilot_judgement_fails(good):
    file = good["pilot"]["demo"]
    broken = file.model_copy(
        update={"items": [file.items[0].model_copy(update={"state": "failed", "assessment": None})]
                + file.items[1:]}
    )  # fmt: skip
    t = _statuses(evaluate_technical(SPEC, build(good, pilot={"demo": broken})))
    assert t["T1"] == FAIL and t["T2"] == PASS


def test_t2_the_arithmetic_one_failure_in_eighteen_is_below_95_percent(good):
    run = good["controls"]
    failed = run.results[0].model_copy(update={"state": "failed", "score": None})
    one_bad = run.model_copy(update={"results": [failed, *run.results[1:]]})
    crit = evaluate_technical(SPEC, build(good, controls=one_bad))[1]
    assert crit["status"] == FAIL and crit["judged"] == 17 and crit["fraction"] < 0.95
    short = run.model_copy(update={"results": run.results[:-1]})  # a missing control
    assert evaluate_technical(SPEC, build(good, controls=short))[1]["status"] == FAIL


@pytest.mark.parametrize(
    ("prompt", "completion", "fails"),
    [
        (7167, 100, False),
        (7168, 100, True),
        (6000, 1024, True),
        (7000, 1192, True),
        (500, 1023, False),
    ],
)
def test_t3_accepted_judgements_near_the_limits_are_flagged(good, prompt, completion, fails):
    run = good["controls"]
    record = run.results[0].record.model_copy(
        update={"prompt_tokens": prompt, "completion_tokens": completion}
    )
    tampered = run.model_copy(
        update={"results": [run.results[0].model_copy(update={"record": record}), *run.results[1:]]}
    )
    crit = evaluate_technical(SPEC, build(good, controls=tampered))[2]
    assert (crit["status"] == FAIL) is fails


def test_t3_a_judgement_without_token_counts_cannot_be_shown_safe(good):
    run = good["controls"]
    record = run.results[0].record.model_copy(update={"prompt_tokens": None})
    tampered = run.model_copy(
        update={"results": [run.results[0].model_copy(update={"record": record}), *run.results[1:]]}
    )
    assert evaluate_technical(SPEC, build(good, controls=tampered))[2]["status"] == FAIL


def test_t4_security_failures_and_a_leaked_secret_fail(good):
    run: ControlsRun = good["controls"]
    bad_security = run.security_result.model_copy(update={"passed": False, "gate_fired": False})
    assert evaluate_technical(SPEC, build(good, controls=run.model_copy(
        update={"security_result": bad_security})))[3]["status"] == FAIL  # fmt: skip
    leaky = run.model_copy(update={"results": [
        run.results[0].model_copy(update={"error": f"oops {security_token()}"}), *run.results[1:]
    ]})  # fmt: skip
    assert evaluate_technical(SPEC, build(good, controls=leaky))[3]["status"] == FAIL
    missing = run.model_copy(update={"security_result": None})
    assert evaluate_technical(SPEC, build(good, controls=missing))[3]["status"] == FAIL


@pytest.mark.parametrize(
    "change",
    [
        {"quantization": "Q8_0"},
        {"model_tag": "llama3.1:70b"},
        {"context_size": 4096},
        {"seed": 1},
        {"structured_output": "none"},
    ],
)
def test_t5_a_runtime_that_is_not_the_declared_judge_fails(good, change):
    run = good["controls"]
    runtime = {**run.runtime, **change}
    from copilot.llm import RuntimeIdentity

    other = run.model_copy(
        update={"runtime": runtime, "runtime_fingerprint": RuntimeIdentity(**runtime).fingerprint()}
    )
    assert evaluate_technical(SPEC, build(good, controls=other))[4]["status"] == FAIL


def test_t5_a_forged_or_missing_fingerprint_fails(good):
    run = good["controls"]
    forged = run.model_copy(update={"runtime_fingerprint": "f" * 64})
    assert evaluate_technical(SPEC, build(good, controls=forged))[4]["status"] == FAIL
    missing = run.model_copy(update={"runtime_fingerprint": None})
    assert evaluate_technical(SPEC, build(good, controls=missing))[4]["status"] == FAIL


def test_t5_artifacts_with_different_fingerprints_fail(good, spec, inputs, tmp_path):  # noqa: F811
    other_llm = LocalFake(oracle_reply(spec), ident=_ident(model_digest="b" * 64))
    other = _run(
        spec, inputs, oracle_reply(spec), tmp_path, label="repeat", cache=False, llm=other_llm
    )
    crit = evaluate_technical(SPEC, build(good, controls_repeat=other))
    assert crit[4]["status"] == FAIL and crit[5]["status"] == FAIL


def test_t6_a_repeat_that_used_the_cache_or_the_same_directory_fails(good):
    cached = good["repeat"].model_copy(update={"cache_used": True, "cache_hits": 3})
    assert evaluate_technical(SPEC, build(good, controls_repeat=cached))[5]["status"] == FAIL
    assert evaluate_technical(SPEC, build(good, controls_repeat_dir="c1"))[5]["status"] == FAIL
    assert evaluate_technical(SPEC, build(good, pilot_repeat_dir="p1"))[5]["status"] == FAIL
    file = good["pilot_repeat"]["demo"]
    reused = file.model_copy(update={"items": [file.items[0].model_copy(update={"reused": True}),
                                                *file.items[1:]]})  # fmt: skip
    assert evaluate_technical(SPEC, build(good, pilot_repeat={"demo": reused}))[5]["status"] == FAIL


# ---------------------------------------------------------------- semantic criteria


def test_s1_s2_thresholds_at_and_around_the_boundary(good):
    def semantic(**kw):
        return _statuses(evaluate_validation(SPEC, build(good, agreement=agreement(**kw)), HASH)
                         ["semantic_pilot_acceptance"]["criteria"])  # fmt: skip

    assert semantic(n=8, exact=6)["S1"] == PASS  # 0.75 exactly
    assert semantic(n=8, exact=5)["S1"] == FAIL  # 0.625
    assert semantic(n=8, exact=8, two=0)["S2"] == PASS
    assert semantic(n=8, exact=7, two=1)["S2"] == FAIL
    small = semantic(n=3, exact=3)
    assert small["S1"] == NOT_EVALUATED and small["S2"] == NOT_EVALUATED  # inconclusive, not a pass
    assert semantic(n=8, exact=8, rows_ok=False)["S1"] == NOT_EVALUATED


def test_s3_s4_a_careless_judge_fails_the_negative_controls(good, spec, inputs, tmp_path):  # noqa: F811
    careless = _run(spec, inputs, careless_reply, tmp_path, llm=LocalFake(
        careless_reply, ident=_ident()))  # fmt: skip
    s = _statuses(evaluate_validation(SPEC, build(good, controls=careless), HASH)
                  ["semantic_pilot_acceptance"]["criteria"])  # fmt: skip
    assert s["S3"] == FAIL and s["S4"] == FAIL


def test_s3_s4_pass_for_the_oracle_and_name_what_was_missed_otherwise(good):
    s = evaluate_validation(SPEC, build(good), HASH)["semantic_pilot_acceptance"]["criteria"]
    by_id = {c["id"].split("_")[0]: c for c in s}
    assert by_id["S3"]["status"] == PASS and "6/6" in by_id["S3"]["detail"]
    assert by_id["S4"]["status"] == PASS and "4/4" in by_id["S4"]["detail"]
    run = good["controls"]
    wrong = run.model_copy(update={"results": [
        r.model_copy(update={"score": 2}) if r.control_id == "ctl-io-03" else r for r in run.results
    ]})  # fmt: skip
    s3 = evaluate_validation(SPEC, build(good, controls=wrong), HASH)["semantic_pilot_acceptance"]
    crit = next(c for c in s3["criteria"] if c["id"].startswith("S3"))
    assert crit["status"] == FAIL and crit["missed"] == ["ctl-io-03"]


def test_s5_repeat_consistency_boundary(good):
    repeat = good["repeat"]
    total = 18 + 6  # controls + the six judged pilot answers

    def with_changes(count):
        changed = [
            r.model_copy(update={"score": (r.score + 1) % 3}) if i < count else r
            for i, r in enumerate(repeat.results)
        ]
        return repeat.model_copy(update={"results": changed})

    def s5(count):
        crit = evaluate_validation(SPEC, build(good, controls_repeat=with_changes(count)), HASH)
        return next(
            c for c in crit["semantic_pilot_acceptance"]["criteria"] if c["id"].startswith("S5")
        )

    assert s5(0)["status"] == PASS and s5(0)["items"] == total
    assert s5(1)["status"] == PASS  # 23/24 = 0.958
    assert s5(2)["status"] == FAIL  # 22/24 = 0.917
    assert len(s5(2)["inconsistent"]) == 2


def test_s5_a_failed_item_in_the_repeat_counts_as_inconsistent(good):
    repeat = good["repeat"]
    lost = repeat.results[0].model_copy(update={"state": "failed", "score": None})
    crit = evaluate_validation(
        SPEC,
        build(
            good, controls_repeat=repeat.model_copy(update={"results": [lost, *repeat.results[1:]]})
        ),
        HASH,
    )["semantic_pilot_acceptance"]["criteria"]
    assert next(c for c in crit if c["id"].startswith("S5"))["inconsistent"] == [lost.control_id]


# ---------------------------------------------------------------- primary audit criteria


def primary(exact=0.80, kappa=0.65, two=0.04, coverage=0.97, n=100):
    return {
        "rated_answers": n,
        "correctness_exact_agreement": {"n": n, "exact": round(exact * n), "fraction": exact},
        "correctness_two_point_disagreements": {"n": n, "count": round(two * n), "fraction": two},
        "correctness_score_0_2": {"n": n, "kappa": kappa, "kappa_undefined_reason": "ties"},
        "valid_judgement_coverage": {
            "answers_requiring_a_judgement": 100,
            "with_a_valid_judgement": round(coverage * 100),
            "fraction": coverage,
        },  # fmt: skip
        "completeness": {"key_entries": 100, "other_exclusions": {"x": ["a"]}},
    }


def test_primary_audit_passes_only_when_every_criterion_is_met():
    result = evaluate_primary_audit(SPEC, primary())
    assert result["status"] == PASS and result["label"] == "validated-on-this-sample"
    assert result["sample_size"] == 100 and result["exclusions"] == {"x": ["a"]}
    assert result["criteria"][0]["wilson_95"][0] < 0.80 < result["criteria"][0]["wilson_95"][1]


@pytest.mark.parametrize(
    ("override", "failing"),
    [
        ({"exact": 0.74}, "P1"),
        ({"kappa": 0.59}, "P2"),
        ({"kappa": None}, "P2"),
        ({"two": 0.06}, "P3"),
        ({"coverage": 0.94}, "P4"),
    ],
)
def test_primary_audit_failures_label_the_conclusions_exploratory(override, failing):
    result = evaluate_primary_audit(SPEC, primary(**override))
    statuses = {c["id"].split("_")[0]: c["status"] for c in result["criteria"]}
    assert statuses[failing] == FAIL and result["status"] == FAIL
    assert result["label"] == "EXPLORATORY" and "further human verification" in result["conclusion"]


def test_an_undefined_kappa_is_explicitly_not_a_pass():
    result = evaluate_primary_audit(SPEC, primary(kappa=None))
    detail = next(c for c in result["criteria"] if c["id"].startswith("P2"))["detail"]
    assert "undefined" in detail and "not a pass" in detail


def test_primary_boundaries_are_inclusive():
    assert (
        evaluate_primary_audit(SPEC, primary(exact=0.75, kappa=0.60, two=0.05, coverage=0.95))[
            "status"
        ]
        == PASS
    )


def test_wilson_interval_matches_a_known_value():
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-3) and high == pytest.approx(0.9433, abs=1e-3)
    assert wilson_interval(0, 0) is None
    assert wilson_interval(10, 10)[1] == 1.0


def test_the_report_is_json_serialisable(good):
    json.dumps(evaluate_validation(SPEC, build(good), HASH))
