# ruff: noqa: F811  (imported pytest fixtures are re-declared as test arguments)
"""Milestone 11B closeout: judge-validation-v1 applied to a hosted judge, offline.

The hosted profile changes no threshold. These tests show (a) the profile is pinned to the exact v1
file and adapts only T3/T5/T6, (b) a hosted run is judged by the same thresholds - a careless
hosted judge fails exactly as the local one did - (c) wrong or inconsistent model ids fail, and
(d) the hosted preflight and call budget make no provider call. Nothing here calls a provider.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from copilot.config import Settings
from copilot.evaluation.generation import __main__ as cli
from copilot.evaluation.generation.judge import JudgeCache, LLMJudge
from copilot.evaluation.generation.judge_preflight import hosted_call_budget
from copilot.evaluation.generation.judge_validation import (
    FAIL,
    HOSTED_PROFILE_PATH,
    NOT_EVALUATED,
    PASS,
    VALIDATION_SPEC_PATH,
    HostedContext,
    HostedProfileError,
    ValidationInputs,
    evaluate_technical,
    evaluate_validation,
    load_hosted_profile,
    load_validation_spec,
    spec_sha256,
)
from copilot.evaluation.generation.judging import judge_run
from copilot.llm import FakeLLMClient, GuardedLLMClient
from tests.unit.test_generation_judging import _generation, _judge_reply, _refs
from tests.unit.test_judge_controls import (
    _run,
    careless_reply,
    inputs,  # noqa: F401  (module fixture)
    oracle_reply,
    spec,  # noqa: F401  (module fixture)
)

ROOT = Path(__file__).resolve().parents[2]
SPEC = load_validation_spec(ROOT / VALIDATION_SPEC_PATH)
SPEC_HASH = spec_sha256(ROOT / VALIDATION_SPEC_PATH)
PROFILE = load_hosted_profile(ROOT / HOSTED_PROFILE_PATH, ROOT / VALIDATION_SPEC_PATH)
PROFILE_HASH = spec_sha256(ROOT / HOSTED_PROFILE_PATH)
MODEL = "judge-model-under-test"
CAP = 4096
CTX = HostedContext(MODEL, CAP)


class HostedFake(FakeLLMClient):
    """A hosted-style judge: provider 'gemini', no local runtime identity to fingerprint."""

    provider = "gemini"

    def __init__(self, reply, model: str = MODEL):
        super().__init__(reply, model=model)


def _hosted(reply, model: str = MODEL) -> HostedFake:
    return HostedFake(reply, model)


def _statuses(criteria) -> dict[str, str]:
    return {c["id"].split("_")[0]: c["status"] for c in criteria}


def _agreement(n=8, exact=8, two=0):
    return {
        "rows_accounted_for": True,
        "correctness_exact_agreement": {"n": n, "exact": exact, "fraction": exact / n},
        "correctness_two_point_disagreements": {"n": n, "count": two, "fraction": two / n},
    }


# ---------------------------------------------------------------- the profile


def test_the_profile_is_pinned_to_the_exact_v1_file_and_changes_no_threshold():
    assert PROFILE["base_spec"]["sha256"] == SPEC_HASH
    assert PROFILE["judge"]["provider"] == "gemini" and PROFILE["judge"]["model"] is None
    assert PROFILE["judge"]["judge_prompt_version"] == SPEC["judge"]["judge_prompt_version"]
    assert PROFILE["judge"]["rubric_version"] == SPEC["judge"]["rubric_version"]
    assert PROFILE["judge"]["temperature"] == SPEC["judge"]["temperature"] == 0.0
    # Only T3/T5/T6 are adapted; everything else is inherited by reference, not restated.
    assert {k for k in PROFILE["adapted"]} == {"T3", "T5", "T6"}
    assert set(PROFILE["inherited_unchanged"]["criteria"]) == {
        "T1", "T2", "T4", "S1", "S2", "S3", "S4", "S5", "P1", "P2", "P3", "P4",
    }  # fmt: skip
    t3 = SPEC["technical_acceptance"]["T3_no_accepted_context_overflow_or_truncation"]
    assert PROFILE["adapted"]["T3"]["max_violations"] == t3["max_violations"] == 0
    text = json.dumps(PROFILE)
    for numeric_threshold in ("min_fraction", "required_fraction", "exact_agreement_min"):
        assert numeric_threshold not in text


def test_a_profile_for_another_v1_file_or_provider_is_refused(tmp_path):
    bad = dict(PROFILE, base_spec={**PROFILE["base_spec"], "sha256": "0" * 64})
    path = tmp_path / "p.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(HostedProfileError, match="different judge-validation-v1"):
        load_hosted_profile(path, ROOT / VALIDATION_SPEC_PATH)
    other = dict(PROFILE, judge={**PROFILE["judge"], "provider": "ollama"})
    path.write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(HostedProfileError, match="provider"):
        load_hosted_profile(path, ROOT / VALIDATION_SPEC_PATH)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(HostedProfileError):
        load_hosted_profile(path, ROOT / VALIDATION_SPEC_PATH)


def test_the_hosted_context_needs_an_explicit_model_and_a_real_cap():
    with pytest.raises(HostedProfileError):
        HostedContext("  ", 4096)
    with pytest.raises(HostedProfileError):
        HostedContext("m", 8)


def test_a_hosted_evaluation_needs_both_the_profile_and_the_context(good_hosted):
    with pytest.raises(HostedProfileError):
        evaluate_validation(SPEC, good_hosted, SPEC_HASH, hosted=CTX)
    with pytest.raises(HostedProfileError):
        evaluate_validation(SPEC, good_hosted, SPEC_HASH, profile=PROFILE)


# ---------------------------------------------------------------- a hosted run


@pytest.fixture(scope="module")
def good_hosted(spec, inputs, tmp_path_factory):  # noqa: F811
    base = tmp_path_factory.mktemp("hosted")
    first = _run(spec, inputs, None, base / "c1", llm=_hosted(oracle_reply(spec)))
    repeat = _run(
        spec,
        inputs,
        None,
        base / "c2",
        label="repeat",
        cache=False,
        llm=_hosted(oracle_reply(spec)),
    )
    run, ctx = _generation(base / "gen")

    def pilot(directory: Path, *, cache: bool):
        judge = LLMJudge(
            _hosted(_judge_reply), max_output_tokens=CAP, retry_sleep=lambda s: None,
            cache=JudgeCache(directory / "cache") if cache else None,
        )  # fmt: skip
        return judge_run(
            run, {q.id: q for q in ctx.questions}, _refs(ctx), judge,
            checkpoint_path=directory / "judgements-demo.json",
        )  # fmt: skip

    return ValidationInputs(
        controls=first, controls_repeat=repeat, controls_dir="c1", controls_repeat_dir="c2",
        pilot={"demo": pilot(base / "p1", cache=True)},
        pilot_repeat={"demo": pilot(base / "p2", cache=False)},
        pilot_dir="p1", pilot_repeat_dir="p2",
        questions={"demo": {q.id: q for q in ctx.questions}}, agreement=_agreement(),
    )  # fmt: skip


def _evaluate(inp: ValidationInputs, ctx: HostedContext = CTX) -> dict:
    return evaluate_validation(
        SPEC, inp, SPEC_HASH, hosted=ctx, profile=PROFILE, profile_hash=PROFILE_HASH
    )


def _with(base: ValidationInputs, **over) -> ValidationInputs:
    values = {name: getattr(base, name) for name in base.__dataclass_fields__}
    values.update(over)
    return ValidationInputs(**values)


def test_a_well_behaved_hosted_run_meets_every_pilot_criterion_without_a_fingerprint(good_hosted):
    assert good_hosted.controls.runtime_fingerprint is None
    report = _evaluate(good_hosted)
    assert report["overall"] == "PILOT_CRITERIA_MET", report
    ids = [c["id"].split("_")[0] for c in report["technical_acceptance"]["criteria"]]
    assert ids == ["T1", "T2", "T3", "T4", "T5H", "T6"]
    assert report["hosted_profile"] == {
        "version": "judge-validation-v1-hosted",
        "sha256": PROFILE_HASH,
        "expected_judge": f"gemini/{MODEL}",
        "output_cap_tokens": CAP,
        "not_applicable": sorted(PROFILE["not_applicable"]),
    }
    assert report["self_preference"]["same_model_as_generator"] is False
    assert "same vendor family" in report["self_preference"]["note"]
    assert "validation_spec" in report and report["validation_spec"]["sha256"] == SPEC_HASH


def test_the_local_profile_still_rejects_hosted_artifacts_for_lack_of_a_fingerprint(good_hosted):
    local = evaluate_validation(SPEC, good_hosted, SPEC_HASH)
    assert _statuses(local["technical_acceptance"]["criteria"])["T5"] == FAIL


def test_thresholds_are_not_relaxed_a_careless_hosted_judge_fails_like_the_local_one(
    spec,
    inputs,
    good_hosted,
    tmp_path,  # noqa: F811
):
    careless = _run(spec, inputs, None, tmp_path, llm=_hosted(careless_reply))
    report = _evaluate(_with(good_hosted, controls=careless))
    semantic = _statuses(report["semantic_pilot_acceptance"]["criteria"])
    assert semantic["S3"] == FAIL and semantic["S4"] == FAIL
    assert report["overall"] == FAIL


def test_one_unparsed_control_still_fails_t2_for_a_hosted_judge(good_hosted):
    run = good_hosted.controls
    failed = run.results[0].model_copy(update={"state": "failed", "score": None})
    one_bad = run.model_copy(update={"results": [failed, *run.results[1:]]})
    t = _statuses(evaluate_technical(SPEC, _with(good_hosted, controls=one_bad), CTX))
    assert t["T2"] == FAIL


# ---------------------------------------------------------------- T5H identity and T6


def test_t5h_a_different_or_mixed_model_id_fails(good_hosted):
    other = _evaluate(good_hosted, HostedContext("some-other-model", CAP))
    crit = next(c for c in other["technical_acceptance"]["criteria"] if c["id"].startswith("T5H"))
    assert crit["status"] == FAIL and len(crit["problems"]) >= 4

    run = good_hosted.controls
    mixed = run.model_copy(update={"judge_model": "intruder"})
    t = _evaluate(_with(good_hosted, controls=mixed))["technical_acceptance"]["criteria"]
    assert _statuses(t)["T5H"] == FAIL
    wrong_provider = run.model_copy(update={"judge_provider": "ollama"})
    t = _evaluate(_with(good_hosted, controls=wrong_provider))["technical_acceptance"]["criteria"]
    assert _statuses(t)["T5H"] == FAIL


def test_t6_passes_with_the_same_hosted_model_and_fails_when_the_repeat_used_another(good_hosted):
    base = _statuses(_evaluate(good_hosted)["technical_acceptance"]["criteria"])
    assert base["T6"] == PASS
    repeat = good_hosted.controls_repeat.model_copy(update={"judge_model": "intruder"})
    t = _statuses(
        _evaluate(_with(good_hosted, controls_repeat=repeat))["technical_acceptance"]["criteria"]
    )
    assert t["T6"] == FAIL
    # the independence checks of v1 are unchanged
    cached = good_hosted.controls_repeat.model_copy(update={"cache_used": True, "cache_hits": 2})
    t = _statuses(
        _evaluate(_with(good_hosted, controls_repeat=cached))["technical_acceptance"]["criteria"]
    )
    assert t["T6"] == FAIL
    t = _statuses(
        _evaluate(_with(good_hosted, controls_repeat_dir="c1"))["technical_acceptance"]["criteria"]
    )
    assert t["T6"] == FAIL


def test_missing_hosted_evidence_is_not_evaluated_never_passed():
    report = _evaluate(ValidationInputs())
    assert report["overall"] == "INCOMPLETE"
    assert _statuses(report["technical_acceptance"]["criteria"])["T5H"] == NOT_EVALUATED


# ---------------------------------------------------------------- T3 output cap only


@pytest.mark.parametrize(
    ("completion", "reasoning", "fails"),
    [
        (100, None, False),
        (CAP - 1, 0, False),
        (CAP, 0, True),
        (CAP - 10, 10, True),
        (CAP - 11, 10, False),
        (None, None, True),
    ],
)
def test_t3_applies_the_output_cap_including_reasoning_tokens(
    good_hosted, completion, reasoning, fails
):
    run = good_hosted.controls
    record = run.results[0].record.model_copy(
        update={"completion_tokens": completion, "reasoning_tokens": reasoning}
    )
    tampered = run.model_copy(
        update={"results": [run.results[0].model_copy(update={"record": record}), *run.results[1:]]}
    )
    crit = evaluate_technical(SPEC, _with(good_hosted, controls=tampered), CTX)[2]
    assert (crit["status"] == FAIL) is fails


def test_t3_does_not_apply_a_local_window_to_a_hosted_prompt(good_hosted):
    run = good_hosted.controls
    record = run.results[0].record.model_copy(update={"prompt_tokens": 200_000})
    big = run.model_copy(
        update={"results": [run.results[0].model_copy(update={"record": record}), *run.results[1:]]}
    )
    assert evaluate_technical(SPEC, _with(good_hosted, controls=big), CTX)[2]["status"] == PASS


# ---------------------------------------------------------------- self-preference


def test_self_preference_is_reported_when_the_judge_is_the_generation_model(good_hosted):
    file = good_hosted.pilot["demo"].model_copy(update={"same_model_as_generator": True})
    report = _evaluate(_with(good_hosted, pilot={"demo": file}))
    assert report["self_preference"]["same_model_as_generator"] is True
    assert "self-preference bias" in report["self_preference"]["note"]
    assert report["overall"] == "PILOT_CRITERIA_MET"  # documented, not scored


def test_self_preference_is_unknown_without_pilot_judgements(good_hosted):
    report = _evaluate(_with(good_hosted, pilot={}, pilot_repeat={}, agreement=None))
    assert report["self_preference"]["same_model_as_generator"] is None


# ---------------------------------------------------------------- the call budget


def test_the_budget_is_exact_arithmetic_with_and_without_retries():
    b = hosted_call_budget(controls=18, pilot_answers=8, primary_answers=144, max_attempts=5)
    s = b["stages"]
    assert b["retry_attempts_per_call"] == 5
    assert (
        s["controls_first_pass"]["max_calls"],
        s["controls_first_pass"]["max_calls_with_retries"],
    ) == (18, 90)
    assert (s["pilot_judging"]["max_calls"], s["pilot_judging"]["max_calls_with_retries"]) == (
        8,
        40,
    )
    assert (s["primary_judging"]["max_calls"], s["primary_judging"]["max_calls_with_retries"]) == (
        144,
        720,
    )
    assert s["validation_and_pilot_total"] == {"max_calls": 52, "max_calls_with_retries": 260}
    one = hosted_call_budget(controls=18, pilot_answers=8, primary_answers=144, max_attempts=1)
    assert one["stages"]["primary_judging"]["max_calls_with_retries"] == 144
    with pytest.raises(ValueError):
        hosted_call_budget(controls=1, pilot_answers=1, primary_answers=1, max_attempts=0)


# ---------------------------------------------------------------- CLI (no provider is ever built)


@pytest.fixture
def no_provider(monkeypatch):
    """A guard: any attempt to construct a judge client fails the test."""

    def boom(*args, **kwargs):  # pragma: no cover - only reached on a bug
        raise AssertionError("a provider client was constructed")

    monkeypatch.setattr(cli, "create_client", boom)
    monkeypatch.setattr(
        cli, "get_settings", lambda: Settings(_env_file=None, GEMINI_API_KEY="not-a-real-key")
    )


def _preflight(*more: str) -> list[str]:
    return [
        "judge-preflight", "--root", str(ROOT), "--judge-provider", "gemini",
        "--judge-model", MODEL, "--judgements-dir", "data/judge-validation/_test-hosted/controls",
        *more,
    ]  # fmt: skip


def test_hosted_preflight_is_offline_and_reports_the_budget(no_provider, capsys):
    code = cli.main(_preflight("--confirm-model", "gemini/gemini-3.6-flash"))
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert code == 0 and result["hosted_calls_made"] == 0 and result["ok"] is True
    stages = result["budget"]["stages"]
    assert stages["controls_first_pass"]["max_calls"] == 18
    assert stages["pilot_judging"]["max_calls"] == 8
    assert stages["primary_judging"]["max_calls"] == 144
    assert stages["primary_judging"]["max_calls_with_retries"] == 144 * 5
    assert result["same_model_as_generator"] is False
    assert result["hosted_profile"]["sha256"] == PROFILE_HASH
    assert result["validation_spec_sha256"] == SPEC_HASH
    assert not (ROOT / "data/judge-validation/_test-hosted").exists()  # nothing was written
    assert "not-a-real-key" not in captured.out + captured.err


def test_hosted_preflight_flags_the_same_model_and_a_missing_key(no_provider, monkeypatch, capsys):
    code = cli.main(
        [*_preflight("--confirm-model", f"gemini/{MODEL}", "--retry-max-attempts", "3")]
    )
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert code == 0 and result["same_model_as_generator"] is True
    assert "self-preference bias" in captured.err
    assert result["budget"]["stages"]["controls_first_pass"]["max_calls_with_retries"] == 54

    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    code = cli.main(_preflight())
    captured = capsys.readouterr()
    assert code == 3 and json.loads(captured.out)["ok"] is False
    assert "GEMINI_API_KEY" in captured.err and "generation model is unknown" in captured.err


def test_hosted_preflight_refuses_local_only_options_and_the_primary_directory(no_provider, capsys):
    assert cli.main(_preflight("--warm-up")) == 2
    assert "local judge only" in capsys.readouterr().err
    argv = _preflight()
    argv[argv.index("--judgements-dir") + 1] = "data/generation-eval/primary/controls"
    assert cli.main(argv) == 2
    assert "primary" in capsys.readouterr().err


def test_judge_validate_hosted_needs_a_declared_model_and_refuses_the_flag_otherwise(
    no_provider, capsys
):
    assert cli.main(["judge-validate", "--root", str(ROOT), "--judge-profile", "hosted"]) == 2
    assert "--expect-judge-model" in capsys.readouterr().err
    assert cli.main(["judge-validate", "--root", str(ROOT), "--expect-judge-model", MODEL]) == 2
    assert "hosted only" in capsys.readouterr().err


def test_judge_controls_and_validate_run_end_to_end_for_a_hosted_judge(
    spec,
    monkeypatch,
    tmp_path,
    capsys,  # noqa: F811
):
    monkeypatch.setattr(
        cli, "create_client", lambda settings, **kw: GuardedLLMClient(_hosted(oracle_reply(spec)))
    )
    monkeypatch.setattr(
        cli, "get_settings", lambda: Settings(_env_file=None, GEMINI_API_KEY="not-a-real-key")
    )
    target = tmp_path / "controls"
    base = ["--root", str(ROOT), "--judge-provider", "gemini", "--judge-model", MODEL]
    assert cli.main(["judge-controls", *base, "--judgements-dir", str(target)]) == 0
    summary = json.loads(capsys.readouterr().out.split("wrote ")[0])
    assert summary["judged"] == 18 and summary["all_expectations_met"] == 18

    argv = ["judge-validate", "--root", str(ROOT), "--judge-profile", "hosted",
            "--controls-dir", str(target)]  # fmt: skip
    code = cli.main([*argv, "--expect-judge-model", MODEL])
    report = json.loads(capsys.readouterr().out)
    assert code == 4 and report["overall"] == "INCOMPLETE"  # no pilot or repeat evidence yet
    assert report["hosted_profile"]["expected_judge"] == f"gemini/{MODEL}"
    assert (target / "judge-validation-report-hosted.json").is_file()
    assert not (target / "judge-validation-report.json").exists()
    t = _statuses(report["technical_acceptance"]["criteria"])
    assert t["T2"] == PASS and t["T4"] == PASS and t["T5H"] == PASS

    code = cli.main([*argv, "--expect-judge-model", "another-model"])
    report = json.loads(capsys.readouterr().out)
    assert code == 3 and _statuses(report["technical_acceptance"]["criteria"])["T5H"] == FAIL
