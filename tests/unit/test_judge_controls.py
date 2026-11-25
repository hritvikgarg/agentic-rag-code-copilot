# ruff: noqa: F811  (imported pytest fixtures are re-declared as test arguments)
"""Milestone 11B: the synthetic judge controls - design, scoring consistency, isolation, running.

The committed controls (``generation-eval/judge-controls/controls-v1.json``) are checked against the
committed benchmarks and the rubric. The runner is exercised with an *oracle* judge that answers
exactly what each control expects, and with deliberately bad judges, so the tests prove that the
controls pass a sensible judge and fail a careless one. No network, no model.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import httpx
import pytest

from copilot.evaluation.generation import __main__ as cli
from copilot.evaluation.generation.judge import JudgeCache, LLMJudge
from copilot.evaluation.generation.judge_controls import (
    CONTROLS_PATH,
    MODEL_KINDS,
    ControlsError,
    ControlsMismatchError,
    check_against_benchmarks,
    controls_sha256,
    load_controls,
    load_results,
    results_path,
    run_controls,
    security_token,
)
from copilot.evaluation.generation.judge_validation import VALIDATION_SPEC_PATH, spec_sha256
from copilot.evaluation.generation.pilot import PILOT_SPEC_PATH, load_pilot_spec
from copilot.evaluation.generation.protocol import PROTOCOL_PATH, load_protocol
from copilot.evaluation.generation.rubric import Assessment, FactVerdict, score_assessment
from copilot.evaluation.generation.schema import load_generation_benchmark
from copilot.llm import create_llm_client
from copilot.llm import factory as llm_factory
from tests.unit.test_judge_runtime_identity import LocalFake, identity
from tests.unit.test_ollama_client import chat_ok, installed

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "copilot"
needs_repos = pytest.mark.skipif(
    not (ROOT / "data/snapshots/a282ec471ebf1387cb4feafcfe732ad24a431c7a/tree").is_dir(),
    reason="the frozen copilot snapshot is not present in this checkout",
)

# What each kind MUST mean, written independently of controls-v1.json and of any judge.
KIND_CONTRACT = {
    "correct_substantive": {"scores": {2}, "abstained": False, "types": {"answered"}},
    "partially_correct": {"scores": {1}, "abstained": False, "types": {"answered"}},
    "confident_fabrication": {"scores": {0}, "abstained": False, "types": {"answered"}},
    "contradicted_fact": {"scores": {0}, "abstained": False, "types": {"answered"}},
    "correct_abstention": {"scores": {2}, "abstained": True, "types": {"insufficient_evidence"}},
    "incorrect_refusal": {"scores": {0}, "abstained": True, "types": {"insufficient_evidence"}},
    "generic_refusal": {"scores": {0}, "abstained": False, "types": {"generic_refusal"}},
    "mixed_answer_and_abstention": {"scores": {1}, "abstained": True, "types": {"mixed"}},
    "invented_symbol": {"scores": {0}, "abstained": False, "types": {"answered"}},
}


@pytest.fixture(scope="module")
def spec():
    return load_controls(ROOT / CONTROLS_PATH)


def _questions():
    protocol = load_protocol(ROOT / PROTOCOL_PATH)
    return {
        r.repository_id: {
            q.id: q for q in load_generation_benchmark(ROOT / r.benchmark_path).questions
        }
        for r in protocol.repositories
    }


# ---------------------------------------------------------------- design of the committed suite


def test_the_suite_covers_every_required_kind_in_both_repositories(spec):
    assert len(spec.controls) == 18 and len(spec.parser_controls) == 10
    assert set(MODEL_KINDS) == {c.kind for c in spec.controls} == set(KIND_CONTRACT)
    for kind in MODEL_KINDS:
        repos = {c.repository_id for c in spec.controls if c.kind == kind}
        assert repos == {"inventory-orders", "copilot"}, kind


def test_every_expected_outcome_matches_the_independent_contract(spec):
    for c in spec.controls:
        contract = KIND_CONTRACT[c.kind]
        assert set(c.expected.correctness_scores) == contract["scores"], c.control_id
        assert c.expected.abstained is contract["abstained"], c.control_id
        assert set(c.expected.response_types) == contract["types"], c.control_id


def test_the_parser_and_security_controls_cover_the_malformed_cases(spec):
    kinds = {c.kind for c in spec.parser_controls}
    expected = {"prose_instead_of_json", "truncated_json", "missing_response_type"}
    expected |= {"response_type_contradicts_abstained", "groundedness_out_of_range"}
    assert expected <= kinds
    assert sum(c.expected == "accept" for c in spec.parser_controls) == 2
    assert spec.security_control.control_id == "ctl-sec-01"


@needs_repos
def test_controls_name_real_non_pilot_questions_with_matching_fact_ids(spec):
    pilot = load_pilot_spec(ROOT / PILOT_SPEC_PATH).questions
    check_against_benchmarks(spec, _questions(), pilot)  # raises on any mismatch
    used = {(c.repository_id, c.question_id) for c in spec.controls}
    assert all(q not in pilot.get(r, ()) for r, q in used)


@needs_repos
def test_a_control_on_a_pilot_question_is_refused(spec):
    pilot = {"inventory-orders": ("io-discount-tiers",)}
    with pytest.raises(ControlsError, match="pilot question"):
        check_against_benchmarks(spec, _questions(), pilot)


@needs_repos
def test_control_answers_are_not_the_benchmarks_reference_text(spec):
    questions = _questions()
    for c in spec.controls:
        q = questions[c.repository_id][c.question_id]
        assert c.answer.strip() != q.acceptable_answer.strip()
        for fact in q.key_facts:
            assert fact.statement not in c.answer, f"{c.control_id} copies a key-fact statement"


@needs_repos
def test_expected_outcomes_are_consistent_with_the_rubric(spec):
    """Every combination of allowed verdicts must score inside the expected scores."""
    questions = _questions()
    for c in spec.controls:
        q = questions[c.repository_id][c.question_id]
        ids = list(c.expected.fact_verdicts)
        unsupported_options = {"none": [0], "at_least_one": [1], "any": [0, 1]}[
            c.expected.unsupported_claims
        ]
        for combo in itertools.product(*(c.expected.fact_verdicts[i] for i in ids)):
            for n_unsupported in unsupported_options:
                a = Assessment(
                    evaluator_kind="judge", evaluator_id="x",
                    fact_verdicts={i: FactVerdict(v) for i, v in zip(ids, combo, strict=True)},
                    material_unsupported_claims=tuple("claim" for _ in range(n_unsupported)),
                    groundedness=2, abstained=c.expected.abstained,
                    response_type=c.expected.response_types[0],
                )  # fmt: skip
                assert score_assessment(q, a).score in c.expected.correctness_scores, (
                    c.control_id, combo, n_unsupported,
                )  # fmt: skip


def test_the_validation_specification_points_at_these_controls(spec):
    validation = json.loads((ROOT / VALIDATION_SPEC_PATH).read_text())
    assert validation["controls"]["specification"] == CONTROLS_PATH
    assert validation["controls"]["model_controls_expected"] == len(spec.controls)


# ---------------------------------------------------------------- isolation from primary metrics


def test_no_metric_or_report_code_imports_the_controls():
    allowed = {"__main__.py", "judge_validation.py", "judge_controls.py"}
    offenders = [
        p.name
        for p in (SRC / "evaluation").rglob("*.py")
        if "judge_controls" in p.read_text() and p.name not in allowed
    ]
    assert offenders == []
    for name in ("analysis.py", "audit.py", "audit_agree.py", "audit_population.py", "judging.py"):
        assert "judge_controls" not in (SRC / "evaluation/generation" / name).read_text()


def test_control_ids_are_in_their_own_namespace(spec):
    questions = _questions() if (ROOT / PROTOCOL_PATH).is_file() else {}
    benchmark_ids = {q for repo in questions.values() for q in repo}
    ids = [c.control_id for c in spec.controls] + [c.control_id for c in spec.parser_controls]
    assert all(i.startswith("ctl-") for i in ids) and not set(ids) & benchmark_ids


# ---------------------------------------------------------------- running the controls (oracle)


def oracle_reply(spec):
    """A judge that answers exactly what each control expects (found by its answer text)."""
    by_answer = {c.answer: c for c in spec.controls}

    def reply(request) -> str:
        for answer, c in by_answer.items():
            if answer in request.user_prompt:
                exp = c.expected
                return json.dumps(
                    {
                        "fact_verdicts": {f: v[0] for f, v in exp.fact_verdicts.items()},
                        "material_unsupported_claims": ["an invented detail"]
                        if exp.unsupported_claims == "at_least_one" else [],
                        "groundedness": 2,
                        "abstained": exp.abstained,
                        "response_type": exp.response_types[0],
                        "notes": "oracle",
                    }
                )  # fmt: skip
        raise AssertionError("an unknown answer reached the oracle")

    return reply


def careless_reply(request) -> str:
    """Says every fact is supported and nothing is wrong - whatever the answer says."""
    ids = re.findall(r"Key fact (f\d+):", request.user_prompt)
    return json.dumps(
        {
            "fact_verdicts": {i: "supported" for i in ids}, "groundedness": 2,
            "abstained": False, "response_type": "answered",
        }
    )  # fmt: skip


@pytest.fixture(scope="module")
def inputs():
    if not (ROOT / "data/snapshots/a282ec471ebf1387cb4feafcfe732ad24a431c7a/tree").is_dir():
        pytest.skip("the frozen copilot snapshot is not present in this checkout")
    from copilot.config import Settings

    protocol = load_protocol(ROOT / PROTOCOL_PATH)
    questions, references = cli._benchmarks(protocol, ROOT, Settings(_env_file=None), [])
    return questions, references


def _run(spec, inputs, reply, tmp_path, *, label="pass1", cache=True, llm=None, prior=None, **kw):
    questions, references = inputs
    llm = llm or LocalFake(reply, ident=identity(provider="ollama"))
    judge = LLMJudge(
        llm, max_output_tokens=1024, retry_sleep=lambda s: None,
        cache=JudgeCache(tmp_path / "cache") if cache else None,
    )  # fmt: skip
    return run_controls(
        spec, judge, questions, references, pass_label=label,
        spec_sha256=controls_sha256(ROOT / CONTROLS_PATH),
        validation_spec_sha256=spec_sha256(ROOT / VALIDATION_SPEC_PATH),
        checkpoint_path=results_path(tmp_path, label), prior=prior, **kw,
    )  # fmt: skip


def test_the_oracle_judge_meets_every_expectation(spec, inputs, tmp_path):
    run = _run(spec, inputs, oracle_reply(spec), tmp_path)
    assert [r.control_id for r in run.results] == [c.control_id for c in spec.controls]
    assert all(r.state == "judged" and r.passed for r in run.results)
    summary = run.summary()
    assert summary["judged"] == 18 and summary["all_expectations_met"] == 18
    assert summary["parse_success_fraction"] == 1.0 and summary["context_limit_failures"] == 0
    assert all(
        cell["controls"] == 2 and cell["passed"] == 2 for cell in summary["by_kind"].values()
    )
    assert summary["parser_controls_passed"] == summary["parser_controls"] == 10
    assert run.security_result is not None and run.security_result.passed
    assert run.security_result.gate_fired and not run.security_result.secret_in_artifact
    assert run.runtime_fingerprint == identity(provider="ollama").fingerprint()
    assert run.namespace == "synthetic-judge-controls" and run.pass_label == "pass1"
    assert run.validation_spec_sha256 == spec_sha256(ROOT / VALIDATION_SPEC_PATH)
    assert security_token() not in results_path(tmp_path, "pass1").read_text()


def test_a_careless_judge_fails_the_controls(spec, inputs, tmp_path):
    run = _run(spec, inputs, careless_reply, tmp_path)
    failed = {r.kind for r in run.results if not r.passed}
    # it praises fabrications, contradictions and invented symbols, and misses abstentions
    assert {"confident_fabrication", "contradicted_fact", "invented_symbol",
            "correct_abstention", "incorrect_refusal", "generic_refusal",
            "mixed_answer_and_abstention", "partially_correct"} <= failed  # fmt: skip
    assert next(r for r in run.results if r.control_id == "ctl-io-03").score == 2


def test_an_unparsable_judge_output_is_a_failed_control_not_a_default(spec, inputs, tmp_path):
    broken = spec.controls[0].answer

    def reply2(request):
        return "not json" if broken in request.user_prompt else oracle_reply(spec)(request)

    run = _run(spec, inputs, reply2, tmp_path)
    assert run.summary()["judged"] == 17 and run.summary()["parse_success_fraction"] < 0.95
    failed = next(r for r in run.results if r.state == "failed")
    assert failed.control_id == spec.controls[0].control_id and failed.score is None
    assert failed.passed is False


def test_the_parser_controls_need_no_model(spec, inputs):
    from copilot.evaluation.generation.judge_controls import run_parser_controls

    questions, _ = inputs
    results = run_parser_controls(spec, questions)
    assert all(r.passed for r in results)
    assert {r.outcome for r in results if r.expected == "reject"} == {"rejected"}
    assert {r.outcome for r in results if r.expected == "accept"} == {"accepted"}


def test_the_secret_gate_fires_before_the_provider_and_the_secret_is_never_stored(
    spec, inputs, tmp_path
):
    llm = LocalFake(oracle_reply(spec), ident=identity(provider="ollama"))
    run = _run(spec, inputs, None, tmp_path, llm=llm)
    assert (
        llm.call_count == 18
    )  # the eighteen model controls only - the security control never called
    assert run.security_result.error and "RepositorySecretRiskError" in run.security_result.error
    assert all(security_token() not in r.model_dump_json() for r in run.results)


def test_resume_reuses_judged_controls_and_refuses_a_different_runtime(spec, inputs, tmp_path):
    first = _run(spec, inputs, oracle_reply(spec), tmp_path)
    again_llm = LocalFake(oracle_reply(spec), ident=identity(provider="ollama"))
    again = _run(
        spec,
        inputs,
        None,
        tmp_path,
        llm=again_llm,
        prior=load_results(results_path(tmp_path, "pass1")),
    )
    assert again_llm.call_count == 0 and len(again.results) == len(first.results)

    other = LocalFake(oracle_reply(spec), ident=identity(provider="ollama", model_digest="d" * 64))
    with pytest.raises(ControlsMismatchError, match="runtime_fingerprint"):
        _run(spec, inputs, None, tmp_path, llm=other, prior=first)
    assert other.call_count == 0
    with pytest.raises(ControlsMismatchError, match="pass_label"):
        _run(spec, inputs, oracle_reply(spec), tmp_path, label="repeat", prior=first)


def test_a_repeat_pass_without_a_cache_asks_the_judge_again(spec, inputs, tmp_path):
    _run(spec, inputs, oracle_reply(spec), tmp_path / "a")
    llm = LocalFake(oracle_reply(spec), ident=identity(provider="ollama"))
    repeat = _run(spec, inputs, None, tmp_path / "b", label="repeat", cache=False, llm=llm)
    assert repeat.cache_used is False and repeat.cache_hits == 0 and llm.call_count == 18
    # with a shared cache the second pass would have been answered from it
    llm2 = LocalFake(oracle_reply(spec), ident=identity(provider="ollama"))
    cached = _run(spec, inputs, None, tmp_path / "a", label="repeat", llm=llm2)
    assert cached.cache_hits == 18 and llm2.call_count == 0


# ---------------------------------------------------------------- the CLI, end to end


class OracleServer:
    def __init__(self, spec, digest="a" * 64):
        self.reply = oracle_reply(spec)
        self.digest = digest
        self.calls: list[httpx.Request] = []

    def __call__(self, req):
        self.calls.append(req)
        path = req.url.path
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.12.3"})
        if path == "/api/tags":
            return httpx.Response(200, json=installed(digest=self.digest))
        if path == "/api/chat":
            body = json.loads(req.content)

            class R:
                user_prompt = body["messages"][1]["content"]

            return httpx.Response(200, json=chat_ok(self.reply(R)))
        return httpx.Response(404, json={})


@pytest.fixture
def oracle_ollama(spec, monkeypatch):
    server = OracleServer(spec)
    real = llm_factory.OllamaLLMClient
    monkeypatch.setattr(
        llm_factory,
        "OllamaLLMClient",
        lambda **kw: real(transport=httpx.MockTransport(server), **kw),
    )
    monkeypatch.setattr(cli, "create_client", create_llm_client)
    return server


def _controls_argv(directory: Path, *more: str) -> list[str]:
    return [
        "judge-controls", "--root", str(ROOT), "--judge-provider", "ollama",
        "--judge-model", "llama3.1:8b", "--judgements-dir", str(directory), *more,
    ]  # fmt: skip


@needs_repos
def test_judge_controls_cli_writes_only_into_its_own_directory(oracle_ollama, tmp_path, capsys):
    target = tmp_path / "controls"
    assert cli.main(_controls_argv(target)) == 0
    files = sorted(p.name for p in target.rglob("*") if p.is_file() and "cache" not in str(p))
    assert files == ["judge-controls-pass1.json"]
    summary = json.loads(capsys.readouterr().out.split("wrote ")[0])
    assert summary["judged"] == 18 and summary["all_expectations_met"] == 18
    body = json.loads(oracle_ollama.calls[-1].content)
    assert body["format"] == "json" and body["options"]["num_ctx"] == 8192
    paths = {c.url.path for c in oracle_ollama.calls}
    assert paths <= {"/api/chat", "/api/tags", "/api/version"}


@needs_repos
def test_judge_controls_repeat_bypasses_the_cache_and_unloads_the_model(
    oracle_ollama, tmp_path, capsys
):
    assert cli.main(_controls_argv(tmp_path / "one")) == 0
    chats_before = sum(c.url.path == "/api/chat" for c in oracle_ollama.calls)
    assert cli.main(_controls_argv(tmp_path / "two", "--repeat")) == 0
    run = load_results(results_path(tmp_path / "two", "repeat"))
    assert run.cache_used is False and run.cache_hits == 0
    chats_after = sum(c.url.path == "/api/chat" for c in oracle_ollama.calls)
    assert chats_after - chats_before == 18
    body = json.loads([c for c in oracle_ollama.calls if c.url.path == "/api/chat"][-1].content)
    assert body["keep_alive"] == 0
    assert not (tmp_path / "two" / "cache-judge").exists()


@needs_repos
def test_judge_controls_cli_refuses_the_primary_directory_and_a_missing_directory(
    oracle_ollama, tmp_path, capsys
):
    assert cli.main(_controls_argv(ROOT / "data/generation-eval/primary/controls")) == 2
    assert "primary directory" in capsys.readouterr().err
    argv = ["judge-controls", "--root", str(ROOT), "--judge-provider", "ollama",
            "--judge-model", "llama3.1:8b"]  # fmt: skip
    assert cli.main(argv) == 2 and "--judgements-dir" in capsys.readouterr().err
    assert not (ROOT / "data/generation-eval/primary/controls").exists()


@needs_repos
def test_judge_validate_cli_reports_incomplete_evidence_and_never_passes_it(
    oracle_ollama, tmp_path, capsys
):
    assert cli.main(_controls_argv(tmp_path / "one")) == 0
    capsys.readouterr()
    code = cli.main(
        ["judge-validate", "--root", str(ROOT), "--controls-dir", str(tmp_path / "one")]
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 4 and report["overall"] == "INCOMPLETE"
    assert (tmp_path / "one" / "judge-validation-report.json").is_file()


@needs_repos
def test_judge_validate_cli_refuses_results_judged_under_another_specification(
    oracle_ollama, tmp_path, capsys
):
    assert cli.main(_controls_argv(tmp_path / "one")) == 0
    capsys.readouterr()
    stored = results_path(tmp_path / "one", "pass1")
    data = json.loads(stored.read_text(encoding="utf-8"))
    data["validation_spec_sha256"] = "0" * 64
    stored.write_text(json.dumps(data), encoding="utf-8")
    code = cli.main(
        ["judge-validate", "--root", str(ROOT), "--controls-dir", str(tmp_path / "one")]
    )
    assert code != 0
    assert "different validation specification" in capsys.readouterr().err
    assert not (tmp_path / "one" / "judge-validation-report.json").exists()
