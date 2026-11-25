"""Milestone 11B closeout: the preserved failed local-judge validation stays exactly as recorded.

These tests pin the committed specification and control definitions to the hashes stored with the
preserved results, re-derive the preserved report from the preserved raw results with the current
evaluator, and check the preserved artifacts for credentials and machine-specific paths. They make
no model call.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from copilot.evaluation.generation.judge_controls import load_results, security_token
from copilot.evaluation.generation.judge_validation import (
    VALIDATION_SPEC_PATH,
    ValidationInputs,
    evaluate_validation,
    load_validation_spec,
    spec_sha256,
)

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "generation-eval/judge-validation/results/ollama-llama3.1-8b-q4km-v1"
CONTROLS = ROOT / "generation-eval/judge-controls/controls-v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def meta() -> dict:
    return json.loads((RESULT / "execution-metadata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def run():
    return load_results(RESULT / "judge-controls-pass1.json")


def test_the_specification_and_controls_are_unchanged_since_the_run(meta):
    repo = meta["repository"]
    assert spec_sha256(ROOT / VALIDATION_SPEC_PATH) == repo["validation_spec_sha256"]
    assert _sha(CONTROLS) == repo["controls_sha256"]


def test_the_preserved_files_match_their_recorded_hashes(meta):
    for name, entry in meta["files"].items():
        assert _sha(RESULT / name) == entry["sha256"], name


def test_the_raw_results_belong_to_the_recorded_run(meta, run):
    assert run.validation_spec_sha256 == meta["repository"]["validation_spec_sha256"]
    assert run.controls_sha256 == meta["repository"]["controls_sha256"]
    assert (run.judge_provider, run.judge_model) == ("ollama", "llama3.1:8b")
    assert (run.judge_prompt_version, run.rubric_version) == ("gen-judge-2", "gen-rubric-1")
    assert run.runtime_fingerprint == meta["runtime"]["runtime_fingerprint"]
    assert run.runtime["model_digest"] == meta["ollama"]["model_digest"]
    assert run.runtime["quantization"] == "Q4_K_M"
    assert run.runtime["runtime_version"] == meta["ollama"]["server_version"]


def test_the_recorded_numbers_follow_from_the_raw_results(run):
    results = run.results
    assert len(results) == 18
    assert sum(1 for r in results if r.state == "judged") == 15
    assert sorted(r.control_id for r in results if r.state == "failed") == [
        "ctl-cp-06",
        "ctl-cp-07",
        "ctl-io-07",
    ]
    assert sum(1 for r in results if r.passed) == 8
    assert all(r.record is not None and r.record.prompt_tokens for r in results)
    assert sum(1 for p in run.parser_results if p.passed) == 10 == len(run.parser_results)
    assert run.security_result is not None and run.security_result.passed


def test_the_preserved_report_is_reproduced_by_the_evaluator(run):
    stored = json.loads((RESULT / "judge-validation-report.json").read_text(encoding="utf-8"))
    spec = load_validation_spec(ROOT / VALIDATION_SPEC_PATH)
    fresh = evaluate_validation(
        spec, ValidationInputs(controls=run), spec_sha256(ROOT / VALIDATION_SPEC_PATH)
    )
    assert fresh["overall"] == stored["overall"] == "FAIL"
    for section in ("technical_acceptance", "semantic_pilot_acceptance"):
        assert {c["id"]: c["status"] for c in fresh[section]["criteria"]} == {
            c["id"]: c["status"] for c in stored[section]["criteria"]
        }
    statuses = {c["id"].split("_")[0]: c for c in stored["technical_acceptance"]["criteria"]}
    assert statuses["T2"]["status"] == "FAIL" and "15/18" in statuses["T2"]["detail"]
    semantic = {c["id"].split("_")[0]: c for c in stored["semantic_pilot_acceptance"]["criteria"]}
    assert semantic["S3"]["status"] == "FAIL" and "4/6" in semantic["S3"]["detail"]
    assert semantic["S4"]["status"] == "FAIL" and "3/4" in semantic["S4"]["detail"]
    for unrun in ("S1", "S2", "S5"):
        assert semantic[unrun]["status"] == "NOT_EVALUATED"


_PRIVATE = re.compile(
    r"[A-Za-z]:\\Users|/home/|/sessions/|/root/"
    r"|AIza[0-9A-Za-z_-]{20,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY"
)


def test_the_preserved_artifacts_hold_no_credential_or_private_path():
    for path in RESULT.iterdir():
        text = path.read_text(encoding="utf-8")
        assert not _PRIVATE.search(text), path.name
        assert security_token() not in text, path.name
        assert "ghp_" not in text, path.name


def test_only_the_expected_files_are_kept():
    assert sorted(p.name for p in RESULT.iterdir()) == [
        "execution-metadata.json",
        "judge-controls-pass1.json",
        "judge-validation-report.json",
    ]
