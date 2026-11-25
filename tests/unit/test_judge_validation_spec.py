"""Milestone 11B: the judge-validation specification is frozen before any judge ran.

The pinned hash below was recorded when the file was committed, before any local or hosted judge
execution on the pilot or the synthetic controls. If it fails, a threshold was edited after the
fact: write ``judge-validation-v2`` instead (and keep v1 and every result judged under it).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from copilot.evaluation.generation.judge_validation import (
    VALIDATION_SPEC_PATH,
    ValidationSpecError,
    load_validation_spec,
    spec_sha256,
)

ROOT = Path(__file__).resolve().parents[2]
PINNED_SHA256 = "aa8af417b0eceb0347abebdb3fefa9ca833f7e737521e3f0511f002f049aa81a"


def test_the_committed_specification_is_unchanged():
    assert spec_sha256(ROOT / VALIDATION_SPEC_PATH) == PINNED_SHA256


def test_the_specification_declares_the_agreed_thresholds():
    spec = load_validation_spec(ROOT / VALIDATION_SPEC_PATH)
    tech, sem, primary = (
        spec["technical_acceptance"],
        spec["semantic_pilot_acceptance"],
        spec["primary_audit_acceptance"],
    )
    assert tech["T2_control_parse_success"]["min_fraction"] == 0.95
    assert tech["T3_no_accepted_context_overflow_or_truncation"]["max_violations"] == 0
    assert tech["T4_no_security_bypass"]["max_violations"] == 0
    assert sem["S1_human_exact_agreement"]["min_fraction"] == 0.75
    assert sem["S2_no_two_point_disagreement"]["max_count"] == 0
    assert sem["S5_repeat_consistency"]["min_fraction"] == 0.95
    assert primary["exact_agreement_min"] == 0.75
    assert primary["weighted_kappa_min"] == 0.60
    assert primary["severe_two_point_disagreement_max_fraction"] == 0.05
    assert primary["valid_judgement_coverage_min"] == 0.95
    assert "NOT a pass" in primary["undefined_kappa"]
    judge = spec["judge"]
    assert (judge["provider"], judge["model"], judge["quantization"]) == (
        "ollama", "llama3.1:8b", "Q4_K_M",
    )  # fmt: skip
    assert (judge["context_size"], judge["max_output_tokens"], judge["temperature"]) == (
        8192, 1024, 0.0,
    )  # fmt: skip


def test_a_malformed_specification_is_refused(tmp_path):
    bad = tmp_path / "s.json"
    bad.write_text("{}")
    with pytest.raises(ValidationSpecError):
        load_validation_spec(bad)
    bad.write_text("not json")
    with pytest.raises(ValidationSpecError):
        load_validation_spec(bad)
    with pytest.raises(ValidationSpecError):
        load_validation_spec(tmp_path / "missing.json")
