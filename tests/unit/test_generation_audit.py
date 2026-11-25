"""Milestone 11A: audit sampling, blinded worksheets and agreement statistics."""

from __future__ import annotations

import pytest

from copilot.evaluation.generation.audit import (
    AuditAnswer,
    AuditUnit,
    agreement,
    build_worksheet,
    cohen_kappa,
    parse_human_row,
    raw_agreement,
    sample_audit,
    weighted_kappa,
)
from copilot.evaluation.generation.reference import FactReference, ReferenceMaterial
from copilot.evaluation.generation.rubric import EvaluationError, FactVerdict


def _units():
    units = []
    for repo, n_ans, n_un in (("copilot", 34, 8), ("inventory-orders", 24, 6)):
        units += [AuditUnit(repo, f"a{i:02d}", True) for i in range(n_ans)]
        units += [AuditUnit(repo, f"u{i:02d}", False) for i in range(n_un)]
    return units


def test_sampling_is_reproducible_stratified_and_about_twenty_percent():
    units = _units()
    sample = sample_audit(units, seed=7)
    assert sample == sample_audit(list(reversed(units)), seed=7)  # independent of input order
    assert sample != sample_audit(units, seed=8)
    repos = {u.repository_id for u in sample}
    assert repos == {"copilot", "inventory-orders"}
    assert {u.answerable for u in sample} == {True, False}
    for repo in repos:
        for answerable in (True, False):
            assert any(u.repository_id == repo and u.answerable == answerable for u in sample)
    assert 0.18 <= len(sample) / len(units) <= 0.26  # target 20 % (rounding per stratum)
    assert len({(u.repository_id, u.question_id) for u in sample}) == len(sample)


def test_sampling_validates_fraction_and_handles_tiny_strata():
    with pytest.raises(ValueError):
        sample_audit(_units(), fraction=0, seed=1)
    one = sample_audit([AuditUnit("r", "only", False)], seed=1)
    assert [u.question_id for u in one] == ["only"]


def _ref(answerable=True):
    facts = (FactReference("f1", "a key fact", (("a.py:1-2", "code"),)),) if answerable else ()
    return ReferenceMaterial("q", answerable, facts, "no owner", "say so", "good answer")


def _worksheet(seed=3):
    sampled = [("r", "q1"), ("r", "q2"), ("r", "q3")]
    answers = [
        AuditAnswer(r, q, arm, f"{arm} answer for {q} [Source 1]")
        for r, q in sampled
        for arm in ("plain", "rag")
    ]
    questions = {(r, q): f"Question {q}?" for r, q in sampled}
    references = {(r, q): _ref() for r, q in sampled}
    return build_worksheet(answers, questions, references, seed=seed)


def test_worksheet_is_blinded_complete_and_unrated():
    sheet = _worksheet()
    assert len(sheet.rows) == 6 and len(sheet.key) == 6
    for row in sheet.rows:
        assert "arm" not in row and "question_id" not in row  # arm and ids live only in the key
        assert "[Source" not in row["answer"] and "[cite]" in row["answer"]
        assert row["reference_key_facts"].startswith("f1:")
        assert row["reference_source_evidence"].startswith("[a.py:1-2]")
        assert all(row[f] == "" for f in row if f.startswith("human_"))  # nothing pre-filled
    assert {(k.question_id, k.arm) for k in sheet.key} == {
        (q, a) for q in ("q1", "q2", "q3") for a in ("plain", "rag")
    }
    assert [r["item_id"] for r in sheet.rows] == sorted(r["item_id"] for r in sheet.rows)


def test_worksheet_order_is_shuffled_but_reproducible():
    one, two, other = _worksheet(3), _worksheet(3), _worksheet(4)
    assert one.key == two.key and one.to_csv() == two.to_csv()
    assert one.key != other.key
    arms = [k.arm for k in one.key]
    assert arms != sorted(arms)  # not grouped by arm


def test_worksheet_exports():
    sheet = _worksheet()
    csv_text = sheet.to_csv()
    assert csv_text.splitlines()[0].startswith("item_id,repository_id,question")
    assert sheet.to_jsonl().count("\n") == 6 and "A-0001" in sheet.key_json()


def test_unanswerable_reference_is_explained_not_blank():
    answers = [AuditAnswer("r", "u1", "rag", "I do not know")]
    sheet = build_worksheet(answers, {("r", "u1"): "Who?"}, {("r", "u1"): _ref(False)}, seed=1)
    assert "cannot answer" in sheet.rows[0]["reference_key_facts"]


def test_human_rows_parse_blank_complete_and_partial():
    blank = {"item_id": "A-0001"}
    assert parse_human_row(blank, ["f1"], "ann") is None
    row = {
        "item_id": "A-0001",
        "human_fact_verdicts": "f1=supported",
        "human_groundedness": "2",
        "human_abstained": "no",
        "human_unsupported_claims": "invented helper | made-up flag",
    }
    assessment = parse_human_row(row, ["f1"], "ann")
    assert assessment is not None and assessment.evaluator_kind == "human"
    assert assessment.fact_verdicts == {"f1": FactVerdict.SUPPORTED}
    assert assessment.material_unsupported_claims == ("invented helper", "made-up flag")
    with pytest.raises(EvaluationError):
        parse_human_row({"item_id": "A-2", "human_groundedness": "2"}, ["f1"], "ann")
    with pytest.raises(EvaluationError):
        parse_human_row({**row, "human_fact_verdicts": "f1=perhaps"}, ["f1"], "ann")
    with pytest.raises(EvaluationError):
        parse_human_row({**row, "human_abstained": "maybe"}, ["f1"], "ann")
    with pytest.raises(EvaluationError):
        parse_human_row({**row, "human_groundedness": "7"}, ["f1"], "ann")


def test_unanswerable_human_rows_need_no_fact_verdicts():
    row = {"item_id": "A-3", "human_groundedness": "2", "human_abstained": "yes"}
    assessment = parse_human_row(row, [], "ann")
    assert assessment is not None and assessment.abstained and assessment.fact_verdicts == {}


# --------------------------------------------------------------------------- agreement


def test_cohen_kappa_by_hand():
    a, b = [1, 1, 0, 0], [1, 0, 0, 0]
    assert raw_agreement(a, b) == 0.75
    assert cohen_kappa(a, b) == pytest.approx(0.5)  # (0.75 - 0.5) / (1 - 0.5)
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == pytest.approx(-1.0)


def test_kappa_is_undefined_without_variation():
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) is None
    assert weighted_kappa([2, 2, 2], [2, 2, 2]) is None
    assert cohen_kappa([], []) is None and raw_agreement([], []) is None


def test_weighted_kappa_penalises_distance():
    assert weighted_kappa([0, 1, 2, 1], [0, 1, 2, 1]) == pytest.approx(1.0)
    assert weighted_kappa([0, 2, 0, 2], [2, 0, 2, 0]) == pytest.approx(-1.0)
    near = weighted_kappa([0, 1, 2, 2, 0, 1], [1, 1, 2, 1, 0, 1])
    far = weighted_kappa([0, 1, 2, 2, 0, 1], [2, 1, 2, 0, 0, 1])
    assert near is not None and far is not None and near > far


def test_agreement_result_reports_both_numbers():
    result = agreement([1, 1, 0, 0], [1, 0, 0, 0])
    assert result.to_json() == {
        "n": 4, "raw_agreement": 0.75, "kappa": pytest.approx(0.5), "kappa_kind": "cohen",
    }  # fmt: skip
    assert agreement([0, 1, 2], [0, 1, 2], ordinal=True).kappa == pytest.approx(1.0)
    with pytest.raises(ValueError):
        raw_agreement([1], [1, 2])
