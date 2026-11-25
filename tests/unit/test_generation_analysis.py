"""Milestone 11A: statistics and reports - hand-checked numbers, traceability, test labelling."""

from __future__ import annotations

import json

import pytest

from copilot.evaluation.generation.analysis import (
    RepoInputs,
    analyse,
    build_report,
    build_rows,
    paired_summary,
    recompute_primary,
    render_markdown,
)
from copilot.evaluation.generation.execution import write_run
from copilot.evaluation.generation.inventory import build_inventory
from copilot.evaluation.generation.judge import LLMJudge
from copilot.evaluation.generation.judging import judge_run, write_judgements
from copilot.llm import FakeLLMClient
from copilot.llm.retry import RetryPolicy
from tests.unit.test_generation_execution import _ctx, _protocol, _run
from tests.unit.test_generation_judging import _refs

PROTOCOL = _protocol().model_copy(update={"bootstrap_resamples": 2000})


def _row(repo, qid, arm, score, *, answerable=True, state="scored", **extra):
    row = {
        "repository_id": repo, "question_id": qid, "arm": arm, "answerable": answerable,
        "side_state": "successful", "origin": "generated", "score_state": state,
        "abstention_label": "answered", "test_artifact": False,
    }  # fmt: skip
    if score is not None:
        row["score"] = score
        row["completeness"] = score / 2
        row["groundedness"] = score
    row.update(extra)
    return row


def _rows(repo, plain, rag):
    out = []
    for i, (p, r) in enumerate(zip(plain, rag, strict=True)):
        out.append(_row(repo, f"{repo}-{i}", "plain", p))
        out.append(_row(repo, f"{repo}-{i}", "rag", r))
    return out


def test_paired_summary_by_hand():
    summary = paired_summary(
        [(0, 2), (1, 1), (1, 2), (2, 2)], seed=1, resamples=2000, confidence=0.95
    )
    assert summary["n"] == 4 and summary["mean_plain"] == 1.0 and summary["mean_rag"] == 1.75
    assert summary["mean_difference"] == pytest.approx(0.75)  # (2 + 0 + 1 + 0) / 4
    assert summary["ci_low"] <= 0.75 <= summary["ci_high"] and summary["ci_low"] >= 0.0
    again = paired_summary(
        [(0, 2), (1, 1), (1, 2), (2, 2)], seed=1, resamples=2000, confidence=0.95
    )
    assert again == summary  # deterministic for a seed


def test_ci_degenerate_cases_and_empty_pairs():
    same = paired_summary([(0, 1)] * 6, seed=1, resamples=500, confidence=0.95)
    assert same["ci_low"] == same["ci_high"] == 1.0 and same["ci_excludes_zero"] is True
    zero = paired_summary([(1, 1)] * 6, seed=1, resamples=500, confidence=0.95)
    assert zero["mean_difference"] == 0 and zero["ci_excludes_zero"] is False
    empty = paired_summary([], seed=1, resamples=500, confidence=0.95)
    assert empty["n"] == 0 and empty["mean_difference"] is None and empty["ci_low"] is None


def test_primary_is_per_repository_and_combined_and_traceable_to_rows():
    rows = _rows("a", [0, 1, 1, 2], [2, 1, 2, 2]) + _rows("b", [1, 1], [0, 1])
    result = analyse(rows, PROTOCOL)
    assert result["per_repository"]["a"]["primary"]["mean_difference"] == pytest.approx(0.75)
    assert result["per_repository"]["b"]["primary"]["mean_difference"] == pytest.approx(-0.5)
    assert result["combined"]["primary"]["n"] == 6
    assert result["combined"]["primary"]["mean_difference"] == pytest.approx((3.0 - 1.0) / 6)
    for repo in (None, "a", "b"):
        reported = (result["combined"] if repo is None else result["per_repository"][repo])[
            "primary"
        ]["mean_difference"]
        assert recompute_primary(rows, repo) == pytest.approx(reported)


def test_incomplete_pairs_are_listed_and_never_dropped_silently_or_imputed():
    rows = _rows("a", [0, 1, 2], [2, 2, 2])
    rows[3] = _row("a", "a-1", "rag", None, state="judge_failed")  # a-1's RAG judgement failed
    rows[4] = _row("a", "a-2", "plain", None, state="side_unavailable", side_state="failed")
    primary = analyse(rows, PROTOCOL)["combined"]["primary"]
    assert primary["n"] == 1 and primary["answerable_questions"] == 3
    excluded = {e["question_id"]: e for e in primary["excluded_pairs"]}
    assert set(excluded) == {"a-1", "a-2"}
    assert (
        excluded["a-1"]["rag"] == "judge_failed" and excluded["a-2"]["plain"] == "side_unavailable"
    )


def test_unanswerable_questions_are_not_part_of_the_primary_endpoint():
    rows = _rows("a", [0, 0], [2, 2]) + [
        _row("a", "u1", "plain", 0, answerable=False),
        _row("a", "u1", "rag", 2, answerable=False),
    ]
    assert analyse(rows, PROTOCOL)["combined"]["primary"]["n"] == 2


def test_reliability_counts_every_requested_side():
    rows = _rows("a", [1, 1], [1, 1])
    rows[0] = _row("a", "a-0", "plain", None, state="side_unavailable", side_state="failed")
    rows[3] = _row("a", "a-1", "rag", None, state="side_unavailable", side_state="not_attempted")
    rel = analyse(rows, PROTOCOL)["combined"]["reliability"]
    assert rel["plain"]["requested"] == 2 and rel["plain"]["failed"] == 1
    assert rel["rag"]["not_attempted"] == 1 and rel["rag"]["success_rate"] == 0.5


def test_latency_and_tokens_exclude_cache_and_historical_sides():
    rows = _rows("a", [1], [1])
    rows[0].update(latency_seconds=2.0, prompt_tokens=100)
    rows[1].update(origin="historical")
    arms = analyse(rows, PROTOCOL)["combined"]["per_arm"]
    assert (
        arms["plain"]["latency_seconds"]["mean"] == 2.0
        and arms["plain"]["tokens"]["prompt_tokens"]["total"] == 100
    )
    assert arms["rag"]["latency_seconds"]["n"] == 0
    assert arms["rag"]["latency_seconds"]["excluded_cache_or_historical"] == 1


# --------------------------------------------------------------------------- end to end


@pytest.fixture
def pipeline(tmp_path):
    def reply(request):
        return (
            "rag answer [Source 1]" if "BEGIN SOURCE 1" in request.user_prompt else "plain answer"
        )

    def judge_reply(request):
        text = request.user_prompt
        if "Key fact" not in text:
            return json.dumps(
                {
                    "fact_verdicts": {},
                    "groundedness": 2,
                    "abstained": False,
                    "response_type": "answered",
                }
            )
        verdict = "supported" if "rag answer" in text else "missing"
        return json.dumps(
            {"fact_verdicts": {"f1": verdict}, "groundedness": 2 if verdict == "supported" else 1,
             "abstained": False, "response_type": "answered"}
        )  # fmt: skip

    ctx = _ctx(n=4)
    run = _run(ctx, tmp_path, FakeLLMClient(reply))
    judge = LLMJudge(
        FakeLLMClient(judge_reply, model="judge-m"),
        retry_policy=RetryPolicy(initial_delay_seconds=0.0, max_attempts=2),
        retry_sleep=lambda s: None,
    )
    judgements = judge_run(
        run, {q.id: q for q in ctx.questions}, _refs(ctx), judge,
        checkpoint_path=tmp_path / "j.json",
    )  # fmt: skip
    write_run(run, tmp_path / "run.json")
    write_judgements(judgements, tmp_path / "j.json")
    inputs = RepoInputs(
        run=run, questions={q.id: q for q in ctx.questions}, inventory=build_inventory([], "r1"),
        judgements=judgements, run_file="run.json", judgement_file="j.json",
        independent=False, kind="synthetic-authored",
    )  # fmt: skip
    return inputs, tmp_path


def test_full_pipeline_report_is_labelled_traceable_and_does_not_declare_a_winner(pipeline):
    inputs, tmp = pipeline
    files = {"run:r1": tmp / "run.json", "judgements:r1": tmp / "j.json"}
    report = build_report([inputs], PROTOCOL, input_files=files)
    assert report["test_artifact"] is True
    assert any("TEST ARTIFACT" in c for c in report["caveats"])
    primary = report["analysis"]["combined"]["primary"]
    assert primary["n"] == 3 and primary["mean_plain"] == 0.0 and primary["mean_rag"] == 2.0
    assert primary["mean_difference"] == pytest.approx(2.0)
    assert recompute_primary(report["rows"]) == pytest.approx(primary["mean_difference"])
    assert len(report["input_sha256"]["run:r1"]) == 64
    assert {r["run_file"] for r in report["rows"]} == {"run.json"}
    unans = [r for r in report["rows"] if not r["answerable"]]
    assert {r["abstention_label"] for r in unans} == {"answered"}  # both arms answered a trap
    markdown = render_markdown(report)
    assert markdown.splitlines()[2].startswith("> **TEST ARTIFACT")
    for forbidden in ("winner", "outperform", "significantly better", "proves"):
        assert forbidden not in markdown.lower()
    json.dumps(report)  # serialisable


def test_rows_carry_programmatic_metrics_for_answers(pipeline):
    inputs, _ = pipeline
    rows = build_rows(inputs)
    rag = next(r for r in rows if r["arm"] == "rag" and r["answerable"])
    plain = next(r for r in rows if r["arm"] == "plain" and r["answerable"])
    assert rag["citation_applicable"] is True and rag["citations_total"] == 1
    assert plain["citation_applicable"] is False  # no explicit locations in a plain answer
    assert rag["retrieval_hit"] is True and "retrieval_hit" not in plain
    assert rag["blind_id"] != plain["blind_id"] and "plain" not in str(plain["blind_id"])


def test_a_run_without_judgements_reports_no_scores_instead_of_inventing_them(pipeline):
    inputs, tmp = pipeline
    bare = RepoInputs(
        run=inputs.run, questions=inputs.questions, inventory=inputs.inventory, judgements=None,
        run_file="run.json",
    )  # fmt: skip
    report = build_report([bare], PROTOCOL, input_files={"run:r1": tmp / "run.json"})
    primary = report["analysis"]["combined"]["primary"]
    assert primary["n"] == 0 and primary["mean_difference"] is None
    assert any("no judgements" in c for c in report["caveats"])
