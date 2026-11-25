"""M11A corrections: audit eligibility/exclusions, finalization, rule-item verification, and the
strict audit-agree accounting. FakeLLMClient only; no network."""

# ruff: noqa: F811  (imported pytest fixtures are re-declared as test arguments)
from __future__ import annotations

import csv
import io
import json

import pytest

from copilot.evaluation.generation import __main__ as cli
from copilot.evaluation.generation.analysis import RepoInputs
from copilot.evaluation.generation.audit import agreement, sample_audit
from copilot.evaluation.generation.audit_agree import evaluate_agreement
from copilot.evaluation.generation.audit_population import (
    classify_questions,
    describe_empty_sample,
    finalization_problems,
    population_report,
)
from copilot.evaluation.generation.audit_rules import (
    build_rule_worksheet,
    evaluate_rule_check,
    rule_items,
)
from copilot.evaluation.generation.execution import GenerationEntry, SideRecord
from copilot.evaluation.generation.judging import JudgedItem
from tests.unit.test_generation_analysis import pipeline  # noqa: F401  (fixture)
from tests.unit.test_generation_cli import _args, no_network, project  # noqa: F401  (fixtures)

# ---------------------------------------------------------------- population / eligibility


def _with_entries(inputs: RepoInputs, edit) -> RepoInputs:
    entries = [edit(e) for e in inputs.run.entries]
    return RepoInputs(
        run=inputs.run.model_copy(update={"entries": entries}), questions=inputs.questions,
        inventory=inputs.inventory, judgements=inputs.judgements, run_file=inputs.run_file,
        judgement_file=inputs.judgement_file, independent=inputs.independent, kind=inputs.kind,
    )  # fmt: skip


def _side_update(side: SideRecord, **update) -> SideRecord:
    return side.model_copy(update=update)


def test_a_complete_experiment_is_final_and_every_question_is_eligible(pipeline):  # noqa: F811
    inputs, _ = pipeline
    statuses = classify_questions([inputs])
    assert len(statuses) == 4 and all(s.eligible for s in statuses)
    assert finalization_problems([inputs]) == []
    sample = sample_audit([s.unit() for s in statuses], fraction=0.5, seed=7)
    report = population_report([inputs], statuses, sample, problems=[], provisional=False)
    assert report["total_benchmark_questions"] == 4 and report["total_responses"] == 8
    assert report["completed_pairs"] == 4 and report["eligible_pairs"] == 4
    assert report["excluded_pairs"] == 0 and report["provisional"] is False
    n = report["selected_sample_size"]
    assert report["sample_fraction_of_eligible"] == n / 4
    assert report["sample_fraction_of_full_population"] == n / 4
    assert sum(c["selected"] for c in report["strata"].values()) == n


def test_every_exclusion_has_a_reason_and_the_counts_add_up(pipeline):  # noqa: F811
    inputs, _ = pipeline

    def edit(entry: GenerationEntry) -> GenerationEntry:
        if entry.question_id == "r1-q0":  # RAG abstained without calling the model
            rag = _side_update(entry.rag, status="insufficient_evidence", answer="not enough")
            return entry.model_copy(update={"rag": rag})
        if entry.question_id == "r1-q1":  # RAG call failed
            rag = _side_update(entry.rag, state="failed", status=None, answer=None, error="boom")
            return entry.model_copy(update={"rag": rag})
        if entry.question_id == "r1-q2":  # plain never attempted
            plain = _side_update(entry.plain, state="not_attempted", status=None, answer=None)
            return entry.model_copy(update={"plain": plain})
        return entry

    inputs = _with_entries(inputs, edit)
    # r1-q3: both answers fine, but the judge rating of its plain answer is missing
    kept = [i for i in inputs.judgements.items if (i.question_id, i.arm) != ("r1-q3", "plain")]
    inputs = RepoInputs(
        run=inputs.run, questions=inputs.questions, inventory=inputs.inventory,
        judgements=inputs.judgements.model_copy(update={"items": kept}),
        run_file="r", judgement_file="j", independent=False, kind="k",
    )  # fmt: skip
    statuses = {s.question_id: s for s in classify_questions([inputs])}
    assert statuses["r1-q0"].reasons == ("rule_scored_system_abstention",)
    assert statuses["r1-q1"].reasons == ("generation_failed",)
    assert statuses["r1-q2"].reasons == ("not_attempted",)
    assert statuses["r1-q3"].reasons == ("judge_rating_missing",)
    report = population_report(
        [inputs], list(statuses.values()), [], problems=finalization_problems([inputs]),
        provisional=True,
    )  # fmt: skip
    assert report["eligible_pairs"] == 0 and report["excluded_pairs"] == 4
    assert sum(report["exclusions_by_primary_reason"].values()) == 4
    assert report["rule_scored_system_abstentions"] == 1 and report["failed_responses"] == 1
    assert report["not_attempted_responses"] == 1
    assert report["completed_pairs"] == 2  # q0 and q3: both arms returned something
    assert report["questions_with_missing_or_failed_judge_ratings"] == 1
    assert report["provisional"] and report["provisional_reasons"]
    message = describe_empty_sample(report)
    assert "no eligible pair" in message and "rule-scored system abstentions=1" in message


def test_a_stopped_run_with_no_judgements_is_not_finalized(pipeline):  # noqa: F811
    inputs, _ = pipeline
    bare = RepoInputs(
        run=inputs.run, questions=inputs.questions, inventory=inputs.inventory, judgements=None,
        run_file="r", independent=False, kind="k",
    )  # fmt: skip
    assert any("no judgement file" in p for p in finalization_problems([bare]))
    assert all("judge_rating_missing" in s.reasons for s in classify_questions([bare]))


# ---------------------------------------------------------------- the CLI


def _run_and_judge(project, monkeypatch, judge: bool = True):  # noqa: F811
    assert cli.main(_args(project, "run", "--confirm-model", "fake/fake-model",
                          "--retry-initial-delay", "0")) == 0  # fmt: skip
    if not judge:
        return
    from copilot.llm import FakeLLMClient

    def reply(request):
        if "Key fact" in request.user_prompt:
            ok = "sum of the items" in request.user_prompt.split("BEGIN ANSWER")[1]
            body = {"fact_verdicts": {"f1": "supported" if ok else "missing"},
                    "groundedness": 2 if ok else 1, "abstained": False,
                    "response_type": "answered"}  # fmt: skip
        else:
            body = {"fact_verdicts": {}, "groundedness": 2, "abstained": False,
                    "response_type": "answered"}  # fmt: skip
        return json.dumps(body)

    monkeypatch.setattr(cli, "create_client", lambda s: FakeLLMClient(reply, model=s.llm_model))
    judge_argv = _args(
        project, "judge", "--confirm-model", "fake/fake-model", "--judge-model", "judge-m",
        "--retry-initial-delay", "0",
    )  # fmt: skip
    assert cli.main(judge_argv) == 0


def test_official_sample_is_refused_before_judging_and_writes_no_worksheet(
    project,
    monkeypatch,
    capsys,  # noqa: F811
):
    _run_and_judge(project, monkeypatch, judge=False)
    capsys.readouterr()
    assert cli.main(_args(project, "audit-worksheet")) == 2
    assert "not finalized" in capsys.readouterr().err
    assert not list((project / "out").glob("audit-*"))


def test_a_provisional_sample_with_nothing_eligible_is_an_error_not_an_empty_success(
    project,
    monkeypatch,
    capsys,  # noqa: F811
):
    _run_and_judge(project, monkeypatch, judge=False)
    capsys.readouterr()
    assert cli.main(_args(project, "audit-worksheet", "--provisional")) == 2
    err = capsys.readouterr().err
    assert "no eligible pair to audit" in err and "judge_rating_missing=4" in err
    assert not list((project / "out").glob("audit-worksheet*"))
    report = json.loads(
        (project / "out/audit-population-PROVISIONAL-TEST-ARTIFACT.json").read_text()
    )
    assert report["eligible_pairs"] == 0 and report["provisional"] is True


def test_official_sample_reports_percentages_against_eligible_and_full_population(
    project,
    monkeypatch,
    capsys,  # noqa: F811
):
    _run_and_judge(project, monkeypatch)
    capsys.readouterr()
    assert cli.main(_args(project, "audit-worksheet")) == 0
    out = capsys.readouterr().out
    assert "of 4 eligible pair(s)" in out and "of 4 benchmark question(s)" in out
    assert "PROVISIONAL" not in out
    report = json.loads((project / "out/audit-population-TEST-ARTIFACT.json").read_text())
    assert report["provisional"] is False and report["eligible_pairs"] == 4
    rows = list(csv.DictReader(io.StringIO(
        (project / "out/audit-worksheet-TEST-ARTIFACT.csv").read_text())))  # fmt: skip
    assert rows and "audit_status" not in rows[0]  # official sheets carry no provisional label


def test_provisional_artifacts_are_labelled_in_name_and_in_every_row(
    project,
    monkeypatch,
    capsys,  # noqa: F811
):
    _run_and_judge(project, monkeypatch)
    path = project / "out/judgements-demo.json"
    data = json.loads(path.read_text())
    broken = next(i for i in data["items"] if i["state"] == "judged")
    broken.update(state="failed", assessment=None)
    path.write_text(json.dumps(data))
    capsys.readouterr()
    assert cli.main(_args(project, "audit-worksheet")) == 2  # official: refused
    assert cli.main(_args(project, "audit-worksheet", "--provisional")) == 0
    sheet = project / "out/audit-worksheet-PROVISIONAL-TEST-ARTIFACT.csv"
    rows = list(csv.DictReader(io.StringIO(sheet.read_text())))
    assert rows and all(r["audit_status"].startswith("PROVISIONAL") for r in rows)
    assert not (project / "out/audit-worksheet-TEST-ARTIFACT.csv").exists()


def test_the_sample_is_reproducible_and_blinded_as_before(project, monkeypatch, capsys):  # noqa: F811
    _run_and_judge(project, monkeypatch)
    assert cli.main(_args(project, "audit-worksheet")) == 0
    first = (project / "out/audit-worksheet-TEST-ARTIFACT.csv").read_text()
    assert cli.main(_args(project, "audit-worksheet")) == 0
    assert (project / "out/audit-worksheet-TEST-ARTIFACT.csv").read_text() == first
    header = first.splitlines()[0]
    assert "arm" not in header.split(",") and "question_id" not in header.split(",")


# ---------------------------------------------------------------- rule-item verification


def test_rule_scored_abstentions_get_their_own_worksheet(pipeline):  # noqa: F811
    inputs, _ = pipeline

    def edit(entry):
        if entry.question_id in ("r1-q0", "r1-q3"):
            rag = _side_update(entry.rag, status="insufficient_evidence", answer="no evidence")
            return entry.model_copy(update={"rag": rag})
        return entry

    items = rule_items([_with_entries(inputs, edit)])
    assert {i.question_id for i in items} == {"r1-q0", "r1-q3"}
    text, key, sampled = build_rule_worksheet(items, fraction=1.0, seed=3)
    rows = list(csv.DictReader(io.StringIO(text)))
    assert len(rows) == len(sampled) == 2 and all(r["item_id"].startswith("R-") for r in rows)
    assert all(r["human_makes_no_repository_claims"] == "" for r in rows)  # nothing pre-filled
    assert {k["item_id"] for k in json.loads(key)} == {r["item_id"] for r in rows}
    assert build_rule_worksheet([], fraction=0.2, seed=3) == ("", "[]\n", [])


def test_rule_check_is_reported_separately_and_accounts_for_every_row():
    key = [
        {"item_id": f"R-000{i}", "repository_id": "r", "question_id": f"q{i}"} for i in (1, 2, 3, 4)
    ]
    filled = {"human_makes_no_repository_claims": "yes", "human_is_genuine_abstention": "yes"}
    rows = [
        {"item_id": "R-0001", **filled},
        {"item_id": "R-0002", "human_makes_no_repository_claims": "yes",
         "human_is_genuine_abstention": "no"},  # disputed
        {"item_id": "R-0003", "human_makes_no_repository_claims": "maybe",
         "human_is_genuine_abstention": "yes"},  # invalid
        {"item_id": "A-0001", **filled},  # a judge-worksheet id: unexpected here
    ]  # fmt: skip
    result = evaluate_rule_check(rows, key)
    assert result["kind"].startswith("rule-item verification")
    assert (result["rated"], result["rule_confirmed"]) == (2, 1)
    assert result["rule_confirmed_rate"] == 0.5 and result["rule_disputed"] == ["R-0002"]
    assert [i["item_id"] for i in result["invalid_ratings"]] == ["R-0003"]
    assert result["unexpected_item_ids"] == ["A-0001"]
    assert result["key_items_missing_from_worksheet"] == ["R-0004"]
    assert result["complete"] is False


# ---------------------------------------------------------------- strict audit-agree


def _agree_inputs(inputs: RepoInputs):
    questions = {"r1": dict(inputs.questions)}
    judged = {"r1": inputs.judgements.by_key()}
    key, rows = [], []
    for n, ((qid, arm), item) in enumerate(sorted(judged["r1"].items()), start=1):
        item_id = f"A-{n:04d}"
        key.append({"item_id": item_id, "repository_id": "r1", "question_id": qid, "arm": arm})
        a = item.assessment
        rows.append(
            {
                "item_id": item_id,
                "human_fact_verdicts": ";".join(
                    f"{k}={v.value}" for k, v in a.fact_verdicts.items()
                ),
                "human_groundedness": str(a.groundedness),
                "human_abstained": "yes" if a.abstained else "no",
                "human_unsupported_claims": "", "human_notes": "",
            }
        )  # fmt: skip
    return rows, key, questions, judged


def test_a_perfect_rater_has_every_row_accounted_for(pipeline):  # noqa: F811
    inputs, _ = pipeline
    rows, key, questions, judged = _agree_inputs(inputs)
    result = evaluate_agreement(rows, key, questions, judged, rater="h")
    c = result["completeness"]
    assert (c["worksheet_rows"], c["human_rated"], c["eligible_rows"]) == (8, 8, 8)
    assert c["rows_with_valid_judge_rating"] == 8 and c["unrated_rows"] == 0
    assert result["rows_accounted_for"] and result["fully_rated"]
    assert result["correctness_score_0_2"]["raw_agreement"] == 1.0
    assert "correctness_score_0_2_unweighted" in result and "abstained" in result


def test_undefined_kappa_is_null_with_a_reason(pipeline):  # noqa: F811
    inputs, _ = pipeline
    rows, key, questions, judged = _agree_inputs(inputs)
    result = evaluate_agreement(rows, key, questions, judged, rater="h")
    # nobody ever abstained -> chance agreement is 1 -> kappa undefined
    assert result["abstained"]["kappa"] is None and result["abstained"]["raw_agreement"] == 1.0
    assert "single category" in result["abstained"]["kappa_undefined_reason"]
    assert agreement([], []).to_json()["kappa_undefined_reason"] == "no eligible rated pairs"


def test_invalid_duplicate_unexpected_missing_and_unrated_rows_are_reported_never_skipped(
    pipeline,  # noqa: F811
):
    inputs, _ = pipeline
    rows, key, questions, judged = _agree_inputs(inputs)
    rows[0]["human_fact_verdicts"] = "f1=perhaps"  # invalid verdict
    for field in ("human_fact_verdicts", "human_groundedness", "human_abstained"):
        rows[1][field] = ""  # unrated
    rows.append(dict(rows[2]))  # duplicate id
    rows.append({"item_id": "R-0001", "human_groundedness": "2"})  # a rule-sheet id
    del rows[3]  # a key item missing from the worksheet
    judged["r1"].pop(("r1-q1", "rag"), None)  # a judge rating that does not exist
    result = evaluate_agreement(rows, key, questions, judged, rater="h")
    c = result["completeness"]
    assert [i["item_id"] for i in c["invalid_ratings"]] == [rows[0]["item_id"]]
    assert c["unrated_rows"] == 1 and c["duplicate_item_ids"] == [rows[2]["item_id"]]
    assert (
        c["unexpected_item_ids"] == ["R-0001"] and len(c["key_items_missing_from_worksheet"]) == 1
    )
    assert result["rows_accounted_for"] is False and result["fully_rated"] is False
    assert c["eligible_rows"] + c["unrated_rows"] + len(c["invalid_ratings"]) <= 8


def test_a_missing_judge_rating_is_listed_and_a_rule_item_is_not_a_judge_rating(pipeline):  # noqa: F811
    inputs, _ = pipeline
    rows, key, questions, judged = _agree_inputs(inputs)
    target = next(e for e in key if (e["question_id"], e["arm"]) == ("r1-q1", "rag"))
    judged["r1"].pop(("r1-q1", "rag"))
    rule = next(e for e in key if (e["question_id"], e["arm"]) == ("r1-q2", "rag"))
    judged["r1"][("r1-q2", "rag")] = JudgedItem(
        question_id="r1-q2", arm="rag", blind_id="J-x", state="rule"
    )
    result = evaluate_agreement(rows, key, questions, judged, rater="h")
    c = result["completeness"]
    assert c["failed_or_missing_judge_ratings"] == [target["item_id"]]
    assert c["other_exclusions"]["rule_scored_item_not_a_judge_rating"] == [rule["item_id"]]
    assert c["eligible_rows"] == 6 and result["rated_answers"] == 6


def test_cli_audit_agree_exits_3_on_a_bad_row_and_reports_it(project, monkeypatch, capsys):  # noqa: F811
    _run_and_judge(project, monkeypatch)
    assert cli.main(_args(project, "audit-worksheet")) == 0
    sheet = project / "out/audit-worksheet-TEST-ARTIFACT.csv"
    rows = list(csv.DictReader(io.StringIO(sheet.read_text())))
    rows[0].update(human_fact_verdicts="f1=nonsense", human_groundedness="2", human_abstained="no")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    filled = project / "out/filled.csv"
    filled.write_text(buffer.getvalue())
    capsys.readouterr()
    code = cli.main(_args(project, "audit-agree", "--worksheet", str(filled), "--key",
                          str(project / "out/audit-key-TEST-ARTIFACT.json")))  # fmt: skip
    assert code == 3
    result = json.loads((project / "out/audit-agreement.json").read_text())
    assert result["completeness"]["invalid_ratings"] and result["rows_accounted_for"] is False


@pytest.mark.parametrize("flag", ["--provisional"])
def test_provisional_flag_exists_only_on_the_worksheet_command(flag, project, capsys):  # noqa: F811
    with pytest.raises(SystemExit):
        cli.main(_args(project, "audit-agree", "--worksheet", "x", "--key", "y", flag))
