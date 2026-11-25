"""M11A corrections: judge-assisted abstention labels, groundedness forms, historical sensitivity.

All figures are small hand-computed cases (no provider, no network).
"""

from __future__ import annotations

import pytest

from copilot.evaluation.generation.analysis import (
    analyse,
    concordance,
    groundedness_composition,
    historical_sensitivity,
    primary_abstention_label,
    render_markdown,
)
from copilot.evaluation.generation.execution import SideRecord
from copilot.evaluation.generation.refusal import abstention_metrics, classify_response
from tests.unit.test_generation_analysis import PROTOCOL, _row

# ---------------------------------------------------------------- the cue rules (diagnostic)


@pytest.mark.parametrize(
    "text",
    [
        "Nothing in the repository shows who deployed it.",
        "There is no mention of this in the code.",
        "The retrieved sources do not say.",
        "The provided files do not show how it is deployed.",
    ],
)
def test_cue_rules_recognise_paraphrased_insufficient_evidence(text):
    assert classify_response(text).label == "insufficient_evidence"


def test_cue_rules_keep_generic_refusal_and_answers_apart():
    assert classify_response("I cannot help with that.").label == "generic_refusal"
    assert classify_response("It returns the sum of the items.").label == "answered"


# ---------------------------------------------------------------- the primary label


def _side(text="some answer", *, state="successful", status="answered", arm="rag"):
    return SideRecord(
        arm=arm, state=state, prompt_version="v", status=status if state == "successful" else None,
        answer=text if state == "successful" else None,
    )  # fmt: skip


def _judged(response_type, *, abstained=False, kind="judge"):
    return {"evaluator_kind": kind, "judged_response_type": response_type,
            "judged_abstained": abstained}  # fmt: skip


@pytest.mark.parametrize(
    ("response_type", "label"),
    [
        ("answered", "answered"),
        ("insufficient_evidence", "insufficient_evidence"),
        ("generic_refusal", "generic_refusal"),
        ("mixed", "ambiguous"),
        ("ambiguous", "ambiguous"),
    ],
)
def test_a_generated_response_takes_the_judge_response_type(response_type, label):
    side = _side("Nothing in the repository shows who deployed it.")
    assert primary_abstention_label(side, _judged(response_type)) == (label, "judge")


def test_judge_label_wins_over_the_cue_rules_even_when_they_disagree():
    # the text says nothing the cue rules recognise, the judge says it is an abstention
    side = _side("Deployment is outside what these sources cover.")
    assert classify_response(side.answer).label == "answered"
    assert primary_abstention_label(side, _judged("insufficient_evidence", abstained=True)) == (
        "insufficient_evidence",
        "judge",
    )


def test_a_system_abstention_is_labelled_by_the_system_rule():
    side = _side("The repository evidence is insufficient.", status="insufficient_evidence")
    assert primary_abstention_label(side, {}) == ("insufficient_evidence", "system")


def test_a_failed_or_missing_response_is_no_response_not_a_refusal():
    for state in ("failed", "not_attempted"):
        assert primary_abstention_label(_side(state=state), {}) == ("no_response", "none")


def test_a_response_without_a_valid_judge_rating_is_unrated_never_regex_labelled():
    side = _side("The retrieved sources do not say.")  # the cue rules would call this an abstention
    assert classify_response(side.answer).label == "insufficient_evidence"
    assert primary_abstention_label(side, {}) == ("unrated", "none")  # judge failed / not run


def test_abstained_flag_alone_does_not_classify():
    # an older judge file or human rating: abstained=true but no response_type
    side = _side("It is not shown.")
    assert primary_abstention_label(side, _judged(None, abstained=True)) == ("unrated", "none")


# ---------------------------------------------------------------- metrics and concordance


def test_unrated_is_counted_but_excluded_from_every_denominator():
    items = [
        (False, "insufficient_evidence"), (False, "answered"), (False, "unrated"),
        (True, "answered"), (True, "generic_refusal"), (True, "unrated"), (True, "no_response"),
    ]  # fmt: skip
    m = abstention_metrics(items)
    assert (m.unanswerable_evaluable, m.unrated_unanswerable) == (2, 1)
    assert m.appropriate_abstention_rate == 0.5 and m.unsupported_answer_rate == 0.5
    assert (m.answerable_evaluable, m.unrated_answerable, m.no_response_answerable) == (2, 1, 1)
    assert m.false_refusal_rate == 0.5


def _r(arm, qid, label, regex, source="judge", answerable=False):
    return _row("r", qid, arm, 0, answerable=answerable, abstention_label=label,
                abstention_label_regex=regex, abstention_label_source=source)  # fmt: skip


def test_concordance_counts_denominators_and_disagreements_by_hand():
    rows = [
        _r("plain", "q1", "insufficient_evidence", "insufficient_evidence"),  # agree
        _r("plain", "q2", "insufficient_evidence", "answered"),  # cue rules miss it
        _r("rag", "q1", "answered", "answered"),  # agree
        _r("rag", "q2", "ambiguous", "insufficient_evidence"),  # disagree
        _r("rag", "q3", "insufficient_evidence", "insufficient_evidence", source="system"),
        _r("rag", "q4", "unrated", "insufficient_evidence", source="none"),
    ]
    c = concordance(rows)
    assert (c["compared"], c["agree"]) == (4, 2) and c["agreement_rate"] == 0.5
    assert c["system_not_compared"] == 1 and c["unrated_not_compared"] == 1
    assert {(d["arm"], d["question_id"]) for d in c["disagreements"]} == {
        ("plain", "q2"),
        ("rag", "q2"),
    }
    assert c["confusion"]["judge=insufficient_evidence regex=answered"] == 1


def test_analysis_reports_judge_metrics_regex_diagnostic_and_unrated_separately():
    rows = [
        _r("plain", "q1", "insufficient_evidence", "answered"),  # judge: abstained; cue rules miss
        _r("plain", "q2", "answered", "answered"),
        _r("plain", "q3", "unrated", "insufficient_evidence", source="none"),
    ]
    arm = analyse(rows, PROTOCOL)["combined"]["per_arm"]["plain"]
    judge, regex = arm["abstention"], arm["abstention_regex_diagnostic"]
    assert judge["unanswerable_evaluable"] == 2 and judge["appropriate_abstention_rate"] == 0.5
    assert judge["unrated_unanswerable"] == 1
    assert regex["unanswerable_evaluable"] == 3 and regex[
        "appropriate_abstention_rate"
    ] == pytest.approx(1 / 3)
    assert arm["abstention_sources"]["unrated_missing_judge_rating"] == 1
    assert arm["abstention_concordance"]["compared"] == 2


# ---------------------------------------------------------------- groundedness forms


def _g(arm, qid, g, kind="judge", abstained=False, *, answerable=True, **extra):
    substantive = {} if (kind == "rule" or abstained) else {"groundedness_substantive": g}
    return _row("r", qid, arm, g, answerable=answerable, evaluator_kind=kind,
                judged_abstained=abstained, **substantive, **extra)  # fmt: skip


def test_groundedness_is_reported_overall_and_for_substantive_answers_with_denominators():
    rows = [
        # q1: both arms answer; plain g=1, rag g=2
        _g("plain", "q1", 1),
        _g("rag", "q1", 2),
        # q2: rag is a rule-scored system abstention on an answerable question (g=2 by convention)
        _g("plain", "q2", 0),
        _g("rag", "q2", 2, kind="rule", abstained=True),
        # q3: plain judged answer, rag's judge rating is missing
        _g("plain", "q3", 2),
        _row("r", "q3", "rag", None, state="judge_failed"),
    ]
    block = analyse(rows, PROTOCOL)["combined"]
    overall = block["secondary_paired"]["groundedness"]
    substantive = block["secondary_paired"]["groundedness_substantive"]
    assert overall["n"] == 2 and overall["mean_difference"] == pytest.approx((1 + 2) / 2)  # q1, q2
    assert substantive["n"] == 1 and substantive["mean_difference"] == pytest.approx(1.0)  # q1 only
    rag = block["groundedness_composition"]["rag"]
    assert (rag["responses"], rag["scored_overall"], rag["judge_scored"]) == (3, 2, 1)
    assert (rag["rule_scored_system_abstentions"], rag["substantive_generated"]) == (1, 1)
    assert rag["missing_judge_ratings"] == 1
    assert rag["mean_overall"] == pytest.approx(2.0) and rag["mean_rule_scored"] == 2.0
    plain = block["groundedness_composition"]["plain"]
    assert plain["substantive_generated"] == 3 and plain["mean_substantive"] == pytest.approx(1.0)


def test_rule_scored_abstention_raises_groundedness_but_not_correctness():
    # RAG abstains on every answerable question: perfect groundedness, zero correctness
    rows = []
    for i in range(3):
        rows.append(_g("plain", f"q{i}", 1, score=1) if False else _row(
            "r", f"q{i}", "plain", 1, evaluator_kind="judge", judged_abstained=False,
            groundedness_substantive=1))  # fmt: skip
        rows.append(_row("r", f"q{i}", "rag", 0, evaluator_kind="rule", judged_abstained=True))
    for r in rows:
        if r["arm"] == "rag":
            r["groundedness"] = 2
    block = analyse(rows, PROTOCOL)["combined"]
    assert block["primary"]["mean_difference"] == pytest.approx(-1.0)  # correctness: worse
    assert block["secondary_paired"]["groundedness"]["mean_difference"] == pytest.approx(1.0)
    assert block["secondary_paired"]["groundedness_substantive"]["n"] == 0  # no substantive pair


def test_groundedness_composition_handles_an_empty_arm():
    assert groundedness_composition([])["plain"]["mean_overall"] is None


# ---------------------------------------------------------------- historical sensitivity


def test_historical_sensitivity_excludes_every_question_with_a_reused_side():
    rows = []
    # four answerable questions; q1 and q2 have a reused historical RAG side
    for qid, (p, r, hist) in {"q1": (0, 2, True), "q2": (1, 2, True), "q3": (1, 1, False),
                              "q4": (2, 2, False)}.items():  # fmt: skip
        rows.append(_row("r", qid, "plain", p))
        rows.append(_row("r", qid, "rag", r, origin="historical" if hist else "generated"))
    s = historical_sensitivity(rows, seed=1, resamples=500, confidence=0.95)
    assert s["historical_sides"] == 2 and s["questions_with_a_historical_side"] == 2
    assert s["complete_pairs_excluded"] == 2
    assert s["excluded_question_ids"] == ["r/q1", "r/q2"]
    assert s["primary_all"]["n"] == 4 and s["primary_all"]["mean_difference"] == pytest.approx(0.75)
    assert s["primary_excluding_historical"]["n"] == 2
    assert s["primary_excluding_historical"]["mean_difference"] == pytest.approx(0.0)
    assert s["primary_excluding_historical"]["ci_low"] is not None


def test_historical_sensitivity_with_nothing_reused_matches_the_primary():
    rows = [_row("r", "q1", "plain", 1), _row("r", "q1", "rag", 2)]
    s = historical_sensitivity(rows, seed=1, resamples=200, confidence=0.95)
    assert s["complete_pairs_excluded"] == 0
    assert (
        s["primary_all"]["mean_difference"] == s["primary_excluding_historical"]["mean_difference"]
    )


def test_markdown_shows_both_groundedness_forms_the_diagnostic_and_the_sensitivity():
    rows = [
        _g("plain", "q1", 1), _g("rag", "q1", 2, origin="historical"),
        _r("plain", "q2", "insufficient_evidence", "answered"),
    ]  # fmt: skip
    report = {
        "test_artifact": False, "protocol_hash": "h", "caveats": [],
        "primary_endpoint": {"name": "n", "definition": "d"},
        "analysis": analyse(rows, PROTOCOL),
    }  # fmt: skip
    text = render_markdown(report)
    needles = (
        "Groundedness: two forms",
        "cue-rule diagnostic",
        "Historical-response sensitivity",
        "unrated (no valid judge rating)",
        "raises this number while correctness is 0",
    )
    for needle in needles:
        assert needle in text
