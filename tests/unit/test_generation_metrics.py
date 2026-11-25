"""Milestone 11A: inventory, citation checker, reference detector, abstention metrics, rubric."""

from __future__ import annotations

import os

import pytest

from copilot.config import Settings
from copilot.evaluation.benchmark import Region
from copilot.evaluation.generation.citations import RetrievedRegion, check_citations
from copilot.evaluation.generation.inventory import build_inventory
from copilot.evaluation.generation.references import find_references
from copilot.evaluation.generation.refusal import abstention_metrics, classify_response
from copilot.evaluation.generation.rubric import (
    Assessment,
    EvaluationError,
    FactVerdict,
    score_assessment,
)
from copilot.evaluation.generation.schema import Answerability, GenerationQuestion, KeyFact
from copilot.ingestion import IngestionPolicy, ingest_repository
from copilot.llm import FakeLLMClient
from copilot.rag import answer_plain

POLICY = IngestionPolicy.from_settings(Settings(_env_file=None))

ORDERS = '''\
"""Orders."""
import json
import requests_stub_not_really
from invorders.models import order


class OrderService:
    limit = 5

    def place_order(self, order_id):
        return json.dumps({"id": order_id})

    def cancel(self, order_id):
        return order_id


def helper_function(value):
    return value
'''
CONFIG_A = "SETTING_ONE = 1\n"


@pytest.fixture
def inventory(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src" / "pkg" / "services").mkdir(parents=True)
    (repo / "src" / "pkg" / "orders.py").write_text(ORDERS, encoding="utf-8")
    (repo / "src" / "pkg" / "config.py").write_text(CONFIG_A, encoding="utf-8")
    (repo / "src" / "pkg" / "services" / "config.py").write_text("OTHER = 2\n", encoding="utf-8")
    (repo / "src" / "pkg" / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    (repo / "docs").mkdir()
    (repo / "docs" / "guide.md").write_text(
        "Set `INVORDERS_TAX_RATE_BPS` to change tax.\n", "utf-8"
    )
    ingestion = ingest_repository(repo, POLICY, repository_name="demo")
    return build_inventory(ingestion.files, "demo")


# --------------------------------------------------------------------------- inventory


def test_inventory_contents(inventory):
    assert "src/pkg/orders.py" in inventory.paths
    assert inventory.line_count("src/pkg/orders.py") == len(ORDERS.splitlines())
    assert "pkg.orders" in inventory.modules and "pkg.services.config" in inventory.modules
    assert {"OrderService", "place_order", "helper_function"} <= set(inventory.defined)
    assert "OrderService.place_order" in inventory.qualified
    assert "OrderService.limit" in inventory.qualified
    assert {"json", "requests_stub_not_really", "invorders"} <= inventory.imported_roots
    assert inventory.unparsable == ("src/pkg/broken.py",)
    assert inventory.basenames["config.py"] == (
        "src/pkg/config.py",
        "src/pkg/services/config.py",
    )
    assert "INVORDERS_TAX_RATE_BPS" in inventory.text_tokens


def test_building_an_inventory_never_executes_repository_code(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "boom.py").write_text(
        "import os\nos.environ['M11A_EXEC_SENTINEL'] = '1'\nraise SystemExit(3)\n", encoding="utf-8"
    )
    files = ingest_repository(repo, POLICY, repository_name="demo").files
    inv = build_inventory(files, "demo")
    assert inv.parsed_python_files == 1
    assert "M11A_EXEC_SENTINEL" not in os.environ


# --------------------------------------------------------------------------- references


def _by_text(report):
    return {r.text: r for r in report.references}


def test_real_and_invented_paths_and_symbols(inventory):
    report = find_references(
        "See `src/pkg/orders.py` where `OrderService.place_order` is defined, and "
        "`OrderService.cancel_all` in src/pkg/fantasy.py.",
        inventory,
    )
    refs = _by_text(report)
    assert refs["src/pkg/orders.py"].verdict == "verified"
    assert refs["OrderService.place_order"].verdict == "verified"
    assert refs["OrderService.cancel_all"].verdict == "not_found"
    assert refs["src/pkg/fantasy.py"].verdict == "not_found"
    assert report.not_found == 2 and report.verified_fraction == pytest.approx(0.5)


def test_ambiguous_basename_and_unique_basename(inventory):
    refs = _by_text(find_references("Edit config.py and also orders.py.", inventory))
    assert refs["config.py"].verdict == "ambiguous"
    assert refs["orders.py"].verdict == "verified"


def test_hypothetical_and_suggested_references_are_not_flagged(inventory):
    report = find_references(
        "You could create `src/pkg/new_feature.py` with a `NewFeature` class, for example.",
        inventory,
    )
    assert report.not_found == 0
    assert {r.verdict for r in report.references} == {"cannot_evaluate"}


def test_stdlib_builtins_and_generic_words_are_ignored(inventory):
    report = find_references(
        "It calls `os.replace` and `len()`, using `self` and `value`.", inventory
    )
    assert report.not_found == 0
    assert all(r.verdict == "ignored" for r in report.references)
    assert report.counted == 0 and report.verified_fraction is None


def test_dependencies_external_are_not_flagged_unless_claimed_as_used(inventory):
    plain_import = find_references("```python\nimport numpy\nimport json\n```", inventory)
    refs = _by_text(plain_import)
    assert refs["numpy"].verdict == "cannot_evaluate"
    assert refs["json"].verdict == "ignored"
    claim = find_references("The project relies on `flask` for its web layer.", inventory)
    assert _by_text(claim)["flask"].verdict == "not_found"
    real = find_references("The project uses `requests_stub_not_really`.", inventory)
    assert _by_text(real)["requests_stub_not_really"].verdict == "verified"


def test_a_repository_own_module_import_is_verified(inventory):
    refs = _by_text(
        find_references("```python\nfrom pkg.orders import OrderService\n```", inventory)
    )
    assert refs["pkg"].verdict == "verified"


def test_urls_are_not_paths_and_env_names_found_in_docs_are_verified(inventory):
    report = find_references(
        "Docs at https://example.com/readme.md; set `INVORDERS_TAX_RATE_BPS` first.", inventory
    )
    refs = _by_text(report)
    assert "readme.md" not in refs and "example.com/readme.md" not in refs
    assert refs["INVORDERS_TAX_RATE_BPS"].verdict == "verified"


def test_call_syntax_without_backticks_is_checked(inventory):
    refs = _by_text(find_references("Then helper_function(1) and ghost_helper(2) run.", inventory))
    assert refs["helper_function"].verdict == "verified"
    assert refs["ghost_helper"].verdict == "not_found"


# --------------------------------------------------------------------------- citations


def _region(path, start, end, number):
    return RetrievedRegion(source_number=number, file_path=path, start_line=start, end_line=end)


SOURCES = {1: _region("src/pkg/orders.py", 6, 12, 1), 2: _region("src/pkg/config.py", 1, 1, 2)}
EXPECTED = [Region(file_path="src/pkg/orders.py", start_line=10, end_line=11)]


def test_valid_source_label_overlaps_retrieved_and_expected(inventory):
    report = check_citations(
        "Placing is in [Source 1].",
        arm="rag",
        inventory=inventory,
        sources=SOURCES,
        expected=EXPECTED,
    )
    assert report.applicable and not report.missing
    check = report.citations[0]
    assert check.status == "valid" and check.in_retrieved_evidence and check.overlaps_expected
    assert report.retrieved_uncited == (2,)  # returned by retrieval, not used by the answer
    assert report.valid_fraction == 1.0 and report.expected_overlap_fraction == 1.0


def test_unknown_source_number_is_invented(inventory):
    report = check_citations("See [Source 7].", arm="rag", inventory=inventory, sources=SOURCES)
    assert report.citations[0].status == "unknown_source" and report.invented == 1


def test_a_valid_but_irrelevant_citation_does_not_overlap_the_expected_region(inventory):
    report = check_citations(
        "Config is in [Source 2].",
        arm="rag",
        inventory=inventory,
        sources=SOURCES,
        expected=EXPECTED,
    )
    check = report.citations[0]
    assert check.status == "valid" and check.overlaps_expected is False
    assert report.expected_overlap_fraction == 0.0


def test_a_rag_answer_without_citations_is_missing_not_invalid(inventory):
    report = check_citations(
        "It is done somewhere.", arm="rag", inventory=inventory, sources=SOURCES
    )
    assert report.applicable and report.missing and report.citations == ()
    assert report.valid_fraction is None


def test_plain_answers_are_not_penalised_for_lacking_source_labels(inventory):
    report = check_citations("Orders are placed by a service.", arm="plain", inventory=inventory)
    assert not report.applicable and not report.missing
    assert report.not_applicable_reason is not None


def test_plain_answer_with_explicit_locations_is_checked(inventory):
    text = "Look at src/pkg/orders.py:10-11, src/pkg/nowhere.py:5 and src/pkg/orders.py:900."
    report = check_citations(text, arm="plain", inventory=inventory, expected=EXPECTED)
    statuses = {c.raw: c.status for c in report.citations}
    assert statuses["src/pkg/orders.py:10-11"] == "valid"
    assert statuses["src/pkg/nowhere.py:5"] == "file_missing"
    assert statuses["src/pkg/orders.py:900"] == "invalid_range"
    assert report.invented == 2 and report.applicable
    assert report.retrieved_support_fraction is None  # no retrieval in the plain arm


def test_malformed_citations(inventory):
    report = check_citations(
        "See src/pkg/orders.py:abc and src/pkg/orders.py:12-4.", arm="plain", inventory=inventory
    )
    assert {c.status for c in report.citations} == {"malformed"}


def test_location_cited_by_rag_but_not_in_retrieved_evidence(inventory):
    report = check_citations(
        "Also src/pkg/orders.py:1-2.", arm="rag", inventory=inventory, sources=SOURCES
    )
    assert report.citations[0].status == "valid"
    assert report.citations[0].in_retrieved_evidence is False


def test_no_answer_means_not_applicable(inventory):
    assert not check_citations(None, arm="rag", inventory=inventory).applicable
    assert not check_citations("text", arm="rag", inventory=inventory, answered=False).applicable


# --------------------------------------------------------------------------- abstention


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("The retrieved evidence does not show how tax is computed.", "insufficient_evidence"),
        ("I cannot determine this from the repository files provided.", "insufficient_evidence"),
        ("I don't have access to the repository, so I can't say.", "insufficient_evidence"),
        ("There is insufficient repository evidence to answer.", "insufficient_evidence"),
        ("I'm sorry, but I can't help with that.", "generic_refusal"),
        ("Placing reserves stock [Source 1]. The evidence does not show tax rules.", "ambiguous"),
        ("Orders are placed by OrderService.place_order, which reserves stock.", "answered"),
    ],
)
def test_classification_by_hand_cases(text, label):
    assert classify_response(text).label == label


def test_missing_responses_are_never_refusals():
    assert classify_response(None, state="failed").label == "no_response"
    assert classify_response("", state="successful").label == "no_response"
    assert classify_response("any text", state="not_attempted").label == "no_response"


def test_system_level_insufficient_evidence_is_an_abstention_from_the_system():
    label = classify_response(None, state="successful", system_insufficient=True)
    assert (label.label, label.source) == ("insufficient_evidence", "system")


def test_classification_works_on_fake_llm_output():
    fake = FakeLLMClient("The repository evidence does not show the monitoring service.")
    answer = answer_plain(
        "Which monitoring service is used?", fake, settings=Settings(_env_file=None)
    )
    assert classify_response(answer.answer).label == "insufficient_evidence"


def test_abstention_metrics_on_a_hand_built_set():
    items = [
        (False, "insufficient_evidence"),
        (False, "insufficient_evidence"),
        (False, "answered"),
        (False, "generic_refusal"),
        (False, "no_response"),
        (True, "answered"),
        (True, "answered"),
        (True, "insufficient_evidence"),
        (True, "ambiguous"),
        (True, "no_response"),
    ]
    metrics = abstention_metrics(items)
    assert metrics.unanswerable_total == 5 and metrics.unanswerable_evaluable == 4
    assert metrics.answerable_total == 5 and metrics.answerable_evaluable == 4
    assert metrics.appropriate_abstention_rate == 0.5
    assert metrics.unsupported_answer_rate == 0.25
    assert metrics.generic_refusal_rate_on_unanswerable == 0.25
    assert metrics.false_refusal_rate == 0.25
    assert (metrics.ambiguous_answerable, metrics.ambiguous_unanswerable) == (1, 0)
    assert (metrics.no_response_answerable, metrics.no_response_unanswerable) == (1, 1)


def test_rates_are_none_without_a_denominator():
    metrics = abstention_metrics([(True, "no_response"), (True, "no_response")])
    assert metrics.false_refusal_rate is None and metrics.appropriate_abstention_rate is None


# --------------------------------------------------------------------------- rubric


def _answerable(n_facts: int = 3) -> GenerationQuestion:
    facts = tuple(
        KeyFact(
            id=f"f{i}",
            statement=f"fact number {i} statement",
            regions=(Region(file_path="a.py", start_line=1, end_line=2),),
        )
        for i in range(1, n_facts + 1)
    )
    return GenerationQuestion(
        id="q", repository_id="r", question="Q?", category="c",
        answerability=Answerability.ANSWERABLE, key_facts=facts, acceptable_answer="x",
    )  # fmt: skip


def _unanswerable() -> GenerationQuestion:
    return GenerationQuestion(
        id="u", repository_id="r", question="U?", category="c",
        answerability=Answerability.UNANSWERABLE, unanswerable_rationale="r",
        expected_abstention="e",
    )  # fmt: skip


def _assess(verdicts=(), claims=(), abstained=False, groundedness=2):
    return Assessment(
        evaluator_kind="human",
        evaluator_id="tester",
        fact_verdicts={f"f{i}": FactVerdict(v) for i, v in enumerate(verdicts, start=1)},
        material_unsupported_claims=tuple(claims),
        groundedness=groundedness,
        abstained=abstained,
    )


@pytest.mark.parametrize(
    ("verdicts", "claims", "score"),
    [
        (("supported", "supported", "supported"), (), 2),
        (("supported", "supported", "missing"), (), 1),
        (("supported", "missing", "missing"), (), 1),
        (("missing", "missing", "missing"), (), 0),
        (("supported", "supported", "supported"), ("invented helper",), 1),
        (("supported", "supported", "contradicted"), (), 1),
        (("supported", "contradicted", "contradicted"), (), 0),
        (("contradicted", "contradicted", "missing"), (), 0),
    ],
)
def test_answerable_scoring_rules(verdicts, claims, score):
    result = score_assessment(_answerable(), _assess(verdicts, claims))
    assert result.score == score
    assert result.completeness == pytest.approx(verdicts.count("supported") / 3)


@pytest.mark.parametrize(
    ("abstained", "claims", "score", "appropriate"),
    [(True, (), 2, True), (True, ("guess",), 1, False), (False, (), 0, False)],
)
def test_unanswerable_scoring_rules(abstained, claims, score, appropriate):
    result = score_assessment(_unanswerable(), _assess((), claims, abstained=abstained))
    assert result.score == score and result.appropriate_abstention is appropriate


def test_assessments_that_do_not_match_the_question_are_errors_not_defaults():
    with pytest.raises(EvaluationError):
        score_assessment(_answerable(), _assess(("supported", "supported")))
    with pytest.raises(EvaluationError):
        score_assessment(_answerable(2), _assess(("supported", "supported", "supported")))
    with pytest.raises(EvaluationError):
        score_assessment(_unanswerable(), _assess(("supported",)))


def test_scoring_is_independent_of_keyword_overlap():
    # The rubric consumes verdicts only; text never enters score_assessment.
    import inspect

    assert "answer" not in inspect.signature(score_assessment).parameters
