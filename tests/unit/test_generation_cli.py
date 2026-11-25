"""Milestone 11A: the generation CLI end to end on a tiny self-contained project (no network)."""

from __future__ import annotations

import csv
import io
import json
import socket
from pathlib import Path

import pytest

from copilot.config import Settings
from copilot.evaluation.benchmark import Region
from copilot.evaluation.generation import __main__ as cli
from copilot.evaluation.generation.protocol import (
    PROTOCOL_PATH,
    default_protocol,
    load_protocol,
)
from copilot.evaluation.generation.schema import (
    Answerability,
    GenerationQuestion,
    KeyFact,
    benchmark_digest,
    dump_generation_benchmark,
    load_generation_benchmark,
    seal_generation_regions,
)
from copilot.ingestion import IngestionPolicy, ingest_repository
from copilot.llm import FakeLLMClient
from tests.rag_helpers import FakeRetriever, make_result

ROOT = Path(__file__).resolve().parents[2]
CODE = "def total(items):\n    return sum(items)\n\n\ndef count(items):\n    return len(items)\n"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A one-repository project: source tree, sealed benchmark, frozen protocol."""
    root = tmp_path / "proj"
    (root / "repo/src").mkdir(parents=True)
    (root / "repo/src/a.py").write_text(CODE)
    (root / "bench").mkdir()
    settings = Settings(_env_file=None)
    files = ingest_repository(
        root / "repo", IngestionPolicy.from_settings(settings), repository_name="demo"
    ).files
    region = Region(file_path="src/a.py", start_line=1, end_line=2)
    questions = [
        GenerationQuestion(
            id=f"q{i}",
            repository_id="demo",
            question=f"What does function number {i} do?",
            category="c",
            answerability=Answerability.ANSWERABLE,
            acceptable_answer="sums",
            key_facts=(KeyFact(id="f1", statement="returns the sum of items", regions=(region,)),),
        )  # fmt: skip
        for i in range(1, 4)
    ] + [
        GenerationQuestion(
            id="u1",
            repository_id="demo",
            question="Who deployed the demo service?",
            category="c",
            answerability=Answerability.UNANSWERABLE,
            unanswerable_rationale="not in repo",
            expected_abstention="says it is not shown",
        )  # fmt: skip
    ]
    sealed = seal_generation_regions(questions, files)
    (root / "bench/demo.jsonl").write_text(dump_generation_benchmark(sealed))
    bench = load_generation_benchmark(root / "bench/demo.jsonl")
    from copilot.evaluation.generation.protocol import RepositorySpec

    spec = RepositorySpec(
        repository_id="demo", kind="synthetic-authored", repository_name="demo",
        source_root="repo", benchmark_path="bench/demo.jsonl",
        benchmark_digest=benchmark_digest(bench), index_dir="idx", index_id="0123456789abcdef",
    )  # fmt: skip
    protocol = default_protocol((spec,))
    proto_path = root / PROTOCOL_PATH
    proto_path.parent.mkdir(parents=True)
    from copilot.evaluation.generation.protocol import dump_protocol

    proto_path.write_text(dump_protocol(protocol))

    retriever = FakeRetriever([make_result(CODE, rank=1, path="src/a.py", start=1, end=2)])
    monkeypatch.setattr(cli, "open_retriever", lambda spec, root, settings: retriever)

    def gen_reply(request):
        if "BEGIN SOURCE 1" in request.user_prompt:
            return "It returns the sum of the items [Source 1]."
        return "It probably adds numbers."

    monkeypatch.setattr(cli, "create_client", lambda s: FakeLLMClient(gen_reply, model=s.llm_model))
    monkeypatch.setenv("COPILOT_LLM_MODEL", "fake-model")
    return root


def _args(project: Path, *more: str) -> list[str]:
    return [*more, "--root", str(project), "--output-dir", "out"]


def test_plan_is_offline_constructs_no_provider_and_writes_nothing(project, monkeypatch, capsys):
    def boom(settings):
        raise AssertionError("a provider client was constructed during a dry run")

    monkeypatch.setattr(cli, "create_client", boom)
    assert cli.main(_args(project, "plan")) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "NOT CONFIRMED" in out
    assert "demo: 4 questions (3 answerable, 1 unanswerable)" in out
    assert "generation calls <= 8" in out and "judge calls <= 8" in out
    assert "out/generation-demo.json" in out and "out/judgements-demo.json" in out
    assert "index demo: idx (MISSING)" in out
    assert not (project / "out").exists()


def test_run_refuses_an_index_that_is_not_the_one_frozen_in_the_protocol(
    project, monkeypatch, capsys
):
    class Index:
        index_id = "ffffffffffffffff"

    wrong = FakeRetriever([make_result(CODE, rank=1, path="src/a.py", start=1, end=2)])
    wrong.index = Index()
    monkeypatch.setattr(cli, "open_retriever", lambda spec, root, settings: wrong)
    argv = _args(project, "run", "--confirm-model", "fake/fake-model")
    assert cli.main(argv) == 2
    assert "re-frozen" in capsys.readouterr().err


def test_run_refuses_without_a_confirmed_model_and_with_a_mismatched_one(project, capsys):
    assert cli.main(_args(project, "run")) == 2
    assert "--confirm-model" in capsys.readouterr().err
    assert cli.main(_args(project, "run", "--confirm-model", "fake/another-model")) == 2
    assert "not the confirmed model" in capsys.readouterr().err
    assert not (project / "out").exists()


def test_the_full_offline_pipeline_run_judge_report_audit(project, monkeypatch, capsys):
    # --- generate
    assert cli.main(_args(project, "run", "--confirm-model", "fake/fake-model",
                          "--retry-initial-delay", "0")) == 0  # fmt: skip
    run_file = project / "out/generation-demo.json"
    run = json.loads(run_file.read_text())
    assert run["test_artifact"] is True and len(run["entries"]) == 4

    # --- judge (a different "model" id than the generator)
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
        ok = "sum of the items" in text.split("BEGIN ANSWER")[1]
        return json.dumps({"fact_verdicts": {"f1": "supported" if ok else "missing"},
                           "groundedness": 2 if ok else 1, "abstained": False,
                           "response_type": "answered"})  # fmt: skip

    monkeypatch.setattr(cli, "create_client",
                        lambda s: FakeLLMClient(judge_reply, model=s.llm_model))  # fmt: skip
    judge_argv = _args(
        project, "judge", "--confirm-model", "fake/fake-model", "--judge-model", "judge-m",
        "--retry-initial-delay", "0",
    )  # fmt: skip
    assert cli.main(judge_argv) == 0
    judgements = json.loads((project / "out/judgements-demo.json").read_text())
    assert judgements["judge_model"] == "judge-m" and judgements["same_model_as_generator"] is False
    # --- report
    assert cli.main(_args(project, "report")) == 0
    report_json = project / "out/report-TEST-ARTIFACT.json"
    report = json.loads(report_json.read_text())
    primary = report["analysis"]["combined"]["primary"]
    assert primary["n"] == 3 and primary["mean_difference"] == pytest.approx(2.0)
    markdown = (project / "out/report-TEST-ARTIFACT.md").read_text()
    assert "TEST ARTIFACT" in markdown
    assert not (project / "out/report.json").exists()  # never written under a real-looking name
    # --- audit: sample, fill the worksheet like a human, compare with the judge
    assert cli.main(_args(project, "audit-worksheet")) == 0
    sheet_path = project / "out/audit-worksheet-TEST-ARTIFACT.csv"
    rows = list(csv.DictReader(io.StringIO(sheet_path.read_text())))
    key = {
        e["item_id"]: e
        for e in json.loads((project / "out/audit-key-TEST-ARTIFACT.json").read_text())
    }
    assert rows and all(r["human_groundedness"] == "" for r in rows)  # nothing pre-filled
    assert not any(k in rows[0] for k in ("arm", "question_id"))
    for row in rows:  # a perfect rater: supported iff the answer says so
        answerable = key[row["item_id"]]["question_id"] != "u1"
        ok = "sum of the items" in row["answer"]
        row["human_fact_verdicts"] = ("f1=supported" if ok else "f1=missing") if answerable else ""
        row["human_groundedness"] = "2" if ok else "1"
        row["human_abstained"] = "no"
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    filled = project / "out/filled.csv"
    filled.write_text(out.getvalue())
    capsys.readouterr()
    code = cli.main(_args(project, "audit-agree", "--worksheet", str(filled), "--key",
                          str(project / "out/audit-key-TEST-ARTIFACT.json")))  # fmt: skip
    assert code == 0
    result = json.loads((project / "out/audit-agreement.json").read_text())
    assert result["rated_answers"] == len(rows)
    assert result["correctness_score_0_2"]["raw_agreement"] == 1.0


def test_run_is_resumable_through_the_cli(project, capsys):
    argv = _args(project, "run", "--confirm-model", "fake/fake-model", "--retry-initial-delay", "0")
    assert cli.main(argv) == 0
    before = (project / "out/generation-demo.json").read_text()
    assert cli.main(argv) == 0
    assert (project / "out/generation-demo.json").read_text() == before  # nothing re-asked


def test_report_refuses_runs_from_another_protocol(project, capsys):
    argv = _args(project, "run", "--confirm-model", "fake/fake-model", "--retry-initial-delay", "0")
    assert cli.main(argv) == 0
    proto = json.loads((project / PROTOCOL_PATH).read_text())
    proto["top_k"] = 3
    (project / PROTOCOL_PATH).write_text(json.dumps(proto))
    assert cli.main(_args(project, "report")) == 2
    assert "another protocol" in capsys.readouterr().err


def test_benchmark_drift_is_refused(project, capsys):
    path = project / "bench/demo.jsonl"
    path.write_text(path.read_text().replace("function number 1", "function number one"))
    assert cli.main(_args(project, "plan")) == 2
    assert "differs from the one frozen" in capsys.readouterr().err


# --------------------------------------------------------------------------- the committed protocol


def test_protocol_init_does_not_overwrite_a_frozen_protocol(tmp_path, capsys):
    target = tmp_path / "p.json"
    target.write_text("{}")
    assert cli.main(["protocol-init", "--root", str(ROOT), "--protocol", str(target)]) == 2
    assert target.read_text() == "{}"


@pytest.mark.skipif(
    not (ROOT / "generation-eval/benchmarks/copilot_generation_055a8d5.jsonl").is_file(),
    reason="committed benchmarks not present",
)
def test_the_committed_protocol_matches_the_committed_benchmarks():
    committed = load_protocol(ROOT / PROTOCOL_PATH)
    assert not committed.model_confirmed  # the model is confirmed only at run time
    regenerated = default_protocol(cli._specs(ROOT))
    assert committed == regenerated, "benchmark or protocol drifted after the protocol was frozen"
    assert committed.plain_prompt_version == "plain-v2" and committed.max_output_tokens == 8192
    assert [r.repository_id for r in committed.repositories] == ["copilot", "inventory-orders"]
    assert committed.primary_endpoint.name.startswith("paired difference")
