# ruff: noqa: F811  (imported pytest fixtures are re-declared as test arguments)
"""M11A corrections: protocol v1.1 amendment and the isolated, bounded pilot (FakeLLM only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from copilot.evaluation.benchmark import Region
from copilot.evaluation.generation import __main__ as cli
from copilot.evaluation.generation.judge import JUDGE_PROMPT_VERSION
from copilot.evaluation.generation.pilot import (
    PILOT_SPEC_PATH,
    PilotMismatchError,
    build_pilot_spec,
    check_namespaces,
    load_pilot_spec,
    select_pilot_questions,
)
from copilot.evaluation.generation.protocol import (
    PROTOCOL_PATH,
    PROTOCOL_V1_FILE_SHA256,
    PROTOCOL_V1_PATH,
    default_protocol,
    dump_protocol,
    load_protocol,
    protocol_hash,
)
from copilot.evaluation.generation.schema import (
    Answerability,
    GenerationQuestion,
    KeyFact,
    benchmark_digest,
    load_generation_benchmark,
)
from copilot.llm import FakeLLMClient
from tests.unit.test_generation_cli import _args, no_network, project  # noqa: F401  (fixtures)

REGION = Region(file_path="a.py", start_line=1, end_line=2)
ROOT = Path(__file__).resolve().parents[2]
COMMITTED = (ROOT / PROTOCOL_V1_PATH).is_file() and (ROOT / PROTOCOL_PATH).is_file()
needs_committed = pytest.mark.skipif(not COMMITTED, reason="committed protocol files not present")

# ---------------------------------------------------------------- the amendment


@needs_committed
def test_the_original_v1_protocol_file_is_preserved_byte_for_byte():
    raw = (ROOT / PROTOCOL_V1_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PROTOCOL_V1_FILE_SHA256
    v1 = load_protocol(ROOT / PROTOCOL_V1_PATH)
    assert v1.amendment is None and v1.judge.prompt_version == "gen-judge-1"
    assert dump_protocol(v1).encode() == raw  # still serialises to exactly the frozen bytes
    assert protocol_hash(v1).startswith("0ae81364")


@needs_committed
def test_v1_1_changes_only_what_the_amendment_lists():
    v1 = load_protocol(ROOT / PROTOCOL_V1_PATH)
    v11 = load_protocol(ROOT / PROTOCOL_PATH)
    assert v11.amendment is not None and v11.amendment.version == "1.1"
    assert v11.amendment.amends_file_sha256 == PROTOCOL_V1_FILE_SHA256
    assert v11.amendment.made_before_any_real_generation is True
    assert protocol_hash(v11) != protocol_hash(v1) and not v11.model_confirmed
    # what must not change
    for field in (
        "arm_a", "arm_b", "plain_prompt_version", "rag_prompt_version", "prompt_digests",
        "temperature", "max_output_tokens", "top_k", "retriever", "repositories",
        "primary_endpoint", "audit_fraction", "audit_seed", "bootstrap_resamples",
        "bootstrap_seed", "confidence",
    ):  # fmt: skip
        assert getattr(v11, field) == getattr(v1, field), field
    assert v11.judge.rubric_version == v1.judge.rubric_version == "gen-rubric-1"
    # what changes
    assert v11.judge.prompt_version == "gen-judge-2" == JUDGE_PROMPT_VERSION
    assert v11.secondary_endpoints != v1.secondary_endpoints
    assert any("historical" in e.name for e in v11.secondary_endpoints)


@needs_committed
def test_the_committed_v1_1_protocol_matches_the_committed_benchmarks():
    assert load_protocol(ROOT / PROTOCOL_PATH) == default_protocol(cli._specs(ROOT))


# ---------------------------------------------------------------- pilot selection


def _q(qid, answerable, text="t"):
    if answerable:
        return GenerationQuestion(
            id=qid, repository_id="r", question=text, category="c",
            answerability=Answerability.ANSWERABLE, acceptable_answer="a",
            key_facts=(KeyFact(id="f1", statement="returns the total", regions=(REGION,)),),
        )  # fmt: skip
    return GenerationQuestion(
        id=qid, repository_id="r", question=text, category="c",
        answerability=Answerability.UNANSWERABLE, unanswerable_rationale="x",
        expected_abstention="y",
    )  # fmt: skip


def test_selection_is_deterministic_and_reads_only_ids_and_answerability():
    pool = [_q(f"a{i}", True) for i in range(8)] + [_q(f"u{i}", False) for i in range(4)]
    first = select_pilot_questions("r", pool)
    assert first == select_pilot_questions("r", list(reversed(pool)))  # order-independent
    altered = [_q(q.id, q.is_answerable, text="completely different wording") for q in pool]
    assert select_pilot_questions("r", altered) == first  # question text is never read
    assert first[0].startswith("a") and first[1].startswith("u")


def test_selection_needs_both_kinds_of_question():
    with pytest.raises(ValueError, match="unanswerable"):
        select_pilot_questions("r", [_q("a1", True)])


@needs_committed
def test_the_committed_pilot_specification_is_the_recorded_deterministic_selection():
    spec = load_pilot_spec(ROOT / PILOT_SPEC_PATH)
    stored = load_protocol(ROOT / PROTOCOL_PATH)
    assert spec.protocol_sha256 == protocol_hash(stored) and spec.recorded_before_any_live_execution
    assert sorted(spec.questions) == ["copilot", "inventory-orders"]
    benchmarks, digests = {}, {}
    for repo in stored.repositories:
        bench = load_generation_benchmark(ROOT / repo.benchmark_path)
        benchmarks[repo.repository_id] = bench.questions
        digests[repo.repository_id] = benchmark_digest(bench)
        ids = spec.questions[repo.repository_id]
        by_id = {q.id: q for q in bench.questions}
        assert len(ids) == 2 and by_id[ids[0]].is_answerable and not by_id[ids[1]].is_answerable
    assert build_pilot_spec(stored.name, protocol_hash(stored), benchmarks, digests) == spec
    assert spec.bounds == {
        "questions": 4, "generation_calls_max": 8, "judge_calls_max": 8,
        "provider_calls_max_before_retries": 16,
    }  # fmt: skip


def test_the_pilot_directory_may_not_overlap_the_primary_one(tmp_path):
    check_namespaces(tmp_path / "pilot", tmp_path / "primary")
    for bad in (tmp_path / "primary", tmp_path / "primary" / "pilot", tmp_path):
        with pytest.raises(PilotMismatchError):
            check_namespaces(bad, tmp_path / "primary")


# ---------------------------------------------------------------- the pilot through the CLI


def _pargs(project: Path, command: str, *more: str) -> list[str]:
    return [command, "--root", str(project), "--output-dir", "pilot-out", "--pilot", *more]


def _judge_reply(request):
    if "Key fact" in request.user_prompt:
        ok = "sum of the items" in request.user_prompt.split("BEGIN ANSWER")[1]
        body = {"fact_verdicts": {"f1": "supported" if ok else "missing"},
                "groundedness": 2, "abstained": False, "response_type": "answered"}  # fmt: skip
    else:
        body = {"fact_verdicts": {}, "groundedness": 2, "abstained": False,
                "response_type": "answered"}  # fmt: skip
    return json.dumps(body)


def test_pilot_init_records_one_answerable_and_one_unanswerable_question(project, capsys):
    assert cli.main(["pilot-init", "--root", str(project)]) == 0
    spec = load_pilot_spec(project / PILOT_SPEC_PATH)
    assert list(spec.questions) == ["demo"] and len(spec.questions["demo"]) == 2
    assert spec.questions["demo"][1] == "u1" and spec.bounds["generation_calls_max"] == 4
    assert cli.main(["pilot-init", "--root", str(project)]) == 0  # identical: unchanged
    path = project / PILOT_SPEC_PATH
    path.write_text(path.read_text().replace('"q', '"x'))  # tamper
    capsys.readouterr()
    assert cli.main(["pilot-init", "--root", str(project)]) == 2
    assert "never overwritten" in capsys.readouterr().err


def test_plan_pilot_is_offline_lists_the_questions_and_bounds_and_writes_nothing(
    project, monkeypatch, capsys
):
    assert cli.main(["pilot-init", "--root", str(project)]) == 0

    def boom(settings):
        raise AssertionError("a provider client was constructed during a dry run")

    monkeypatch.setattr(cli, "create_client", boom)
    capsys.readouterr()
    assert cli.main(_pargs(project, "plan", "--confirm-model", "fake/fake-model")) == 0
    out = capsys.readouterr().out
    assert "PILOT plain-vs-rag-pilot-v1" in out and "pilot questions demo:" in out
    assert "generation calls <= 4" in out and "judge calls <= 4" in out
    assert "provider_calls_max_before_retries" in out
    assert not (project / "pilot-out").exists()


def test_a_pilot_defaults_to_its_own_directory_never_the_primary_one(project, capsys):
    assert cli.main(["pilot-init", "--root", str(project)]) == 0
    capsys.readouterr()
    assert cli.main(["plan", "--root", str(project), "--pilot"]) == 0
    assert "data/generation-eval/pilot" in capsys.readouterr().out


def test_pilot_run_judge_report_stay_in_the_pilot_namespace(project, monkeypatch, capsys):
    assert cli.main(["pilot-init", "--root", str(project)]) == 0
    gen = FakeLLMClient("It returns the sum of the items [Source 1].", model="fake-model")
    monkeypatch.setattr(cli, "create_client", lambda s: gen)
    assert cli.main(_pargs(project, "run", "--confirm-model", "fake/fake-model",
                           "--retry-initial-delay", "0")) == 0  # fmt: skip
    run = json.loads((project / "pilot-out/generation-demo.json").read_text())
    assert run["scope"] == "pilot" and len(run["entries"]) == 2
    assert gen.call_count <= 4  # <= 2 questions x 2 arms: the documented bound
    judge = FakeLLMClient(_judge_reply, model="judge-m")
    monkeypatch.setattr(cli, "create_client", lambda s: judge)
    judge_argv = _pargs(
        project, "judge", "--confirm-model", "fake/fake-model", "--judge-model", "judge-m",
        "--retry-initial-delay", "0",
    )  # fmt: skip
    assert cli.main(judge_argv) == 0
    assert judge.call_count <= 4
    assert cli.main(_pargs(project, "report")) == 0
    report = json.loads((project / "pilot-out/report-PILOT-TEST-ARTIFACT.json").read_text())
    assert report["scope"] == "pilot" and any(c.startswith("PILOT:") for c in report["caveats"])
    assert {r["question_id"] for r in report["rows"]} == set(
        load_pilot_spec(project / PILOT_SPEC_PATH).questions["demo"]
    )
    assert (
        "PILOT - not the primary experiment"
        in (project / "pilot-out/report-PILOT-TEST-ARTIFACT.md").read_text()
    )
    assert not (project / "out").exists()  # nothing was written to the primary namespace


def test_pilot_and_primary_artifacts_never_mix(project, monkeypatch, capsys):
    assert cli.main(["pilot-init", "--root", str(project)]) == 0
    base = ["--confirm-model", "fake/fake-model", "--retry-initial-delay", "0"]
    assert cli.main(_pargs(project, "run", *base)) == 0
    # a primary run pointed at the pilot directory must refuse, not resume or overwrite
    before = (project / "pilot-out/generation-demo.json").read_text()
    primary = ["run", "--root", str(project), "--output-dir", "pilot-out", *base]
    capsys.readouterr()
    assert cli.main(primary) == 2
    assert "cannot resume" in capsys.readouterr().err
    assert (project / "pilot-out/generation-demo.json").read_text() == before
    # primary report / judge on a pilot run are refused too
    assert cli.main(["report", "--root", str(project), "--output-dir", "pilot-out"]) == 2
    assert "never mixed" in capsys.readouterr().err
    # and a pilot pointed at an existing primary run is refused
    assert cli.main(["run", "--root", str(project), "--output-dir", "out", *base]) == 0
    capsys.readouterr()
    clash = ["run", "--root", str(project), "--output-dir", "out", "--pilot", *base]
    assert cli.main(clash) == 2
    assert (
        "cannot resume" in capsys.readouterr().err or "must be separate" in capsys.readouterr().err
    )


def test_pilot_refuses_historical_reuse_and_a_stale_specification(project, capsys):
    assert cli.main(["pilot-init", "--root", str(project)]) == 0
    capsys.readouterr()
    argv = _pargs(project, "run", "--confirm-model", "fake/fake-model", "--historical", "h.json")
    assert cli.main(argv) == 2 and "--historical" in capsys.readouterr().err
    proto_path = project / PROTOCOL_PATH
    proto = json.loads(proto_path.read_text())
    proto["top_k"] = 3  # a different protocol than the one the pilot was recorded against
    proto_path.write_text(json.dumps(proto))
    assert cli.main(_pargs(project, "plan")) == 2
    assert "different protocol" in capsys.readouterr().err


def test_pilot_without_a_recorded_specification_is_refused(project, capsys):
    assert cli.main(_pargs(project, "plan")) == 2
    err = capsys.readouterr().err
    assert "pilot-v1.json" in err or "No such file" in err
