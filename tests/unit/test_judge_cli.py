# ruff: noqa: F811  (imported pytest fixtures are re-declared as test arguments)
"""Milestone 11B: choosing an independent (local) judge on the command line.

The generation model is a fake; the local judge is the *real* factory + ``OllamaLLMClient`` talking
to an in-process ``httpx.MockTransport``, so the whole call chain (settings -> factory -> guard ->
client -> judge -> artifacts) is exercised without a socket, a server or a model.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import httpx
import pytest

from copilot.evaluation.generation import __main__ as cli
from copilot.llm import FakeLLMClient, create_llm_client
from copilot.llm import factory as llm_factory
from tests.security import secret_helpers as h
from tests.unit.test_generation_cli import _args, no_network, project  # noqa: F401  (fixtures)
from tests.unit.test_generation_pilot import _judge_reply, _pargs
from tests.unit.test_ollama_client import chat_ok, installed

DIGEST = "a" * 64
LOCAL = "pilot-judge-ollama"


def _verdict(user_prompt: str, *, strict: bool = False) -> str:
    """Judge A (strict=False): supports the fact iff the answer says so. Judge B: never does and
    abstains on unanswerable questions - so the two disagree on every pilot answer."""
    if "Key fact" in user_prompt:
        ok = not strict and "sum of the items" in user_prompt.split("BEGIN ANSWER")[1]
        body = {
            "fact_verdicts": {"f1": "supported" if ok else "missing"},
            "groundedness": 2,
            "abstained": False,
            "response_type": "answered",
        }
    elif strict:
        body = {
            "fact_verdicts": {},
            "groundedness": 2,
            "abstained": True,
            "response_type": "insufficient_evidence",
        }
    else:
        body = {
            "fact_verdicts": {},
            "groundedness": 2,
            "abstained": False,
            "response_type": "answered",
        }
    return json.dumps(body)


class OllamaServer:
    def __init__(self, *, strict=False, digest=DIGEST, ps=None, chat_extra=None, down=False):
        self.calls: list[httpx.Request] = []
        self.strict, self.digest, self.ps, self.down = strict, digest, ps, down
        self.chat_extra = chat_extra or {}

    @property
    def paths(self) -> list[str]:
        return [c.url.path for c in self.calls]

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.calls.append(req)
        if self.down:
            raise httpx.ConnectError("refused", request=req)
        path = req.url.path
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.12.3"})
        if path == "/api/tags":
            return httpx.Response(200, json=installed(digest=self.digest))
        if path == "/api/ps":
            if self.ps is None:
                return httpx.Response(200, json={"models": []})
            return httpx.Response(200, json={"models": [self.ps]})
        if path == "/api/chat":
            body = json.loads(req.content)
            text = _verdict(body["messages"][1]["content"], strict=self.strict)
            return httpx.Response(200, json=chat_ok(text, **self.chat_extra))
        return httpx.Response(404, json={})


@pytest.fixture
def ollama(monkeypatch):
    """Route the real factory's Ollama client to a scripted in-process server."""
    holder: dict[str, OllamaServer] = {"server": OllamaServer()}
    real = llm_factory.OllamaLLMClient

    def build(**kw):
        return real(transport=httpx.MockTransport(holder["server"]), **kw)

    monkeypatch.setattr(llm_factory, "OllamaLLMClient", build)

    def use(server: OllamaServer) -> OllamaServer:
        holder["server"] = server
        monkeypatch.setattr(cli, "create_client", create_llm_client)
        return server

    return use


def _generate_pilot(project, monkeypatch):
    assert cli.main(["pilot-init", "--root", str(project)]) == 0
    gen = FakeLLMClient("It returns the sum of the items [Source 1].", model="fake-model")
    monkeypatch.setattr(cli, "create_client", lambda s: gen)
    argv = _pargs(
        project, "run", "--confirm-model", "fake/fake-model", "--retry-initial-delay", "0"
    )
    assert cli.main(argv) == 0


def _local_judge(project, *more, directory=LOCAL):
    return _pargs(
        project, "judge", "--confirm-model", "fake/fake-model", "--judge-provider", "ollama",
        "--judge-model", "llama3.1:8b", "--judgements-dir", directory,
        "--retry-initial-delay", "0", *more,
    )  # fmt: skip


def _gemini_style_judge(project, monkeypatch):
    judge = FakeLLMClient(_judge_reply, model="hosted-judge")
    monkeypatch.setattr(cli, "create_client", lambda s: judge)
    argv = _pargs(
        project, "judge", "--confirm-model", "fake/fake-model", "--judge-model", "hosted-judge",
        "--retry-initial-delay", "0",
    )  # fmt: skip
    assert cli.main(argv) == 0


# ---------------------------------------------------------------- selection and validation


def test_a_local_judge_needs_its_own_judgements_dir(project, monkeypatch, capsys, ollama):
    _generate_pilot(project, monkeypatch)
    server = ollama(OllamaServer())
    argv = _pargs(
        project, "judge", "--confirm-model", "fake/fake-model", "--judge-provider", "ollama",
        "--judge-model", "llama3.1:8b",
    )  # fmt: skip
    assert cli.main(argv) == 2
    assert "--judgements-dir" in capsys.readouterr().err
    assert server.calls == []
    same = _local_judge(project, directory="pilot-out")  # the generation output directory
    assert cli.main(same) == 2
    assert "must differ" in capsys.readouterr().err and server.calls == []


def test_a_judge_model_is_required_and_a_pilot_cannot_judge_into_the_primary_directory(
    project, monkeypatch, capsys, ollama
):
    _generate_pilot(project, monkeypatch)
    ollama(OllamaServer())
    no_model = _pargs(project, "judge", "--confirm-model", "fake/fake-model")
    assert cli.main(no_model) == 2 and "--judge-model" in capsys.readouterr().err
    primary = _local_judge(project, directory="data/generation-eval/primary/x")
    assert cli.main(primary) == 2
    assert "primary directory" in capsys.readouterr().err


@pytest.mark.parametrize(
    "flags",
    [["--judge-max-output-tokens", "99999"], ["--judge-timeout", "0"], ["--judge-num-ctx", "8"]],
)
def test_out_of_range_local_settings_are_refused(project, monkeypatch, capsys, ollama, flags):
    _generate_pilot(project, monkeypatch)
    server = ollama(OllamaServer())
    assert cli.main(_local_judge(project, *flags)) == 2
    assert server.calls == [] and not (project / LOCAL).exists()


def test_the_local_judge_runs_through_the_real_factory_and_records_its_identity(
    project, monkeypatch, capsys, ollama
):
    _generate_pilot(project, monkeypatch)
    server = ollama(OllamaServer())
    assert cli.main(_local_judge(project)) == 0
    out = capsys.readouterr().out
    assert "{'judged': 4" in out

    chats = [c for c in server.calls if c.url.path == "/api/chat"]
    assert 1 <= len(chats) <= 4
    body = json.loads(chats[0].content)
    assert body["format"] == "json" and body["stream"] is False
    assert body["options"] == {
        "temperature": 0.0,
        "num_predict": 1024,
        "num_ctx": 8192,
        "seed": 20260,
    }
    assert body["model"] == "llama3.1:8b"
    assert all(p in {"/api/chat", "/api/tags", "/api/version"} for p in server.paths)

    file = json.loads((project / LOCAL / "judgements-demo.json").read_text())
    assert (file["judge_provider"], file["judge_model"]) == ("ollama", "llama3.1:8b")
    assert file["judge_runtime"]["model_digest"] == DIGEST
    assert file["judge_runtime"]["max_output_tokens"] == 1024
    assert len(file["judge_runtime_fingerprint"]) == 64
    assert (project / LOCAL / "cache-judge").is_dir()
    # the generation namespace received nothing new
    assert not (project / "pilot-out/judgements-demo.json").exists()
    assert not (project / "pilot-out/cache-judge").exists()


def test_a_hosted_judgement_file_is_untouched_and_cannot_be_resumed_by_a_local_judge(
    project, monkeypatch, capsys, ollama
):
    _generate_pilot(project, monkeypatch)
    _gemini_style_judge(project, monkeypatch)
    hosted = project / "pilot-out/judgements-demo.json"
    before = hosted.read_bytes()
    server = ollama(OllamaServer())
    assert cli.main(_local_judge(project)) == 0
    assert hosted.read_bytes() == before

    # a local judge pointed at a directory that holds the hosted file must refuse, not overwrite
    other = project / "mixed"
    other.mkdir()
    (other / "judgements-demo.json").write_bytes(before)
    calls_before = len(server.calls)
    capsys.readouterr()
    assert cli.main(_local_judge(project, directory="mixed")) == 2
    assert "cannot resume judging" in capsys.readouterr().err
    assert (other / "judgements-demo.json").read_bytes() == before
    assert [c for c in server.calls[calls_before:] if c.url.path == "/api/chat"] == []


def test_resume_with_a_changed_model_file_is_refused_and_nothing_is_overwritten(
    project, monkeypatch, capsys, ollama
):
    _generate_pilot(project, monkeypatch)
    ollama(OllamaServer())
    assert cli.main(_local_judge(project)) == 0
    path = project / LOCAL / "judgements-demo.json"
    before = path.read_bytes()
    server = ollama(OllamaServer(digest="b" * 64))  # the model was re-pulled: different weights
    capsys.readouterr()
    assert cli.main(_local_judge(project)) == 2
    assert "judge_runtime_fingerprint" in capsys.readouterr().err
    assert path.read_bytes() == before
    assert "/api/chat" not in server.paths


def test_an_identical_rerun_resumes_without_new_requests_and_no_cache_bypasses_the_cache(
    project, monkeypatch, capsys, ollama
):
    _generate_pilot(project, monkeypatch)
    server = ollama(OllamaServer())
    assert cli.main(_local_judge(project)) == 0
    first = server.paths.count("/api/chat")
    assert cli.main(_local_judge(project)) == 0
    assert server.paths.count("/api/chat") == first  # everything reused

    assert cli.main(_local_judge(project, "--no-cache", directory="repeat")) == 0
    assert server.paths.count("/api/chat") > first
    assert not (project / "repeat/cache-judge").exists()


def test_a_context_limit_answer_is_recorded_and_never_counted_as_a_judgement(
    project, monkeypatch, capsys, ollama
):
    _generate_pilot(project, monkeypatch)
    ollama(OllamaServer(chat_extra={"done_reason": "length"}))
    assert cli.main(_local_judge(project)) == 0
    out = capsys.readouterr().out
    assert "'judged': 0" in out and "context-limit failures: 4" in out
    file = json.loads((project / LOCAL / "judgements-demo.json").read_text())
    failed = [i for i in file["items"] if i["state"] == "failed"]
    assert len(failed) == 4 and all(i["assessment"] is None for i in failed)
    assert all(i["record"]["error"].startswith("LLMContextLimitError") for i in failed)


def test_secret_bearing_input_never_reaches_the_local_server_or_the_cache(
    project, monkeypatch, capsys, ollama
):
    _generate_pilot(project, monkeypatch)
    run_path = project / "pilot-out/generation-demo.json"
    run = json.loads(run_path.read_text())
    token = h.github_token()
    for entry in run["entries"]:
        for arm in ("plain", "rag"):
            entry[arm]["answer"] = f"use {token} to call it"
    run_path.write_text(json.dumps(run))
    server = ollama(OllamaServer())
    assert cli.main(_local_judge(project)) == 0
    assert "/api/chat" not in server.paths  # the gate fired before any request
    assert all(token not in c.content.decode() for c in server.calls if c.content)
    written = "".join(p.read_text() for p in (project / LOCAL).rglob("*.json"))
    assert token not in written


# ---------------------------------------------------------------- report, compare, audit


def test_report_with_a_judgements_dir_writes_beside_those_judgements(
    project, monkeypatch, capsys, ollama
):
    _generate_pilot(project, monkeypatch)
    _gemini_style_judge(project, monkeypatch)
    assert cli.main(_pargs(project, "report")) == 0
    hosted_report = project / "pilot-out/report-PILOT-TEST-ARTIFACT.json"
    before = hosted_report.read_bytes()
    ollama(OllamaServer())
    assert cli.main(_local_judge(project)) == 0
    assert cli.main(_pargs(project, "report", "--judgements-dir", LOCAL)) == 0
    local_report = project / LOCAL / "report-PILOT-TEST-ARTIFACT.json"
    assert local_report.is_file() and hosted_report.read_bytes() == before
    judge = json.loads(local_report.read_text())["judge"]["demo"]
    assert judge["provider"] == "ollama" and judge["runtime"]["model_digest"] == DIGEST
    assert "runtime fingerprint" in (project / LOCAL / "report-PILOT-TEST-ARTIFACT.md").read_text()


def test_judge_compare_hand_computed_per_item_differences(project, monkeypatch, capsys, ollama):
    _generate_pilot(project, monkeypatch)
    _gemini_style_judge(project, monkeypatch)  # judge A: the answerable pair scores 2, u1 scores 0
    ollama(OllamaServer(strict=True))  # judge B: facts missing -> 0; abstains on u1 -> 2
    assert cli.main(_local_judge(project)) == 0
    capsys.readouterr()
    assert cli.main(_pargs(project, "judge-compare", "--dir-b", LOCAL)) == 0
    result = json.loads((project / LOCAL / "judge-compare-PILOT.json").read_text())["demo"]
    assert result["identity_verified"] is True and result["identity_problems"] == []
    assert result["answers_in_both"] == 4 and result["scored_by_both"] == 4
    rows = {(r["question_id"], r["arm"]): (r["score_a"], r["score_b"]) for r in result["items"]}
    assert sorted(rows.values()) == [(0, 2), (0, 2), (2, 0), (2, 0)]
    assert result["exact_agreement"] == {"count": 0, "n": 4, "fraction": 0.0}
    assert result["two_point_disagreements"] == {"count": 4, "n": 4}
    assert result["confusion_a_rows_b_columns_scores_0_1_2"] == [[0, 0, 2], [0, 0, 0], [2, 0, 0]]
    assert result["chance_corrected"]["kappa"] is None
    assert "noise" in result["chance_corrected"]["kappa_undefined_reason"]
    assert result["missing_in_a"] == [] and result["missing_in_b"] == []
    assert result["judge_a"] == "fake/hosted-judge" and result["judge_b"].startswith("ollama/")


def test_judge_compare_reports_missing_judgements_and_refuses_different_answers(
    project, monkeypatch, capsys, ollama
):
    _generate_pilot(project, monkeypatch)
    _gemini_style_judge(project, monkeypatch)
    ollama(OllamaServer(chat_extra={"done_reason": "length"}))  # B has no valid judgement at all
    assert cli.main(_local_judge(project)) == 0
    assert cli.main(_pargs(project, "judge-compare", "--dir-b", LOCAL)) == 0
    result = json.loads((project / LOCAL / "judge-compare-PILOT.json").read_text())["demo"]
    assert result["scored_by_both"] == 0 and len(result["missing_in_b"]) == 4

    ollama(OllamaServer())
    path = project / LOCAL / "judgements-demo.json"
    path.unlink()
    assert cli.main(_local_judge(project)) == 0
    tampered = json.loads(path.read_text())
    tampered["items"][0]["record"]["input_sha256"] = "0" * 64  # "another answer"
    path.write_text(json.dumps(tampered))
    capsys.readouterr()
    assert cli.main(_pargs(project, "judge-compare", "--dir-b", LOCAL)) == 3
    assert "judge inputs differ" in capsys.readouterr().out


def test_judge_compare_refuses_the_same_directory(project, monkeypatch, capsys):
    _generate_pilot(project, monkeypatch)
    assert cli.main(_pargs(project, "judge-compare", "--dir-b", "pilot-out")) == 2


def _fill_like_a_human(worksheet: Path) -> Path:
    rows = list(csv.DictReader(io.StringIO(worksheet.read_text())))
    for row in rows:
        answerable = row["human_fact_verdicts"] != "n/a" and "Who deployed" not in row["question"]
        row["human_fact_verdicts"] = "f1=supported" if answerable else ""
        row["human_groundedness"] = "2"
        row["human_abstained"] = "no"
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    filled = worksheet.with_name("filled.csv")
    filled.write_text(out.getvalue())
    return filled


def test_the_pilot_worksheet_is_drawn_before_judging_and_agreement_has_no_kappa(
    project, monkeypatch, capsys, ollama
):
    _generate_pilot(project, monkeypatch)
    # No judgement file exists yet, yet every answer with text is on the blinded worksheet.
    assert cli.main(_pargs(project, "audit-worksheet")) == 0
    sheet = project / "pilot-out/audit-worksheet-PILOT-TEST-ARTIFACT.csv"
    rows = list(csv.DictReader(io.StringIO(sheet.read_text())))
    assert len(rows) == 4 and all("PILOT" in r["audit_status"] for r in rows)
    assert not any(k in rows[0] for k in ("arm", "question_id", "judge"))
    key = project / "pilot-out/audit-key-PILOT-TEST-ARTIFACT.json"
    population = json.loads(
        (project / "pilot-out/audit-population-PILOT-TEST-ARTIFACT.json").read_text()
    )
    assert population["eligible_pairs"] == 2 and population["selected_sample_size"] == 2
    assert not (project / "pilot-out/judgements-demo.json").exists()

    ollama(OllamaServer())
    assert cli.main(_local_judge(project)) == 0
    filled = _fill_like_a_human(sheet)
    capsys.readouterr()
    code = cli.main(
        _pargs(project, "audit-agree", "--worksheet", str(filled), "--key", str(key),
               "--judgements-dir", LOCAL)
    )  # fmt: skip
    assert code == 0
    result = json.loads((project / LOCAL / "audit-agreement-PILOT.json").read_text())
    assert result["pilot"] is True and result["rated_answers"] == 4
    for name in ("correctness_score_0_2", "abstained", "key_fact_supported"):
        assert result[name]["kappa"] is None
        assert "too small" in result[name]["kappa_undefined_reason"]
    # human: answerable pair 2/2, unanswerable pair 2 each (rated "answered" => not what the rubric
    # scores, so only the shape of the counts is asserted, not a particular agreement value)
    assert result["correctness_exact_agreement"]["n"] == 4
    assert result["valid_judgement_coverage"] == {
        "answers_requiring_a_judgement": 4, "with_a_valid_judgement": 4, "fraction": 1.0,
    }  # fmt: skip
    assert len(result["per_item"]) == 4
    assert not (project / "pilot-out/audit-agreement-PILOT.json").exists()


# ---------------------------------------------------------------- preflight


def _preflight(project, *more):
    return [
        "judge-preflight", "--root", str(project), "--judge-provider", "ollama",
        "--judge-model", "llama3.1:8b", *more,
    ]  # fmt: skip


def test_preflight_reports_the_runtime_and_never_pulls(project, capsys, ollama, monkeypatch):
    server = ollama(OllamaServer())
    assert cli.main(_preflight(project, "--expect-quantization", "Q4_K_M")) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True and result["ollama_version"] == "0.12.3"
    assert result["model"]["digest"] == DIGEST and result["model"]["quantization"] == "Q4_K_M"
    assert result["runtime_identity"]["context_size"] == 8192
    assert len(result["runtime_fingerprint"]) == 64 and result["loopback"] is True
    assert result["loaded"] is None and any("not loaded" in w for w in result["warnings"])
    assert set(server.paths) <= {"/api/version", "/api/tags", "/api/ps"}


def test_preflight_fails_on_an_unexpected_quantization(project, capsys, ollama):
    ollama(OllamaServer())
    assert cli.main(_preflight(project, "--expect-quantization", "Q8_0")) == 3
    assert json.loads(capsys.readouterr().out)["checks"][0]["ok"] is False


def test_preflight_gpu_residency(project, capsys, ollama):
    full = {"name": "llama3.1:8b", "size": 6_000_000_000, "size_vram": 6_000_000_000}
    ollama(OllamaServer(ps=full))
    assert cli.main(_preflight(project, "--require-gpu")) == 0
    assert json.loads(capsys.readouterr().out)["loaded"]["gpu_fraction"] == 1.0
    partial = {"name": "llama3.1:8b", "size": 8_000_000_000, "size_vram": 6_000_000_000}
    ollama(OllamaServer(ps=partial))
    assert cli.main(_preflight(project, "--require-gpu")) == 3
    result = json.loads(capsys.readouterr().out)
    assert result["loaded"]["gpu_fraction"] == 0.75 and any("CPU" in w for w in result["warnings"])


def test_preflight_warm_up_sends_one_constant_guarded_prompt(project, capsys, ollama):
    server = ollama(OllamaServer())
    assert cli.main(_preflight(project, "--warm-up")) == 0
    chats = [c for c in server.calls if c.url.path == "/api/chat"]
    assert len(chats) == 1
    body = json.loads(chats[0].content)
    assert "ready" in body["messages"][1]["content"] and body["options"]["num_predict"] <= 64
    assert "warm_up" in json.loads(capsys.readouterr().out)


def test_preflight_errors_are_clear_and_secret_free(project, capsys, ollama):
    ollama(OllamaServer(down=True))
    assert cli.main(_preflight(project)) == 2
    assert "could not reach Ollama" in capsys.readouterr().err
    bad = OllamaServer()
    bad.tags = None
    server = ollama(OllamaServer())
    server.digest = "b" * 64
    missing = _preflight(project)
    missing[missing.index("llama3.1:8b")] = "not-installed:1b"
    assert cli.main(missing) == 2
    assert "ollama pull not-installed:1b" in capsys.readouterr().err
    assert all("pull" not in p for p in server.paths)


def test_a_hosted_preflight_never_builds_a_client_and_needs_the_hosted_profile(
    project, capsys, monkeypatch
):
    """Since the closeout the hosted preflight exists (offline); it is exercised in full in
    ``test_hosted_judge_validation.py``. Here: it builds no client, and a project without the
    hosted profile is refused rather than passed."""

    def boom(*args, **kwargs):  # pragma: no cover - only reached on a bug
        raise AssertionError("a provider client was constructed")

    monkeypatch.setattr(cli, "create_client", boom)
    argv = ["judge-preflight", "--root", str(project), "--judge-provider", "gemini",
            "--judge-model", "x", "--judgements-dir", str(project / "v")]  # fmt: skip
    assert cli.main(argv) == 2
    assert "hosted-judge profile" in capsys.readouterr().err
