"""``python -m copilot.agents`` with a real (hash-embedded) index, a fake model, no network."""

from pathlib import Path

import pytest

from copilot.agents import __main__ as cli
from copilot.config import Settings
from copilot.embeddings import HashEmbedder
from copilot.llm import FakeLLMClient
from copilot.vectorstore import build_repository_index
from tests.agent_helpers import ScriptedRetriever, chunk
from tests.retrieval_helpers import write_repo
from tests.security import secret_helpers as h

QUESTION = "deterministic chunk id sha256 hash payload"


@pytest.fixture
def repo(tmp_path):
    return write_repo(tmp_path / "demo repo")  # a space on purpose


@pytest.fixture
def index_dir(repo, tmp_path):
    return build_repository_index(
        repo, embedder=HashEmbedder(dimension=64), settings=Settings(_env_file=None),
        indexes_dir=tmp_path / "indexes", text_style="raw",
    ).index_path  # fmt: skip


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No test may build a hosted client: the developer's real ``.env`` may hold a key and a model
    id, which ``get_settings`` would load. Tests that need a client install a fake explicitly."""
    from copilot.llm import LLMConfigurationError

    def no_hosted_client(settings):
        raise LLMConfigurationError("COPILOT_LLM_MODEL is not set (test stub)")

    monkeypatch.setattr(cli, "create_llm_client", no_hosted_client)
    monkeypatch.setattr(cli, "create_embedder", lambda settings: HashEmbedder(dimension=64))
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)


def run(capsys, *argv):
    code = cli.main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture
def no_hosted_model(monkeypatch):
    def refuse(settings):
        raise AssertionError("the hosted model must not be created in --fake-llm mode")

    monkeypatch.setattr(cli, "create_llm_client", refuse)


def test_fake_llm_mode_runs_the_whole_workflow_offline(capsys, no_hosted_model, repo, index_dir):
    code, out, err = run(capsys, "ask", index_dir, "--repo", repo, "--fake-llm", QUESTION)
    assert code == 0
    assert out.index("ANSWER") < out.index("SOURCES") < out.index("WORKFLOW")
    assert "offline fake model" in out and "no real model was called" in out
    assert "[Source 1] src/pkg/chunk_ids.py:" in out  # POSIX relative path with a line range
    assert "intent:" in out and "retrieval rounds:" in out and "termination:" in out
    assert "context: " in out and "estimated tokens" in out
    assert "round 1 rank " in out  # provenance: retrieval round and rank of each source
    assert "no model is called" in err
    assert str(repo) not in out + err and "\\" not in out
    assert "def make_chunk_id" not in out  # citations and metadata only, never source text
    assert "TRACE" not in out  # off by default


def test_trace_flag_prints_the_structured_trace(capsys, no_hosted_model, repo, index_dir):
    code, out, _ = run(capsys, "ask", index_dir, "--repo", repo, "--fake-llm", "--trace", QUESTION)
    assert code == 0
    trace = out[out.index("TRACE") :]
    for node in ("classify_intent", "retrieve", "merge_evidence", "assess_evidence", "generate"):
        assert node in trace
    assert QUESTION not in trace and "def make_chunk_id" not in trace  # no query or source text


def test_max_rounds_is_validated_by_the_parser(capsys, repo, index_dir):
    for bad in ("0", "6", "-1", "two"):
        with pytest.raises(SystemExit) as exc:
            cli.main(["ask", str(index_dir), "--repo", str(repo), "--max-rounds", bad, QUESTION])
        assert exc.value.code == 2
    assert "must be between 1 and 5" in capsys.readouterr().err


def test_max_rounds_reaches_the_workflow(capsys, no_hosted_model, repo, index_dir):
    code, out, _ = run(
        capsys, "ask", index_dir, "--repo", repo, "--fake-llm", "--max-rounds", 1,
        "Explain how zebra quantum telemetry works",  # nothing like it in the repository
    )  # fmt: skip
    assert code == 0 and "retrieval rounds: 1 of at most 1" in out


def test_a_blank_question_exits_2_before_the_embedder_is_created(
    capsys, monkeypatch, repo, index_dir
):
    def boom(settings):
        raise AssertionError("embedder must not be created")

    monkeypatch.setattr(cli, "create_embedder", boom)
    code, out, err = run(capsys, "ask", index_dir, "--repo", repo, "--fake-llm", "   ")
    assert code == 2 and err.startswith("error:") and out == ""


def test_a_missing_model_configuration_exits_2_without_a_traceback(capsys, repo, index_dir):
    code, _, err = run(capsys, "ask", index_dir, "--repo", repo, QUESTION)
    assert code == 2 and "COPILOT_LLM_MODEL is not set" in err and "Traceback" not in err
    assert "note:" not in err  # nothing was about to be sent anywhere


def test_a_configured_model_goes_through_the_guarded_rag_path(capsys, monkeypatch, repo, index_dir):
    client = FakeLLMClient("The id is a hash [Source 1].")
    monkeypatch.setattr(cli, "create_llm_client", lambda settings: client)
    code, out, err = run(capsys, "ask", index_dir, "--repo", repo, QUESTION)
    assert code == 0 and "The id is a hash" in out and "fake/fake-model" in err
    assert client.call_count == 1


def test_a_secret_in_the_question_exits_1_and_nothing_is_sent(capsys, monkeypatch, repo, index_dir):
    client = FakeLLMClient("never returned")
    monkeypatch.setattr(cli, "create_llm_client", lambda settings: client)
    secret = h.github_token("cli")
    code, out, err = run(capsys, "ask", index_dir, "--repo", repo, f"{QUESTION} {secret}")
    assert code == 1 and client.call_count == 0
    assert "nothing was sent" in err and not h.contains_secret(out + err, secret)


def test_a_provider_failure_exits_2_with_the_typed_error(capsys, monkeypatch, repo, index_dir):
    from copilot.llm import LLMRateLimitError

    client = FakeLLMClient(error=LLMRateLimitError("quota exhausted (HTTP 429)"))
    monkeypatch.setattr(cli, "create_llm_client", lambda settings: client)
    code, out, _ = run(capsys, "ask", index_dir, "--repo", repo, QUESTION)
    assert code == 2 and "LLMRateLimitError" in out and "429" in out


def test_arguments_are_forwarded_unchanged_including_windows_style_paths(capsys, monkeypatch):
    seen = {}

    def fake_open(index_dir, repo, **kwargs):
        seen.update(index_dir=index_dir, repo=repo, **kwargs)
        return ScriptedRetriever([[chunk("sha256 hash payload", path="src/pkg/ids.py")]])

    monkeypatch.setattr(cli.Retriever, "open", staticmethod(fake_open))
    win_repo = r"C:\Users\someone\indexed-tree\agentic-rag-code-copilot"
    win_index = r"data\indexes\4aaeab9c650ab76c"
    code, out, _ = run(
        capsys, "ask", win_index, "--repo", win_repo, "--ignore-dir", "data", "--ignore-dir",
        "tests", "--fake-llm", "Explain how hash payload works",
    )  # fmt: skip
    assert code == 0
    assert (seen["index_dir"], seen["repo"]) == (win_index, win_repo)
    assert seen["ignore_directories"] == ["data", "tests"]
    assert "src/pkg/ids.py" in out and "\\" not in out  # citations stay POSIX


def test_graph_command_prints_the_mermaid_diagram(capsys):
    code, out, _ = run(capsys, "graph")
    assert code == 0 and "graph TD" in out
    for node in ("classify_intent", "retrieve", "assess_evidence", "generate", "abstain"):
        assert node in out


def test_the_baseline_cli_is_unchanged_and_independent():
    from copilot.rag import __main__ as rag_cli

    source = Path(rag_cli.__file__).read_text(encoding="utf-8")
    assert "copilot.agents" not in source
    with pytest.raises(SystemExit):
        rag_cli.main(["agents"])  # not a baseline subcommand


def test_partial_evidence_is_flagged_in_the_cli_output(capsys, monkeypatch):
    monkeypatch.setattr(
        cli.Retriever,
        "open",
        staticmethod(lambda *a, **k: ScriptedRetriever([[chunk("nothing relevant", path="x.py")]])),
    )
    code, out, _ = run(capsys, "ask", "idx", "--repo", "r", "--fake-llm", "authentication")
    assert code == 0
    assert "evidence: partial" in out and "warning: partial_evidence:" in out
