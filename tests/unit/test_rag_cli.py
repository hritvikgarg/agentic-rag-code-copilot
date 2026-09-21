"""``python -m copilot.rag`` with a fake embedder and a fake LLM (no model, no network)."""

import pytest

from copilot.config import Settings
from copilot.embeddings import HashEmbedder
from copilot.llm import FakeLLMClient, LLMAuthenticationError, LLMConfigurationError
from copilot.rag import __main__ as cli
from copilot.vectorstore import build_repository_index
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


@pytest.fixture
def fake(monkeypatch):
    client = FakeLLMClient("The id is a hash [Source 1].")
    monkeypatch.setattr(cli, "create_llm_client", lambda settings: client)
    monkeypatch.setattr(cli, "create_embedder", lambda settings: HashEmbedder(dimension=64))
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)
    return client


def run(capsys, *argv):
    code = cli.main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_ask_prints_the_answer_then_the_sources(capsys, fake, repo, index_dir):
    code, out, err = run(capsys, "ask", index_dir, "--repo", repo, "--top-k", 2, QUESTION)
    assert code == 0
    assert out.index("ANSWER") < out.index("The id is a hash") < out.index("SOURCES")
    assert "[Source 1] src/pkg/chunk_ids.py:1-4" in out  # POSIX relative, file:start-end
    assert str(repo) not in out + err and "\\" not in out  # no absolute or Windows-style paths
    assert "note:" in err and "fake/fake-model" in err
    assert "def make_chunk_id" not in out  # no source dump by default
    assert fake.call_count == 1


def test_show_context_prints_metadata_only(capsys, fake, repo, index_dir):
    code, out, _ = run(capsys, "ask", index_dir, "--repo", repo, "--show-context", QUESTION)
    assert code == 0 and "CONTEXT (metadata only)" in out
    assert "chunks in context:" in out and "prompt version: rag-v1" in out
    assert "sha256(payload)" not in out and "def make_chunk_id" not in out


def test_unknown_source_numbers_produce_a_warning(capsys, monkeypatch, repo, index_dir):
    monkeypatch.setattr(cli, "create_llm_client", lambda s: FakeLLMClient("see [Source 9]"))
    monkeypatch.setattr(cli, "create_embedder", lambda s: HashEmbedder(dimension=64))
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)
    code, out, _ = run(capsys, "ask", index_dir, "--repo", repo, QUESTION)
    assert (
        code == 0 and "warning: the answer mentions source numbers that were not supplied: 9" in out
    )


def test_plain_prints_the_answer_and_says_it_is_not_grounded(capsys, fake):
    code, out, err = run(capsys, "plain", "How are chunk ids generated?")
    assert code == 0 and "ANSWER (plain LLM, no repository evidence)" in out
    assert "NO repository context" in err
    assert fake.requests[0].user_prompt == "How are chunk ids generated?"


def test_a_blank_question_exits_2_before_anything_is_loaded(capsys, fake, repo, index_dir):
    code, _, err = run(capsys, "ask", index_dir, "--repo", repo, "   ")
    assert code == 2 and err.startswith("error:") and fake.call_count == 0
    code, _, err = run(capsys, "plain", "")
    assert code == 2 and fake.call_count == 0


def test_a_secret_in_the_question_is_refused_with_exit_1_and_nothing_is_sent(
    capsys, fake, repo, index_dir
):
    secret = h.github_token("cli")
    code, out, err = run(capsys, "ask", index_dir, "--repo", repo, f"why is {secret} rejected?")
    assert code == 1 and fake.call_count == 0
    assert "refused" in err and "nothing was sent" in err
    assert not h.contains_secret(out + err, secret)
    code, out, err = run(capsys, "plain", f"is {secret} valid?")
    assert code == 1 and fake.call_count == 0 and not h.contains_secret(out + err, secret)


def test_a_secret_in_the_repository_is_refused_with_exit_1(capsys, fake, repo, tmp_path):
    secret = h.github_token("cli-repo")
    (repo / "src/pkg/config.py").write_text(
        f"# chunk id sha256 hash payload settings\nTOKEN = '{secret}'\n", encoding="utf-8"
    )
    index = build_repository_index(
        repo, embedder=HashEmbedder(dimension=64), settings=Settings(_env_file=None),
        indexes_dir=tmp_path / "idx", text_style="raw",
    ).index_path  # fmt: skip
    code, out, err = run(capsys, "ask", index, "--repo", repo, "--top-k", 5, QUESTION)
    assert code == 1 and fake.call_count == 0
    assert "src/pkg/config.py" in err and not h.contains_secret(out + err, secret)


def test_configuration_errors_exit_2_without_a_traceback(capsys, monkeypatch, repo, index_dir):
    def missing(settings):
        raise LLMConfigurationError("COPILOT_LLM_MODEL is not set")

    monkeypatch.setattr(cli, "create_llm_client", missing)
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)
    code, _, err = run(capsys, "ask", index_dir, "--repo", repo, QUESTION)
    assert code == 2 and "error: COPILOT_LLM_MODEL is not set" in err and "Traceback" not in err


def test_provider_errors_exit_2_with_the_safe_message(capsys, monkeypatch, repo, index_dir):
    failing = FakeLLMClient(
        error=LLMAuthenticationError("Gemini rejected the credentials (HTTP 401)")
    )
    monkeypatch.setattr(cli, "create_llm_client", lambda s: failing)
    monkeypatch.setattr(cli, "create_embedder", lambda s: HashEmbedder(dimension=64))
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)
    code, _, err = run(capsys, "ask", index_dir, "--repo", repo, QUESTION)
    assert code == 2 and "HTTP 401" in err


def test_a_missing_index_exits_2(capsys, fake, repo, tmp_path):
    code, _, err = run(capsys, "ask", tmp_path / "nope", "--repo", repo, QUESTION)
    assert code == 2 and "error: index directory does not exist" in err
    assert fake.call_count == 0


def test_the_cli_has_no_bypass_flag(capsys):
    for argv in (["ask", "--help"], ["plain", "--help"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
        text = capsys.readouterr().out.lower()
        assert not any(
            word in text for word in ("force", "bypass", "skip-scan", "no-scan", "unsafe")
        )
