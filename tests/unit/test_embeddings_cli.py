"""CLI behaviour with a stub embedder (no model, no network)."""

import json

import pytest

from copilot.embeddings import EmbeddingModelUnavailableError, HashEmbedder
from copilot.embeddings import __main__ as cli

MARKER = "UNIQUE_SOURCE_MARKER_91c2"


class StubEmbedder(HashEmbedder):
    def count_tokens(self, texts):
        return [2 * n + 1 for n in super().count_tokens(texts)]  # deliberately "underestimated"

    def runtime_limits(self):
        return {
            "tokenizer_truncation_max_length": 200,
            "config_max_position_embeddings": 200,
            "tokenizer_config_model_max_length": 200,
        }


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "a.py").write_text(f"{MARKER} = 1\n" + "".join(f"v{i} = {i}\n" for i in range(150)))
    (tmp_path / "README.md").write_text("# Title\n\ntext\n")
    return tmp_path


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setattr(cli, "create_embedder", lambda settings: StubEmbedder(dimension=32))


def test_info_reports_verified_fastembed_facts_offline(capsys):
    assert cli.main(["info"]) == 0
    out = capsys.readouterr().out
    assert "jinaai/jina-embeddings-v2-base-code" in out and "768" in out
    assert "onnx/model.onnx" in out and "model cache dir" in out


def test_smoke_prints_summary_without_vectors_or_contents(stub, repo, capsys):
    assert cli.main(["smoke", str(repo), "-n", "3"]) == 0
    out = capsys.readouterr().out
    assert "dimension:           32" in out and "vectors embedded:    3" in out
    assert "all finite:          True" in out and "load time" in out
    assert MARKER not in out and "[" not in out  # no contents, no printed arrays/vectors


def test_smoke_without_a_repository_uses_builtin_samples(stub, capsys):
    assert cli.main(["smoke"]) == 0
    assert "vectors embedded:    3" in capsys.readouterr().out


def test_tokens_report_sections_and_no_contents(stub, repo, capsys):
    assert cli.main(["tokens", str(repo), "--caps", "50,100", "--examples", "2"]) == 0
    out = capsys.readouterr().out
    for section in ("estimated vs actual tokens", "candidate caps", "prefix overhead"):
        assert section in out
    assert "a.py:" in out  # metadata only
    assert MARKER not in out


def test_tokens_json_output(stub, repo, capsys):
    assert cli.main(["tokens", str(repo), "--caps", "50", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"]["dimension"] == 32
    assert payload["comparison"]["sample_count"] > 0
    assert [c["cap"] for c in payload["caps"]] == [50]


def test_representations_command(stub, repo, capsys):
    assert cli.main(["representations", str(repo), "-n", "3"]) == 0
    out = capsys.readouterr().out
    assert "cosine(raw, prefixed)" in out and "not whether retrieval improves" in out


def test_model_load_failure_is_a_clean_error_exit(monkeypatch, repo, capsys):
    class Broken:
        @property
        def info(self):
            raise EmbeddingModelUnavailableError("could not load embedding model 'x' (403)")

    monkeypatch.setattr(cli, "create_embedder", lambda settings: Broken())
    assert cli.main(["smoke", str(repo)]) == 2
    captured = capsys.readouterr()
    assert "error: could not load embedding model" in captured.err
    assert "Traceback" not in captured.err


def test_sizes_command_needs_no_model_and_reports_bytes_bound(monkeypatch, repo, capsys):
    def no_embedder(settings):
        raise AssertionError("the sizes command must not create an embedder")

    monkeypatch.setattr(cli, "create_embedder", no_embedder)
    assert cli.main(["sizes", str(repo), "--caps", "50,100", "--model-limit", "8192"]) == 0
    out = capsys.readouterr().out
    assert "NO tokenizer used" in out and "bytes-bound>limit" in out
    assert MARKER not in out
