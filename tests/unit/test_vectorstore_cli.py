"""``python -m copilot.vectorstore`` commands with a fake embedder (no model, no network)."""

import json

import pytest

from copilot.embeddings import HashEmbedder
from copilot.vectorstore import __main__ as cli
from tests.fixtures.synthetic_repo import MAIN_PY_MARKER

FILES = {
    "src/app/main.py": f'MARKER = "{MAIN_PY_MARKER}"\n\n\ndef main():\n    return 0\n',
    "src/app/util.py": "def util(x):\n    return x + 1\n",
    "README.md": "# Demo\n\nHello\n",
}


class ForbiddenEmbedder:
    """Fails if anything touches the model: proves a command does not need it."""

    @property
    def info(self):
        raise AssertionError("the embedding model must not be loaded")

    def embed_documents(self, texts):
        raise AssertionError("the embedding model must not be used")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "demo-repo"
    for relative, text in FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


@pytest.fixture
def fake_embedder(monkeypatch):
    monkeypatch.setattr(cli, "create_embedder", lambda settings: HashEmbedder(dimension=64))


@pytest.fixture
def built(repo, tmp_path, fake_embedder, capsys):
    index_root = tmp_path / "indexes"
    assert cli.main(["build", str(repo), "--index-dir", str(index_root)]) == 0
    out = capsys.readouterr().out
    (path,) = list(index_root.iterdir())
    return path, out


def test_build_prints_the_required_report_without_contents_or_vectors(built, repo, capsys):
    path, out = built
    for label in (
        "repository:",
        "files indexed:",
        "chunks:",
        "vectors:",
        "dimension:",
        "representation:",
        "index type:",
        "index id:",
        "time (s):",
        "index.faiss",
        "chunks.jsonl",
        "manifest.json",
    ):
        assert label in out, label
    assert "demo-repo" in out and "IndexFlatIP" in out and path.name in out
    assert "prefixed (v1)" in out and "64" in out
    assert MAIN_PY_MARKER not in out and "def util" not in out


def test_build_warns_about_the_model_cache_on_stderr(repo, tmp_path, fake_embedder, capsys):
    cli.main(["build", str(repo), "--index-dir", str(tmp_path / "ix")])
    err = capsys.readouterr().err
    assert "downloads the embedding model" in err and "0.64 GB" in err


def test_build_into_the_default_indexes_directory(repo, tmp_path, fake_embedder, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["build", str(repo)]) == 0
    (path,) = list((tmp_path / "data" / "indexes").iterdir())
    assert len(path.name) == 16


def test_build_twice_requires_force(repo, tmp_path, fake_embedder, capsys):
    args = ["build", str(repo), "--index-dir", str(tmp_path / "ix")]
    assert cli.main(args) == 0
    assert cli.main(args) == 2
    assert "already exists" in capsys.readouterr().err
    assert cli.main([*args, "--force"]) == 0


def test_build_of_a_repository_without_accepted_files_fails_before_loading_the_model(
    tmp_path, monkeypatch, capsys
):
    empty = tmp_path / "empty-repo"
    empty.mkdir()
    (empty / "notes.txt").write_text("unsupported\n")
    monkeypatch.setattr(cli, "create_embedder", lambda settings: ForbiddenEmbedder())
    assert cli.main(["build", str(empty), "--index-dir", str(tmp_path / "ix")]) == 2
    assert "no accepted files" in capsys.readouterr().err
    assert not (tmp_path / "ix").exists()


def test_build_of_a_missing_path_is_a_clean_error(tmp_path, fake_embedder, capsys):
    assert cli.main(["build", str(tmp_path / "nope"), "--index-dir", str(tmp_path / "ix")]) == 2
    assert "error:" in capsys.readouterr().err


def test_info_reads_the_manifest_without_the_model(built, monkeypatch, capsys):
    path, _ = built
    monkeypatch.setattr(cli, "create_embedder", lambda settings: ForbiddenEmbedder())
    assert cli.main(["info", str(path)]) == 0
    out = capsys.readouterr().out
    assert path.name in out and "demo-repo" in out and "IndexFlatIP" in out
    assert "line v1" in out and "fake/hash-embedder" in out and "chunks / vectors" in out


def test_info_json_is_the_manifest(built, capsys):
    path, _ = built
    assert cli.main(["info", str(path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["index_id"] == path.name and data["spec"]["repository_name"] == "demo-repo"


def test_info_on_a_missing_index_is_a_clean_error(tmp_path, capsys):
    assert cli.main(["info", str(tmp_path / "nope")]) == 2
    assert "error:" in capsys.readouterr().err


def test_validate_passes_without_loading_the_model(built, monkeypatch, capsys):
    path, _ = built
    monkeypatch.setattr(cli, "create_embedder", lambda settings: ForbiddenEmbedder())
    assert cli.main(["validate", str(path)]) == 0
    out = capsys.readouterr().out
    assert "OK" in out and "no embedding model was loaded" in out
    assert "NOT checked" in out  # honest: without --repo it only proves internal consistency


def use_fake_model_settings(monkeypatch):
    """The index was built with the fake embedder, so the configured model id must match it."""
    from copilot.config import get_settings

    monkeypatch.setenv("COPILOT_EMBEDDING_MODEL", "fake/hash-embedder")
    get_settings.cache_clear()


def test_validate_against_the_repository_passes_when_unchanged(built, repo, monkeypatch, capsys):
    path, _ = built
    use_fake_model_settings(monkeypatch)
    assert cli.main(["validate", str(path), "--repo", str(repo)]) == 0
    assert "compatible with the repository" in capsys.readouterr().out


def test_validate_reports_a_stale_index_after_the_source_changed(built, repo, monkeypatch, capsys):
    path, _ = built
    use_fake_model_settings(monkeypatch)
    (repo / "src" / "app" / "util.py").write_text("def util(x):\n    return x + 2\n")
    assert cli.main(["validate", str(path), "--repo", str(repo)]) == 1
    err = capsys.readouterr().err
    assert "INCOMPATIBLE" in err and "repository_fingerprint" in err


def test_validate_reports_a_chunking_config_change(built, repo, monkeypatch, capsys):
    path, _ = built
    monkeypatch.setenv("COPILOT_CHUNK_SIZE_LINES", "30")
    use_fake_model_settings(monkeypatch)
    assert cli.main(["validate", str(path), "--repo", str(repo)]) == 1
    assert "chunk_params" in capsys.readouterr().err


def test_validate_reports_a_model_change(built, repo, monkeypatch, capsys):
    path, _ = built
    monkeypatch.setenv("COPILOT_EMBEDDING_MODEL", "another/model")
    from copilot.config import get_settings

    get_settings.cache_clear()
    assert cli.main(["validate", str(path), "--repo", str(repo)]) == 1
    assert "embedding_model_id" in capsys.readouterr().err


def test_validate_reports_corruption(built, capsys):
    path, _ = built
    (path / "index.faiss").write_bytes(b"damaged")
    assert cli.main(["validate", str(path)]) == 1
    assert "INVALID" in capsys.readouterr().err


def test_validate_on_a_missing_directory(tmp_path, capsys):
    assert cli.main(["validate", str(tmp_path / "nope")]) == 1
    assert "INVALID" in capsys.readouterr().err
