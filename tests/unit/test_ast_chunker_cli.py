"""CLI selection of the "ast" strategy (chunking inspector and vector-index commands)."""

from __future__ import annotations

import json

import pytest

from copilot.chunking.__main__ import main as chunking_main
from copilot.config import Settings
from copilot.embeddings import HashEmbedder
from copilot.vectorstore import __main__ as vs_cli

FILES = {
    "app/service.py": (
        "import os\n\n\ndef run():\n    return os.getcwd()\n\n\nclass S:\n    x = 1\n"
    ),
    "README.md": "# Demo\n\nHello\n",
}


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "demo-repo"
    for relative, text in FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    return root


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    monkeypatch.setattr(vs_cli, "create_embedder", lambda settings: HashEmbedder(dimension=64))
    monkeypatch.setattr(vs_cli, "setup_logging", lambda *a, **k: None)


def test_chunking_cli_default_is_the_baseline(repo, capsys):
    assert chunking_main([str(repo), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["params"] == {"size_lines": 60, "overlap_lines": 10, "max_tokens": 512}
    assert "segmentation" not in json.dumps(payload)


def test_chunking_cli_prints_ast_strategy_segmentation_and_symbols(repo, capsys):
    assert chunking_main([str(repo), "--strategy", "ast", "--samples", "5"]) == 0
    out = capsys.readouterr().out
    assert "strategy:            ast v1" in out
    assert "by segmentation:     {'ast': 3, 'fallback': 1}" in out
    assert "function" in out and "  run" in out and "  S" in out
    assert "return os.getcwd" not in out  # still never prints source text


def test_vectorstore_build_selects_the_strategy_and_reports_it(repo, tmp_path, capsys):
    root = tmp_path / "indexes"
    assert (
        vs_cli.main(["build", str(repo), "--index-dir", str(root), "--chunking-strategy", "ast"])
        == 0
    )
    assert "chunking:            ast v1" in capsys.readouterr().out
    (index_dir,) = list(root.iterdir())
    assert vs_cli.main(["info", str(index_dir)]) == 0
    assert "chunking:            ast v1" in capsys.readouterr().out


def test_vectorstore_build_default_stays_the_baseline(repo, tmp_path, capsys):
    root = tmp_path / "indexes"
    assert vs_cli.main(["build", str(repo), "--index-dir", str(root)]) == 0
    assert "chunking:            line v1" in capsys.readouterr().out


def test_invalid_strategy_name_is_a_clear_argparse_error(repo, tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        vs_cli.main(["build", str(repo), "--chunking-strategy", "semantic"])
    assert exit_info.value.code == 2
    assert "invalid choice: 'semantic'" in capsys.readouterr().err


def test_validate_checks_the_selected_strategy_against_the_index(
    repo, tmp_path, capsys, monkeypatch
):
    model_id = HashEmbedder(dimension=64).info.model_id
    monkeypatch.setattr(
        vs_cli, "get_settings", lambda: Settings(_env_file=None, embedding_model=model_id)
    )
    root = tmp_path / "indexes"
    vs_cli.main(["build", str(repo), "--index-dir", str(root), "--chunking-strategy", "ast"])
    (index_dir,) = list(root.iterdir())
    capsys.readouterr()

    args = ["validate", str(index_dir), "--repo", str(repo)]
    assert vs_cli.main([*args, "--chunking-strategy", "ast"]) == 0
    assert "compatible with the repository" in capsys.readouterr().out

    # without the flag the baseline is expected, so the same index is refused for its chunking
    assert vs_cli.main(args) == 1
    assert "chunking" in capsys.readouterr().err
