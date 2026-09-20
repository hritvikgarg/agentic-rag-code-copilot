import json

from copilot.chunking.__main__ import main


def test_cli_prints_statistics_and_never_file_contents(tmp_path, capsys):
    (tmp_path / "a.py").write_text("SECRET_LOOKING_MARKER = 1\n")
    assert main([str(tmp_path), "--samples", "3"]) == 0
    out = capsys.readouterr().out
    assert "chunks:" in out and "a.py:1-1" in out
    assert "SECRET_LOOKING_MARKER" not in out


def test_cli_json_output(tmp_path, capsys):
    (tmp_path / "a.py").write_text("x = 1\n")
    assert main([str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stats"]["chunk_count"] == 1
    assert payload["params"]["size_lines"] == 60


def test_cli_reserved_strategy_is_an_error_not_a_fallback(tmp_path, capsys):
    (tmp_path / "a.py").write_text("x = 1\n")
    assert main([str(tmp_path), "--strategy", "ast"]) == 2
    assert "Milestone 9" in capsys.readouterr().err


def test_cli_invalid_repository_is_an_error(tmp_path, capsys):
    assert main([str(tmp_path / "missing")]) == 2
    assert "error:" in capsys.readouterr().err
