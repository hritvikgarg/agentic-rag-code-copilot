"""``python -m copilot.security scan``: exit codes, output safety, ingestion reuse."""

import json

import pytest

from copilot.security import __main__ as cli
from tests.security import secret_helpers as h

TOKEN = h.github_token("cli")


@pytest.fixture(autouse=True)
def _no_log_handler(monkeypatch):
    monkeypatch.setattr(cli, "setup_logging", lambda *args, **kwargs: None)


def make_repo(tmp_path, *, leak: bool):
    root = tmp_path / "demo repo"  # a space on purpose
    (root / "src").mkdir(parents=True)
    (root / "src" / "ok.py").write_text("x = 1\n", encoding="utf-8")
    if leak:
        (root / "src" / "leak.py").write_text(f'a = 1\ntoken_value = "{TOKEN}"\n', encoding="utf-8")
    return root


def test_clean_repository_exits_zero(tmp_path, capsys):
    assert cli.main(["scan", str(make_repo(tmp_path, leak=False))]) == 0
    out = capsys.readouterr().out
    assert "findings:         0" in out and "no potential secrets" in out


def test_findings_exit_one_and_output_is_safe(tmp_path, capsys):
    root = make_repo(tmp_path, leak=True)
    assert cli.main(["scan", str(root)]) == 1
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "src/leak.py:2:16  [high] github-token" in captured.out
    assert "REFUSED for external LLM use" in captured.err
    assert not h.contains_secret(text, TOKEN)
    assert str(root) not in text and "\\" not in captured.out
    assert "...".join([TOKEN[:4], TOKEN[-2:]]) not in text  # preview only with --show-preview


def test_show_preview_prints_only_a_masked_value(tmp_path, capsys):
    assert cli.main(["scan", str(make_repo(tmp_path, leak=True)), "--show-preview"]) == 1
    out = capsys.readouterr().out
    assert f"{TOKEN[:4]}...{TOKEN[-2:]}" in out
    assert not h.contains_secret(out, TOKEN)


def test_json_output_has_metadata_only(tmp_path, capsys):
    assert cli.main(["scan", str(make_repo(tmp_path, leak=True)), "--json"]) == 1
    raw = capsys.readouterr().out
    payload = json.loads(raw)
    assert payload["findings_count"] == 1
    assert payload["counts_by_rule"] == {"github-token": 1}
    (finding,) = payload["findings"]
    assert finding["file_path"] == "src/leak.py" and "preview" not in finding
    assert not h.contains_secret(raw, TOKEN)


def test_ignore_dir_is_honoured_like_in_the_index_build(tmp_path, capsys):
    root = make_repo(tmp_path, leak=True)
    assert cli.main(["scan", str(root), "--ignore-dir", "src"]) == 0
    assert "files scanned:    0" in capsys.readouterr().out


def test_sensitive_named_files_are_not_scanned_at_all(tmp_path, capsys):
    root = make_repo(tmp_path, leak=False)
    (root / ".env").write_text(f"TOKEN={TOKEN}\n", encoding="utf-8")
    assert cli.main(["scan", str(root)]) == 0
    assert "files scanned:    1" in capsys.readouterr().out


@pytest.mark.parametrize("bad", ["does-not-exist", "C:\\definitely\\missing dir"])
def test_invalid_repository_exits_two(tmp_path, capsys, bad):
    assert cli.main(["scan", str(tmp_path / bad)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")


def test_truncated_ingestion_is_an_incomplete_scan_and_exits_two(tmp_path, capsys, monkeypatch):
    root = make_repo(tmp_path, leak=False)
    (root / "src" / "more.py").write_text("y = 2\n", encoding="utf-8")
    monkeypatch.setenv("COPILOT_MAX_REPO_FILES", "1")
    assert cli.main(["scan", str(root)]) == 2
    assert "scan is incomplete" in capsys.readouterr().err
