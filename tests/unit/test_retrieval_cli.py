"""``python -m copilot.retrieval search`` with a fake embedder (no model, no network)."""

import pytest

from copilot.config import Settings
from copilot.embeddings import HashEmbedder
from copilot.models.chunk import ChunkType
from copilot.retrieval import RetrievalResult
from copilot.retrieval import __main__ as cli
from copilot.vectorstore import build_repository_index

FILES = {
    "src/app/main.py": (
        "class Service:\n"
        "    def run(self):\n"
        "        \tindented_with_tab = 1\n"
        "        return 'discover repository files'\n"
    ),
    "src/app/logs.py": "def setup_logging(level):\n    return level  # configure logging\n",
    "notes/long.md": "\n".join(f"line {i} of a long note about widgets" for i in range(1, 41))
    + "\n",
}


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "demo repo"  # a space on purpose
    for relative, text in FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    return root


@pytest.fixture
def index_dir(repo, tmp_path):
    report = build_repository_index(
        repo,
        embedder=HashEmbedder(dimension=64),
        settings=Settings(_env_file=None),
        indexes_dir=tmp_path / "indexes",
    )
    return report.index_path


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    monkeypatch.setattr(cli, "create_embedder", lambda settings: HashEmbedder(dimension=64))
    # main() installs a log handler bound to the current (captured) stderr; keep tests independent
    monkeypatch.setattr(cli, "setup_logging", lambda *args, **kwargs: None)


def run(capsys, *argv):
    code = cli.main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def result(text: str, **overrides) -> RetrievalResult:
    values = {
        "rank": 1,
        "score": 0.5,
        "position": 0,
        "chunk_id": "0123456789abcdef",
        "repository_name": "repo",
        "file_path": "src/a.py",
        "language": "python",
        "chunk_type": ChunkType.LINE_WINDOW,
        "chunk_index": 0,
        "start_line": 3,
        "end_line": 3 + text.count("\n"),
        "token_estimate": 1,
        "content_sha256": "0" * 64,
        "text": text,
    }
    values.update(overrides)
    return RetrievalResult(**values)


def test_preview_preserves_indentation_and_shows_only_the_first_lines():
    text = "\n".join(["def f():", "    if x:", "        return 1"] + ["pad"] * 20)
    lines = cli.preview(text, max_lines=3)
    assert lines[:3] == ["def f():", "    if x:", "        return 1"]
    assert lines[3] == "... (+20 more lines)" and len(lines) == 4


def test_preview_of_a_short_chunk_has_no_footer():
    assert cli.preview("a\nb", max_lines=8) == ["a", "b"]


def test_preview_truncates_long_lines_and_removes_control_characters():
    (line,) = cli.preview("x" * 500, width=40)
    assert len(line) == 40 and line.endswith("...")
    (cleaned,) = cli.preview("safe\x1b[31mred\x07")
    assert "\x1b" not in cleaned and "\x07" not in cleaned and "safe" in cleaned


def test_a_result_block_shows_rank_score_location_id_and_preview():
    block = cli.format_result(result("def f():\n    return 1\n"), max_lines=8)
    assert block.splitlines()[0].startswith("#1") and "score 0.5000" in block
    assert "src/a.py:3-5" in block and "chunk 0123456789abcdef" in block
    assert "    |     return 1" in block  # indentation survives after the "| " gutter


def test_search_prints_ranked_results_with_relative_paths_only(index_dir, repo, capsys, tmp_path):
    code, out, err = run(
        capsys, "search", index_dir, "--repo", repo, "configure logging", "--top-k", 2
    )
    assert code == 0
    assert "#1" in out and "#2" in out and "#3" not in out
    assert "src/app/logs.py:1-" in out.split("#2")[0]  # the topical file ranks first
    assert "chunk " in out and "    | " in out
    for forbidden in (str(tmp_path), str(repo), str(index_dir)):
        assert forbidden not in out + err


def test_search_never_dumps_a_whole_file(index_dir, repo, capsys):
    code, out, _ = run(
        capsys, "search", index_dir, "--repo", repo, "long note widgets", "--top-k", 1
    )
    assert code == 0
    assert "line 40 of a long note" not in out  # the chunk has 40 lines; only 8 are previewed
    assert "more lines)" in out


def test_preview_lines_option(index_dir, repo, capsys):
    _, out, _ = run(
        capsys, "search", index_dir, "--repo", repo, "long note widgets", "--top-k", 1,
        "--preview-lines", 2,
    )  # fmt: skip
    assert "line 2 of a long note" in out and "line 3 of a long note" not in out


def test_search_reports_the_index_summary(index_dir, repo, capsys):
    _, out, _ = run(capsys, "search", index_dir, "--repo", repo, "logging", "--top-k", 3)
    first = out.splitlines()[0]
    assert first.startswith("index ") and "repository demo repo" in first and "top_k=3" in first


@pytest.mark.parametrize("query", ["", "   "])
def test_an_empty_query_is_a_clean_error(index_dir, repo, capsys, query):
    code, out, err = run(capsys, "search", index_dir, "--repo", repo, query)
    assert code == 2 and out == "" and "error: query is empty" in err


@pytest.mark.parametrize("top_k", ["0", "51", "-3"])
def test_an_invalid_top_k_is_a_clean_error(index_dir, repo, capsys, top_k):
    code, _, err = run(capsys, "search", index_dir, "--repo", repo, "logging", "--top-k", top_k)
    assert code == 2 and "top_k" in err


def test_a_stale_repository_is_a_clean_error(index_dir, repo, capsys, tmp_path):
    (repo / "src/app/logs.py").write_text("x = 1\n", newline="\n")
    code, out, err = run(capsys, "search", index_dir, "--repo", repo, "logging")
    assert code == 2 and out == "" and "fingerprint" in err and str(tmp_path) not in err


def test_ignore_dir_must_repeat_the_build_flags(repo, tmp_path, capsys):
    report = build_repository_index(
        repo,
        embedder=HashEmbedder(dimension=64),
        settings=Settings(_env_file=None),
        indexes_dir=tmp_path / "ix",
        ignore_directories=["notes"],
    )
    code, _, err = run(capsys, "search", report.index_path, "--repo", repo, "widgets")
    assert code == 2 and "ignore-dir" in err
    code, out, _ = run(
        capsys, "search", report.index_path, "--repo", repo, "widgets", "--ignore-dir", "notes"
    )
    assert code == 0 and "notes/long.md" not in out


def test_a_missing_index_is_a_clean_error(repo, tmp_path, capsys):
    code, _, err = run(capsys, "search", tmp_path / "nowhere", "--repo", repo, "logging")
    assert code == 2 and "error:" in err and str(tmp_path) not in err


def test_a_missing_repository_is_a_clean_error(index_dir, tmp_path, capsys):
    code, _, err = run(capsys, "search", index_dir, "--repo", tmp_path / "nowhere", "logging")
    assert code == 2 and "error:" in err


def test_an_embedder_that_is_not_the_index_model_is_a_clean_error(
    index_dir, repo, capsys, monkeypatch
):
    monkeypatch.setattr(cli, "create_embedder", lambda settings: HashEmbedder(dimension=32))
    code, _, err = run(capsys, "search", index_dir, "--repo", repo, "logging")
    assert code == 2 and "incompatible" in err


def test_a_query_with_non_ascii_text_is_accepted(index_dir, repo, capsys):
    code, out, _ = run(capsys, "search", index_dir, "--repo", repo, "función módulo ünï 日本語")
    assert code == 0 and out.startswith("index ")
