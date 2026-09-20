"""``python -m copilot.evaluation`` commands with the fake embedder."""

import json

import pytest

from copilot.embeddings import HashEmbedder
from copilot.evaluation import __main__ as cli
from copilot.evaluation import dump_benchmark, load_benchmark
from copilot.evaluation.benchmark import BenchmarkQuestion
from copilot.vectorstore import build_repository_index
from tests.retrieval_helpers import write_repo

QS = [
    {
        "id": "q-chunk-id",
        "question": "deterministic chunk id sha256 hash payload",
        "relevant": [{"file_path": "src/pkg/chunk_ids.py", "start_line": 1, "end_line": 4}],
    },
    {
        "id": "q-logging",
        "question": "configure logging handlers formatter log level",
        "relevant": [{"file_path": "src/pkg/logging_setup.py", "start_line": 1, "end_line": 4}],
    },
]


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setattr(cli, "create_embedder", lambda settings: HashEmbedder(dimension=256))
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)


@pytest.fixture
def repo(tmp_path):
    return write_repo(tmp_path / "bench-repo")


@pytest.fixture
def bench(tmp_path):
    path = tmp_path / "bench.jsonl"
    text = dump_benchmark([BenchmarkQuestion.model_validate(q) for q in QS])
    path.write_text(text, encoding="utf-8", newline="\n")
    meta = {
        "name": "tiny",
        "repository_name": "bench-repo",
        "commit": "abc",
        "ignore_directories": [],
    }
    (tmp_path / "bench.meta.json").write_text(json.dumps(meta))
    return path


def run(capsys, *argv):
    code = cli.main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_verify_refuses_an_unsealed_benchmark_then_seal_makes_it_pass(bench, repo, capsys):
    code, _, err = run(capsys, "verify", bench, "--repo", repo)
    assert code == 2 and "not sealed" in err
    code, out, _ = run(capsys, "seal", bench, "--repo", repo)
    assert code == 0 and "sealed 2 regions" in out
    assert all(r.sha256 for q in load_benchmark(bench).questions for r in q.relevant)
    code, out, _ = run(capsys, "verify", bench, "--repo", repo)
    assert code == 0 and out.startswith("OK: 2 questions")


def test_verify_detects_a_changed_repository(bench, repo, capsys):
    run(capsys, "seal", bench, "--repo", repo)
    (repo / "src/pkg/chunk_ids.py").write_text("x = 1\ny = 2\nz = 3\nw = 4\n", newline="\n")
    code, _, err = run(capsys, "verify", bench, "--repo", repo)
    assert code == 2 and "does not match" in err


def test_seal_refuses_regions_outside_the_file(tmp_path, repo, capsys):
    bad = dict(
        QS[0], relevant=[{"file_path": "src/pkg/chunk_ids.py", "start_line": 1, "end_line": 99}]
    )
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(bad) + "\n")
    code, _, err = run(capsys, "seal", path, "--repo", repo)
    assert code == 2 and "does not exist" in err


def test_run_scores_an_existing_index(bench, repo, tmp_path, capsys):
    run(capsys, "seal", bench, "--repo", repo)
    report = build_repository_index(
        repo,
        embedder=HashEmbedder(dimension=256),
        indexes_dir=tmp_path / "ix",
        repository_name="bench-repo",
        text_style="raw",
    )
    code, out, _ = run(capsys, "run", bench, "--repo", repo, "--index", report.index_path)
    assert code == 0
    assert "Hit@1: 100.0%  (2/2)" in out and "MRR: 1.000" in out and "mean lines per hit" in out
    assert "q-chunk-id" in out and "src/pkg/chunk_ids.py:1-" in out
    assert str(tmp_path) not in out


def test_matrix_prints_a_table_and_writes_json(bench, repo, tmp_path, capsys):
    run(capsys, "seal", bench, "--repo", repo)
    output = tmp_path / "out" / "matrix.json"
    code, out, err = run(
        capsys, "matrix", bench, "--repo", repo, "--work-dir", tmp_path / "wd",
        "--caps", "512", "768", "--styles", "raw", "--output", output,
    )  # fmt: skip
    assert code == 0 and "Hit@1" in out and "No winner is declared" in out
    assert "cap=512 style=raw" in err and "cap=768 style=raw" in err
    data = json.loads(output.read_text(encoding="utf-8"))
    assert [c["chunk_cap"] for c in data["configs"]] == [512, 768]
    assert str(tmp_path) not in out


def test_a_missing_benchmark_file_is_a_clean_error(repo, tmp_path, capsys):
    code, _, err = run(capsys, "verify", tmp_path / "nope.jsonl", "--repo", repo)
    assert code == 2 and "error:" in err and str(tmp_path) not in err


def test_the_ignore_dirs_default_to_the_meta_file(tmp_path, repo, capsys):
    path = tmp_path / "b.jsonl"
    path.write_text(json.dumps(QS[0]) + "\n")
    meta = {
        "name": "t",
        "repository_name": "bench-repo",
        "commit": "abc",
        "ignore_directories": ["src"],
    }
    (tmp_path / "b.meta.json").write_text(json.dumps(meta))
    code, _, err = run(
        capsys, "seal", path, "--repo", repo
    )  # src/ is pruned, so the region is gone
    assert code == 2 and "does not exist" in err
    code, out, _ = run(capsys, "seal", path, "--repo", repo, "--ignore-dir", "docs")  # flag wins
    assert code == 0 and "sealed" in out
