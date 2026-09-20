"""The evaluation runner and matrix on a tiny repository with the deterministic fake embedder.

These tests check the *plumbing* (index per configuration, fixed conditions, scoring, output).
They say nothing about retrieval quality: the embedder is a bag-of-words hash.
"""

import json
from pathlib import Path

import pytest

from copilot.config import Settings
from copilot.embeddings import HashEmbedder
from copilot.evaluation import (
    Benchmark,
    BenchmarkFormatError,
    BenchmarkMeta,
    BenchmarkQuestion,
    evaluate_retriever,
    run_configuration,
    run_matrix,
    seal_regions,
)
from copilot.evaluation.runner import format_table, to_jsonable, verify_benchmark
from copilot.ingestion import ingest_repository
from copilot.retrieval import Retriever
from copilot.vectorstore import build_repository_index
from tests.retrieval_helpers import write_repo

QUESTIONS = [
    ("q-ingest", "discover repository files and read their text", "src/pkg/ingest.py", 1, 4),
    ("q-chunk-id", "deterministic chunk id sha256 hash payload", "src/pkg/chunk_ids.py", 1, 4),
    (
        "q-logging",
        "configure logging handlers formatter log level",
        "src/pkg/logging_setup.py",
        2,
        3,
    ),
    ("q-miss", "zebra quantum harvest", "src/pkg/ingest.py", 1, 1),  # unrelated: should miss
]


@pytest.fixture
def repo(tmp_path):
    return write_repo(tmp_path / "bench-repo")


@pytest.fixture
def embedder():
    return HashEmbedder(dimension=256)


@pytest.fixture
def settings():
    return Settings(_env_file=None)


@pytest.fixture
def benchmark(repo, settings):
    raw = [
        BenchmarkQuestion(
            id=qid,
            question=text,
            relevant=[{"file_path": path, "start_line": s, "end_line": e}],
        )
        for qid, text, path, s, e in QUESTIONS
    ]
    files = ingest_repository(repo).files
    meta = BenchmarkMeta(name="tiny", repository_name="bench-repo", commit="c0ffee0")
    return Benchmark(questions=tuple(seal_regions(raw, files)), meta=meta)


def test_the_benchmark_verifies_against_its_repository(benchmark, repo, settings):
    verify_benchmark(benchmark, repo, settings=settings)


def test_a_changed_repository_is_refused_before_anything_is_built(
    benchmark, repo, settings, embedder, tmp_path
):
    (repo / "src/pkg/ingest.py").write_text("def other():\n    pass\n", newline="\n")
    with pytest.raises(BenchmarkFormatError, match="does not match"):
        run_matrix(
            repo,
            benchmark,
            embedder=embedder,
            settings=settings,
            indexes_dir=tmp_path / "ix",
            caps=(512,),
            styles=("raw",),
        )
    assert not (tmp_path / "ix").exists()  # nothing was built


def test_evaluate_retriever_scores_hits_and_misses(benchmark, repo, embedder, tmp_path):
    report = build_repository_index(
        repo,
        embedder=embedder,
        settings=Settings(_env_file=None),
        indexes_dir=tmp_path / "ix",
        text_style="raw",
    )
    retriever = Retriever.open(report.index_path, repo, embedder=embedder)
    outcomes, summary = evaluate_retriever(retriever, benchmark)
    by_id = {o.question_id: o for o in outcomes}
    assert by_id["q-ingest"].first_hit_rank == 1
    assert by_id["q-chunk-id"].first_hit_rank == 1
    assert by_id["q-logging"].first_hit_rank == 1
    assert summary.n_questions == 4 and summary.depth == 10
    assert summary.hit_counts[1] >= 3 and 0 < summary.mrr <= 1
    hit = by_id["q-chunk-id"]
    assert hit.first_hit_lines == 4 and len(hit.retrieved_locations) == len(hit.retrieved_lines)
    assert all(loc.count(":") == 1 for loc in hit.retrieved_locations)  # path:start-end only


def test_the_matrix_builds_one_index_per_configuration_and_holds_the_rest_fixed(
    benchmark, repo, settings, embedder, tmp_path
):
    result = run_matrix(
        repo,
        benchmark,
        embedder=embedder,
        settings=settings,
        indexes_dir=tmp_path / "ix",
        caps=(512, 768),
        styles=("prefixed", "raw"),
        repository_name="bench-repo",
    )
    keys = [(c.chunk_cap, c.text_style) for c in result.configs]
    assert keys == [(512, "prefixed"), (512, "raw"), (768, "prefixed"), (768, "raw")]
    assert len({c.index_id for c in result.configs}) == 4  # each configuration is its own index
    assert {c.summary.n_questions for c in result.configs} == {4}
    assert {c.summary.depth for c in result.configs} == {10}
    cond = result.conditions
    assert cond["caps"] == [512, 768] and cond["index_type"] == "IndexFlatIP"
    assert cond["chunk_size_lines"] == settings.chunk_size_lines
    assert cond["benchmark_independent"] is False and cond["benchmark_commit"] == "c0ffee0"
    assert len(list((tmp_path / "ix").iterdir())) == 4


def test_the_chunk_cap_really_changes_the_chunks(benchmark, repo, settings, embedder, tmp_path):
    small = run_configuration(
        repo, benchmark, embedder=embedder, settings=settings, chunk_cap=16, text_style="raw",
        indexes_dir=tmp_path / "ix",
    )  # fmt: skip
    large = run_configuration(
        repo, benchmark, embedder=embedder, settings=settings, chunk_cap=512, text_style="raw",
        indexes_dir=tmp_path / "ix",
    )  # fmt: skip
    assert small.chunks > large.chunks
    assert small.mean_chunk_lines <= large.mean_chunk_lines
    assert small.index_bytes > 0 and small.seconds_embed >= 0


def test_the_matrix_is_deterministic(benchmark, repo, settings, embedder, tmp_path):
    def run(name):
        return run_matrix(
            repo, benchmark, embedder=embedder, settings=settings,
            indexes_dir=tmp_path / name, caps=(512,), styles=("prefixed", "raw"),
        )  # fmt: skip

    a, b = run("one"), run("two")
    assert [c.outcomes for c in a.configs] == [c.outcomes for c in b.configs]
    assert [c.index_id for c in a.configs] == [c.index_id for c in b.configs]


def test_the_table_lists_every_configuration_and_declares_no_winner(
    benchmark, repo, settings, embedder, tmp_path
):
    result = run_matrix(
        repo, benchmark, embedder=embedder, settings=settings,
        indexes_dir=tmp_path / "ix", caps=(512, 1024), styles=("raw",),
    )  # fmt: skip
    table = format_table(result)
    for cell in ("Hit@1", "Hit@10", "MRR", "ln/hit", "ln/ret", "512", "1024", "raw"):
        assert cell in table
    assert "best" not in table.lower() and "winner" not in table.lower()


def test_the_json_record_is_serialisable_and_free_of_host_paths(
    benchmark, repo, settings, embedder, tmp_path
):
    result = run_matrix(
        repo, benchmark, embedder=embedder, settings=settings,
        indexes_dir=tmp_path / "ix", caps=(512,), styles=("raw",),
    )  # fmt: skip
    text = json.dumps(to_jsonable(result))
    for forbidden in (str(tmp_path), tmp_path.name, str(Path.home())):
        assert forbidden not in text
    data = json.loads(text)
    config = data["configs"][0]
    assert len(config["questions"]) == 4 and "hit_at" in config["summary"]
    assert config["questions"][0]["retrieved"][0].count(":") == 1


def test_a_benchmark_meta_can_state_it_is_independent_only_explicitly():
    assert BenchmarkMeta(name="n", repository_name="r", commit="c").independent is False
