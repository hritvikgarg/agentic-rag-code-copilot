"""Benchmark parsing, region verification and sealing."""

import json

import pytest

from copilot.evaluation import (
    Benchmark,
    BenchmarkFormatError,
    dump_benchmark,
    load_benchmark,
    region_text,
    seal_regions,
    verify_regions,
)
from copilot.evaluation.benchmark import Region
from copilot.ingestion import ingest_repository
from copilot.models.chunk import sha256_text

GOOD = {
    "id": "q1-ingest",
    "question": "Where is ingestion implemented?",
    "relevant": [{"file_path": "src/a.py", "start_line": 2, "end_line": 3}],
}


def write(path, *lines):
    text = "".join((line if isinstance(line, str) else json.dumps(line)) + "\n" for line in lines)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src/a.py").write_text("line1\nline2\nline3\nline4\n", newline="\n")
    (root / "src/b.py").write_text("b1\nb2\n", newline="\n")
    return root


@pytest.fixture
def files(repo):
    return ingest_repository(repo).files


def questions(**changes):
    return [Benchmark_q({**GOOD, **changes})]


def Benchmark_q(data):  # noqa: N802 (a tiny factory reads better with this name in the tests)
    from copilot.evaluation import BenchmarkQuestion

    return BenchmarkQuestion.model_validate(data)


# ---- loading ---------------------------------------------------------------------------------


def test_a_valid_file_loads(tmp_path):
    path = write(tmp_path / "b.jsonl", GOOD, "", {**GOOD, "id": "q2"})
    benchmark = load_benchmark(path)
    assert [q.id for q in benchmark.questions] == ["q1-ingest", "q2"]
    assert benchmark.questions[0].relevant[0].sha256 is None and benchmark.meta is None


def test_several_regions_per_question_are_supported(tmp_path):
    two = {
        **GOOD,
        "relevant": [GOOD["relevant"][0], {"file_path": "x/b.py", "start_line": 1, "end_line": 1}],
    }
    assert len(load_benchmark(write(tmp_path / "b.jsonl", two)).questions[0].relevant) == 2


def test_the_meta_file_is_read_when_present(tmp_path):
    path = write(tmp_path / "bench.jsonl", GOOD)
    meta = {
        "name": "demo",
        "repository_name": "repo",
        "commit": "abc123",
        "ignore_directories": ["tests"],
    }
    (tmp_path / "bench.meta.json").write_text(json.dumps(meta))
    loaded = load_benchmark(path)
    assert loaded.meta.commit == "abc123" and loaded.meta.ignore_directories == ("tests",)
    assert loaded.meta.independent is False  # never claimed by default


@pytest.mark.parametrize(
    ("bad", "fragment"),
    [
        ("{not json", "JSONDecodeError"),
        ({**GOOD, "extra": 1}, "extra"),
        ({k: v for k, v in GOOD.items() if k != "relevant"}, "relevant"),
        ({**GOOD, "relevant": []}, "relevant"),
        ({**GOOD, "question": "   "}, "question"),
        ({**GOOD, "id": "Has Space"}, "id"),
        ({**GOOD, "id": ""}, "id"),
        ({**GOOD, "relevant": [{"file_path": "a.py", "start_line": 5, "end_line": 4}]}, "end_line"),
        (
            {**GOOD, "relevant": [{"file_path": "a.py", "start_line": 0, "end_line": 4}]},
            "start_line",
        ),
        (
            {**GOOD, "relevant": [{"file_path": "/abs/a.py", "start_line": 1, "end_line": 4}]},
            "file_path",
        ),
        (
            {**GOOD, "relevant": [{"file_path": "../a.py", "start_line": 1, "end_line": 4}]},
            "file_path",
        ),
        (
            {**GOOD, "relevant": [{"file_path": "a\\b.py", "start_line": 1, "end_line": 4}]},
            "file_path",
        ),
        (
            {**GOOD, "relevant": [{"file_path": "C:/a.py", "start_line": 1, "end_line": 4}]},
            "file_path",
        ),
        (
            {
                **GOOD,
                "relevant": [
                    {"file_path": "a.py", "start_line": 1, "end_line": 4, "sha256": "xyz"}
                ],
            },
            "sha256",
        ),
    ],
)
def test_malformed_entries_are_refused_with_the_line_number(tmp_path, bad, fragment):
    path = write(tmp_path / "b.jsonl", GOOD if bad != GOOD else bad, bad)
    with pytest.raises(BenchmarkFormatError, match=r"line \d") as caught:
        load_benchmark(path)
    assert fragment in str(caught.value)


def test_duplicate_ids_empty_files_and_missing_files_are_refused(tmp_path):
    with pytest.raises(BenchmarkFormatError, match="duplicate"):
        load_benchmark(write(tmp_path / "d.jsonl", GOOD, GOOD))
    with pytest.raises(BenchmarkFormatError, match="no questions"):
        load_benchmark(write(tmp_path / "e.jsonl", "", "  "))
    with pytest.raises(BenchmarkFormatError, match="cannot read"):
        load_benchmark(tmp_path / "missing.jsonl")


def test_a_bad_meta_file_is_refused(tmp_path):
    path = write(tmp_path / "bench.jsonl", GOOD)
    (tmp_path / "bench.meta.json").write_text('{"name": "x"}')
    with pytest.raises(BenchmarkFormatError, match="meta"):
        load_benchmark(path)


def test_dump_and_load_round_trip(tmp_path):
    original = questions()
    path = tmp_path / "rt.jsonl"
    path.write_text(dump_benchmark(original), encoding="utf-8", newline="\n")
    assert list(load_benchmark(path).questions) == original
    assert dump_benchmark(original).endswith("\n") and "\r" not in dump_benchmark(original)


def test_non_ascii_questions_survive_a_round_trip(tmp_path):
    original = questions(question="¿Dónde se configura el registro? 日本語")
    path = tmp_path / "u.jsonl"
    path.write_text(dump_benchmark(original), encoding="utf-8", newline="\n")
    assert load_benchmark(path).questions[0].question == original[0].question


# ---- verification and sealing -----------------------------------------------------------------


def test_region_text_is_the_inclusive_line_slice(files):
    a = next(f for f in files if f.relative_path == "src/a.py")
    assert region_text(a, Region(file_path="src/a.py", start_line=2, end_line=3)) == "line2\nline3"


def test_an_unsealed_region_is_reported(files):
    (problem,) = verify_regions(
        questions(relevant=[{**GOOD["relevant"][0], "file_path": "src/a.py"}]), files
    )
    assert "not sealed" in problem.problem


def test_sealing_pins_the_text_and_verification_then_passes(files):
    sealed = seal_regions(
        questions(relevant=[{"file_path": "src/a.py", "start_line": 2, "end_line": 3}]), files
    )
    assert sealed[0].relevant[0].sha256 == sha256_text("line2\nline3")
    assert verify_regions(sealed, files) == []


def test_changed_source_text_is_detected(repo, files):
    sealed = seal_regions(
        questions(relevant=[{"file_path": "src/a.py", "start_line": 2, "end_line": 3}]), files
    )
    (repo / "src/a.py").write_text("line1\nCHANGED\nline3\nline4\n", newline="\n")
    (problem,) = verify_regions(sealed, ingest_repository(repo).files)
    assert "differs" in problem.problem


def test_a_shifted_region_is_detected(repo, files):
    sealed = seal_regions(
        questions(relevant=[{"file_path": "src/a.py", "start_line": 2, "end_line": 3}]), files
    )
    (repo / "src/a.py").write_text("NEW\nline1\nline2\nline3\nline4\n", newline="\n")
    assert verify_regions(sealed, ingest_repository(repo).files)


def test_missing_file_and_out_of_range_lines_are_reported(files):
    missing = questions(relevant=[{"file_path": "src/none.py", "start_line": 1, "end_line": 2}])
    assert "not in the ingested" in verify_regions(missing, files)[0].problem
    beyond = questions(relevant=[{"file_path": "src/b.py", "start_line": 1, "end_line": 99}])
    assert "only 2 lines" in verify_regions(beyond, files)[0].problem


def test_sealing_refuses_regions_that_do_not_exist(files):
    with pytest.raises(BenchmarkFormatError, match="does not exist"):
        seal_regions(
            questions(relevant=[{"file_path": "src/b.py", "start_line": 1, "end_line": 99}]), files
        )


def test_the_region_hash_is_the_same_for_lf_and_crlf_checkouts(tmp_path):
    """Windows checkouts with CRLF must not invalidate a benchmark sealed on LF."""
    lf, crlf = tmp_path / "lf", tmp_path / "crlf"
    for root, data in ((lf, b"a\nb\nc\n"), (crlf, b"a\r\nb\r\nc\r\n")):
        root.mkdir()
        (root / "m.py").write_bytes(data)
    q = questions(relevant=[{"file_path": "m.py", "start_line": 2, "end_line": 3}])
    sealed = seal_regions(q, ingest_repository(lf).files)
    assert verify_regions(sealed, ingest_repository(crlf).files) == []


def test_benchmark_container_holds_questions_and_meta():
    assert Benchmark(questions=tuple(questions())).meta is None
