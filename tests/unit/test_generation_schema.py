"""Milestone 11A: generation-benchmark schema, validation, sealing and the two frozen benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from copilot.config import Settings
from copilot.evaluation import BenchmarkFormatError, load_benchmark
from copilot.evaluation.generation.schema import (
    Answerability,
    GenerationQuestion,
    KeyFact,
    Region,
    benchmark_digest,
    dump_generation_benchmark,
    load_generation_benchmark,
    seal_generation_regions,
    verify_generation_regions,
)
from copilot.ingestion import IngestionPolicy, ingest_repository
from copilot.vectorstore import repository_fingerprint

ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "generation-eval" / "benchmarks"
COPILOT = BENCH_DIR / "copilot_generation_055a8d5.jsonl"
INVENTORY = BENCH_DIR / "inventory_orders_generation.jsonl"
INV_ROOT = ROOT / "generation-eval" / "repos" / "inventory-orders"
FREEZE = ROOT / "generation-eval" / "repos" / "inventory-orders.freeze.json"
RETRIEVAL = ROOT / "benchmarks" / "copilot_self_055a8d5.jsonl"
SNAPSHOT_TREE = (
    ROOT / "data" / "snapshots" / "a282ec471ebf1387cb4feafcfe732ad24a431c7a" / "tree"
)  # git-ignored: absent on a fresh clone

POLICY = IngestionPolicy.from_settings(Settings(_env_file=None))


def _region(path: str = "src/a.py", start: int = 1, end: int = 2) -> Region:
    return Region(file_path=path, start_line=start, end_line=end)


def _answerable(**over) -> GenerationQuestion:
    base = {
        "id": "q1",
        "repository_id": "demo",
        "question": "What does f do?",
        "category": "implementation",
        "answerability": Answerability.ANSWERABLE,
        "key_facts": (KeyFact(id="f1", statement="f returns one value", regions=(_region(),)),),
        "acceptable_answer": "says f returns one",
    }
    base.update(over)
    return GenerationQuestion(**base)


def _unanswerable(**over) -> GenerationQuestion:
    base = {
        "id": "u1",
        "repository_id": "demo",
        "question": "Who owns it?",
        "category": "process",
        "answerability": Answerability.UNANSWERABLE,
        "unanswerable_rationale": "not in the repository",
        "expected_abstention": "says the repository does not show it",
    }
    base.update(over)
    return GenerationQuestion(**base)


# --------------------------------------------------------------------------- question rules


def test_a_valid_answerable_and_unanswerable_question():
    assert _answerable().is_answerable
    assert not _unanswerable().is_answerable


def test_answerable_needs_key_facts_and_an_acceptable_answer():
    with pytest.raises(ValueError, match="key fact"):
        _answerable(key_facts=())
    with pytest.raises(ValueError, match="acceptable_answer"):
        _answerable(acceptable_answer=" ")


def test_answerable_must_not_carry_abstention_fields():
    with pytest.raises(ValueError, match="abstention"):
        _answerable(unanswerable_rationale="x")


def test_unanswerable_rules():
    with pytest.raises(ValueError, match="cannot have key facts"):
        _unanswerable(key_facts=_answerable().key_facts)
    with pytest.raises(ValueError, match="rationale"):
        _unanswerable(unanswerable_rationale="")
    with pytest.raises(ValueError, match="general_knowledge_sufficient"):
        _unanswerable(general_knowledge_sufficient=True)


def test_duplicate_fact_ids_and_bad_ids_are_rejected():
    fact = KeyFact(id="f1", statement="some real statement", regions=(_region(),))
    with pytest.raises(ValueError, match="duplicate key-fact"):
        _answerable(key_facts=(fact, fact))
    with pytest.raises(ValueError):
        _answerable(id="Bad Id")


def test_question_text_must_be_clean_and_non_empty():
    with pytest.raises(ValueError):
        _answerable(question="   ")
    with pytest.raises(ValueError, match="whitespace"):
        _answerable(question=" padded ")


@pytest.mark.parametrize("bad", ["../escape.py", "/abs.py", "a\\b.py", "a/../b.py"])
def test_regions_reject_unsafe_paths(bad):
    with pytest.raises(ValueError):
        _region(path=bad)


def test_region_order_is_checked():
    with pytest.raises(ValueError):
        _region(start=5, end=3)


def test_relevant_regions_is_the_deduplicated_union_of_fact_regions():
    facts = (
        KeyFact(id="f1", statement="first statement", regions=(_region(), _region(end=9))),
        KeyFact(id="f2", statement="second statement", regions=(_region(),)),
    )
    q = _answerable(key_facts=facts)
    assert [(r.start_line, r.end_line) for r in q.relevant_regions] == [(1, 2), (1, 9)]


# --------------------------------------------------------------------------- file loading


def _write(tmp_path: Path, questions) -> Path:
    path = tmp_path / "bench.jsonl"
    path.write_text(dump_generation_benchmark(questions), encoding="utf-8")
    return path


def test_roundtrip_counts_and_digest(tmp_path):
    path = _write(tmp_path, [_answerable(), _unanswerable()])
    bench = load_generation_benchmark(path)
    assert bench.counts() == {"total": 2, "answerable": 1, "unanswerable": 1}
    assert bench.repository_id == "demo"
    assert benchmark_digest(bench) == benchmark_digest(load_generation_benchmark(path))
    changed = _write(tmp_path, [_answerable(question="What does g do?"), _unanswerable()])
    assert benchmark_digest(load_generation_benchmark(changed)) != benchmark_digest(bench)


def test_duplicate_ids_texts_and_mixed_repositories_are_rejected(tmp_path):
    with pytest.raises(BenchmarkFormatError, match="duplicate question ids"):
        load_generation_benchmark(_write(tmp_path, [_answerable(), _answerable(question="Other?")]))
    with pytest.raises(BenchmarkFormatError, match="duplicate question text"):
        load_generation_benchmark(_write(tmp_path, [_answerable(), _answerable(id="q2")]))
    with pytest.raises(BenchmarkFormatError, match="exactly one repository_id"):
        load_generation_benchmark(
            _write(tmp_path, [_answerable(), _unanswerable(repository_id="other")])
        )


def test_unknown_fields_and_bad_json_are_rejected(tmp_path):
    path = tmp_path / "b.jsonl"
    record = json.loads(dump_generation_benchmark([_answerable()]))
    record["surprise"] = 1
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(BenchmarkFormatError, match="line 1"):
        load_generation_benchmark(path)
    path.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(BenchmarkFormatError, match="line 1"):
        load_generation_benchmark(path)
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(BenchmarkFormatError, match="no questions"):
        load_generation_benchmark(path)


def test_meta_repository_must_match(tmp_path):
    path = _write(tmp_path, [_answerable()])
    meta = {
        "name": "x",
        "repository_id": "other",
        "repository_name": "other",
        "source_kind": "synthetic-authored",
    }
    path.with_name("bench.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(BenchmarkFormatError, match="meta repository_id"):
        load_generation_benchmark(path)


# --------------------------------------------------------------------------- sealing


def _tiny_repo(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("def f():\n    return 1\n\nx = 2\n", encoding="utf-8")
    return ingest_repository(repo, POLICY, repository_name="demo").files


def test_seal_then_verify_and_detect_drift(tmp_path):
    files = _tiny_repo(tmp_path)
    question = _answerable()
    unsealed = load_generation_benchmark(_write(tmp_path, [question, _unanswerable()]))
    problems = verify_generation_regions(unsealed, files)
    assert any("not sealed" in p.problem for p in problems)

    sealed = seal_generation_regions([question, _unanswerable()], files)
    assert all(r.sha256 for r in sealed[0].key_facts[0].regions)
    sealed_bench = load_generation_benchmark(_write(tmp_path, sealed))
    assert verify_generation_regions(sealed_bench, files) == []

    (tmp_path / "repo" / "src" / "a.py").write_text("def f():\n    return 99\n", encoding="utf-8")
    drifted = ingest_repository(tmp_path / "repo", POLICY, repository_name="demo").files
    assert any("differs" in p.problem for p in verify_generation_regions(sealed_bench, drifted))


def test_sealing_a_missing_file_or_range_fails(tmp_path):
    files = _tiny_repo(tmp_path)
    missing = _answerable(
        key_facts=(
            KeyFact(id="f1", statement="refers to nothing", regions=(_region("src/zzz.py"),)),
        )
    )
    with pytest.raises(BenchmarkFormatError):
        seal_generation_regions([missing], files)
    too_long = _answerable(
        key_facts=(KeyFact(id="f1", statement="beyond the file", regions=(_region(end=500),)),)
    )
    with pytest.raises(BenchmarkFormatError):
        seal_generation_regions([too_long], files)


# --------------------------------------------------------------------------- the real benchmarks


def test_copilot_benchmark_shape():
    bench = load_generation_benchmark(COPILOT)
    assert bench.counts() == {"total": 42, "answerable": 34, "unanswerable": 8}
    assert bench.repository_id == "copilot"
    assert bench.meta is not None and bench.meta.independent is False
    assert 0.15 <= bench.counts()["unanswerable"] / bench.counts()["total"] <= 0.25
    assert all(r.sha256 for q in bench.questions for f in q.key_facts for r in f.regions)
    assert all(not q.general_knowledge_sufficient for q in bench.questions if not q.is_answerable)


def test_copilot_answerable_questions_preserve_the_retrieval_benchmark():
    original = {q.id: q for q in load_benchmark(RETRIEVAL).questions}
    bench = load_generation_benchmark(COPILOT)
    answerable = [q for q in bench.questions if q.is_answerable]
    assert {q.id for q in answerable} == set(original)
    for q in answerable:
        assert q.question == original[q.id].question  # the exact text, same for both arms
        allowed = original[q.id].relevant
        for region in q.relevant_regions:
            assert any(
                region.file_path == a.file_path
                and a.start_line <= region.start_line
                and region.end_line <= a.end_line
                for a in allowed
            ), f"{q.id}: fact region outside the original ground truth"


def test_inventory_benchmark_shape():
    bench = load_generation_benchmark(INVENTORY)
    assert bench.counts() == {"total": 30, "answerable": 24, "unanswerable": 6}
    assert bench.repository_id == "inventory-orders"
    assert bench.meta is not None and "Synthetic" in bench.meta.limitations[0]
    assert len(bench.category_counts()) >= 6
    assert all(r.sha256 for q in bench.questions for f in q.key_facts for r in f.regions)


def test_inventory_benchmark_matches_the_frozen_repository():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    ingestion = ingest_repository(INV_ROOT, POLICY, repository_name="inventory-orders")
    assert repository_fingerprint(ingestion.files) == freeze["repository_fingerprint"]
    assert len(ingestion.files) == freeze["ingested_file_count"]
    bench = load_generation_benchmark(INVENTORY)
    assert bench.meta is not None
    assert bench.meta.repository_fingerprint == freeze["repository_fingerprint"]
    assert verify_generation_regions(bench, ingestion.files) == []


@pytest.mark.skipif(not SNAPSHOT_TREE.is_dir(), reason="frozen snapshot not present (git-ignored)")
def test_copilot_benchmark_matches_the_frozen_snapshot():
    policy = POLICY.with_extra_ignored_directories("data", "tests", "submission")
    files = ingest_repository(
        SNAPSHOT_TREE, policy, repository_name="agentic-rag-code-copilot"
    ).files
    assert verify_generation_regions(load_generation_benchmark(COPILOT), files) == []


def test_no_unanswerable_question_text_collides_with_an_answerable_one():
    for path in (COPILOT, INVENTORY):
        bench = load_generation_benchmark(path)
        texts = [q.question.casefold() for q in bench.questions]
        assert len(texts) == len(set(texts))
