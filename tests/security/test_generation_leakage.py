"""Milestone 11A: benchmark answers must never be reachable through repository ingestion.

The plain-vs-RAG comparison is only meaningful if the RAG arm cannot retrieve the answer key.
Key facts, acceptable answers and rationales live in ``generation-eval/benchmarks/*.jsonl`` which
(1) sits outside every repository root that is indexed, (2) has an extension ingestion does not
accept, and (3) is re-checked here by content: no ingested file may contain a key-fact statement.
"""

from __future__ import annotations

from pathlib import Path

from copilot.config import Settings
from copilot.evaluation.generation.schema import GenerationBenchmark, load_generation_benchmark
from copilot.ingestion import IngestionPolicy, ingest_repository

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "generation-eval"
BENCHMARKS = sorted((EVAL / "benchmarks").glob("*.jsonl"))
INV_ROOT = EVAL / "repos" / "inventory-orders"
POLICY = IngestionPolicy.from_settings(Settings(_env_file=None))


def _answer_key_strings(bench: GenerationBenchmark) -> list[str]:
    strings: list[str] = []
    for q in bench.questions:
        strings += [f.statement for f in q.key_facts]
        strings += [
            s for s in (q.acceptable_answer, q.unanswerable_rationale, q.expected_abstention) if s
        ]
    return strings


def test_benchmarks_exist_and_are_outside_every_repository_root():
    assert len(BENCHMARKS) == 2
    for path in BENCHMARKS:
        assert INV_ROOT not in path.parents
        assert "src" not in path.relative_to(ROOT).parts


def test_the_benchmark_extension_is_not_ingestible():
    assert POLICY.language_for("copilot_generation_055a8d5.jsonl") is None


def test_the_synthetic_repository_contains_no_benchmark_or_answer_key_files():
    names = [p.name for p in INV_ROOT.rglob("*") if p.is_file()]
    assert not any(n.endswith((".jsonl", ".meta.json")) for n in names)
    assert not any("benchmark" in n.lower() or "rubric" in n.lower() for n in names)


def test_no_ingested_file_anywhere_under_generation_eval_contains_an_answer_key_string():
    keys = [s for path in BENCHMARKS for s in _answer_key_strings(load_generation_benchmark(path))]
    assert len(keys) > 100
    ingestion = ingest_repository(EVAL, POLICY, repository_name="generation-eval")
    assert ingestion.files, "expected the synthetic repository to be ingested"
    for file in ingestion.files:
        for key in keys:
            assert key not in file.content, f"{file.relative_path} leaks an answer-key string"


def test_the_ingested_synthetic_repository_never_includes_benchmark_paths():
    files = ingest_repository(INV_ROOT, POLICY, repository_name="inventory-orders").files
    assert all(not f.relative_path.startswith(("benchmarks/", "generation-eval/")) for f in files)
    assert all(f.extension != ".jsonl" for f in files)
