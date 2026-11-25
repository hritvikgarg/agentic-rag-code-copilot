"""Milestone 11A: reuse checks for historical comparison responses (read-only, no provider)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from copilot.evaluation.comparison import (
    ComparisonEntry,
    ComparisonRun,
    PlainRecord,
    RagRecord,
    SourceRecord,
)
from copilot.evaluation.generation.compat import (
    PINNED_PROMPT_DIGESTS,
    check_historical_run,
    frozen_target,
)
from copilot.evaluation.generation.schema import load_generation_benchmark
from copilot.rag import prompts

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL = ROOT / "data/evaluation/compare-8192-4aaeab9c650ab76c.json"
INDEX = ROOT / "data/indexes/4aaeab9c650ab76c"
BENCH = next((ROOT / "generation-eval/benchmarks").glob("copilot_generation_*.jsonl"), None)


def test_pinned_digests_match_the_prompt_texts_in_code():
    for version, text in (
        ("plain-v1", prompts.PLAIN_SYSTEM_PROMPT),
        ("plain-v2", prompts.PLAIN_SYSTEM_PROMPT_V2),
        ("rag-v1", prompts.RAG_SYSTEM_PROMPT),
    ):
        assert hashlib.sha256(text.encode()).hexdigest() == PINNED_PROMPT_DIGESTS[version]


# --------------------------------------------------------------------------- synthetic index


@pytest.fixture
def index_dir(tmp_path: Path) -> Path:
    """A tiny index directory: a real-format manifest plus chunks.jsonl (no vectors needed)."""
    from copilot.vectorstore.manifest import IndexSpec

    root = tmp_path / "idx"
    root.mkdir()
    chunks = [
        {"chunk_id": "c1", "file_path": "a.py", "start_line": 1, "end_line": 20},
        {"chunk_id": "c2", "file_path": "b.py", "start_line": 5, "end_line": 30},
    ]
    (root / "chunks.jsonl").write_text("\n".join(json.dumps(c) for c in chunks) + "\n")
    spec = IndexSpec(
        repository_name="repo",
        repository_fingerprint="f" * 64,
        chunking_strategy="line",
        chunking_strategy_version=1,
        chunk_params={"size_lines": 60, "overlap_lines": 10, "max_tokens": 512},
        chunk_id_schema="chunk-id/1",
        embedding_model_id="jinaai/jina-embeddings-v2-base-code",
        embedding_dimension=768,
        embedding_normalized=True,
        text_style="prefixed",
        representation_version=1,
        index_type="IndexFlatIP",
        metric="inner_product",
    )
    from copilot.vectorstore import manifest as m

    index_id = m.compute_index_id(spec) if hasattr(m, "compute_index_id") else "0" * 16
    manifest = {
        "schema_version": "vector-index-manifest/1",
        "index_id": index_id,
        "spec": spec.model_dump(),
        "provenance": {
            "created_at": "2026-01-01T00:00:00Z",
            "embedding_pooling": "mean",
            "embedding_runtime": "fastembed-onnx",
            "embedding_runtime_version": "0",
            "faiss_version": "0",
            "ingestion_truncated": False,
            "source_file_count": 2,
        },
        "vector_count": 2,
        "chunk_count": 2,
        "chunk_ids_sha256": "0" * 64,
        "artifacts": {},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    root.joinpath("INDEX_ID").write_text(index_id)
    return root


def _index_id(index_dir: Path) -> str:
    return (index_dir / "INDEX_ID").read_text()


def _entry(qid: str, *, plain_ok=True, rag_ok=True, chunk="c1", lines=(1, 20)) -> ComparisonEntry:
    source = SourceRecord(
        source_number=1, file_path="a.py", start_line=lines[0], end_line=lines[1],
        chunk_id=chunk, score=0.5, rank=1,
    )  # fmt: skip
    return ComparisonEntry(
        question_id=qid,
        question=f"text of {qid}",
        expected_regions=("a.py:1-2",),
        plain=PlainRecord(answer="plain" if plain_ok else None, error=None if plain_ok else "x"),
        rag=RagRecord(
            status="answered" if rag_ok else "error",
            answer="rag [Source 1]" if rag_ok else None,
            error=None if rag_ok else "x",
            sources=(source,) if rag_ok else (),
            retrieved_count=1 if rag_ok else 0,
        ),
    )


def _run(index_dir: Path, **over) -> ComparisonRun:
    base = dict(
        benchmark_name="b", repository_name="repo", index_id=_index_id(index_dir),
        llm_provider="gemini", llm_model="m", plain_prompt_version="plain-v1",
        rag_prompt_version="rag-v1", top_k=5, temperature=0.0, max_output_tokens=8192,
        entries=[_entry("q1"), _entry("q2", plain_ok=False, rag_ok=False)],
    )  # fmt: skip
    base.update(over)
    return ComparisonRun(**base)


def _target(index_dir: Path, **over):
    args = dict(
        questions={"q1": "text of q1", "q2": "text of q2"}, repository_name="repo",
        index_id=_index_id(index_dir), llm_provider="gemini", llm_model="m", max_output_tokens=8192,
    )  # fmt: skip
    args.update(over)
    return frozen_target(**args)


def _check(run, target, index_dir):
    return check_historical_run(run, target, source_name="t.json", index_dir=index_dir)


def test_rag_side_is_reusable_when_every_condition_holds(index_dir):
    report = _check(_run(index_dir), _target(index_dir), index_dir)
    assert report.reusable("rag") == ["q1"]  # q2 failed
    assert report.successful_sides == {"plain": 1, "rag": 1}


def test_plain_v1_is_never_reusable_in_a_plain_v2_experiment(index_dir):
    report = _check(_run(index_dir), _target(index_dir), index_dir)
    assert report.reusable("plain") == []
    reasons = next(v for v in report.verdicts if v.arm == "plain" and v.question_id == "q1").reasons
    assert any("plain-v1" in r and "plain-v2" in r for r in reasons)
    # ...but it would be reusable by a (hypothetical) plain-v1 protocol with identical settings
    assert _check(
        _run(index_dir), _target(index_dir, plain_prompt_version="plain-v1"), index_dir
    ).reusable("plain") == ["q1"]


@pytest.mark.parametrize(
    "target_change,run_change",
    [
        ({"llm_model": "other"}, {}),
        ({"llm_provider": "other"}, {}),
        ({"max_output_tokens": 1024}, {}),
        ({"temperature": 0.5}, {}),
        ({"top_k": 3}, {}),
        ({"repository_name": "other"}, {}),
        ({}, {"index_id": "0" * 16}),
        ({}, {"rag_prompt_version": "rag-v0"}),
    ],
)
def test_any_setting_difference_blocks_reuse(index_dir, target_change, run_change):
    report = _check(_run(index_dir, **run_change), _target(index_dir, **target_change), index_dir)
    assert report.reusable("rag") == []


def test_changed_question_text_blocks_only_that_question(index_dir):
    target = _target(index_dir, questions={"q1": "reworded", "q2": "text of q2"})
    report = _check(_run(index_dir), target, index_dir)
    assert report.reusable("rag") == []
    assert any("question text differs" in r for v in report.verdicts for r in v.reasons)


def test_drifted_prompt_text_blocks_reuse(index_dir):
    target = _target(index_dir)
    texts = dict(target.prompt_texts)
    texts["rag-v1"] += " edited"
    drifted = type(target)(**{**target.__dict__, "prompt_texts": texts})
    assert _check(_run(index_dir), drifted, index_dir).reusable("rag") == []


def test_provenance_must_match_the_index(index_dir):
    unknown = _run(index_dir, entries=[_entry("q1", chunk="nope")])
    assert _check(unknown, _target(index_dir), index_dir).reusable("rag") == []
    wrong_lines = _run(index_dir, entries=[_entry("q1", lines=(1, 99))])
    assert _check(wrong_lines, _target(index_dir), index_dir).reusable("rag") == []
    no_sources = _entry("q1").model_copy(
        update={"rag": RagRecord(status="answered", answer="a", sources=(), retrieved_count=0)}
    )
    assert (
        _check(_run(index_dir, entries=[no_sources]), _target(index_dir), index_dir).reusable("rag")
        == []
    )


def test_missing_index_means_nothing_is_reusable(index_dir, tmp_path):
    report = check_historical_run(
        _run(index_dir), _target(index_dir), source_name="t", index_dir=tmp_path / "absent"
    )
    assert report.reusable("rag") == []


def test_a_different_retriever_blocks_reuse(index_dir):
    target = _target(index_dir)
    other = type(target)(
        **{**target.__dict__, "retriever": {**target.retriever, "text_style": "plain"}}
    )
    assert _check(_run(index_dir), other, index_dir).reusable("rag") == []


# --------------------------------------------------------------------------- real historical files


@pytest.mark.skipif(
    not (HISTORICAL.is_file() and INDEX.is_dir() and BENCH is not None),
    reason="historical comparison data are not present",
)
def test_real_historical_8192_file_counts_and_files_are_untouched():
    before = hashlib.sha256(HISTORICAL.read_bytes()).hexdigest()
    benchmark = load_generation_benchmark(BENCH)
    run = ComparisonRun.model_validate_json(HISTORICAL.read_text(encoding="utf-8"))
    target = frozen_target(
        questions={q.id: q.question for q in benchmark.questions},
        repository_name="agentic-rag-code-copilot", index_id="4aaeab9c650ab76c",
        llm_provider=run.llm_provider, llm_model=run.llm_model, max_output_tokens=8192,
    )  # fmt: skip
    report = check_historical_run(run, target, source_name=HISTORICAL.name, index_dir=INDEX)
    assert report.successful_sides == {"plain": 9, "rag": 8}
    assert report.reusable("plain") == []  # plain-v1 vs plain-v2
    assert len(report.reusable("rag")) == 8  # every condition verified, incl. chunk provenance
    assert hashlib.sha256(HISTORICAL.read_bytes()).hexdigest() == before
    # the same file under a different confirmed model is not reusable at all
    other = frozen_target(
        questions=dict(target.questions), repository_name="agentic-rag-code-copilot",
        index_id="4aaeab9c650ab76c", llm_provider="gemini", llm_model="some-other-model",
        max_output_tokens=8192,
    )  # fmt: skip
    assert check_historical_run(run, other, source_name="x", index_dir=INDEX).reusable("rag") == []
