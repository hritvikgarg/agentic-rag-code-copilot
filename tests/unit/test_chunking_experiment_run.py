"""Scoring, budgets, paired comparison and the build/run pipeline (offline, fake embedder)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from copilot.config.settings import Settings
from copilot.embeddings import HashEmbedder
from copilot.evaluation.benchmark import Benchmark, BenchmarkQuestion, Region
from copilot.evaluation.chunking_experiment import analysis, builds, execute
from copilot.evaluation.chunking_experiment.artifacts import ArtifactError
from copilot.evaluation.chunking_experiment.config import ArmConfig, canonical_json
from copilot.evaluation.chunking_experiment.relevance import region_coverage
from copilot.evaluation.chunking_experiment.scoring import (
    BudgetScore,
    QuestionScore,
    score_budget,
    score_question,
)
from copilot.evaluation.metrics import QuestionOutcome, summarize
from copilot.retrieval import RetrievalResult


def _result(rank, path="a.py", start=1, end=10, tokens=50, chunk_id=None) -> RetrievalResult:
    text = "\n".join(f"line {n}" for n in range(start, end + 1))
    return RetrievalResult(
        rank=rank,
        score=1.0 / rank,
        position=rank - 1,
        chunk_id=chunk_id or f"{rank:016x}",
        repository_name="repo",
        file_path=path,
        language="python",
        chunk_type="line_window",
        chunk_index=rank - 1,
        start_line=start,
        end_line=end,
        token_estimate=tokens,
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        text=text,
    )


def _question(regions=(("a.py", 20, 30),)) -> BenchmarkQuestion:
    return BenchmarkQuestion(
        id="q1",
        question="where?",
        relevant=tuple(Region(file_path=f, start_line=s, end_line=e) for f, s, e in regions),
    )


# --- coverage ---------------------------------------------------------------------------------


def test_region_coverage_is_the_fraction_of_region_lines_inside_the_union_of_chunks():
    region = Region(file_path="a.py", start_line=11, end_line=20)  # 10 lines
    assert region_coverage([_result(1, "a.py", 1, 15)], [region]) == pytest.approx(0.5)
    union = [_result(1, "a.py", 1, 15), _result(2, "a.py", 14, 18)]
    assert region_coverage(union, [region]) == pytest.approx(0.8)
    assert region_coverage([_result(1, "b.py", 1, 99)], [region]) == 0.0  # other file
    assert region_coverage([], [region]) == 0.0
    assert region_coverage([_result(1, "a.py", 1, 99)], [region]) == 1.0


def test_coverage_pools_all_regions_of_a_question():
    regions = [Region(file_path="a.py", start_line=1, end_line=10),
               Region(file_path="b.py", start_line=1, end_line=30)]  # fmt: skip
    chunks = [_result(1, "a.py", 1, 10), _result(2, "b.py", 1, 10)]
    assert region_coverage(chunks, regions) == pytest.approx(20 / 40)


# --- scoring ----------------------------------------------------------------------------------


def test_first_hit_rank_follows_the_any_overlap_rule_and_the_top_10_window():
    pool = [_result(r, "a.py", 100 * r, 100 * r + 9) for r in range(1, 6)]
    pool.append(_result(6, "a.py", 25, 40))  # overlaps the region 20-30 by 6 lines
    s = score_question(pool, _question(), budgets=[512])
    assert s.first_hit_rank == 6 and s.hit_at(5) is False and s.hit_at(6) is True
    assert s.reciprocal_rank == pytest.approx(1 / 6)
    adjacent = score_question([_result(1, "a.py", 31, 40)], _question(), budgets=[512])
    assert adjacent.first_hit_rank is None  # sharing no line (30 vs 31) is not a hit
    beyond = [_result(r, "z.py", 1, 5) for r in range(1, 11)] + [_result(11, "a.py", 20, 30)]
    assert score_question(beyond, _question(), budgets=[512]).first_hit_rank is None


def test_hit_at_k_and_mrr_agree_with_the_existing_summarize_convention():
    pools = [
        [_result(1, "a.py", 20, 30)],
        [_result(1, "x.py", 1, 5), _result(2, "a.py", 20, 30)],
        [_result(r, "x.py", 1, 5) for r in range(1, 6)] + [_result(6, "a.py", 25, 26)],
        [_result(1, "x.py", 1, 5)],
    ]
    scores = [score_question(p, _question(), budgets=[512]) for p in pools]
    outcomes = [
        QuestionOutcome(f"q{i}", s.first_hit_rank, s.first_hit_lines, (), ())
        for i, s in enumerate(scores)
    ]
    reference = summarize(outcomes, (1, 3, 5, 10), depth=10)
    for k in (1, 3, 5, 10):
        assert sum(s.hit_at(k) for s in scores) == reference.hit_counts[k]
    assert sum(s.reciprocal_rank for s in scores) / len(scores) == pytest.approx(reference.mrr)


# --- fixed-budget simulation ------------------------------------------------------------------


def test_context_never_exceeds_the_budget_and_skips_non_fitting_chunks_then_continues():
    big = _result(1, "x.py", 1, 400, tokens=900)  # estimated cost far above the budget
    small = _result(2, "a.py", 20, 22, tokens=10)
    q = _question()
    for budget in (512, 1024, 2048):
        score = score_budget([big, small], q, budget)
        assert score.tokens <= budget
    tight = score_budget([big, small], q, 512)
    assert tight.included_chunk_ids == (small.chunk_id,)  # big is dropped whole, small still fits
    assert tight.dropped_for_budget == 1 and tight.hit is True
    assert 0 < tight.utilization <= 1


def test_a_chunk_is_never_truncated_to_fit_and_larger_budgets_include_at_least_as_much():
    pool = [_result(r, "a.py", 100 * r, 100 * r + 59) for r in range(1, 20)]
    included = [score_budget(pool, _question(), b).chunks for b in (512, 1024, 2048)]
    assert included == sorted(included) and included[0] < included[-1]


def test_duplicate_spans_are_deduplicated_by_the_production_builder():
    a = _result(1, "a.py", 20, 30)
    b = _result(2, "a.py", 20, 30, chunk_id="f" * 16)
    assert score_budget([a, b], _question(), 2048).chunks == 1


# --- paired comparison ------------------------------------------------------------------------


def _fake_scores(ranks):
    return [
        QuestionScore(
            question_id=f"q{i}",
            first_hit_rank=r,
            first_hit_lines=10 if r else None,
            coverage_at={5: 1.0 if r and r <= 5 else 0.0, 10: 1.0 if r else 0.0},
            retrieved=(),
            retrieved_tokens=100,
            retrieved_lines=(10,),
            distinct_files=2,
            budgets={512: BudgetScore(512, bool(r), 1.0 if r else 0.0, 3, 2, 300, 0.58, (), 0)},
        )
        for i, r in enumerate(ranks)
    ]


def _compare(scores):
    return analysis.compare_arms(
        scores,
        comparisons=[("X", "Y")],
        budgets=[512],
        coverage_threshold=0.5,
        primary={"comparison": ["X", "Y"], "endpoint": "hit@5", "alpha": 0.05},
        seed=11,
        resamples=500,
        confidence=0.95,
    )


def test_compare_arms_reports_the_paired_table_difference_and_mcnemar():
    x = _fake_scores([1, 1, 2, 3, 8, None, None, 4, 1, 1])
    y = _fake_scores([None, 1, 6, 3, 2, None, 2, None, 9, 1])
    row = _compare({"X": x, "Y": y})["comparisons"]["X-Y"]["binary"]["hit@5"]
    # X hits@5: q0..q3, q7, q8, q9 -> 7 ; Y hits@5: q1, q3, q4, q6, q9 -> 5
    assert (row["count_x"], row["count_y"]) == (7, 5)
    assert row["table"] == {"both": 3, "only_x": 4, "only_y": 2, "neither": 1}
    assert row["improved"] == 4 and row["regressed"] == 2
    assert row["difference_pp"] == pytest.approx(20.0)
    assert row["mcnemar_p"] == pytest.approx(2 * (1 + 6 + 15) / 64)  # n=6, tail 0..2
    assert row["ci_pp"][0] <= 20.0 <= row["ci_pp"][1]
    assert row["is_primary"] is True and row["holm_p"] is None


def test_non_primary_tests_get_holm_adjusted_p_values_and_mrr_gets_a_paired_ci():
    x = _fake_scores([1, 1, 2, 3, 8, None, None, 4, 1, 1])
    y = _fake_scores([None, 1, 6, 3, 2, None, 2, None, 9, 1])
    result = _compare({"X": x, "Y": y})
    row = result["comparisons"]["X-Y"]["binary"]["hit@1"]
    assert row["holm_p"] >= row["mcnemar_p"] and row["is_primary"] is False
    mrr = result["comparisons"]["X-Y"]["continuous"]["mrr@10"]
    expected = (sum(s.reciprocal_rank for s in x) - sum(s.reciprocal_rank for s in y)) / 10
    assert mrr["mean_difference"] == pytest.approx(expected)
    assert mrr["ci"][0] <= expected <= mrr["ci"][1]
    assert result["holm_family_size"] == len(result["comparisons"]["X-Y"]["binary"]) - 1


def test_compare_arms_is_deterministic_and_rejects_mismatched_questions():
    x = _fake_scores([1, 2, None, 4])
    y = _fake_scores([3, None, 2, 1])
    assert _compare({"X": x, "Y": y}) == _compare({"X": x, "Y": y})
    with pytest.raises(ValueError):
        _compare({"X": x, "Y": _fake_scores([1, 2, 3])})
    shuffled = [*reversed(y)]
    with pytest.raises(ValueError):
        _compare({"X": x, "Y": shuffled})


# --- build + run pipeline (fake embedder) ---------------------------------------------------


REPO = {
    "app/db.py": "".join(
        f'def connect_{i}(url):\n    """Open database connection number {i}."""\n'
        f"    pool = make_pool(url, {i})\n    return pool\n\n\n"
        for i in range(8)
    ),
    "app/auth.py": "".join(
        f'def check_token_{i}(token):\n    """Validate authentication token {i}."""\n'
        f"    return verify(token, {i})\n\n\n"
        for i in range(8)
    ),
    "README.md": "# Demo\n\nThis project connects to a database.\n" * 6,
}


def _arms() -> list[ArmConfig]:
    base = {"size_lines": 12, "overlap_lines": 2, "max_tokens": 200}
    ast = {"max_tokens": 200, "fallback_size_lines": 12, "fallback_overlap_lines": 2,
           "fallback": "line@1"}  # fmt: skip
    return [
        ArmConfig("A", "baseline", "line", 1, base),
        ArmConfig("B", "ast", "ast", 1, ast),
        ArmConfig("C", "sized", "line_sized", 1, {"python_size_lines": 6,
                                                  "python_overlap_lines": 0, **base}),
        ArmConfig("D", "merged", "ast_merged", 1, {**ast, "merge_max_tokens": 100,
                                                  "base": "ast@1"}),
    ]  # fmt: skip


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for rel, text in REPO.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    return root


def _config(repo: Path) -> dict:
    return {
        "snapshot": {"tree": str(repo)},
        "budget": {"pool_depth": 12, "budgets": [200, 400]},
        "coverage": {"threshold": 0.5},
        "latency": {"depth": 10, "warmup_queries": 1, "repetitions": 2},
        "statistics": {
            "seed": 5,
            "bootstrap_resamples": 200,
            "confidence": 0.95,
            "primary": {"comparison": ["B", "A"], "endpoint": "hit@5", "alpha": 0.05},
            "comparisons": [["B", "A"], ["B", "C"], ["C", "A"], ["D", "A"], ["D", "B"]],
        },
    }


def _benchmark() -> Benchmark:
    return Benchmark(
        questions=(
            BenchmarkQuestion(
                id="q-db",
                question="open database connection pool",
                relevant=(Region(file_path="app/db.py", start_line=1, end_line=4),),
            ),
            BenchmarkQuestion(
                id="q-auth",
                question="validate authentication token",
                relevant=(Region(file_path="app/auth.py", start_line=1, end_line=3),),
            ),
        )
    )


@pytest.fixture
def built(tmp_path):
    repo = _repo(tmp_path)
    settings = Settings(_env_file=None)
    embedder = HashEmbedder(dimension=64)
    results_dir = tmp_path / "results"
    index_root = tmp_path / "exp-indexes"
    records = builds.build_all(
        _arms(),
        tree=str(repo),
        repository_name="repo",
        ignore_directories=(),
        settings=settings,
        embedder=embedder,
        index_root=index_root,
        results_dir=results_dir,
        protected=[tmp_path / "protected"],
        progress=lambda _m: None,
    )
    return repo, settings, embedder, results_dir, index_root, records


def test_build_all_records_every_arm_in_its_own_directory_with_distinct_ids(built):
    repo, _, _, results_dir, index_root, records = built
    assert [r["arm_id"] for r in records] == ["A", "B", "C", "D"]
    assert len({r["index_id"] for r in records}) == 4
    for r in records:
        assert (index_root / r["index_path"] / "manifest.json").is_file()
        assert r["vector_count"] == r["chunk_count"] > 0 and r["dimension"] == 64
        assert r["bytes"]["index_total"] > 0 and r["model_cold_start_seconds"] >= 0
        assert set(r["seconds"]) == {
            "ingest",
            "chunk",
            "embed",
            "index_and_save",
            "total",
            "embed_wall_final_call",
        }
        assert r["embedding_mode"] == "single-call"
        assert (results_dir / "builds" / f"{r['arm_id']}.json").is_file()
    assert len({r["repository_fingerprint"] for r in records}) == 1  # same repository for all
    assert {r["strategy"] for r in records} == {"line", "ast", "line_sized", "ast_merged"}


def test_a_second_build_refuses_to_overwrite_and_a_protected_root_is_rejected(built, tmp_path):
    repo, settings, embedder, results_dir, index_root, _ = built
    common = dict(
        tree=str(repo), repository_name="repo", ignore_directories=(), settings=settings,
        embedder=embedder, progress=lambda _m: None,
    )  # fmt: skip
    with pytest.raises(ArtifactError):
        builds.build_all(_arms(), index_root=index_root, results_dir=results_dir,
                         protected=[tmp_path / "protected"], **common)  # fmt: skip
    with pytest.raises(ArtifactError):
        builds.build_all(_arms(), index_root=tmp_path / "data" / "indexes" / "x",
                         results_dir=tmp_path / "r2", protected=[tmp_path / "data" / "indexes"],
                         **common)  # fmt: skip
    with pytest.raises(ArtifactError):
        builds.build_all(_arms(), index_root=tmp_path / "data", results_dir=tmp_path / "r3",
                         protected=[tmp_path / "data" / "indexes"], **common)  # fmt: skip


def test_full_run_produces_consistent_results_and_leaves_inputs_untouched(built, tmp_path):
    repo, settings, embedder, results_dir, index_root, records = built

    def digest(root: Path) -> str:
        h = hashlib.sha256()
        for p in sorted(root.rglob("*")):
            if p.is_file():
                h.update(p.relative_to(root).as_posix().encode() + p.read_bytes())
        return h.hexdigest()

    before = (digest(repo), digest(index_root))
    config = _config(repo)
    by_arm = {r["arm_id"]: r for r in records}
    retrievers = execute.open_retrievers(
        config, by_arm, index_root=index_root, embedder=embedder, settings=settings, ignores=()
    )
    results = execute.run_experiment(
        config, by_arm, _benchmark(), retrievers, progress=lambda _m: None
    )
    assert (digest(repo), digest(index_root)) == before  # retrieval is read-only

    assert set(results["per_question"]) == {"A", "B", "C", "D"}
    for rows in results["per_question"].values():
        assert [r["question_id"] for r in rows] == ["q-db", "q-auth"]
        for row in rows:
            assert len(row["retrieved"]) <= 10
            assert all(b["tokens"] <= int(k) for k, b in row["budgets"].items())
    assert results["latency"]["ranking_mismatches_vs_scoring_pass"] == 0
    assert results["latency"]["summary"]["A"]["samples"] == 2 * 2  # repetitions x questions
    agg = results["aggregates"]["arms"]["A"]
    assert agg["n_questions"] == 2 and "hit@5" in agg["binary"]
    assert results["paired"]["primary_key"] == "B-A/hit@5"
    assert set(results["aggregates"]["efficiency"]) == {"A", "B", "C", "D"}
    json.dumps(results)  # fully JSON-serialisable

    written = execute.write_results(
        results_dir, results, config_sha256="0" * 64, started_utc="t0", finished_utc="t1"
    )
    assert {p.name for p in written} == set(execute.RESULT_FILES)
    with pytest.raises(ArtifactError):
        execute.write_results(
            results_dir, results, config_sha256="0" * 64, started_utc="t0", finished_utc="t1"
        )
    again = execute.write_results(
        results_dir, results, config_sha256="0" * 64, started_utc="t0", finished_utc="t1",
        overwrite=True,
    )  # fmt: skip
    assert len(again) == len(written)
    assert not [p for p in results_dir.iterdir() if p.name.endswith(".tmp")]


def test_every_arm_is_scored_on_identical_questions_and_the_run_is_deterministic(built):
    repo, settings, embedder, _, index_root, records = built
    by_arm = {r["arm_id"]: r for r in records}
    config = _config(repo)

    def once():
        retrievers = execute.open_retrievers(
            config, by_arm, index_root=index_root, embedder=embedder, settings=settings,
            ignores=(),
        )  # fmt: skip
        scores, ids = execute.retrieval_pass(
            retrievers, _benchmark(), pool_depth=12, budgets=[200, 400], progress=lambda _m: None
        )
        return json.dumps({a: [s.to_json() for s in v] for a, v in scores.items()}), ids

    first, second = once(), once()
    assert first == second


def test_missing_build_records_are_reported_by_arm(tmp_path):
    (tmp_path / "builds").mkdir()
    (tmp_path / "builds" / "A.json").write_text(canonical_json({"arm_id": "A"}))
    with pytest.raises(ArtifactError, match="B"):
        builds.load_build_records(tmp_path, ["A", "B"])


# --- design verification and CLI guard rails ------------------------------------------------


def test_verify_design_rejects_a_modified_frozen_config(tmp_path):
    config = tmp_path / "frozen-config.json"
    config.write_text(canonical_json({"schema": "chunking-experiment-config/1", "arms": []}))
    sha = tmp_path / "frozen-config.sha256"
    sha.write_text(hashlib.sha256(config.read_bytes()).hexdigest() + "  frozen-config.json\n")
    assert execute.verify_design(config, sha)["schema"] == "chunking-experiment-config/1"
    config.write_text(canonical_json({"schema": "chunking-experiment-config/1", "arms": [1]}))
    from copilot.evaluation.chunking_experiment.config import ExperimentConfigError

    with pytest.raises(ExperimentConfigError):
        execute.verify_design(config, sha)


def test_prepare_rejects_a_benchmark_changed_after_freezing(tmp_path):
    from copilot.evaluation.chunking_experiment.config import ExperimentConfigError

    bench = tmp_path / "b.jsonl"
    bench.write_text("{}\n")
    config = {"benchmark": {"path": str(bench), "sha256": "0" * 64, "questions": 1}}
    with pytest.raises(ExperimentConfigError):
        execute.prepare(config, Settings(_env_file=None))


def test_cli_forces_offline_mode_and_fails_cleanly_without_a_design(tmp_path, monkeypatch, capsys):
    import os

    from copilot.evaluation.chunking_experiment.__main__ import main

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    code = main(["run", "--design-dir", str(tmp_path / "none"), "--results-dir", str(tmp_path)])
    assert code == 2 and "frozen" in capsys.readouterr().err
    assert os.environ["HF_HUB_OFFLINE"] == "1"


# --- resumable embedding and phased execution -----------------------------------------------


class _Clock:
    """Deterministic ``perf_counter`` replacement: time only moves when told to."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _SlowEmbedder:
    """Wraps an embedder; every document batch 'takes' ``cost`` seconds of fake time."""

    def __init__(self, inner, clock, cost=10.0):
        self.inner, self.clock, self.cost, self.calls = inner, clock, cost, 0

    @property
    def info(self):
        return self.inner.info

    def embed_query(self, text):
        return self.inner.embed_query(text)

    def embed_documents(self, texts):
        self.calls += 1
        self.clock.now += self.cost
        return self.inner.embed_documents(texts)


def test_resumable_embedder_returns_the_same_vectors_as_one_continuous_call(tmp_path):
    import numpy as np

    from copilot.evaluation.chunking_experiment.embedding_cache import ResumableEmbedder

    inner = HashEmbedder(dimension=32)
    texts = [f"def f{i}(): return {i} database token" for i in range(11)]
    resumable = ResumableEmbedder(inner, tmp_path / "cache", batch_size=4)
    assert np.array_equal(resumable.embed_documents(texts), inner.embed_documents(texts))
    assert resumable.batches_computed == 3 and resumable.batches_loaded == 0
    again = ResumableEmbedder(inner, tmp_path / "cache", batch_size=4)
    assert np.array_equal(again.embed_documents(texts), inner.embed_documents(texts))
    assert again.batches_computed == 0 and again.batches_loaded == 3
    assert again.recorded_seconds == pytest.approx(resumable.recorded_seconds)
    assert resumable.embed_documents([]).shape == (0, 32)


def test_time_budget_pauses_and_a_rerun_continues_without_recomputing(tmp_path, monkeypatch):
    import numpy as np

    from copilot.evaluation.chunking_experiment import embedding_cache as ec

    clock = _Clock()
    monkeypatch.setattr(ec, "_now", clock)
    inner = HashEmbedder(dimension=32)
    slow = _SlowEmbedder(inner, clock)
    texts = [f"text number {i} about databases" for i in range(10)]  # batch size 2 -> 5 batches

    first = ec.ResumableEmbedder(slow, tmp_path / "c", batch_size=2, time_budget_seconds=15)
    with pytest.raises(ec.TimeBudgetExceeded) as paused:
        first.embed_documents(texts)
    assert (paused.value.done_batches, paused.value.total_batches) == (2, 5)
    assert slow.calls == 2

    second = ec.ResumableEmbedder(slow, tmp_path / "c", batch_size=2, time_budget_seconds=15)
    with pytest.raises(ec.TimeBudgetExceeded) as paused2:
        second.embed_documents(texts)
    assert paused2.value.done_batches == 4 and slow.calls == 4  # two loaded, two new

    third = ec.ResumableEmbedder(slow, tmp_path / "c", batch_size=2, time_budget_seconds=15)
    vectors = third.embed_documents(texts)
    assert np.array_equal(vectors, inner.embed_documents(texts)) and slow.calls == 5
    assert third.recorded_seconds == pytest.approx(5 * 10.0)  # measured time of all five batches


def test_a_changed_text_invalidates_only_its_cached_batch(tmp_path):
    from copilot.evaluation.chunking_experiment.embedding_cache import ResumableEmbedder

    inner = HashEmbedder(dimension=32)
    texts = [f"alpha {i}" for i in range(6)]
    ResumableEmbedder(inner, tmp_path / "c", batch_size=2).embed_documents(texts)
    texts[3] = "completely different"
    run = ResumableEmbedder(inner, tmp_path / "c", batch_size=2)
    run.embed_documents(texts)
    assert (run.batches_loaded, run.batches_computed) == (2, 1)


def test_a_resumable_build_produces_the_same_index_as_a_single_call_build(
    built, tmp_path, monkeypatch
):
    import numpy as np

    from copilot.evaluation.chunking_experiment.embedding_cache import TimeBudgetExceeded
    from copilot.vectorstore import IndexExpectation, VectorIndex

    repo, settings, _, _, index_root, single = built
    import copilot.evaluation.chunking_experiment.embedding_cache as ec

    clock = _Clock()
    monkeypatch.setattr(ec, "_now", clock)
    slow = _SlowEmbedder(HashEmbedder(dimension=64), clock, cost=40.0)
    kwargs = dict(
        tree=str(repo), repository_name="repo", ignore_directories=(),
        settings=settings.model_copy(update={"embedding_batch_size": 4}),
        embedder=slow, index_root=tmp_path / "resumable", results_dir=tmp_path / "res2",
        protected=[tmp_path / "protected"], only=["A"], progress=lambda _m: None,
        cache_root=tmp_path / "cache", time_budget_seconds=60.0,
    )  # fmt: skip
    pauses = 0
    record = None
    while record is None:
        try:
            record = builds.build_all(_arms(), **kwargs)[0]
        except TimeBudgetExceeded:
            pauses += 1
            clock.now = 0.0  # a new process starts with a fresh clock
            assert pauses < 20
    assert pauses >= 1 and record["embedding_mode"] == "resumable-cache"
    reference = next(r for r in single if r["arm_id"] == "A")
    assert record["chunk_ids_sha256"] == reference["chunk_ids_sha256"]
    one = VectorIndex.load(index_root / reference["index_path"], expected=IndexExpectation())
    two = VectorIndex.load(
        tmp_path / "resumable" / record["index_path"], expected=IndexExpectation()
    )
    assert one.count == two.count
    assert all(np.array_equal(one.reconstruct(i), two.reconstruct(i)) for i in range(one.count))
    assert record["seconds"]["embed"] == pytest.approx(40.0 * -(-record["chunk_count"] // 4))


def test_phased_run_equals_the_single_process_run(built):
    repo, settings, embedder, _, index_root, records = built
    by_arm = {r["arm_id"]: r for r in records}
    config = _config(repo)
    retrievers = execute.open_retrievers(
        config, by_arm, index_root=index_root, embedder=embedder, settings=settings, ignores=()
    )
    per_question = execute.retrieval_phase(
        config, _benchmark(), retrievers, progress=lambda _m: None
    )
    ids = execute.top_chunk_ids(per_question)
    parts = [
        execute.latency_repetition(retrievers, _benchmark(), ids, repetition=r, depth=10, warmup=1)
        for r in range(2)
    ]
    assert [p["arm_order"] for p in parts] == [["A", "B", "C", "D"], ["B", "C", "D", "A"]]
    assert all(p["ranking_mismatches"] == 0 for p in parts)
    phased = execute.assemble_results(config, by_arm, per_question, parts)
    whole = execute.run_experiment(
        config, by_arm, _benchmark(), retrievers, progress=lambda _m: None
    )
    assert phased["per_question"] == whole["per_question"]
    assert phased["paired"] == whole["paired"] and phased["aggregates"] == whole["aggregates"]
    assert phased["latency"]["summary"]["A"]["samples"] == 4
