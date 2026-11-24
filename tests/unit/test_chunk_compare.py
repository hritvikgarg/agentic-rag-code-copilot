"""The descriptive baseline-vs-structure-aware comparison (no model, no retrieval)."""

from __future__ import annotations

import hashlib
import json

import pytest

from copilot.chunking.compare import (
    DEFINITIONS,
    compare_chunkers,
    format_comparison,
    python_definitions,
)
from copilot.chunking.compare import main as compare_main
from copilot.ingestion import IngestionPolicy, ingest_repository
from tests.ast_helpers import ast_chunker, line_chunker
from tests.fixtures.ast_demo_repo import FILES, write_demo_repo


@pytest.fixture
def demo(tmp_path):
    return write_demo_repo(tmp_path / "demo")


@pytest.fixture
def comparison(demo):
    ingestion = ingest_repository(demo, IngestionPolicy(), repository_name="demo")
    return compare_chunkers(ingestion, line_chunker(), ast_chunker(), max_tokens=512)


def _tree_hash(root):
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_python_definitions_follow_modules_and_classes_but_not_nested_functions():
    src = (
        "def top():\n    def inner():\n        pass\n\n\n"
        "@deco\nclass Outer:\n    def m(self):\n        pass\n\n"
        "    class In:\n        async def n(self):\n            pass\n"
    )
    found = [(d.kind, d.qualified_name, d.start_line, d.end_line) for d in python_definitions(src)]
    assert found == [
        ("function", "top", 1, 3),
        ("class", "Outer", 6, 13),
        ("function", "Outer.m", 8, 9),
        ("class", "Outer.In", 11, 13),
        ("function", "Outer.In.n", 12, 13),
    ]


def test_unparseable_python_has_no_ground_truth():
    assert python_definitions("def broken(:\n") is None


def test_file_and_definition_counts_match_the_synthetic_repository(comparison):
    assert comparison.files_total == len(FILES)
    assert comparison.files_python == 6
    assert comparison.python_files_unparseable == 1
    assert comparison.baseline.definitions_total == comparison.ast.definitions_total
    assert comparison.ast.definitions_total == 43  # independent of either chunker
    assert comparison.ast.definitions_oversized == 1  # shop/pipeline.py:run_pipeline


def test_baseline_has_no_symbols_and_the_ast_strategy_does(comparison):
    assert comparison.baseline.symbol_chunks_pct == 0.0
    assert comparison.baseline.symbol_lines_pct == 0.0
    assert comparison.ast.symbol_chunks_pct > 50.0
    assert comparison.baseline.segmentation == {}
    assert comparison.ast.segmentation["ast"] > 0 and comparison.ast.segmentation["fallback"] > 0


def test_neither_strategy_loses_source_lines(comparison):
    assert comparison.baseline.line_coverage_pct == 100.0
    assert comparison.ast.line_coverage_pct == 100.0


def test_structural_chunks_have_no_overlap_duplication(comparison):
    assert comparison.ast.overlap_duplicated_lines <= comparison.baseline.overlap_duplicated_lines
    assert comparison.baseline.overlap_duplicated_lines > 0


def test_fallback_reasons_and_split_counts_are_reported(comparison):
    assert comparison.ast.fallback_files == {"syntax_error": 1, "unsupported_language": 3}
    assert comparison.ast.oversized_definitions_split == 1
    assert comparison.baseline.fallback_files == {}
    assert comparison.baseline.oversized_definitions_split == 0


def test_definition_accounting_is_consistent(comparison):
    for profile in (comparison.baseline, comparison.ast):
        fitting = profile.definitions_total - profile.definitions_oversized
        assert profile.definitions_kept_whole + profile.definitions_cut == fitting


def test_token_statistics_are_ordered(comparison):
    for profile in (comparison.baseline, comparison.ast):
        assert profile.tokens_all_min <= profile.tokens_all_median <= profile.tokens_all_max <= 512
        assert (
            profile.tokens_python_min <= profile.tokens_python_median <= profile.tokens_python_max
        )


def test_examples_are_definitions_the_baseline_really_cut(comparison):
    assert comparison.examples, "the synthetic repository contains a definition the baseline cuts"
    for example in comparison.examples:
        assert (
            len(example.baseline_chunks_touching_it) >= 2
        )  # it spans more than one baseline chunk
    assert "ast chunk" in format_comparison(comparison)


def test_path_prefix_restricts_the_analysis(demo):
    ingestion = ingest_repository(demo, IngestionPolicy(), repository_name="demo")
    narrowed = compare_chunkers(
        ingestion, line_chunker(), ast_chunker(), max_tokens=512, path_prefixes=["shop/models"]
    )
    assert (narrowed.files_total, narrowed.files_python) == (1, 1)
    assert narrowed.ast.definitions_total == 3


def test_every_reported_statistic_is_defined():
    assert {"kept_whole_pct", "symbol_lines_pct", "line_coverage_pct"} <= set(DEFINITIONS)


def test_cli_text_and_json_output_and_it_never_modifies_the_repository(demo, capsys):
    before = _tree_hash(demo)
    assert compare_main([str(demo), "--definitions"]) == 0
    text = capsys.readouterr().out
    assert "baseline (line)" in text and "kept_whole_pct" in text
    assert "retrieval accuracy" in text
    assert "return stage_69" not in text  # no source text is printed
    assert compare_main([str(demo), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ast"]["strategy"] == "ast" and payload["baseline"]["strategy"] == "line"
    assert payload["note"].startswith("Descriptive")
    assert _tree_hash(demo) == before


def test_cli_reports_invalid_repository(tmp_path, capsys):
    assert compare_main([str(tmp_path / "missing")]) == 2
    assert "error:" in capsys.readouterr().err
