"""Experimental controls of Milestone 10: ``line_sized`` (Arm C) and ``ast_merged`` (Arm D)."""

from __future__ import annotations

import pytest

from copilot.chunking import available_strategies, create_chunker
from copilot.chunking.errors import ChunkingError
from copilot.chunking.experimental import MergedAstChunker, SizedLineChunker
from copilot.chunking.registry import create_chunker_from_params
from copilot.chunking.windows import split_lines
from copilot.config.settings import Settings
from copilot.models import ChunkType
from tests.ast_helpers import (
    assert_source_reconstructs,
    assert_structural_chunks_do_not_overlap,
    ast_chunker,
    line_chunker,
)
from tests.chunking_helpers import assert_chunk_invariants, make_file


def _module(functions: int, body_lines: int = 3) -> str:
    parts = ['"""Module docstring."""\n', "import os\n"]
    for i in range(functions):
        body = "".join(f"    value_{k} = os.path.join('a', str({i}))\n" for k in range(body_lines))
        parts.append(f"\ndef func_{i}(arg):\n{body}    return arg\n")
    return "\n".join(parts)


def _merged(budget: int, max_tokens: int = 512) -> MergedAstChunker:
    return MergedAstChunker(
        max_tokens=max_tokens,
        fallback_size_lines=60,
        fallback_overlap_lines=10,
        merge_max_tokens=budget,
    )


def _sized(py_size: int, py_overlap: int = 0) -> SizedLineChunker:
    return SizedLineChunker(
        python_size_lines=py_size,
        python_overlap_lines=py_overlap,
        size_lines=60,
        overlap_lines=10,
        max_tokens=512,
    )


# --- isolation from production ------------------------------------------------------------


def test_experimental_strategies_are_not_selectable_production_strategies():
    assert "line_sized" not in available_strategies()
    assert "ast_merged" not in available_strategies()
    with pytest.raises(Exception):  # noqa: B017 - unknown strategy error type is not the point
        create_chunker("ast_merged", settings=Settings())


def test_experimental_chunkers_rebuild_from_manifest_params():
    for chunker in (_sized(15), _merged(400)):
        rebuilt = create_chunker_from_params(chunker.name, chunker.version, chunker.params())
        assert rebuilt.params() == chunker.params()


def test_rebuild_rejects_wrong_version_missing_and_altered_params():
    chunker = _merged(400)
    with pytest.raises(ChunkingError):
        create_chunker_from_params(chunker.name, chunker.version + 1, chunker.params())
    with pytest.raises(ChunkingError):
        create_chunker_from_params(chunker.name, chunker.version, {"max_tokens": 512})
    altered = {**chunker.params(), "base": "ast@999"}
    with pytest.raises(ChunkingError):
        create_chunker_from_params(chunker.name, chunker.version, altered)


def test_production_chunk_ids_are_untouched_by_the_experimental_strategies():
    file = make_file(_module(6))
    before = [c.chunk_id for c in ast_chunker().chunk_file(file)]
    experimental = {c.chunk_id for c in _merged(512).chunk_file(file)}
    experimental |= {c.chunk_id for c in _sized(15).chunk_file(file)}
    assert not experimental & set(before)  # strategy identity is part of every id
    assert before == [c.chunk_id for c in ast_chunker().chunk_file(file)]


# --- SizedLineChunker (Arm C) ---------------------------------------------------------------


def test_sized_chunker_uses_python_window_for_python_and_baseline_for_other_files():
    py = make_file("\n".join(f"x{i} = {i}" for i in range(100)) + "\n", "a.py")
    js = make_file(
        "\n".join(f"let x{i} = {i};" for i in range(100)) + "\n", "a.js", language="javascript"
    )
    sized, base = _sized(15), line_chunker()
    assert max(c.end_line - c.start_line + 1 for c in sized.chunk_file(py)) == 15
    assert [(c.start_line, c.end_line) for c in sized.chunk_file(js)] == [
        (c.start_line, c.end_line) for c in base.chunk_file(js)
    ]
    assert_chunk_invariants(py, sized.chunk_file(py), max_tokens=512)


def test_sized_chunker_with_zero_overlap_tiles_the_file():
    file = make_file("\n".join(f"x{i} = {i}" for i in range(50)) + "\n", "a.py")
    spans = [(c.start_line, c.end_line) for c in _sized(15, 0).chunk_file(file)]
    assert spans == [(1, 15), (16, 30), (31, 45), (46, 50)]


@pytest.mark.parametrize("size,overlap", [(0, 0), (5, 5), (5, -1)])
def test_sized_chunker_rejects_invalid_geometry(size, overlap):
    with pytest.raises(ValueError):
        _sized(size, overlap)


# --- MergedAstChunker (Arm D) ---------------------------------------------------------------


def test_merging_keeps_the_exact_source_text_and_respects_the_budget():
    file = make_file(_module(30))
    base = ast_chunker().chunk_file(file)
    merged = _merged(300).chunk_file(file)
    assert len(merged) < len(base)
    assert_chunk_invariants(file, merged, max_tokens=300)
    assert_source_reconstructs(file, merged)
    assert_structural_chunks_do_not_overlap(merged)
    lines = split_lines(file.content)
    for chunk in merged:
        assert chunk.content == "\n".join(lines[chunk.start_line - 1 : chunk.end_line])


def test_merged_chunk_boundaries_are_a_subset_of_the_ast_boundaries():
    file = make_file(_module(30))
    base = ast_chunker().chunk_file(file)
    merged = _merged(300).chunk_file(file)
    starts = {c.start_line for c in base}
    ends = {c.end_line for c in base}
    assert {c.start_line for c in merged} <= starts
    assert {c.end_line for c in merged} <= ends


def test_multi_unit_chunks_carry_no_invented_symbol_and_single_units_keep_theirs():
    file = make_file(_module(30))
    merged = _merged(300).chunk_file(file)
    multi = [c for c in merged if c.chunk_type is ChunkType.MERGED]
    assert multi
    for chunk in multi:
        assert (chunk.symbol_name, chunk.qualified_name, chunk.parent_class) == (None,) * 3
        assert chunk.segmentation == "ast"
    # A definition too large to share a chunk keeps its own symbol metadata.
    big = make_file(_module(2, body_lines=40), "big.py")
    singles = _merged(250).chunk_file(big)
    assert any(c.chunk_type is ChunkType.FUNCTION and c.qualified_name for c in singles)


def test_budget_below_two_units_merges_nothing_and_matches_the_ast_boundaries():
    file = make_file(_module(8, body_lines=10))
    base = ast_chunker().chunk_file(file)
    functions = [c for c in base if c.chunk_type is ChunkType.FUNCTION]
    budget = max(c.token_estimate for c in functions)  # a unit fits; no two functions do
    merged = _merged(budget).chunk_file(file)
    assert [c for c in merged if c.chunk_type is ChunkType.FUNCTION] and all(
        c.chunk_type is not ChunkType.MERGED or c.token_estimate <= budget for c in merged
    )
    spans = [(c.start_line, c.end_line) for c in merged if c.chunk_type is ChunkType.FUNCTION]
    assert spans == [(c.start_line, c.end_line) for c in functions]


def test_merge_budget_is_never_exceeded_exactly_at_the_boundary():
    file = make_file(_module(12))
    base = ast_chunker().chunk_file(file)
    pair = sum(c.token_estimate for c in base[:2])
    # one token short of the first two units -> they cannot be merged; exactly enough -> they can
    chunker = _merged(max(pair, 1))
    assert all(c.token_estimate <= pair for c in chunker.chunk_file(file))
    assert chunker.chunk_file(file)[0].end_line >= base[0].end_line


def test_merge_budget_may_not_exceed_the_chunker_cap():
    with pytest.raises(ValueError):
        MergedAstChunker(
            max_tokens=512, fallback_size_lines=60, fallback_overlap_lines=10, merge_max_tokens=513
        )


def test_merging_is_deterministic_and_ids_are_unique():
    file = make_file(_module(25))
    first = [(c.chunk_id, c.start_line, c.end_line) for c in _merged(350).chunk_file(file)]
    second = [(c.chunk_id, c.start_line, c.end_line) for c in _merged(350).chunk_file(file)]
    assert first == second
    assert len({c[0] for c in first}) == len(first)


def test_unparseable_python_falls_back_to_baseline_boundaries():
    file = make_file("def broken(:\n" + "x = 1\n" * 130)
    merged = _merged(512).chunk_file(file)
    baseline = line_chunker().chunk_file(file)
    assert [(c.start_line, c.end_line) for c in merged] == [
        (c.start_line, c.end_line) for c in baseline
    ]
    assert {c.segmentation for c in merged} == {"fallback"}
    assert all(c.chunk_type is ChunkType.LINE_WINDOW for c in merged)


def test_non_python_files_are_cut_exactly_like_the_baseline():
    file = make_file(
        "\n".join(f"const a{i} = {i};" for i in range(150)) + "\n", "x.js", language="javascript"
    )
    merged = _merged(512).chunk_file(file)
    assert [(c.start_line, c.end_line, c.content) for c in merged] == [
        (c.start_line, c.end_line, c.content) for c in line_chunker().chunk_file(file)
    ]


def test_split_statements_are_never_merged_with_neighbours():
    big_literal = "DATA = [\n" + "".join(f"    {i},\n" for i in range(400)) + "]\n"
    file = make_file("import os\n\n" + big_literal + "\ndef tail():\n    return 1\n")
    chunks, report = _merged(512).analyze(file)
    assert report.ast_report.split_statements >= 1
    split = [c for c in chunks if c.segmentation == "ast_split"]
    assert split
    assert_source_reconstructs(file, chunks)


def test_analyze_reports_the_merge():
    file = make_file(_module(30))
    chunks, report = _merged(300).analyze(file)
    assert report.output_chunks == len(chunks)
    assert report.input_chunks > report.output_chunks
    assert report.merged_groups == sum(c.chunk_type is ChunkType.MERGED for c in chunks)
