"""Oversized definitions: deterministic splitting within the estimated-token budget."""

from __future__ import annotations

import pytest

from copilot.models import ChunkType
from copilot.utils.tokens import estimate_tokens
from tests.ast_helpers import (
    assert_structural_chunks_do_not_overlap,
    ast_chunker,
    by_qualified,
    chunk_py,
    ranges,
)
from tests.chunking_helpers import make_file


def _function(name: str, statements: int) -> str:
    body = "".join(f"    value_{i} = compute({i}, {i + 1})\n" for i in range(statements))
    return f'def {name}(arg):\n    """Doc for {name}."""\n{body}    return arg\n'


def test_definition_exactly_at_the_budget_stays_whole_and_one_over_is_split():
    src = _function("f", 4)
    tokens = estimate_tokens(src)
    _, whole = chunk_py(src, max_tokens=tokens)
    assert len(whole) == 1 and whole[0].token_estimate == tokens
    _, split = chunk_py(src, max_tokens=tokens - 1)
    assert len(split) > 1


def test_class_exactly_at_the_budget_stays_whole_and_one_over_splits_at_members():
    src = "class K:\n    def a(self):\n        return 1\n\n    def b(self):\n        return 2\n"
    tokens = estimate_tokens(src)
    _, whole = chunk_py(src, max_tokens=tokens)
    assert [(c.chunk_type, c.qualified_name) for c in whole] == [(ChunkType.CLASS, "K")]
    _, split = chunk_py(src, max_tokens=tokens - 1)
    assert [c.qualified_name for c in split] == ["K", "K.a", "K.b"]


def test_oversized_function_splits_at_statement_boundaries_and_keeps_its_symbol():
    src = _function("process", 12)
    file, chunks = chunk_py(src, max_tokens=60)
    assert len(chunks) > 1
    assert {c.qualified_name for c in chunks} == {"process"}
    assert {c.chunk_type for c in chunks} == {ChunkType.FUNCTION}
    assert {c.segmentation for c in chunks} == {
        "ast"
    }  # cut at statements, not by the line splitter
    # signature and docstring are in the first chunk
    assert chunks[0].content.startswith("def process(arg):")
    assert '"""Doc for process."""' in chunks[0].content
    # every part starts on a statement boundary: no part begins mid-statement
    for c in chunks[1:]:
        assert c.content.lstrip().startswith("value_") or c.content.lstrip().startswith("return")
    assert_structural_chunks_do_not_overlap(chunks)
    assert all(c.token_estimate <= 60 for c in chunks)
    assert len(file.content.splitlines()) == chunks[-1].end_line


def test_oversized_class_splits_at_member_boundaries():
    methods = "\n".join(f"    def m{i}(self):\n        return {i} + {i} + {i}\n" for i in range(6))
    src = f'class Big:\n    """Doc."""\n\n    LIMIT = 3\n\n{methods}'
    _, chunks = chunk_py(src, max_tokens=30)
    qualified = by_qualified(chunks)
    head = qualified["Big"][0]
    assert head.chunk_type is ChunkType.CLASS
    assert "LIMIT = 3" in head.content and '"""Doc."""' in head.content
    assert head.content.startswith("class Big:")
    assert [q for q in qualified if q and q.startswith("Big.m")] == [f"Big.m{i}" for i in range(6)]
    assert all(
        c.chunk_type is ChunkType.METHOD for q, cs in qualified.items() if q != "Big" for c in cs
    )
    assert_structural_chunks_do_not_overlap(chunks)


def test_class_without_leading_statements_gets_a_header_only_class_chunk():
    src = "class K(Base):\n" + "".join(
        f"    def m{i}(self):\n        return {i}\n" for i in range(5)
    )
    _, chunks = chunk_py(src, max_tokens=12)
    assert chunks[0].content == "class K(Base):"
    assert chunks[0].qualified_name == "K"


def test_oversized_nested_class_is_split_recursively():
    inner = "".join(f"        def m{i}(self):\n            return {i} + {i}\n" for i in range(5))
    src = f"class Outer:\n    class Inner:\n{inner}\n    def tail(self):\n        return 1\n"
    _, chunks = chunk_py(src, max_tokens=16)
    qualified = by_qualified(chunks)
    assert "Outer.Inner.m0" in qualified and "Outer.Inner.m4" in qualified
    assert qualified["Outer.Inner.m3"][0].parent_class == "Outer.Inner"
    assert qualified["Outer.tail"][0].parent_class == "Outer"


def test_single_oversized_statement_falls_back_to_the_line_splitter_and_keeps_the_symbol():
    items = "".join(f"        'key_{i}': {i},\n" for i in range(40))
    src = f"def table():\n    return {{\n{items}    }}\n"
    _, chunks = chunk_py(src, max_tokens=40)
    assert len(chunks) > 1
    assert {c.qualified_name for c in chunks} == {"table"}
    assert "ast_split" in {c.segmentation for c in chunks}
    assert chunks[0].start_line == 1  # the signature is not lost


def test_oversized_single_line_is_fragmented_with_exact_ranges_and_symbol():
    long_line = "    data = [" + ", ".join(str(i) for i in range(300)) + "]\n"
    src = f"def blob():\n{long_line}    return data\n"
    file, chunks = chunk_py(src, max_tokens=50)
    fragments = [c for c in chunks if c.is_fragment]
    assert len(fragments) > 2
    assert {c.start_line for c in fragments} == {2}  # fragments keep the physical line number
    assert [c.fragment_index for c in fragments] == list(range(len(fragments)))
    assert "".join(c.content for c in fragments) == long_line.rstrip("\n")
    assert {c.qualified_name for c in fragments} == {"blob"}
    assert {c.segmentation for c in fragments} == {"ast_split"}
    assert all(c.token_estimate <= 50 for c in chunks)


def test_oversized_module_statement_is_split_without_a_symbol():
    items = "".join(f"    'k{i}': {i},\n" for i in range(60))
    _, chunks = chunk_py(f"TABLE = {{\n{items}}}\n", max_tokens=30)
    assert len(chunks) > 1
    assert {c.chunk_type for c in chunks} == {ChunkType.MODULE}
    assert {c.symbol_name for c in chunks} == {None}


def test_oversized_one_line_definition_is_fragmented():
    src = "def f(): return [" + ", ".join(str(i) for i in range(200)) + "]\n"
    _, chunks = chunk_py(src, max_tokens=30)
    assert all(c.is_fragment and c.qualified_name == "f" for c in chunks)


def test_splitting_is_deterministic_and_ids_are_unique():
    src = _function("process", 30)
    file = make_file(src)
    first = ast_chunker(50).chunk_file(file)
    assert first == ast_chunker(50).chunk_file(file)
    assert len({c.chunk_id for c in first}) == len(first)


@pytest.mark.parametrize("budget", [1, 2, 3, 5, 8, 13, 21, 40, 80, 160])
def test_any_budget_terminates_loses_nothing_and_respects_the_limit(budget):
    src = (
        '"""Doc."""\nimport os\n\nX = {"a": 1,\n     "b": 2}\n\n\n'
        + _function("f", 5)
        + "\n\nclass C:\n    '''Doc.'''\n    attr = [1, 2, 3]\n\n"
        "    def m(self, a, b):\n        return a + b\n\n"
        "    class In:\n        def n(self):\n            return 'x' * 5\n\n\n"
        "async def g():\n    return await h(1, 2, 3, 4)\n"
    )
    _, chunks = chunk_py(src, max_tokens=budget)  # helper asserts coverage + budget + ranges
    assert all(c.token_estimate <= budget for c in chunks)


def test_budget_is_never_raised_silently_for_oversized_input():
    src = "x = " + " + ".join(["value"] * 500) + "\n"
    _, chunks = chunk_py(src, max_tokens=64)
    assert max(c.token_estimate for c in chunks) <= 64


def test_ranges_of_split_function_are_contiguous_or_separated_by_blank_lines_only():
    src = _function("process", 12)
    file, chunks = chunk_py(src, max_tokens=50)
    lines = file.content.split("\n")
    spans = ranges(chunks)
    for (_, end), (start, _) in zip(spans, spans[1:], strict=False):
        assert all(not line.strip() for line in lines[end : start - 1])
