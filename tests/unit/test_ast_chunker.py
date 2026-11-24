"""Structure-aware segmentation of Python source (the "ast" strategy)."""

from __future__ import annotations

import pytest

from copilot.models import ChunkType
from tests.ast_helpers import (
    assert_structural_chunks_do_not_overlap,
    ast_chunker,
    by_qualified,
    chunk_py,
    ranges,
)
from tests.chunking_helpers import make_file

SMALL_MODULE = '''"""Module doc."""
import os
import sys

LIMIT = 10


def helper(a, b):
    """Add."""
    return a + b


def other():
    return helper(1, 2)
'''


def test_small_module_is_one_header_chunk_and_one_chunk_per_function():
    _, chunks = chunk_py(SMALL_MODULE)
    assert [(c.chunk_type, c.start_line, c.end_line) for c in chunks] == [
        (ChunkType.MODULE, 1, 5),
        (ChunkType.FUNCTION, 8, 10),
        (ChunkType.FUNCTION, 13, 14),
    ]
    assert [c.symbol_name for c in chunks] == [None, "helper", "other"]
    assert {c.segmentation for c in chunks} == {"ast"}
    assert chunks[0].content.startswith('"""Module doc."""')
    assert "LIMIT = 10" in chunks[0].content


def test_chunks_carry_the_ast_strategy_identity_and_dense_indexes():
    _, chunks = chunk_py(SMALL_MODULE)
    assert {(c.chunking_strategy, c.chunking_version) for c in chunks} == {("ast", 1)}
    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_async_function_is_a_function_chunk():
    src = "import asyncio\n\n\nasync def fetch(x):\n    await asyncio.sleep(x)\n    return x\n"
    _, chunks = chunk_py(src)
    fetch = by_qualified(chunks)["fetch"][0]
    assert (fetch.chunk_type, fetch.start_line, fetch.end_line) == (ChunkType.FUNCTION, 4, 6)
    assert fetch.content.startswith("async def fetch")


def test_decorators_are_part_of_the_definition_range():
    src = (
        "import functools\n\n\n@functools.cache\n@other(1,\n       2)\n"
        "def compute(n):\n    return n\n"
    )
    _, chunks = chunk_py(src)
    compute = by_qualified(chunks)["compute"][0]
    assert (compute.start_line, compute.end_line) == (4, 8)
    assert compute.content.splitlines()[0] == "@functools.cache"


def test_decorated_class_and_method_include_their_decorators():
    src = (
        "@dataclass\nclass P:\n    x: int\n\n"
        "    @property\n    def double(self):\n        return self.x * 2\n"
    )
    _, chunks = chunk_py(src, max_tokens=14)  # force the class to split at members
    qualified = by_qualified(chunks)
    assert qualified["P"][0].content.startswith("@dataclass\nclass P:")
    double = qualified["P.double"][0]
    assert double.chunk_type is ChunkType.METHOD
    # original indentation is preserved verbatim
    assert double.content.startswith("    @property\n    def double")


def test_multiline_signature_stays_with_its_body():
    src = (
        "def build(\n    first,\n    second=None,\n    *,\n    flag: bool = False,\n"
        ") -> dict:\n    return {}\n"
    )
    _, chunks = chunk_py(src)
    assert ranges(chunks) == [(1, 7)]
    assert chunks[0].symbol_name == "build"


def test_class_that_fits_is_one_chunk_and_its_methods_are_not_duplicated():
    src = (
        "class Loader:\n"
        '    """Loads."""\n'
        "\n"
        "    def load(self):\n"
        "        return 1\n"
        "\n"
        "    def save(self):\n"
        "        return 2\n"
    )
    _, chunks = chunk_py(src)
    assert len(chunks) == 1
    only = chunks[0]
    assert (only.chunk_type, only.symbol_name, only.qualified_name) == (
        ChunkType.CLASS,
        "Loader",
        "Loader",
    )
    assert only.parent_class is None
    assert "def load" in only.content and "def save" in only.content


def test_methods_of_a_split_class_get_qualified_names_and_the_class_as_parent():
    src = (
        "class RepositoryLoader:\n"
        '    """Loads repositories."""\n'
        "\n"
        "    def load(self, root):\n"
        "        files = [root, root]\n"
        "        return files\n"
        "\n"
        "    def close(self):\n"
        "        return None\n"
    )
    _, chunks = chunk_py(src, max_tokens=22)
    qualified = by_qualified(chunks)
    load = qualified["RepositoryLoader.load"][0]
    assert (load.symbol_name, load.parent_class, load.chunk_type) == (
        "load",
        "RepositoryLoader",
        ChunkType.METHOD,
    )
    head = qualified["RepositoryLoader"][0]
    assert head.chunk_type is ChunkType.CLASS
    assert head.content.startswith("class RepositoryLoader:")
    assert head.parent_class is None
    assert_structural_chunks_do_not_overlap(chunks)


def test_nested_class_methods_use_the_full_qualified_path():
    src = (
        "class OuterClass:\n"
        "    class InnerClass:\n"
        "        def method(self):\n"
        "            value = [1, 2, 3]\n"
        "            return value\n"
        "\n"
        "        def second(self):\n"
        "            return 2\n"
        "\n"
        "    def outer_method(self):\n"
        "        return 1\n"
    )
    _, chunks = chunk_py(src, max_tokens=24)
    qualified = by_qualified(chunks)
    method = qualified["OuterClass.InnerClass.method"][0]
    assert (method.symbol_name, method.parent_class) == ("method", "OuterClass.InnerClass")
    inner = qualified["OuterClass.InnerClass"][0]
    assert (inner.symbol_name, inner.parent_class) == ("InnerClass", "OuterClass")
    assert qualified["OuterClass.outer_method"][0].parent_class == "OuterClass"
    assert_structural_chunks_do_not_overlap(chunks)


def test_nested_function_stays_inside_its_parent_chunk():
    src = "def outer():\n    def inner():\n        return 1\n    return inner()\n"
    _, chunks = chunk_py(src)
    assert len(chunks) == 1
    assert chunks[0].symbol_name == "outer"
    assert "def inner" in chunks[0].content


def test_decorated_property_and_overloads_share_a_name_but_not_a_range():
    src = (
        "class C:\n"
        "    @property\n"
        "    def v(self):\n"
        "        return self._v\n"
        "\n"
        "    @v.setter\n"
        "    def v(self, value):\n"
        "        self._v = value\n"
    )
    _, chunks = chunk_py(src, max_tokens=18)
    parts = by_qualified(chunks)["C.v"]
    assert len(parts) == 2
    assert len({c.chunk_id for c in parts}) == 2
    assert parts[0].content.lstrip().startswith("@property")
    assert parts[1].content.lstrip().startswith("@v.setter")


def test_docstring_and_leading_statements_stay_with_the_signature():
    src = "def f(x):\n    '''Doc.'''\n    y = x\n    return y\n"
    _, chunks = chunk_py(src)
    assert chunks[0].content.startswith("def f(x):\n    '''Doc.'''")


def test_module_statements_between_definitions_form_their_own_chunks():
    src = (
        "import os\n\n\ndef a():\n    return 1\n\n\n"
        "LATER = 2\nLATEST = 3\n\n\ndef b():\n    return LATER\n"
    )
    _, chunks = chunk_py(src)
    assert [(c.chunk_type, c.start_line, c.end_line) for c in chunks] == [
        (ChunkType.MODULE, 1, 1),
        (ChunkType.FUNCTION, 4, 5),
        (ChunkType.MODULE, 8, 9),
        (ChunkType.FUNCTION, 12, 13),
    ]
    assert all(c.symbol_name is None for c in chunks if c.chunk_type is ChunkType.MODULE)


def test_main_guard_and_script_code_are_module_chunks():
    src = (
        "def main():\n    return 0\n\n\nif __name__ == '__main__':\n    raise SystemExit(main())\n"
    )
    _, chunks = chunk_py(src)
    assert [c.chunk_type for c in chunks] == [ChunkType.FUNCTION, ChunkType.MODULE]
    assert chunks[1].content.startswith("if __name__")


def test_comments_travel_forward_with_the_next_definition():
    src = "import os\n\n# ---- helpers ----\n\n# Adds numbers.\ndef add(a, b):\n    return a + b\n"
    _, chunks = chunk_py(src)
    add = by_qualified(chunks)["add"][0]
    assert add.content.startswith("# ---- helpers ----")
    assert "# Adds numbers." in add.content
    assert (add.start_line, add.end_line) == (3, 7)


def test_trailing_comment_at_end_of_file_is_not_lost():
    src = "def f():\n    return 1\n\n# end of file\n# really\n"
    _, chunks = chunk_py(src)
    assert any("# really" in c.content for c in chunks)
    assert chunks[-1].chunk_type is ChunkType.MODULE and chunks[-1].symbol_name is None


def test_comment_only_file_is_chunked_without_symbols():
    _, chunks = chunk_py("# just\n# comments\n")
    assert [c.chunk_type for c in chunks] == [ChunkType.MODULE]
    assert chunks[0].symbol_name is None


def test_statements_sharing_a_line_and_one_line_definitions():
    src = "a = 1; b = 2\n\n\ndef f(): return 1\n\n\nclass E(Exception): pass\n"
    _, chunks = chunk_py(src)
    assert [(c.chunk_type, c.start_line, c.end_line) for c in chunks] == [
        (ChunkType.MODULE, 1, 1),
        (ChunkType.FUNCTION, 4, 4),
        (ChunkType.CLASS, 7, 7),
    ]


def test_unicode_identifiers_strings_and_comments():
    src = (
        'def größe(名前):\n    """Länge – 長さ."""\n    return "héllo ✓"\n\n\nπ = 3.14  # circle\n'
    )
    _, chunks = chunk_py(src)
    assert by_qualified(chunks)["größe"][0].symbol_name == "größe"
    assert any("héllo ✓" in c.content for c in chunks)
    assert chunks[-1].chunk_type is ChunkType.MODULE and "π = 3.14" in chunks[-1].content


@pytest.mark.parametrize("source", ["", "\n", "   \n\n  \n", "\t\n"])
def test_empty_and_whitespace_only_files_produce_no_chunks(source):
    file = make_file(source)
    assert ast_chunker().chunk_file(file) == []


def test_file_without_trailing_newline():
    _, chunks = chunk_py("def f():\n    return 1")
    assert ranges(chunks) == [(1, 2)]


def test_form_feed_and_other_odd_whitespace_do_not_shift_line_numbers():
    src = "def a():\n    return 1\n\x0c\ndef b():\n    return 2\n"
    _, chunks = chunk_py(src)
    assert by_qualified(chunks)["b"][0].start_line == 4


def test_lambdas_comprehensions_and_match_statements_are_ordinary_statements():
    src = (
        "f = lambda x: [i for i in range(x)]\n"
        "\n"
        "def g(cmd):\n"
        "    match cmd:\n"
        "        case 'a':\n"
        "            return 1\n"
        "        case _:\n"
        "            return 0\n"
    )
    _, chunks = chunk_py(src)
    assert [c.symbol_name for c in chunks] == [None, "g"]


def test_structural_chunks_never_overlap_and_cover_every_nonblank_line():
    _, chunks = chunk_py(SMALL_MODULE)
    assert_structural_chunks_do_not_overlap(chunks)
    # chunk_py already asserted coverage and reconstruction


def test_chunking_is_deterministic():
    file = make_file(SMALL_MODULE)
    first = ast_chunker().chunk_file(file)
    second = ast_chunker().chunk_file(file)
    assert first == second


def test_python_coding_cookie_does_not_break_parsing():
    src = "# -*- coding: latin-1 -*-\ndef f():\n    return 1\n"
    _, chunks = chunk_py(src)
    assert by_qualified(chunks)["f"][0].segmentation == "ast"
