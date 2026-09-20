import subprocess
import sys

import pytest

from copilot.chunking import LineChunker
from copilot.chunking.windows import split_lines
from copilot.models import ChunkType
from tests.chunking_helpers import assert_chunk_invariants, make_file

PY_SOURCE = (
    "import os\n\n\n"
    "def first():\n    return 1\n\n\n"
    "class Box:\n    def method(self):\n        return 2\n"
)  # 10 lines


def chunker(size=4, overlap=1, max_tokens=512):
    return LineChunker(size_lines=size, overlap_lines=overlap, max_tokens=max_tokens)


def test_name_and_params():
    c = chunker(4, 1, 100)
    assert c.name == "line"
    assert c.params() == {"size_lines": 4, "overlap_lines": 1, "max_tokens": 100}


@pytest.mark.parametrize("kwargs", [{"size": 0}, {"size": 3, "overlap": 3}, {"max_tokens": 0}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        chunker(**kwargs)


def test_code_file_windows_and_metadata_hand_computed():
    file = make_file(PY_SOURCE)
    chunks = chunker().chunk_file(file)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 4), (4, 7), (7, 10)]
    first = chunks[0]
    assert first.content == "import os\n\n\ndef first():"
    assert first.file_path == "pkg/mod.py"
    assert first.repository_name == "repo"
    assert first.language == "python"
    assert first.chunk_type is ChunkType.LINE_WINDOW
    assert first.chunking_strategy == "line"
    # the baseline is structure-blind: it does not know about functions or classes
    assert all(c.symbol_name is None and c.parent_class is None for c in chunks)
    assert all(c.qualified_name is None for c in chunks)
    assert_chunk_invariants(file, chunks, max_tokens=512)


def test_baseline_can_cut_a_function_in_half():
    """Documents the weakness that structure-aware chunking (M9) is meant to fix."""
    file = make_file("def big():\n    a = 1\n    b = 2\n    c = 3\n    return a + b + c\n")
    chunks = chunker(size=3, overlap=0).chunk_file(file)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 3), (4, 5)]
    assert "return" not in chunks[0].content and "def big" not in chunks[1].content


def test_empty_and_whitespace_only_files_yield_no_chunks():
    assert chunker().chunk_file(make_file("")) == []
    assert chunker().chunk_file(make_file("\n\n  \n")) == []


def test_one_line_file_with_and_without_trailing_newline():
    for text in ("x = 1", "x = 1\n"):
        chunks = chunker().chunk_file(make_file(text))
        assert [(c.start_line, c.end_line, c.content) for c in chunks] == [(1, 1, "x = 1")]


def test_very_long_single_line_becomes_fragments():
    line = "DATA = [" + ", ".join(str(i) for i in range(500)) + "]"
    file = make_file(f"import x\n{line}\nprint(DATA)\n")
    chunks = chunker(size=10, overlap=2, max_tokens=64).chunk_file(file)
    fragments = [c for c in chunks if c.is_fragment]
    assert len(fragments) > 5
    assert {c.start_line for c in fragments} == {2}
    assert_chunk_invariants(file, chunks, max_tokens=64)


def test_unicode_content_is_preserved():
    text = "def größe():\n    return 'ünïcode 你好 🚀'\n"
    file = make_file(text)
    (chunk,) = chunker().chunk_file(file)
    assert chunk.content == text.rstrip("\n")
    assert_chunk_invariants(file, [chunk], max_tokens=512)


def test_crlf_normalised_input_produces_lf_only_chunks():
    file = make_file("a = 1\nb = 2\n")  # ingestion already converted CRLF to LF
    assert all("\r" not in c.content for c in chunker().chunk_file(file))


def test_deterministic_across_runs_and_instances():
    file = make_file(PY_SOURCE)
    first = chunker().chunk_file(file)
    assert first == chunker().chunk_file(file)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in chunker().chunk_file(file)]


def test_every_baseline_chunk_is_a_line_window_regardless_of_file_kind():
    files = [
        make_file("x = 1\n"),
        make_file('{"a": 1}\n', "c/x.json", language="json"),
        make_file("a: 1\n", "c/x.yml", language="yaml"),
        make_file("# Title\ntext\n", "d/x.md", language="markdown"),
    ]
    for f in files:
        (chunk,) = chunker().chunk_file(f)
        assert chunk.chunk_type is ChunkType.LINE_WINDOW
        assert chunk.language == f.language


def test_chunk_records_source_hash_strategy_and_version():
    file = make_file(PY_SOURCE)
    chunk = chunker().chunk_file(file)[0]
    assert chunk.source_sha256 == file.sha256
    assert (chunk.chunking_strategy, chunk.chunking_version) == ("line", 1)


# ---- Strategy A is structure-blind: Markdown/JSON/YAML are windowed exactly like code ---------
MARKDOWN = "# Title\nintro\n\n## Install\nrun it\n\n## Use\nuse it\n\n### Deep\nmore\n"


def spans_and_content(chunks):
    return [(c.start_line, c.end_line, c.content) for c in chunks]


@pytest.mark.parametrize(
    ("path", "language"),
    [("d/x.md", "markdown"), ("d/x.json", "json"), ("d/x.yaml", "yaml"), ("d/x.py", "python")],
)
def test_boundaries_depend_only_on_lines_never_on_file_kind(path, language):
    reference = spans_and_content(chunker(4, 1).chunk_file(make_file(MARKDOWN, "d/ref.py")))
    got = spans_and_content(chunker(4, 1).chunk_file(make_file(MARKDOWN, path, language=language)))
    assert got == reference


def test_markdown_headings_do_not_determine_boundaries():
    chunks = chunker(size=5, overlap=1).chunk_file(
        make_file(MARKDOWN, "docs/guide.md", language="markdown")
    )
    # Heading-aware splitting would give (1,3), (4,6), (7,9), (10,11). Windows ignore headings:
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 5), (5, 9), (9, 11)]
    assert "## Install" in chunks[0].content  # a heading sits in the middle of a chunk
    assert chunks[1].content.startswith("run it")  # a boundary falls inside a section


def test_fenced_code_in_markdown_is_ordinary_text():
    text = "intro\n```python\n# a comment\ndef f():\n    pass\n```\ntail\n"
    chunks = chunker(size=3, overlap=0).chunk_file(make_file(text, "n.md", language="markdown"))
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 3), (4, 6), (7, 7)]


def test_token_cap_shortens_markdown_windows_like_any_other_file():
    text = "\n".join("word " * 30 for _ in range(6)) + "\n"
    file = make_file(text, "n.md", language="markdown")
    chunks = chunker(size=50, overlap=0, max_tokens=70).chunk_file(file)
    assert len(chunks) > 1
    assert_chunk_invariants(file, chunks, max_tokens=70)


def test_baseline_does_not_import_the_markdown_section_splitter():
    """The heading-aware utility is isolated; importing the baseline must not load it."""
    code = (
        "import sys, copilot.chunking;"
        "assert 'copilot.chunking.markdown_sections' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# ---- line semantics: terminal newlines never create a phantom line ----------------------------
@pytest.mark.parametrize(
    ("text", "expected_lines", "expected_chunks"),
    [
        ("", [], []),
        ("abc", ["abc"], [(1, 1, "abc")]),
        ("abc\n", ["abc"], [(1, 1, "abc")]),  # terminal newline: no line 2
        ("abc\n\n", ["abc", ""], [(1, 2, "abc\n")]),  # a real blank line 2, but no line 3
        ("\n", [""], []),  # a blank-only file has nothing to index
        ("a\n\n\nb", ["a", "", "", "b"], [(1, 4, "a\n\n\nb")]),
    ],
)
def test_line_and_terminal_newline_semantics(text, expected_lines, expected_chunks):
    assert split_lines(text) == expected_lines
    chunks = chunker().chunk_file(make_file(text))
    assert [(c.start_line, c.end_line, c.content) for c in chunks] == expected_chunks
    assert all(c.end_line <= len(expected_lines) for c in chunks)  # no phantom citation line


def test_chunk_lines_are_one_based_and_inclusive():
    (chunk,) = chunker(size=2, overlap=0).chunk_file(make_file("first\nsecond\n"))
    assert (chunk.start_line, chunk.end_line) == (1, 2)
    assert chunk.content.split("\n") == ["first", "second"]
