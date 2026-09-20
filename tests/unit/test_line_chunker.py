import pytest

from copilot.chunking import LineChunker
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
    assert first.chunk_type is ChunkType.CODE_WINDOW
    assert first.chunking_strategy == "line"
    # the baseline is structure-blind: it does not know about functions or classes
    assert all(c.symbol_name is None and c.parent_class is None for c in chunks)
    assert all(c.qualified_name is None and c.heading is None for c in chunks)
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
    fragments = [c for c in chunks if c.line_fragment]
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


def test_chunk_ids_depend_on_path_and_content_but_are_stable_for_unchanged_text():
    base = chunker().chunk_file(make_file(PY_SOURCE))
    other_path = chunker().chunk_file(make_file(PY_SOURCE, path="pkg/other.py"))
    edited = chunker().chunk_file(make_file(PY_SOURCE.replace("return 2", "return 3")))
    assert {c.chunk_id for c in base}.isdisjoint(c.chunk_id for c in other_path)
    assert edited[0].chunk_id == base[0].chunk_id  # untouched region keeps its id
    assert edited[-1].chunk_id != base[-1].chunk_id  # edited region gets a new id


def test_json_and_yaml_are_config_windows():
    j = chunker().chunk_file(make_file('{"a": 1}\n', "c/x.json", language="json"))
    y = chunker().chunk_file(make_file("a: 1\n", "c/x.yml", language="yaml"))
    assert j[0].chunk_type is ChunkType.CONFIG_WINDOW
    assert y[0].chunk_type is ChunkType.CONFIG_WINDOW


def test_markdown_uses_heading_sections_with_correct_line_numbers():
    text = "# Title\nintro\n\n## Install\nrun it\n\n## Use\nuse it\n"
    file = make_file(text, "docs/guide.md", language="markdown")
    chunks = chunker(size=50, overlap=5).chunk_file(file)
    assert [(c.start_line, c.end_line, c.heading) for c in chunks] == [
        (1, 3, "Title"),
        (4, 6, "Title > Install"),
        (7, 8, "Title > Use"),
    ]
    assert all(c.chunk_type is ChunkType.DOC_SECTION for c in chunks)
    assert chunks[1].content == "## Install\nrun it\n"
    assert_chunk_invariants(file, chunks, max_tokens=512)


def test_oversized_markdown_section_is_windowed_and_keeps_heading():
    body = "\n".join(f"sentence number {i}." for i in range(25))
    file = make_file(f"# Big\n{body}\n# Next\nshort\n", "d/big.md", language="markdown")
    chunks = chunker(size=10, overlap=2).chunk_file(file)
    big = [c for c in chunks if c.heading == "Big"]
    assert len(big) > 2
    assert [(c.start_line, c.end_line) for c in big][0] == (1, 10)
    assert chunks[-1].heading == "Next"
    assert_chunk_invariants(file, chunks, max_tokens=512)


def test_markdown_without_headings_is_windowed():
    file = make_file("\n".join(f"para {i}" for i in range(12)) + "\n", "n.md", language="markdown")
    chunks = chunker(size=5, overlap=1).chunk_file(file)
    assert chunks and all(c.heading is None for c in chunks)
    assert_chunk_invariants(file, chunks, max_tokens=512)


def test_markdown_section_token_cap_is_enforced():
    text = "# T\n" + "\n".join("word " * 30 for _ in range(6)) + "\n"
    file = make_file(text, "n.md", language="markdown")
    chunks = chunker(size=50, overlap=0, max_tokens=70).chunk_file(file)
    assert len(chunks) > 1
    assert_chunk_invariants(file, chunks, max_tokens=70)
