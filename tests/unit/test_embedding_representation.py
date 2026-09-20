import pytest

from copilot.chunking import LineChunker
from copilot.embeddings.representation import (
    REPRESENTATION_VERSION,
    embedding_text,
    prefixed_text,
)
from tests.chunking_helpers import make_file

SOURCE = "import os\n\n\ndef first():\n    return 1\n"


def chunks_of(text, path="src/auth/service.py", language="python", **kw):
    params = {"size_lines": 60, "overlap_lines": 10, "max_tokens": 512, **kw}
    return LineChunker(
        size_lines=params["size_lines"],
        overlap_lines=params["overlap_lines"],
        max_tokens=params["max_tokens"],
    ).chunk_file(make_file(text, path, language))


def test_prefixed_format_is_exact():
    (chunk,) = chunks_of(SOURCE)
    assert prefixed_text(chunk) == (
        "File: src/auth/service.py\n"
        "Language: python\n"
        "Lines: 1-5\n"
        "\n"
        "import os\n\n\ndef first():\n    return 1"
    )


def test_raw_style_is_exactly_the_chunk_content():
    (chunk,) = chunks_of(SOURCE)
    assert embedding_text(chunk, "raw") == chunk.content


def test_default_style_is_prefixed():
    (chunk,) = chunks_of(SOURCE)
    assert embedding_text(chunk) == prefixed_text(chunk)


def test_chunk_content_is_never_modified_and_stays_the_citation_source():
    chunk = chunks_of(SOURCE)[0]
    before = (chunk.content, chunk.content_sha256, chunk.start_line, chunk.end_line)
    embedding_text(chunk, "prefixed")
    assert (chunk.content, chunk.content_sha256, chunk.start_line, chunk.end_line) == before
    assert prefixed_text(chunk).endswith(chunk.content)  # raw source is a verbatim suffix


def test_prefix_is_deterministic():
    (chunk,) = chunks_of(SOURCE)
    assert prefixed_text(chunk) == prefixed_text(chunk)
    assert prefixed_text(chunk) == prefixed_text(chunks_of(SOURCE)[0])


def test_prefix_reflects_path_language_and_lines_of_each_chunk():
    text = "".join(f"x{i} = {i}\n" for i in range(1, 11))
    chunks = chunks_of(text, "lib/util.js", "javascript", size_lines=4, overlap_lines=1)
    labels = [prefixed_text(c).splitlines()[2] for c in chunks]
    assert labels == ["Lines: 1-4", "Lines: 4-7", "Lines: 7-10"]
    assert all(
        prefixed_text(c).startswith("File: lib/util.js\nLanguage: javascript\n") for c in chunks
    )


def test_fragments_are_labelled_with_their_physical_line_and_part():
    long_line = "v = [" + ", ".join(["0"] * 200) + "]"
    chunks = chunks_of(f"a = 1\n{long_line}\n", max_tokens=40)
    fragments = [c for c in chunks if c.is_fragment]
    assert len(fragments) > 2
    second = prefixed_text(fragments[1]).splitlines()[2]
    assert second == f"Lines: 2 (part 2 of {len(fragments)})"


def test_unicode_paths_and_content_pass_through():
    (chunk,) = chunks_of("naïve = 'ü你好'\n", path="src/données/résumé.py")
    text = prefixed_text(chunk)
    assert "File: src/données/résumé.py" in text and text.endswith("naïve = 'ü你好'")


def test_repository_name_is_not_part_of_the_representation():
    (chunk,) = chunks_of(SOURCE)
    assert chunk.repository_name not in prefixed_text(chunk)


def test_unknown_style_is_rejected():
    (chunk,) = chunks_of(SOURCE)
    with pytest.raises(ValueError, match="unknown embedding text style"):
        embedding_text(chunk, "fancy")  # type: ignore[arg-type]


def test_representation_version_is_pinned():
    assert REPRESENTATION_VERSION == 1  # bump together with any prefix format change
