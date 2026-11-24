"""Fallback to the baseline line splitter, and failure safety of the parser step."""

from __future__ import annotations

import ast
import logging
import warnings
from pathlib import Path

import pytest

from copilot.chunking import AstChunker, chunk_repository
from copilot.chunking import ast_chunker as ast_module
from copilot.ingestion import IngestionPolicy, ingest_repository
from tests.ast_helpers import ast_chunker, chunk_py, line_chunker
from tests.chunking_helpers import assert_chunk_invariants, make_file

BROKEN = "def ok():\n    return 1\n\ndef broken(:\n    pass\n"


def _boundaries(chunks):
    return [(c.start_line, c.end_line, c.content, c.fragment_index) for c in chunks]


def test_unparseable_python_uses_the_line_splitter_with_the_ast_identity():
    file = make_file(BROKEN)
    chunks, report = ast_chunker().analyze(file)
    assert (report.segmentation, report.fallback_reason) == ("fallback", "syntax_error")
    assert {c.segmentation for c in chunks} == {"fallback"}
    assert {(c.chunking_strategy, c.chunking_version) for c in chunks} == {("ast", 1)}
    assert {c.symbol_name for c in chunks} == {None}
    # identical boundaries to the baseline strategy
    assert _boundaries(chunks) == _boundaries(line_chunker().chunk_file(file))
    assert_chunk_invariants(file, chunks, max_tokens=512)


@pytest.mark.parametrize(
    "source",
    [
        "def f(\n    return 1\n",  # unclosed parenthesis
        "if True:\npass\n",  # missing indent
        "class A:\n    def f(self):\n        pass\n  oops\n",  # inconsistent dedent
        "x = (1 +\n",  # incomplete expression
        "def f():\n",  # missing body
        "x = 'unterminated\n",
        "(" * 5000 + "\n",  # nesting far beyond the parser limits
    ],
)
def test_incomplete_or_broken_python_never_raises_and_loses_nothing(source):
    file = make_file(source)
    chunks, report = ast_chunker().analyze(file)
    assert report.segmentation == "fallback"
    assert report.fallback_reason in {"syntax_error", "recursion_limit", "memory_error"}
    assert chunks and {c.segmentation for c in chunks} == {"fallback"}
    assert_chunk_invariants(file, chunks, max_tokens=512)


def test_null_byte_in_python_is_handled_safely():
    file = make_file("x = 1\n\x00\ny = 2\n")
    chunks, report = ast_chunker().analyze(file)
    assert report.segmentation == "fallback"
    assert report.fallback_reason in {"syntax_error", "invalid_source"}
    assert chunks


@pytest.mark.parametrize(
    ("path", "language", "content"),
    [
        ("web/app.js", "javascript", "function a() {\n  return 1;\n}\n" * 30),
        ("docs/readme.md", "markdown", "# Title\n\ntext\n\n## Sub\n\nmore\n" * 20),
        ("conf/app.yaml", "yaml", "a:\n  b: 1\n" * 40),
        ("conf/data.json", "json", '{\n  "a": 1,\n  "b": 2\n}\n' * 25),
        ("src/Main.java", "java", "class Main {\n  void f() {}\n}\n" * 30),
    ],
)
def test_unsupported_languages_match_the_baseline_boundaries(path, language, content):
    file = make_file(content, path, language=language)
    chunks, report = ast_chunker().analyze(file)
    assert (report.segmentation, report.fallback_reason) == ("fallback", "unsupported_language")
    assert {c.segmentation for c in chunks} == {"fallback"}
    assert {c.chunk_type.value for c in chunks} == {"line_window"}
    assert _boundaries(chunks) == _boundaries(line_chunker().chunk_file(file))
    assert_chunk_invariants(file, chunks, max_tokens=512)


def test_python_file_with_crlf_that_was_not_normalised_falls_back_safely():
    file = make_file("def f():\r\n    return 1\r\n")
    _, report = ast_chunker().analyze(file)
    assert report.fallback_reason == "non_normalised_newlines"


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (RecursionError("SECRET-PARSER-TEXT"), "recursion_limit"),
        (MemoryError("SECRET-PARSER-TEXT"), "memory_error"),
        (ValueError("SECRET-PARSER-TEXT"), "invalid_source"),
        (OverflowError("SECRET-PARSER-TEXT"), "invalid_source"),
        (SyntaxError("SECRET-PARSER-TEXT"), "syntax_error"),
    ],
)
def test_parser_resource_failures_fall_back_without_exposing_parser_text(
    monkeypatch, caplog, error, reason
):
    def boom(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(ast_module.ast, "parse", boom)
    file = make_file("def f():\n    return 1\n")
    with caplog.at_level(logging.DEBUG):
        chunks, report = ast_chunker().analyze(file)
    assert report.fallback_reason == reason
    assert chunks and {c.segmentation for c in chunks} == {"fallback"}
    assert "SECRET-PARSER-TEXT" not in caplog.text
    assert all("SECRET-PARSER-TEXT" not in c.content for c in chunks)
    assert reason in caplog.text  # the reason code is logged, never the parser's own message


def test_a_failed_self_check_falls_back_instead_of_losing_source(monkeypatch, caplog):
    """If segmentation ever dropped a non-blank line, the file must fall back, not lose it."""
    real_run = ast_module._Segmenter.run

    def lossy(self):
        drafts = real_run(self)
        return drafts[:-1]  # silently drop the last chunk

    monkeypatch.setattr(ast_module._Segmenter, "run", lossy)
    src = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
    file = make_file(src)
    with caplog.at_level(logging.WARNING):
        chunks, report = ast_chunker().analyze(file)
    assert report.fallback_reason == "internal_check_failed"
    assert any("return 2" in c.content for c in chunks)
    assert "self-check failed" in caplog.text


def test_syntax_warnings_in_indexed_code_do_not_leak():
    src = 'import re\nPATTERN = re.compile("\\d+")\n'  # invalid escape: SyntaxWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, chunks = chunk_py(src)
    assert chunks[0].segmentation == "ast"


def test_indexed_code_is_parsed_never_executed(tmp_path):
    marker = tmp_path / "executed.txt"
    src = (
        f"open({str(marker)!r}, 'w').write('x')\n"
        "raise SystemExit(3)\n"
        "def f():\n    import sys; sys.exit(4)\n"
    )
    _, chunks = chunk_py(src)
    assert chunks and not marker.exists()


def test_the_chunker_source_contains_no_dynamic_execution_or_network_use():
    tree = ast.parse(Path(ast_module.__file__).read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    imported = {
        a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    } | {
        n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
    }
    assert not names & {"exec", "eval", "compile", "__import__", "run_path", "import_module"}
    assert not imported & {
        "importlib",
        "subprocess",
        "socket",
        "httpx",
        "requests",
        "urllib",
        "runpy",
    }


def test_mixed_language_repository_is_fully_indexable(tmp_path):
    files = {
        "app/service.py": "import os\n\n\ndef run():\n    return os.getcwd()\n",
        "app/broken.py": "def oops(:\n    pass\n",
        "web/main.js": "export function a() {\n  return 1;\n}\n",
        "README.md": "# Project\n\nSome text.\n",
        "config/settings.yaml": "debug: true\nlevel: info\n",
        "config/package.json": '{\n  "name": "x"\n}\n',
    }
    for relative, text in files.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
    ingestion = ingest_repository(tmp_path, IngestionPolicy(), repository_name="mixed")
    result = chunk_repository(ingestion, ast_chunker())
    assert {c.file_path for c in result.chunks} == set(files)
    by_file = {c.file_path: c for c in result.chunks}
    assert by_file["app/service.py"].segmentation == "ast"
    assert by_file["app/broken.py"].segmentation == "fallback"
    assert by_file["web/main.js"].segmentation == "fallback"
    assert result.stats.files_without_chunks == 0
    assert isinstance(ast_chunker(), AstChunker)
