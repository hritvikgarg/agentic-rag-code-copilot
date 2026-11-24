"""Identity, compatibility and index integration of the "ast" strategy (all offline)."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from copilot.chunking import AstChunker, create_chunker_from_params
from copilot.chunking.errors import ChunkingError
from copilot.config import Settings
from copilot.embeddings import HashEmbedder
from copilot.embeddings.representation import REPRESENTATION_VERSION
from copilot.models.chunk import CHUNK_ID_SCHEMA, ChunkType
from copilot.retrieval import Retriever
from copilot.vectorstore import (
    IndexCompatibilityError,
    VectorIndex,
    build_repository_index,
    expectation_for_repository,
)
from copilot.vectorstore.records import (
    decode_records,
    encode_records,
    record_from_chunk,
)
from tests.ast_helpers import ast_chunker, line_chunker
from tests.chunking_helpers import make_file
from tests.retrieval_helpers import FILES, write_repo
from tests.vectorstore_helpers import make_chunks, make_files

SOURCE = (
    "import os\n\n\ndef a():\n    return os.getcwd()\n\n\n"
    "class B:\n    def m(self):\n        return 1\n"
)


# ---- ids ---------------------------------------------------------------------------------
def test_identical_source_and_config_give_identical_ids():
    file = make_file(SOURCE)
    assert [c.chunk_id for c in ast_chunker().chunk_file(file)] == [
        c.chunk_id for c in ast_chunker().chunk_file(file)
    ]


def test_ids_are_strategy_specific_even_for_identical_line_ranges():
    file = make_file("x = 1\n")
    (ast_chunk,) = ast_chunker().chunk_file(file)
    (line_chunk,) = line_chunker().chunk_file(file)
    assert (ast_chunk.start_line, ast_chunk.end_line) == (
        line_chunk.start_line,
        line_chunk.end_line,
    )
    assert ast_chunk.chunk_id != line_chunk.chunk_id
    assert ast_chunk.content_sha256 == line_chunk.content_sha256  # same text, different identity


def test_source_change_invalidates_every_id_of_the_file():
    old = {c.chunk_id for c in ast_chunker().chunk_file(make_file(SOURCE))}
    new = {c.chunk_id for c in ast_chunker().chunk_file(make_file(SOURCE + "# edit\n"))}
    assert old.isdisjoint(new)


@pytest.mark.parametrize(
    "changes",
    [{"max_tokens": 511}, {"size": 59}, {"overlap": 9}],
)
def test_any_configuration_change_changes_ids_and_params(changes):
    file = make_file(SOURCE)
    base = ast_chunker()
    other = ast_chunker(**{"max_tokens": 512, **changes})
    assert base.params() != other.params()
    assert {c.chunk_id for c in base.chunk_file(file)}.isdisjoint(
        c.chunk_id for c in other.chunk_file(file)
    )


def test_params_record_the_fallback_splitter_and_its_version():
    assert ast_chunker().params() == {
        "max_tokens": 512,
        "fallback_size_lines": 60,
        "fallback_overlap_lines": 10,
        "fallback": "line@1",
    }
    assert (ast_chunker().name, ast_chunker().version) == ("ast", 1)


def test_symbols_are_not_part_of_the_id_payload_so_the_id_schema_is_unchanged():
    assert CHUNK_ID_SCHEMA == "chunk-id/1"


def test_recorded_parameters_must_match_the_current_fallback_version():
    chunker = ast_chunker()
    rebuilt = create_chunker_from_params("ast", 1, chunker.params())
    assert isinstance(rebuilt, AstChunker) and rebuilt.params() == chunker.params()
    with pytest.raises(
        ChunkingError
    ):  # e.g. the line splitter was bumped after the index was built
        create_chunker_from_params("ast", 1, {**chunker.params(), "fallback": "line@2"})
    with pytest.raises(ChunkingError):
        create_chunker_from_params("ast", 2, chunker.params())
    with pytest.raises(ChunkingError):
        create_chunker_from_params("ast", 1, {"max_tokens": 5})


# ---- the baseline did not move -----------------------------------------------------------
def test_baseline_chunk_ids_and_sidecar_bytes_are_unchanged_by_milestone_9():
    """Goldens computed from the pre-Milestone-9 commit (3f955b5) with the same inputs."""
    chunks = make_chunks(make_files())
    assert [c.chunk_id for c in chunks] == [
        "c3bc973701342c69",
        "0d3f8052a12704c7",
        "6a538ae7e7550416",
        "6997929990a45d2c",
        "a3edac92787d6040",
        "f092c1e4bcedb6b7",
    ]
    sidecar = encode_records([record_from_chunk(c, i) for i, c in enumerate(chunks)])
    assert (
        hashlib.sha256(sidecar).hexdigest()
        == "92f9052daebb142689361cb096409dab09abe1893f4463adf6d912275e238cd7"
    )
    assert b"segmentation" not in sidecar
    assert all(
        c.segmentation is None and c.symbol_name is None and c.chunk_type is ChunkType.LINE_WINDOW
        for c in chunks
    )


def test_baseline_default_windows_are_unchanged():
    sample = "".join(f"def f{i}(x):\n    return x + {i}\n\n" for i in range(40))
    chunks = line_chunker().chunk_file(make_file(sample))
    assert [c.chunk_id for c in chunks] == [
        "f5175d94118ec1e3",
        "f7876e54cb5509ba",
        "5b0ce343ac43e50a",
    ]
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 60), (51, 110), (101, 120)]


def test_old_sidecar_lines_without_segmentation_still_parse():
    chunks = make_chunks(make_files())
    legacy = encode_records([record_from_chunk(c, i) for i, c in enumerate(chunks)])
    assert [r.segmentation for r in decode_records(legacy)] == [None] * len(chunks)


def test_segmentation_round_trips_through_the_sidecar():
    chunks = ast_chunker().chunk_file(make_file(SOURCE))
    records = [record_from_chunk(c, i) for i, c in enumerate(chunks)]
    decoded = decode_records(encode_records(records))
    assert decoded == records
    assert {r.segmentation for r in decoded} == {"ast"}
    assert {r.qualified_name for r in decoded} == {None, "a", "B"}


# ---- index integration (real FAISS, fake hash embedder, no network) ------------------------
@pytest.fixture
def repo(tmp_path):
    return write_repo(tmp_path / "demo-repo")


@pytest.fixture
def embedder():
    return HashEmbedder(dimension=64)


def build(repo, tmp_path, embedder, strategy=None, **settings):
    return build_repository_index(
        repo,
        embedder=embedder,
        settings=Settings(_env_file=None, **settings),
        indexes_dir=tmp_path / "indexes",
        chunking_strategy=strategy,
        overwrite=True,
    )


def test_cli_default_and_explicit_baseline_build_the_same_index(repo, tmp_path, embedder):
    default = build(repo, tmp_path, embedder)
    explicit = build(repo, tmp_path, embedder, strategy="line")
    assert (default.chunking_strategy, default.chunking_strategy_version) == ("line", 1)
    assert default.index_id == explicit.index_id


def test_ast_index_records_strategy_version_and_parameters(repo, tmp_path, embedder):
    report = build(repo, tmp_path, embedder, strategy="ast")
    assert (report.chunking_strategy, report.chunking_strategy_version) == ("ast", 1)
    manifest = json.loads((report.index_path / "manifest.json").read_text())
    spec = manifest.get("spec", manifest)
    assert spec["chunking_strategy"] == "ast" and spec["chunking_strategy_version"] == 1
    assert spec["chunk_params"]["fallback"] == "line@1"
    assert spec["representation_version"] == REPRESENTATION_VERSION == 1
    assert spec["chunk_id_schema"] == "chunk-id/1"


def test_strategies_never_share_an_index_id_or_directory(repo, tmp_path, embedder):
    line = build(repo, tmp_path, embedder, strategy="line")
    ast_report = build(repo, tmp_path, embedder, strategy="ast")
    assert line.index_id != ast_report.index_id
    assert line.index_path != ast_report.index_path
    assert (
        line.index_path.exists() and ast_report.index_path.exists()
    )  # neither overwrote the other


def _expectation(repo, strategy, **settings):
    """What the current configuration expects of an index of ``repo`` (model id unchecked)."""
    expected = expectation_for_repository(
        repo, settings=Settings(_env_file=None, **settings), chunking_strategy=strategy
    )
    return dataclasses.replace(expected, embedding_model_id=None)


def test_an_index_is_only_compatible_with_its_own_strategy(repo, tmp_path, embedder):
    ast_dir = build(repo, tmp_path, embedder, strategy="ast").index_path
    line_dir = build(repo, tmp_path, embedder, strategy="line").index_path
    assert (
        VectorIndex.load(ast_dir, expected=_expectation(repo, "ast")).spec.chunking_strategy
        == "ast"
    )
    assert (
        VectorIndex.load(line_dir, expected=_expectation(repo, "line")).spec.chunking_strategy
        == "line"
    )
    with pytest.raises(IndexCompatibilityError, match="chunking"):
        VectorIndex.load(ast_dir, expected=_expectation(repo, "line"))
    with pytest.raises(IndexCompatibilityError, match="chunking"):
        VectorIndex.load(line_dir, expected=_expectation(repo, "ast"))


def test_changed_ast_configuration_is_refused_for_an_existing_ast_index(repo, tmp_path, embedder):
    ast_dir = build(repo, tmp_path, embedder, strategy="ast").index_path
    with pytest.raises(IndexCompatibilityError, match="chunk"):
        VectorIndex.load(ast_dir, expected=_expectation(repo, "ast", chunk_max_tokens=256))


def test_config_change_changes_the_ast_index_identity(repo, tmp_path, embedder):
    a = build(repo, tmp_path, embedder, strategy="ast")
    b = build(repo, tmp_path, embedder, strategy="ast", chunk_max_tokens=256)
    assert a.index_id != b.index_id


def test_retrieval_over_an_ast_index_returns_symbols_and_exact_lines(repo, tmp_path, embedder):
    report = build(repo, tmp_path, embedder, strategy="ast")
    retriever = Retriever.open(report.index_path, repo, embedder=embedder)
    results = retriever.retrieve("make_chunk_id deterministic sha256 payload", top_k=3)
    top = results[0]
    assert top.file_path == "src/pkg/chunk_ids.py"
    assert (top.symbol_name, top.qualified_name, top.chunk_type) == (
        "make_chunk_id",
        "make_chunk_id",
        ChunkType.FUNCTION,
    )
    source_lines = FILES["src/pkg/chunk_ids.py"].split("\n")
    assert top.text == "\n".join(source_lines[top.start_line - 1 : top.end_line])
    assert (top.start_line, top.end_line) == (1, 4)


def test_baseline_retrieval_over_the_same_repository_still_has_no_symbols(repo, tmp_path, embedder):
    report = build(repo, tmp_path, embedder, strategy="line")
    retriever = Retriever.open(report.index_path, repo, embedder=embedder)
    top = retriever.retrieve("make_chunk_id deterministic sha256 payload", top_k=1)[0]
    assert top.symbol_name is None and top.chunk_type is ChunkType.LINE_WINDOW


def test_ast_index_is_rejected_when_the_repository_changed(repo, tmp_path, embedder):
    from copilot.retrieval import StaleRepositoryError

    report = build(repo, tmp_path, embedder, strategy="ast")
    (repo / "src/pkg/ingest.py").write_text("def x():\n    return 1\n", encoding="utf-8")
    with pytest.raises(StaleRepositoryError):
        Retriever.open(report.index_path, repo, embedder=embedder)


def test_crlf_files_chunk_like_their_lf_versions(tmp_path, embedder):
    lf = tmp_path / "lf"
    crlf = tmp_path / "crlf"
    for root, newline in ((lf, "\n"), (crlf, "\r\n")):
        target = root / "src" / "pkg" / "mod.py"
        target.parent.mkdir(parents=True)
        target.write_bytes(SOURCE.replace("\n", newline).encode("utf-8"))
    from copilot.chunking import chunk_repository
    from copilot.ingestion import IngestionPolicy, ingest_repository

    def spans(root):
        result = chunk_repository(
            ingest_repository(root, IngestionPolicy(), repository_name="r"), ast_chunker()
        )
        return [
            (c.file_path, c.start_line, c.end_line, c.content, c.qualified_name)
            for c in result.chunks
        ]

    assert spans(lf) == spans(crlf)
    assert all("\r" not in text for _, _, _, text, _ in spans(crlf))
