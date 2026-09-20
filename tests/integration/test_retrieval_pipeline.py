"""End to end: real ingestion/chunking/FAISS + deterministic fake embeddings -> retrieval.

Builds a real index of a tiny repository, then retrieves through ``Retriever``. Only the embedding
model is the fake bag-of-words hash embedder, so a query that shares distinctive words with a file
ranks that file first. No model is downloaded and nothing here says anything about Jina quality.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from copilot.config import Settings
from copilot.embeddings import HashEmbedder
from copilot.models.chunk import sha256_text
from copilot.retrieval import (
    EmbedderMismatchError,
    QueryError,
    Retriever,
    StaleRepositoryError,
    retrieve,
)
from copilot.vectorstore import IndexCompatibilityError, VectorIndex, build_repository_index
from copilot.vectorstore.manifest import IndexExpectation
from tests.retrieval_helpers import write_repo


@pytest.fixture
def embedder():
    return HashEmbedder(dimension=256)


@pytest.fixture
def repo(tmp_path):
    return write_repo(tmp_path / "demo-repo")


def build(repo, tmp_path, embedder, *, style="raw", settings=None, **kwargs) -> Path:
    report = build_repository_index(
        repo,
        embedder=embedder,
        settings=settings or Settings(_env_file=None),
        indexes_dir=tmp_path / "indexes",
        text_style=style,
        overwrite=True,
        **kwargs,
    )
    return report.index_path


@pytest.fixture
def index_dir(repo, tmp_path, embedder):
    return build(repo, tmp_path, embedder)


@pytest.fixture
def retriever(index_dir, repo, embedder):
    return Retriever.open(index_dir, repo, embedder=embedder)


@pytest.mark.parametrize(
    ("query", "expected_file"),
    [
        ("discover repository files and read their text", "src/pkg/ingest.py"),
        ("deterministic chunk id sha256 hash payload", "src/pkg/chunk_ids.py"),
        ("configure logging handlers formatter log level", "src/pkg/logging_setup.py"),
        ("tutorial and installation steps guide", "docs/guide.md"),
    ],
)
def test_the_topical_file_is_ranked_first(retriever, query, expected_file):
    results = retriever.retrieve(query, 3)
    assert results[0].file_path == expected_file


def test_ranks_and_scores_are_consistent(retriever, embedder):
    query = "deterministic chunk id sha256 hash payload"
    results = retriever.retrieve(query, 10)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    q = embedder.embed_query(query)
    for r in results:  # the score really is the cosine similarity to the stored vector
        assert r.score == pytest.approx(
            float(q @ retriever.index.reconstruct(r.position)), abs=1e-5
        )


def test_results_preserve_the_sidecar_metadata(retriever):
    for result in retriever.retrieve("logging handlers", 10):
        record = retriever.index.record_at(result.position)
        assert result.chunk_id == record.chunk_id
        assert (result.file_path, result.start_line, result.end_line) == (
            record.file_path,
            record.start_line,
            record.end_line,
        )
        assert result.language == record.language
        assert result.chunk_type == record.chunk_type
        assert result.chunk_index == record.chunk_index
        assert result.content_sha256 == record.content_sha256
        assert result.repository_name == "demo-repo"
        assert (result.symbol_name, result.qualified_name, result.parent_class) == (None,) * 3


def test_the_text_is_the_exact_source_slice_and_matches_its_hash(retriever, repo):
    for result in retriever.retrieve("logging handlers", 10):
        lines = (repo / result.file_path).read_text(encoding="utf-8").split("\n")
        assert result.text == "\n".join(lines[result.start_line - 1 : result.end_line])
        assert sha256_text(result.text) == result.content_sha256


def test_results_expose_no_host_paths(retriever, repo, tmp_path):
    dumped = "".join(r.model_dump_json() + repr(r) for r in retriever.retrieve("logging", 10))
    for forbidden in (str(tmp_path), tmp_path.name, str(repo), repo.as_posix()):
        assert forbidden not in dumped.replace("demo-repo", "")
    for result in retriever.retrieve("logging", 10):
        assert not Path(result.file_path).is_absolute() and "\\" not in result.file_path


def test_top_k_larger_than_the_index_returns_every_chunk_once(retriever):
    results = retriever.retrieve("anything at all", 50)
    assert len(results) == retriever.index.count
    assert len({r.chunk_id for r in results}) == retriever.index.count


def test_retrieval_is_deterministic(index_dir, repo, embedder):
    first = Retriever.open(index_dir, repo, embedder=embedder).retrieve("chunk id hash", 4)
    second = Retriever.open(index_dir, repo, embedder=embedder).retrieve("chunk id hash", 4)
    assert first == second


def test_the_one_shot_function_matches_the_retriever(index_dir, repo, embedder, retriever):
    query = "deterministic chunk id sha256 hash payload"
    assert retrieve(query, 3, repo, index_dir, embedder=embedder) == retriever.retrieve(query, 3)


def test_a_preloaded_index_object_can_be_used(index_dir, repo, embedder):
    index = VectorIndex.load(index_dir, expected=IndexExpectation())
    retriever = Retriever.open(index, repo, embedder=embedder)
    assert retriever.retrieve("logging", 1)[0].file_path == "src/pkg/logging_setup.py"


def test_the_prefixed_representation_index_is_searchable_too(repo, tmp_path, embedder):
    path = build(repo, tmp_path, embedder, style="prefixed")
    results = Retriever.open(path, repo, embedder=embedder).retrieve("configure logging", 4)
    assert len(results) == 4 and results[0].rank == 1


# ---- stale-repository detection ---------------------------------------------------------------


def test_an_edited_file_makes_the_repository_stale(index_dir, repo, embedder):
    (repo / "src/pkg/ingest.py").write_text("def changed():\n    pass\n", newline="\n")
    with pytest.raises(StaleRepositoryError, match="fingerprint"):
        Retriever.open(index_dir, repo, embedder=embedder)


def test_an_added_file_makes_the_repository_stale(index_dir, repo, embedder):
    (repo / "src/pkg/new_module.py").write_text("x = 1\n", newline="\n")
    with pytest.raises(StaleRepositoryError):
        Retriever.open(index_dir, repo, embedder=embedder)


def test_a_missing_source_file_makes_the_repository_stale(index_dir, repo, embedder):
    (repo / "src/pkg/chunk_ids.py").unlink()
    with pytest.raises(StaleRepositoryError):
        Retriever.open(index_dir, repo, embedder=embedder)


def test_a_whitespace_only_edit_is_still_detected(index_dir, repo, embedder):
    path = repo / "docs/guide.md"
    path.write_text(path.read_text() + "\n", newline="\n")
    with pytest.raises(StaleRepositoryError):
        Retriever.open(index_dir, repo, embedder=embedder)


def test_a_different_ignore_set_than_at_build_time_is_stale(repo, tmp_path, embedder):
    path = build(repo, tmp_path, embedder, ignore_directories=["docs"])
    with pytest.raises(StaleRepositoryError, match="ignore-dir"):
        Retriever.open(path, repo, embedder=embedder)
    Retriever.open(path, repo, embedder=embedder, ignore_directories=["docs"])  # same set: fine


def test_the_error_message_shows_no_host_path(index_dir, repo, embedder, tmp_path):
    (repo / "src/pkg/ingest.py").write_text("x = 1\n", newline="\n")
    with pytest.raises(StaleRepositoryError) as caught:
        Retriever.open(index_dir, repo, embedder=embedder)
    assert str(tmp_path) not in str(caught.value)


def test_a_copy_of_the_repository_elsewhere_still_matches(index_dir, repo, tmp_path, embedder):
    clone = tmp_path / "somewhere else" / "clone"
    shutil.copytree(repo, clone)
    retriever = Retriever.open(index_dir, clone, embedder=embedder)
    assert retriever.retrieve("logging", 1)[0].file_path == "src/pkg/logging_setup.py"


def test_a_missing_repository_directory_is_an_ingestion_error(index_dir, tmp_path, embedder):
    from copilot.ingestion import InvalidRepositoryError

    with pytest.raises(InvalidRepositoryError):
        Retriever.open(index_dir, tmp_path / "nowhere", embedder=embedder)


# ---- embedder and configuration compatibility ---------------------------------------------------


def test_a_different_embedder_is_refused(index_dir, repo):
    with pytest.raises(EmbedderMismatchError):
        Retriever.open(index_dir, repo, embedder=HashEmbedder(dimension=64))


def test_changed_chunk_settings_do_not_affect_an_existing_index(repo, tmp_path, embedder):
    built_with = Settings(_env_file=None, chunk_size_lines=5, chunk_overlap_lines=1)
    path = build(repo, tmp_path, embedder, settings=built_with)
    now = Settings(_env_file=None, chunk_size_lines=60, chunk_overlap_lines=10, chunk_max_tokens=99)
    retriever = Retriever.open(path, repo, embedder=embedder, settings=now)
    results = retriever.retrieve("deterministic chunk id", 5)
    assert results and max(r.line_count for r in results) <= 5  # the manifest's chunking was used


def test_a_corrupt_index_is_refused_not_repaired(index_dir, repo, embedder):
    (index_dir / "index.faiss").write_bytes(b"damaged")
    from copilot.vectorstore import IndexCorruptError

    with pytest.raises(IndexCorruptError):
        Retriever.open(index_dir, repo, embedder=embedder)


def test_an_unsupported_manifest_is_incompatible(index_dir, repo, embedder):
    manifest = json.loads((index_dir / "manifest.json").read_text())
    manifest["schema_version"] = "vector-index-manifest/999"
    (index_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(IndexCompatibilityError):
        Retriever.open(index_dir, repo, embedder=embedder)


def test_a_bad_query_raises_after_a_successful_open(retriever):
    with pytest.raises(QueryError):
        retriever.retrieve("   ", 3)


# ---- Windows-style paths: spaces and non-ASCII ---------------------------------------------------


def test_spaces_and_non_ascii_in_repository_and_index_paths(tmp_path, embedder):
    repo = write_repo(tmp_path / "répo with spaces 日本" / "my project")
    files = {"src/módulo.py": "def función():\n    return 'ünï'\n"}
    write_repo(repo, files)
    path = build(repo, tmp_path / "índices con espacio", embedder)
    retriever = Retriever.open(path, repo, embedder=embedder)
    results = retriever.retrieve("función módulo", 8)
    hit = next(r for r in results if r.file_path == "src/módulo.py")
    assert "ünï" in hit.text and not np.isnan(hit.score)
