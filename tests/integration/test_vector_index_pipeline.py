"""End to end: synthetic repository -> chunks -> (fake) embeddings -> FAISS -> save -> reload.

Real ingestion, real chunking and real FAISS; only the embedding model is the deterministic fake,
so nothing is downloaded. Dimension 768 matches the production model's shape.
"""

import shutil

import numpy as np
import pytest

from copilot.chunking import chunk_repository, create_chunker
from copilot.config import Settings
from copilot.embeddings import HashEmbedder
from copilot.ingestion import IngestionPolicy, ingest_repository
from copilot.vectorstore import (
    EmptyIndexError,
    IndexBuildError,
    IndexCompatibilityError,
    IndexExistsError,
    VectorIndex,
    build_repository_index,
    expectation_for_repository,
)
from copilot.vectorstore.manifest import CHUNKS_FILE, INDEX_FILE, MANIFEST_FILE, IndexExpectation
from tests.fixtures.synthetic_repo import (
    EXPECTED_ACCEPTED,
    FILES,
    MAIN_PY_MARKER,
    SYNTHETIC_MAX_FILE_SIZE,
    build_synthetic_repo,
)

MODEL_ID = "fake/hash-embedder"


def settings(**overrides) -> Settings:
    values = {
        "max_file_size_bytes": SYNTHETIC_MAX_FILE_SIZE,
        "embedding_model": MODEL_ID,
        "chunk_size_lines": 20,
        "chunk_overlap_lines": 2,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def embedder():
    return HashEmbedder(dimension=768)


def build(repo, indexes, embedder, **kwargs):
    kwargs.setdefault("settings", settings())
    return build_repository_index(repo, embedder=embedder, indexes_dir=indexes, **kwargs)


def test_report_matches_an_independent_ingest_and_chunk(synthetic_repo, tmp_path, embedder):
    report = build(synthetic_repo, tmp_path / "ix", embedder)

    s = settings()
    ingestion = ingest_repository(
        synthetic_repo, IngestionPolicy.from_settings(s), repository_name=None
    )
    chunks = chunk_repository(ingestion, create_chunker(settings=s)).chunks

    assert report.repository_name == synthetic_repo.name
    assert report.files == len(EXPECTED_ACCEPTED) == len(ingestion.files)
    assert report.chunks == report.vectors == len(chunks) > 0
    assert report.dimension == 768
    assert report.index_type == "IndexFlatIP"
    assert report.text_style == "prefixed" and report.representation_version == 1
    assert report.index_path == tmp_path / "ix" / report.index_id
    assert report.seconds_total >= report.seconds_embed >= 0
    assert not report.ingestion_truncated


def test_artifact_sizes_in_the_report_match_the_files(synthetic_repo, tmp_path, embedder):
    report = build(synthetic_repo, tmp_path / "ix", embedder)
    assert set(report.artifact_sizes) == {INDEX_FILE, CHUNKS_FILE, MANIFEST_FILE}
    for name, size in report.artifact_sizes.items():
        assert (report.index_path / name).stat().st_size == size
    # 768 float32 per chunk plus a small header
    assert report.artifact_sizes[INDEX_FILE] >= report.chunks * 768 * 4


def test_save_reload_validate_round_trip(synthetic_repo, tmp_path, embedder):
    report = build(synthetic_repo, tmp_path / "ix", embedder)
    expected = expectation_for_repository(
        synthetic_repo, settings=settings(), text_style="prefixed"
    )
    loaded = VectorIndex.load(report.index_path, expected=expected)
    assert loaded.index_id == report.index_id and loaded.count == report.chunks
    assert loaded.verify_vectors().vectors_checked == report.chunks
    assert loaded.spec.embedding_dimension == 768


def test_vector_i_is_the_embedding_of_record_i(synthetic_repo, tmp_path, embedder):
    report = build(synthetic_repo, tmp_path / "ix", embedder)
    loaded = VectorIndex.load(report.index_path, expected=IndexExpectation())
    s = settings()
    ingestion = ingest_repository(synthetic_repo, IngestionPolicy.from_settings(s))
    by_id = {c.chunk_id: c for c in chunk_repository(ingestion, create_chunker(settings=s)).chunks}
    from copilot.embeddings.representation import embedding_text

    keys = [(r.file_path, r.chunk_index) for r in loaded.records]
    assert keys == sorted(keys)
    for position in range(0, loaded.count, max(1, loaded.count // 12)):  # a spread of positions
        chunk = by_id[loaded.record_at(position).chunk_id]
        expected = embedder.embed_documents([embedding_text(chunk, "prefixed")])[0]
        assert np.array_equal(loaded.reconstruct(position), expected)


def test_builds_are_reproducible_across_locations_and_file_creation_order(tmp_path, embedder):
    first = build_synthetic_repo(tmp_path / "one" / "same-name")
    second = tmp_path / "two" / "same-name"  # different absolute path, files created in reverse
    second.mkdir(parents=True)
    for relative, content in reversed(list(FILES.items())):
        target = second / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode() if isinstance(content, str) else content)

    a = build(first, tmp_path / "ix-a", embedder)
    b = build(second, tmp_path / "ix-b", embedder)

    assert a.index_id == b.index_id
    for name in (INDEX_FILE, CHUNKS_FILE):  # byte-identical vectors and mapping
        assert (a.index_path / name).read_bytes() == (b.index_path / name).read_bytes()
    ma = VectorIndex.load(a.index_path, expected=IndexExpectation()).manifest
    mb = VectorIndex.load(b.index_path, expected=IndexExpectation()).manifest
    assert ma.model_copy(update={"provenance": mb.provenance}) == mb  # only provenance may differ


def test_a_source_change_gives_a_new_identity_and_flags_the_old_index_stale(
    synthetic_repo, tmp_path, embedder
):
    old = build(synthetic_repo, tmp_path / "ix", embedder)
    (synthetic_repo / "src" / "app" / "main.py").write_text("def main():\n    return 42\n")
    new = build(synthetic_repo, tmp_path / "ix", embedder)
    assert new.index_id != old.index_id  # no collision: both indexes coexist

    expectation = expectation_for_repository(synthetic_repo, settings=settings())
    VectorIndex.load(new.index_path, expected=expectation)  # current index matches
    with pytest.raises(IndexCompatibilityError, match="repository_fingerprint"):
        VectorIndex.load(old.index_path, expected=expectation)  # stale index refused


def test_chunk_configuration_changes_identity_and_is_detected(synthetic_repo, tmp_path, embedder):
    base = build(synthetic_repo, tmp_path / "ix", embedder)
    other = build(synthetic_repo, tmp_path / "ix", embedder, settings=settings(chunk_size_lines=10))
    assert other.index_id != base.index_id
    with pytest.raises(IndexCompatibilityError, match="chunk_params"):
        VectorIndex.load(
            base.index_path,
            expected=expectation_for_repository(
                synthetic_repo, settings=settings(chunk_size_lines=10)
            ),
        )


def test_text_style_changes_identity_and_vectors(synthetic_repo, tmp_path, embedder):
    prefixed = build(synthetic_repo, tmp_path / "ix", embedder, text_style="prefixed")
    raw = build(synthetic_repo, tmp_path / "ix", embedder, text_style="raw")
    assert prefixed.index_id != raw.index_id
    assert (prefixed.index_path / INDEX_FILE).read_bytes() != (
        raw.index_path / INDEX_FILE
    ).read_bytes()
    with pytest.raises(IndexCompatibilityError, match="text_style"):
        VectorIndex.load(
            prefixed.index_path,
            expected=expectation_for_repository(
                synthetic_repo, settings=settings(), text_style="raw"
            ),
        )


def test_embedding_model_change_changes_identity(synthetic_repo, tmp_path):
    a = build(synthetic_repo, tmp_path / "ix", HashEmbedder(dimension=768))
    b = build(synthetic_repo, tmp_path / "ix", HashEmbedder(dimension=384))
    assert a.index_id != b.index_id and a.dimension != b.dimension


def test_the_repository_name_is_part_of_the_identity(synthetic_repo, tmp_path, embedder):
    a = build(synthetic_repo, tmp_path / "ix", embedder)
    b = build(synthetic_repo, tmp_path / "ix", embedder, repository_name="renamed")
    assert a.index_id != b.index_id and b.repository_name == "renamed"


# ------------------------------------------------------------------------- what is (not) stored
def test_no_artifact_contains_source_text_secrets_files_or_host_paths(
    synthetic_repo, tmp_path, embedder
):
    report = build(synthetic_repo, tmp_path / "ix", embedder)
    for name in (MANIFEST_FILE, CHUNKS_FILE):
        text = (report.index_path / name).read_text()
        assert MAIN_PY_MARKER not in text  # no source text
        assert str(tmp_path) not in text and str(synthetic_repo) not in text  # no host path
        for excluded in (".env", "id_rsa", "server.pem", "credentials.json", "secrets.yaml"):
            assert excluded not in text.replace("src/app/secrets.py", "")
    paths = {
        r.file_path
        for r in VectorIndex.load(report.index_path, expected=IndexExpectation()).records
    }
    assert paths <= set(EXPECTED_ACCEPTED) and paths  # only accepted files were indexed


# ------------------------------------------------------------------------- empty and partial
def test_a_repository_with_no_accepted_files_is_refused_before_embedding(tmp_path):
    repo = tmp_path / "nothing"
    repo.mkdir()
    (repo / "notes.txt").write_text("not a supported type\n")
    (repo / ".env").write_text("FAKE=1\n")

    class Explodes(HashEmbedder):
        def embed_documents(self, texts):
            raise AssertionError("must not embed anything")

    with pytest.raises(EmptyIndexError, match="no accepted files"):
        build(repo, tmp_path / "ix", Explodes(dimension=32))
    assert not (tmp_path / "ix").exists()  # nothing was persisted


def test_a_completely_empty_directory_is_refused(tmp_path, embedder):
    (tmp_path / "void").mkdir()
    with pytest.raises(EmptyIndexError):
        build(tmp_path / "void", tmp_path / "ix", embedder)


def test_accepted_files_that_produce_zero_chunks_are_refused(tmp_path, embedder):
    repo = tmp_path / "blank"
    repo.mkdir()
    (repo / "empty.py").write_text("")
    (repo / "spaces.py").write_text("   \n\n  \n")
    with pytest.raises(EmptyIndexError, match="zero chunks"):
        build(repo, tmp_path / "ix", embedder)
    assert not (tmp_path / "ix").exists()


def test_a_truncated_ingestion_is_refused_unless_explicitly_allowed(
    synthetic_repo, tmp_path, embedder
):
    limited = settings(max_repo_files=2)
    with pytest.raises(IndexBuildError, match="stopped early"):
        build(synthetic_repo, tmp_path / "ix", embedder, settings=limited)
    report = build(
        synthetic_repo, tmp_path / "ix", embedder, settings=limited, allow_truncated=True
    )
    assert report.ingestion_truncated and report.files == 2
    manifest = VectorIndex.load(report.index_path, expected=IndexExpectation()).manifest
    assert manifest.provenance.ingestion_truncated is True


def test_rebuilding_the_same_index_needs_overwrite(synthetic_repo, tmp_path, embedder):
    first = build(synthetic_repo, tmp_path / "ix", embedder)
    with pytest.raises(IndexExistsError):
        build(synthetic_repo, tmp_path / "ix", embedder)
    again = build(synthetic_repo, tmp_path / "ix", embedder, overwrite=True)
    assert again.index_id == first.index_id
    assert not [p for p in (tmp_path / "ix").iterdir() if p.name.startswith(".")]


def test_a_failure_during_embedding_leaves_no_index(synthetic_repo, tmp_path):
    class Fails(HashEmbedder):
        def embed_documents(self, texts):
            raise RuntimeError("embedding blew up")

    with pytest.raises(RuntimeError, match="blew up"):
        build(synthetic_repo, tmp_path / "ix", Fails(dimension=32))
    assert not (tmp_path / "ix").exists()


def test_a_copy_of_the_repository_elsewhere_yields_the_same_index_id(
    synthetic_repo, tmp_path, embedder
):
    copy = tmp_path / "copy" / synthetic_repo.name
    shutil.copytree(synthetic_repo, copy)
    assert build(synthetic_repo, tmp_path / "a", embedder).index_id == (
        build(copy, tmp_path / "b", embedder).index_id
    )
