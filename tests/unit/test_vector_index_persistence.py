"""Save/load round trips, manifest and mapping fidelity, corruption and compatibility refusals."""

import shutil

import numpy as np
import pytest

from copilot.vectorstore import (
    IndexBuildError,
    IndexCompatibilityError,
    IndexCorruptError,
    IndexExistsError,
    IndexExpectation,
    IndexManifest,
    VectorIndex,
)
from copilot.vectorstore import faiss_backend as backend
from copilot.vectorstore.index import read_manifest
from copilot.vectorstore.manifest import CHUNKS_FILE, INDEX_FILE, MANIFEST_FILE
from copilot.vectorstore.records import decode_records, encode_records
from tests.vectorstore_helpers import (
    ARTIFACT_NAMES,
    build_test_index,
    edit_manifest,
    read_manifest_json,
    rewrite_artifact,
)


@pytest.fixture
def saved(tmp_path):
    """(index, directory it was saved to)."""
    index, _, _ = build_test_index()
    return index, index.save(tmp_path / "indexes")


def load(directory, **expected):
    return VectorIndex.load(directory, expected=IndexExpectation(**expected))


# ------------------------------------------------------------------------- round trip
def test_save_creates_the_documented_layout(saved, tmp_path):
    index, path = saved
    assert path == tmp_path / "indexes" / index.index_id
    assert sorted(p.name for p in path.iterdir()) == sorted(ARTIFACT_NAMES)
    assert [p.name for p in (tmp_path / "indexes").iterdir()] == [index.index_id]  # no staging


def test_round_trip_preserves_identity_counts_mapping_and_manifest(saved):
    index, path = saved
    expected_vectors = np.stack([index.reconstruct(i) for i in range(index.count)])
    expected_records = index.records
    expected_manifest = index.manifest
    expected_id = index.index_id
    del index  # destroy the in-memory object; only the files remain

    loaded = load(path)
    assert loaded.index_id == expected_id
    assert loaded.count == len(expected_records)
    assert loaded.dimension == 32
    assert loaded.records == expected_records  # same mapping, same order
    assert loaded.manifest == expected_manifest
    actual = np.stack([loaded.reconstruct(i) for i in range(loaded.count)])
    assert np.array_equal(actual, expected_vectors)  # bit-identical vectors


def test_round_trip_gives_identical_low_level_search_results(saved):
    index, path = saved
    queries = np.stack([index.reconstruct(i) for i in range(index.count)])
    before = index._search_positions(queries, 4)
    loaded = load(path)
    after = loaded._search_positions(queries, 4)
    assert np.array_equal(before[1], after[1])
    assert np.array_equal(before[0], after[0])


def test_loaded_index_can_be_saved_again_with_identical_bytes(saved, tmp_path):
    _, path = saved
    again = load(path).save(tmp_path / "copy")
    for name in ARTIFACT_NAMES:
        assert (again / name).read_bytes() == (path / name).read_bytes()


def test_load_with_the_full_expectation_of_its_own_spec_succeeds(saved):
    index, path = saved
    assert load(path, **index.spec.model_dump()).index_id == index.index_id
    assert VectorIndex.load(path, expected=IndexExpectation.from_spec(index.spec))


def test_verify_vectors_passes_on_a_healthy_index(saved):
    check = load(saved[1]).verify_vectors()
    assert check.vectors_checked == 6 and check.dimension == 32
    assert check.max_norm_deviation < 1e-5 and check.min_self_score > 0.999


def test_saving_twice_without_overwrite_is_refused_and_with_overwrite_is_identical(saved, tmp_path):
    index, path = saved
    before = {n: (path / n).read_bytes() for n in ARTIFACT_NAMES}
    with pytest.raises(IndexExistsError):
        index.save(tmp_path / "indexes")
    assert {n: (path / n).read_bytes() for n in ARTIFACT_NAMES} == before
    index.save(tmp_path / "indexes", overwrite=True)
    assert {n: (path / n).read_bytes() for n in ARTIFACT_NAMES} == before


# ------------------------------------------------------------------------- manifest
def test_manifest_json_round_trips_to_the_same_model(saved):
    index, path = saved
    parsed = IndexManifest.model_validate(read_manifest_json(path))
    assert parsed == index.manifest
    assert read_manifest(path) == index.manifest  # readable without loading FAISS


def test_manifest_records_the_compatibility_critical_fields(saved):
    _, path = saved
    raw = read_manifest_json(path)
    spec = raw["spec"]
    assert raw["schema_version"] == "vector-index-manifest/1"
    assert raw["index_id"] and raw["provenance"]["created_at"].endswith("Z")
    assert spec["repository_name"] == "repo" and len(spec["repository_fingerprint"]) == 64
    assert spec["chunking_strategy"] == "line" and spec["chunking_strategy_version"] == 1
    assert spec["chunk_params"] == {"size_lines": 4, "overlap_lines": 1, "max_tokens": 512}
    assert spec["embedding_model_id"] == "fake/hash-embedder"
    assert spec["embedding_dimension"] == 32 and spec["embedding_normalized"] is True
    assert spec["text_style"] == "prefixed" and spec["representation_version"] == 1
    assert spec["index_type"] == "IndexFlatIP" and spec["metric"] == "inner_product"
    assert raw["vector_count"] == raw["chunk_count"] == 6
    assert set(raw["artifacts"]) == {INDEX_FILE, CHUNKS_FILE}
    assert raw["provenance"]["embedding_runtime"] == "python-hash"


def test_no_artifact_contains_host_paths_or_source_text(saved, tmp_path):
    _, path = saved
    for name in (MANIFEST_FILE, CHUNKS_FILE):
        text = (path / name).read_text()
        assert str(tmp_path) not in text and str(path) not in text
        assert "/home/" not in text and "C:\\\\" not in text and "\\\\Users" not in text
        assert "return 1" not in text and "some text" not in text


def test_chunk_mapping_round_trip_from_disk(saved):
    index, path = saved
    from_disk = decode_records((path / CHUNKS_FILE).read_bytes())
    assert from_disk == list(index.records)
    assert encode_records(from_disk) == (path / CHUNKS_FILE).read_bytes()


# ------------------------------------------------------------------------- corruption
def test_a_missing_directory_is_reported(tmp_path):
    with pytest.raises(IndexCorruptError, match="does not exist"):
        load(tmp_path / "nope")


@pytest.mark.parametrize("name", ARTIFACT_NAMES)
def test_a_missing_artifact_is_refused(saved, name):
    _, path = saved
    (path / name).unlink()
    with pytest.raises(IndexCorruptError, match=f"missing artifact: {name}"):
        load(path)


@pytest.mark.parametrize(
    "garbage", [b"", b"{not json", b"\xff\xfe\x00", b"[1, 2, 3]", b'"just a string"', b"null"]
)
def test_a_corrupt_manifest_is_refused(saved, garbage):
    _, path = saved
    (path / MANIFEST_FILE).write_bytes(garbage)
    with pytest.raises(IndexCorruptError):
        load(path)


def test_a_manifest_missing_a_key_is_refused(saved):
    _, path = saved
    edit_manifest(path, lambda m: m.pop("chunk_ids_sha256"))
    with pytest.raises(IndexCorruptError, match="schema"):
        load(path)


def test_a_manifest_with_an_unknown_key_is_refused(saved):
    _, path = saved
    edit_manifest(path, lambda m: m.update(surprise=1))
    with pytest.raises(IndexCorruptError):
        load(path)


def test_a_hand_edited_spec_no_longer_matches_its_index_id(saved):
    _, path = saved
    edit_manifest(path, lambda m: m["spec"].update(embedding_model_id="sneaky/model"))
    with pytest.raises(IndexCorruptError, match="index_id does not match"):
        load(path)


def test_a_modified_index_file_is_refused(saved):
    _, path = saved
    data = bytearray((path / INDEX_FILE).read_bytes())
    data[-1] ^= 0xFF
    (path / INDEX_FILE).write_bytes(bytes(data))
    with pytest.raises(IndexCorruptError, match="index.faiss does not match the manifest"):
        load(path)


def test_a_truncated_index_file_is_refused(saved):
    _, path = saved
    data = (path / INDEX_FILE).read_bytes()
    (path / INDEX_FILE).write_bytes(data[: len(data) // 2])
    with pytest.raises(IndexCorruptError, match="does not match the manifest"):
        load(path)


def test_a_modified_chunk_mapping_is_refused(saved):
    _, path = saved
    (path / CHUNKS_FILE).write_bytes(
        (path / CHUNKS_FILE).read_bytes().replace(b"src/a.py", b"src/z.py")
    )
    with pytest.raises(IndexCorruptError, match="chunks.jsonl does not match"):
        load(path)


def test_undeserialisable_faiss_bytes_are_refused_even_with_a_matching_checksum(saved):
    _, path = saved
    rewrite_artifact(path, INDEX_FILE, b"this is not a faiss index" * 10)
    with pytest.raises(IndexCorruptError, match="could not be deserialised"):
        load(path)


def test_faiss_count_mismatch_is_refused(saved):
    index, path = saved
    vectors = np.stack([index.reconstruct(i) for i in range(index.count)])
    rewrite_artifact(path, INDEX_FILE, backend.serialize(backend.build_flat_ip(vectors[:-1])))
    with pytest.raises(IndexCorruptError, match="FAISS holds 5 vectors, manifest 6"):
        load(path)


def test_faiss_dimension_mismatch_is_refused(saved):
    _, path = saved
    other = np.full((6, 16), 0.25, dtype=np.float32)  # 16-d, unit length
    rewrite_artifact(path, INDEX_FILE, backend.serialize(backend.build_flat_ip(other)))
    with pytest.raises(IndexCorruptError, match="FAISS dimension 16"):
        load(path)


def test_wrong_faiss_index_type_is_refused(saved):
    import faiss

    index, path = saved
    vectors = np.stack([index.reconstruct(i) for i in range(index.count)])
    l2 = faiss.IndexFlatL2(32)
    l2.add(vectors)
    rewrite_artifact(path, INDEX_FILE, backend.serialize(l2))
    with pytest.raises(IndexCorruptError, match="FAISS index is IndexFlatL2"):
        load(path)


def test_manifest_vector_count_that_disagrees_with_faiss_is_refused(saved):
    _, path = saved
    edit_manifest(path, lambda m: m.update(vector_count=7))
    with pytest.raises(IndexCorruptError, match="FAISS holds 6 vectors, manifest 7"):
        load(path)


def test_mapping_shorter_than_the_index_is_refused(saved):
    _, path = saved
    lines = (path / CHUNKS_FILE).read_bytes().split(b"\n")[:-1]
    rewrite_artifact(path, CHUNKS_FILE, b"\n".join(lines[:-1]) + b"\n")
    with pytest.raises(IndexCorruptError, match="chunk mapping has 5 rows, manifest says 6"):
        load(path)


def test_reordered_mapping_rows_are_refused(saved):
    _, path = saved
    lines = (path / CHUNKS_FILE).read_bytes().split(b"\n")[:-1]
    lines[0], lines[1] = lines[1], lines[0]
    rewrite_artifact(path, CHUNKS_FILE, b"\n".join(lines) + b"\n")
    with pytest.raises(IndexCorruptError, match="positions are not 0..n-1"):
        load(path)


def test_duplicate_chunk_ids_in_the_mapping_are_refused(saved):
    _, path = saved
    records = decode_records((path / CHUNKS_FILE).read_bytes())
    records[1] = records[1].model_copy(update={"chunk_id": records[0].chunk_id})
    rewrite_artifact(path, CHUNKS_FILE, encode_records(records))
    with pytest.raises(IndexCorruptError, match="duplicate chunk ids"):
        load(path)


def test_a_substituted_chunk_id_is_caught_by_the_digest(saved):
    _, path = saved
    records = decode_records((path / CHUNKS_FILE).read_bytes())
    records[2] = records[2].model_copy(update={"chunk_id": "f" * 16})
    rewrite_artifact(path, CHUNKS_FILE, encode_records(records))
    with pytest.raises(IndexCorruptError, match="digest"):
        load(path)


def test_a_malformed_mapping_line_is_refused(saved):
    _, path = saved
    rewrite_artifact(path, CHUNKS_FILE, b'{"position": 0}\n')
    with pytest.raises(IndexCorruptError, match="line 1 is invalid"):
        load(path)


def test_unexpected_artifact_listing_is_refused(saved):
    _, path = saved
    edit_manifest(path, lambda m: m["artifacts"].pop(CHUNKS_FILE))
    with pytest.raises(IndexCorruptError, match="unexpected artifacts"):
        load(path)


def test_verify_vectors_catches_unnormalised_stored_vectors(saved):
    index, path = saved
    vectors = np.stack([index.reconstruct(i) for i in range(index.count)]) * 3.0
    rewrite_artifact(path, INDEX_FILE, backend.serialize(backend.build_flat_ip(vectors)))
    loaded = load(path)  # structurally consistent...
    with pytest.raises(IndexCorruptError, match="not unit length"):
        loaded.verify_vectors()  # ...but the deep check refuses it


def test_verify_vectors_catches_nan(saved):
    index, path = saved
    vectors = np.stack([index.reconstruct(i) for i in range(index.count)])
    vectors[0, 0] = np.nan
    rewrite_artifact(path, INDEX_FILE, backend.serialize(backend.build_flat_ip(vectors)))
    with pytest.raises(IndexCorruptError, match="NaN"):
        load(path).verify_vectors()


# ------------------------------------------------------------------------- compatibility
def test_an_incompatible_manifest_schema_is_refused_as_incompatible(saved):
    _, path = saved
    edit_manifest(path, lambda m: m.update(schema_version="vector-index-manifest/999"))
    with pytest.raises(IndexCompatibilityError, match="unsupported manifest schema") as info:
        load(path)
    assert info.value.mismatches[0][0] == "schema_version"


def test_a_missing_schema_version_is_refused_as_incompatible(saved):
    _, path = saved
    edit_manifest(path, lambda m: m.pop("schema_version"))
    with pytest.raises(IndexCompatibilityError):
        load(path)


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("embedding_model_id", "jinaai/jina-embeddings-v2-base-code"),
        ("embedding_dimension", 768),
        ("embedding_normalized", False),
        ("text_style", "raw"),
        ("representation_version", 2),
        ("chunking_strategy", "ast"),
        ("chunking_strategy_version", 2),
        ("chunk_params", {"size_lines": 60, "overlap_lines": 10, "max_tokens": 512}),
        ("chunk_id_schema", "chunk-id/2"),
        ("repository_fingerprint", "0" * 64),
        ("repository_name", "some-other-repo"),
        ("index_type", "IndexHNSWFlat"),
        ("metric", "l2"),
    ],
)
def test_each_incompatible_field_is_refused_and_named(saved, field, wrong):
    _, path = saved
    with pytest.raises(IndexCompatibilityError, match=field) as info:
        load(path, **{field: wrong})
    assert [m[0] for m in info.value.mismatches] == [field]
    assert info.value.mismatches[0][1] == wrong  # expected value
    assert isinstance(info.value, Exception) and not isinstance(info.value, IndexCorruptError)


def test_all_mismatches_are_reported_together(saved):
    _, path = saved
    with pytest.raises(IndexCompatibilityError) as info:
        load(path, embedding_model_id="x/y", text_style="raw", chunk_params={"size_lines": 1})
    assert {m[0] for m in info.value.mismatches} == {
        "embedding_model_id",
        "text_style",
        "chunk_params",
    }


def test_a_failed_load_does_not_modify_or_rebuild_anything(saved):
    _, path = saved
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    with pytest.raises(IndexCompatibilityError):
        load(path, text_style="raw")
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before


def test_compatibility_is_checked_before_any_large_artifact_is_read(saved):
    _, path = saved
    (path / INDEX_FILE).unlink()  # would be "corrupt", but the mismatch is reported first
    with pytest.raises(IndexCompatibilityError):
        load(path, text_style="raw")


def test_an_index_moved_to_another_directory_still_loads(saved, tmp_path):
    index, path = saved
    moved = tmp_path / "somewhere" / "renamed-dir"
    moved.parent.mkdir()
    shutil.copytree(path, moved)
    assert load(moved).index_id == index.index_id


def test_load_requires_an_explicit_expectation(saved):
    _, path = saved
    with pytest.raises(TypeError):
        VectorIndex.load(path)  # type: ignore[call-arg]


def test_save_refuses_when_in_memory_bytes_drifted_from_the_manifest(saved, tmp_path):
    _, path = saved
    loaded = load(path)
    loaded._artifact_bytes = {**loaded._artifact_bytes, CHUNKS_FILE: b"tampered\n"}
    with pytest.raises(IndexCorruptError, match="no longer matches"):
        loaded.save(tmp_path / "elsewhere")
    assert not (tmp_path / "elsewhere").exists()  # nothing was written


def test_build_errors_are_not_confused_with_load_errors():
    assert not issubclass(IndexBuildError, IndexCorruptError)
    assert not issubclass(IndexCompatibilityError, IndexCorruptError)
