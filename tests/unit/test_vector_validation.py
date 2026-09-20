"""Pre-indexing consistency checks: everything wrong must fail loudly and specifically."""

import numpy as np
import pytest

from copilot.vectorstore import EmptyIndexError, IndexBuildError, VectorIndex
from copilot.vectorstore.validation import (
    NORM_TOLERANCE,
    canonical_order_key,
    sort_chunks_canonically,
    validate_vectors,
)
from tests.vectorstore_helpers import build_test_index, make_chunker, make_files, make_provenance

DIM = 32


def unit(n: int = 3, dim: int = DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, dim)).astype(np.float32)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def rebuild(chunks=None, vectors=None, **spec_changes):
    """Call VectorIndex.build with a valid base, replacing chunks/vectors/spec fields."""
    index, base_chunks, _ = build_test_index()
    chunks = base_chunks if chunks is None else chunks
    vectors = unit(len(base_chunks)) if vectors is None else vectors
    spec = index.spec.model_copy(update=spec_changes)
    return VectorIndex.build(
        spec=spec, provenance=make_provenance(), chunks=chunks, vectors=vectors
    )


def test_a_valid_input_builds():
    assert rebuild().count == 6


def test_wrong_dimension_is_rejected():
    n = len(build_test_index()[1])
    with pytest.raises(IndexBuildError, match="dimension 16 does not match"):
        rebuild(vectors=unit(n, dim=16))


def test_nan_and_inf_are_rejected():
    n = len(build_test_index()[1])
    for bad in (np.nan, np.inf, -np.inf):
        vectors = unit(n).copy()
        vectors[1, 2] = bad
        with pytest.raises(IndexBuildError, match="NaN or infinite"):
            rebuild(vectors=vectors)


def test_unnormalised_vectors_are_rejected():
    n = len(build_test_index()[1])
    with pytest.raises(IndexBuildError, match="not unit length"):
        rebuild(vectors=unit(n) * 2.0)


def test_norm_just_inside_the_tolerance_passes_and_just_outside_fails():
    v = unit(2)
    inside = (v * np.float32(1 + NORM_TOLERANCE / 2)).astype(np.float32)
    outside = (v * np.float32(1 + NORM_TOLERANCE * 3)).astype(np.float32)
    validate_vectors(inside, dimension=DIM)
    with pytest.raises(IndexBuildError):
        validate_vectors(outside, dimension=DIM)


def test_zero_vector_is_rejected():
    v = unit(3).copy()
    v[0] = 0
    with pytest.raises(IndexBuildError, match="not unit length"):
        validate_vectors(v, dimension=DIM)


def test_wrong_dtype_is_rejected():
    with pytest.raises(IndexBuildError, match="float32"):
        validate_vectors(unit().astype(np.float64), dimension=DIM)


def test_wrong_rank_and_type_are_rejected():
    with pytest.raises(IndexBuildError, match="2-D"):
        validate_vectors(unit()[0], dimension=DIM)
    with pytest.raises(IndexBuildError, match="numpy array"):
        validate_vectors([[1.0] * DIM], dimension=DIM)


def test_more_vectors_than_chunks_is_rejected():
    n = len(build_test_index()[1])
    with pytest.raises(IndexBuildError, match=f"{n} chunks but {n + 1} vectors"):
        rebuild(vectors=unit(n + 1))


def test_fewer_vectors_than_chunks_is_rejected():
    n = len(build_test_index()[1])
    with pytest.raises(IndexBuildError, match=f"{n} chunks but {n - 1} vectors"):
        rebuild(vectors=unit(n - 1))


def test_duplicate_chunk_ids_are_rejected():
    _, chunks, _ = build_test_index()
    duplicated = [chunks[0], chunks[0], *chunks[2:]]
    with pytest.raises(IndexBuildError, match="duplicate chunk ids"):
        rebuild(chunks=duplicated)


def test_chunks_out_of_canonical_order_are_rejected_not_silently_reordered():
    _, chunks, _ = build_test_index()
    with pytest.raises(IndexBuildError, match="canonical order"):
        rebuild(chunks=list(reversed(chunks)))


def test_empty_vectors_raise_the_empty_index_error():
    with pytest.raises(EmptyIndexError):
        rebuild(vectors=np.empty((0, DIM), dtype=np.float32))


def test_empty_chunks_raise_the_empty_index_error():
    with pytest.raises(EmptyIndexError, match="no chunks"):
        rebuild(chunks=[], vectors=unit(1))


def test_an_empty_error_is_also_a_build_error():
    assert issubclass(EmptyIndexError, IndexBuildError)


def test_chunk_from_another_repository_is_rejected():
    _, chunks, _ = build_test_index()
    foreign = [chunks[0].model_copy(update={"repository_name": "other"}), *chunks[1:]]
    with pytest.raises(IndexBuildError, match="does not belong"):
        rebuild(chunks=foreign)


def test_chunk_from_another_strategy_version_is_rejected():
    _, chunks, _ = build_test_index()
    stale = [chunks[0].model_copy(update={"chunking_version": 2}), *chunks[1:]]
    with pytest.raises(IndexBuildError, match="does not belong"):
        rebuild(chunks=stale)


def test_unsupported_index_type_and_unnormalised_embedder_are_rejected():
    with pytest.raises(IndexBuildError, match="only IndexFlatIP"):
        rebuild(index_type="IndexHNSWFlat")
    with pytest.raises(IndexBuildError, match="normalised"):
        rebuild(embedding_normalized=False)


def test_canonical_order_is_by_code_point_path_then_chunk_index():
    files = make_files({"a/b.py": "x = 1\n", "a.py": "x = 2\n", "B.py": "x = 3\n", "z.py": "y\n"})
    chunker = make_chunker(size=1, overlap=0)
    shuffled = [c for f in reversed(files) for c in chunker.chunk_file(f)]
    ordered = sort_chunks_canonically(shuffled)
    assert [c.file_path for c in ordered] == ["B.py", "a.py", "a/b.py", "z.py"]
    assert [canonical_order_key(c) for c in ordered] == sorted(
        canonical_order_key(c) for c in ordered
    )


def test_chunk_index_is_compared_numerically_not_as_text():
    files = make_files({"big.py": "".join(f"v{i} = {i}\n" for i in range(12))})
    ordered = sort_chunks_canonically(list(reversed(make_chunker(1, 0).chunk_file(files[0]))))
    assert [c.chunk_index for c in ordered] == list(range(12))  # 2 before 10
