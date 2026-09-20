"""``VectorIndex``: an exact FAISS index plus its verified position <-> chunk mapping.

Persisted layout (``<indexes_dir>/<index_id>/``)::

    manifest.json   compatibility/build record (schema, spec, provenance, counts, checksums)
    index.faiss     the FAISS ``IndexFlatIP`` bytes
    chunks.jsonl    line i = citation metadata of vector i (no source text)

**Vector ordering contract.** Vector ``i`` belongs to the ``i``-th chunk in canonical order:
ascending ``(file_path, chunk_index)`` (see ``validation.canonical_order_key``). ``build``
*verifies* that its input is in that order instead of silently reordering, so vectors can never
be paired with the wrong chunk by a caller's mistake; the pipeline sorts explicitly before
embedding.

**Loading is strict.** ``load`` requires an ``IndexExpectation`` and never repairs or rebuilds: a
damaged index raises ``IndexCorruptError``; an intact but mismatching one raises
``IndexCompatibilityError``. Pass ``IndexExpectation()`` to check internal consistency only.

Security note: FAISS deserialisation is not hardened against maliciously crafted files. The
SHA-256 checks catch accidental damage, not an attacker who can edit both the file and the manifest.
Only load indexes this application built.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from copilot.models.chunk import Chunk
from copilot.vectorstore import faiss_backend as backend
from copilot.vectorstore.atomic import write_directory_atomically
from copilot.vectorstore.errors import (
    IndexBuildError,
    IndexCompatibilityError,
    IndexCorruptError,
    SearchInputError,
)
from copilot.vectorstore.manifest import (
    CHUNKS_FILE,
    INDEX_FILE,
    INDEX_TYPE,
    MANIFEST_FILE,
    MANIFEST_SCHEMA,
    METRIC,
    ArtifactInfo,
    IndexExpectation,
    IndexManifest,
    IndexSpec,
    Provenance,
    compute_index_id,
    find_mismatches,
)
from copilot.vectorstore.records import (
    ChunkRecord,
    chunk_ids_digest,
    decode_records,
    encode_records,
    record_from_chunk,
)
from copilot.vectorstore.validation import NORM_TOLERANCE, validate_chunks, validate_vectors


def _artifact_info(data: bytes) -> ArtifactInfo:
    return ArtifactInfo(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def _manifest_bytes(manifest: IndexManifest) -> bytes:
    text = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=True)
    return (text + "\n").encode("utf-8")


@dataclass(frozen=True)
class SearchHit:
    """One search result: the vector's position in the index and its similarity to the query.

    ``score`` is the inner product of two unit vectors, i.e. the cosine similarity in ``[-1, 1]``.
    """

    position: int
    score: float


@dataclass(frozen=True)
class VectorCheck:
    """Result of ``VectorIndex.verify_vectors`` (numbers only)."""

    vectors_checked: int
    dimension: int
    max_norm_deviation: float
    self_match_sampled: int
    min_self_score: float


def read_manifest(directory: str | Path) -> IndexManifest:
    """Read and validate ``manifest.json`` only (no FAISS, no embedding model).

    Raises ``IndexCorruptError`` if it is missing/unparsable or its ``index_id`` does not match the
    spec it contains (a hand-edited manifest), and ``IndexCompatibilityError`` for an unsupported
    ``schema_version``.
    """
    root = Path(directory)
    if not root.is_dir():
        raise IndexCorruptError("index directory does not exist")
    path = root / MANIFEST_FILE
    try:
        raw = json.loads(path.read_bytes())
    except FileNotFoundError as exc:
        raise IndexCorruptError(f"missing artifact: {MANIFEST_FILE}") from exc
    except (OSError, ValueError) as exc:
        raise IndexCorruptError(f"{MANIFEST_FILE} is unreadable or not valid JSON") from exc
    if not isinstance(raw, dict):
        raise IndexCorruptError(f"{MANIFEST_FILE} must contain a JSON object")
    schema = raw.get("schema_version")
    if schema != MANIFEST_SCHEMA:
        raise IndexCompatibilityError(
            f"unsupported manifest schema {schema!r} (this version reads {MANIFEST_SCHEMA!r}); "
            "rebuild the index",
            (("schema_version", MANIFEST_SCHEMA, schema),),
        )
    try:
        manifest = IndexManifest.model_validate(raw)
    except ValueError as exc:
        raise IndexCorruptError(f"{MANIFEST_FILE} does not match the schema") from exc
    if compute_index_id(manifest.spec) != manifest.index_id:
        raise IndexCorruptError("manifest index_id does not match its spec (edited or corrupt)")
    return manifest


class VectorIndex:
    """An in-memory index; create it with ``build`` or ``load`` rather than directly."""

    def __init__(
        self,
        *,
        manifest: IndexManifest,
        records: tuple[ChunkRecord, ...],
        faiss_index: Any,
        artifact_bytes: dict[str, bytes] | None = None,
    ) -> None:
        self._manifest = manifest
        self._records = records
        self._faiss = faiss_index
        self._artifact_bytes = artifact_bytes
        self._position_by_id: dict[str, int] | None = None

    # ------------------------------------------------------------------ introspection
    @property
    def manifest(self) -> IndexManifest:
        return self._manifest

    @property
    def index_id(self) -> str:
        return self._manifest.index_id

    @property
    def spec(self) -> IndexSpec:
        return self._manifest.spec

    @property
    def count(self) -> int:
        """Number of stored vectors (== number of chunks)."""
        return backend.total(self._faiss)

    @property
    def dimension(self) -> int:
        return backend.dimension(self._faiss)

    @property
    def records(self) -> tuple[ChunkRecord, ...]:
        """Sidecar records; ``records[i]`` describes vector ``i``."""
        return self._records

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(r.chunk_id for r in self._records)

    def record_at(self, position: int) -> ChunkRecord:
        """Record of the vector at ``position`` (``IndexError`` when out of range)."""
        if not 0 <= position < len(self._records):
            raise IndexError(f"position {position} out of range (0..{len(self._records) - 1})")
        return self._records[position]

    def position_of(self, chunk_id: str) -> int:
        """Vector position of ``chunk_id`` (``KeyError`` when it is not indexed)."""
        if self._position_by_id is None:
            self._position_by_id = {r.chunk_id: r.position for r in self._records}
        return self._position_by_id[chunk_id]

    def reconstruct(self, position: int) -> np.ndarray:
        """The stored vector at ``position`` (copy)."""
        self.record_at(position)
        return np.asarray(self._faiss.reconstruct(position), dtype=np.float32)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        *,
        spec: IndexSpec,
        provenance: Provenance,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
    ) -> VectorIndex:
        """Validate the inputs and build the index. ``vectors[i]`` must belong to ``chunks[i]``.

        ``chunks`` must already be in canonical order. Raises ``EmptyIndexError`` for no data and
        ``IndexBuildError`` for any inconsistency (dimension, NaN/inf, normalisation, duplicate ids,
        count mismatch, wrong order, chunks that disagree with ``spec``).
        """
        if spec.index_type != INDEX_TYPE or spec.metric != METRIC:
            raise IndexBuildError(f"only {INDEX_TYPE} with metric {METRIC!r} is supported")
        if not spec.embedding_normalized:
            raise IndexBuildError("cosine similarity via inner product needs normalised vectors")
        validate_vectors(vectors, dimension=spec.embedding_dimension)
        validate_chunks(chunks, vector_count=int(vectors.shape[0]))
        for chunk in chunks:
            if (
                chunk.repository_name != spec.repository_name
                or chunk.chunking_strategy != spec.chunking_strategy
                or chunk.chunking_version != spec.chunking_strategy_version
            ):
                raise IndexBuildError(
                    f"chunk {chunk.chunk_id} does not belong to the index spec "
                    "(repository or chunking strategy/version differs)"
                )

        faiss_index = backend.build_flat_ip(vectors)
        if backend.total(faiss_index) != len(chunks):
            raise IndexBuildError("FAISS holds a different number of vectors than chunks")
        records = tuple(record_from_chunk(c, i) for i, c in enumerate(chunks))
        artifact_bytes = {
            INDEX_FILE: backend.serialize(faiss_index),
            CHUNKS_FILE: encode_records(records),
        }
        manifest = IndexManifest(
            schema_version=MANIFEST_SCHEMA,
            index_id=compute_index_id(spec),
            spec=spec,
            provenance=provenance,
            vector_count=len(chunks),
            chunk_count=len(records),
            chunk_ids_sha256=chunk_ids_digest([r.chunk_id for r in records]),
            artifacts={name: _artifact_info(data) for name, data in artifact_bytes.items()},
        )
        return cls(
            manifest=manifest,
            records=records,
            faiss_index=faiss_index,
            artifact_bytes=artifact_bytes,
        )

    # ------------------------------------------------------------------ save
    def save(self, indexes_dir: str | Path, *, overwrite: bool = False) -> Path:
        """Atomically write ``<indexes_dir>/<index_id>/`` and return that directory.

        Raises ``IndexExistsError`` if it exists and ``overwrite`` is false. A failed write leaves
        no directory at the final path (see ``atomic.py``).
        """
        artifacts = self._artifact_bytes or {
            INDEX_FILE: backend.serialize(self._faiss),
            CHUNKS_FILE: encode_records(self._records),
        }
        for name, data in artifacts.items():  # a loaded index re-derives bytes: they must agree
            if _artifact_info(data) != self._manifest.artifacts[name]:
                raise IndexCorruptError(f"in-memory {name} no longer matches the manifest")
        files = {**artifacts, MANIFEST_FILE: _manifest_bytes(self._manifest)}  # manifest last
        return write_directory_atomically(
            Path(indexes_dir) / self.index_id, files, overwrite=overwrite
        )

    # ------------------------------------------------------------------ load / validate
    @classmethod
    def load(cls, directory: str | Path, *, expected: IndexExpectation) -> VectorIndex:
        """Load and fully verify an index, refusing anything damaged or incompatible.

        Order of checks: manifest readable/schema/id -> ``expected`` (cheap, before any large read)
        -> artifact presence, size and SHA-256 -> mapping structure -> FAISS type/dimension/count.
        """
        root = Path(directory)
        manifest = read_manifest(root)

        mismatches = find_mismatches(manifest.spec, expected)
        if mismatches:
            raise IndexCompatibilityError(
                "index is incompatible with the expected configuration: "
                + "; ".join(str(m) for m in mismatches),
                tuple((m.field, m.expected, m.actual) for m in mismatches),
            )

        if set(manifest.artifacts) != {INDEX_FILE, CHUNKS_FILE}:
            raise IndexCorruptError("manifest lists unexpected artifacts")
        loaded: dict[str, bytes] = {}
        for name, info in manifest.artifacts.items():
            try:
                data = (root / name).read_bytes()
            except FileNotFoundError as exc:
                raise IndexCorruptError(f"missing artifact: {name}") from exc
            except OSError as exc:
                raise IndexCorruptError(f"could not read artifact {name}") from exc
            if _artifact_info(data) != info:
                raise IndexCorruptError(f"{name} does not match the manifest (size or SHA-256)")
            loaded[name] = data

        records = tuple(decode_records(loaded[CHUNKS_FILE]))
        if len(records) != manifest.chunk_count:
            raise IndexCorruptError(
                f"chunk mapping has {len(records)} rows, manifest says {manifest.chunk_count}"
            )
        if [r.position for r in records] != list(range(len(records))):
            raise IndexCorruptError("chunk mapping positions are not 0..n-1 in order")
        ids = [r.chunk_id for r in records]
        if len(set(ids)) != len(ids):
            raise IndexCorruptError("chunk mapping contains duplicate chunk ids")
        if chunk_ids_digest(ids) != manifest.chunk_ids_sha256:
            raise IndexCorruptError("chunk id order/content does not match the manifest digest")

        faiss_index = backend.deserialize(loaded[INDEX_FILE])
        actual_type = backend.index_type_name(faiss_index)
        if actual_type != manifest.spec.index_type or not backend.is_inner_product(faiss_index):
            raise IndexCorruptError(
                f"FAISS index is {actual_type}, manifest declares {manifest.spec.index_type}"
            )
        if backend.dimension(faiss_index) != manifest.spec.embedding_dimension:
            raise IndexCorruptError(
                f"FAISS dimension {backend.dimension(faiss_index)} does not match the manifest "
                f"({manifest.spec.embedding_dimension})"
            )
        if not backend.total(faiss_index) == manifest.vector_count == len(records):
            raise IndexCorruptError(
                f"FAISS holds {backend.total(faiss_index)} vectors, manifest "
                f"{manifest.vector_count}, mapping {len(records)}"
            )
        return cls(
            manifest=manifest, records=records, faiss_index=faiss_index, artifact_bytes=loaded
        )

    def verify_vectors(self, *, sample: int = 64) -> VectorCheck:
        """Deep check of the stored vectors (used by the ``validate`` command).

        Confirms every stored vector is finite and unit length, and that a deterministic sample of
        vectors finds itself with similarity ~1 when used as a query. This is a serialisation
        sanity check, **not** retrieval.
        """
        vectors = backend.reconstruct_all(self._faiss)
        if not np.isfinite(vectors).all():
            raise IndexCorruptError("stored vectors contain NaN or infinite values")
        deviation = float(np.abs(np.linalg.norm(vectors, axis=1) - 1.0).max())
        if deviation > NORM_TOLERANCE:
            raise IndexCorruptError(
                f"stored vectors are not unit length (deviation {deviation:.3g})"
            )
        n = vectors.shape[0]
        picks = np.unique(np.linspace(0, n - 1, num=min(n, sample)).astype(np.int64))
        scores, _ = backend.search(self._faiss, vectors[picks], 1)
        min_score = float(scores[:, 0].min())
        if min_score < 1.0 - NORM_TOLERANCE:
            raise IndexCorruptError(f"a stored vector does not find itself (score {min_score:.4f})")
        return VectorCheck(
            vectors_checked=int(n),
            dimension=int(vectors.shape[1]),
            max_norm_deviation=deviation,
            self_match_sampled=int(len(picks)),
            min_self_score=min_score,
        )

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchHit]:
        """The ``top_k`` stored vectors most similar to ``query_vector``, best first.

        * ``query_vector`` must be a 1-D float array of this index's dimension, finite and of unit
          length (the same guarantee the embedding layer gives). Anything else raises
          ``SearchInputError``: a wrong-model or un-normalised query would silently give wrong
          "cosine" scores.
        * ``top_k`` must be an ``int >= 1``. It may exceed the index size: you then get every vector
          (fewer than ``top_k`` results). FAISS pads missing results with position ``-1``; those
          pads are dropped here and are never mapped to a chunk.
        * **Deterministic order:** results are sorted by score descending, then by ascending
          position. Exact ties (equal ``float32`` scores, e.g. identical chunk text) are therefore
          always ordered the same way, and a tie at the ``top_k`` boundary is resolved in favour of
          the lower position: the search is widened until the boundary is unambiguous.
        """
        query = self._validated_query(query_vector)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise SearchInputError(f"top_k must be an integer >= 1, got {top_k!r}")
        count = self.count
        if count == 0:
            return []
        wanted = min(top_k, count)
        fetch = min(count, wanted + 1)  # one extra result reveals a tie at the boundary
        while True:
            scores, positions = backend.search(self._faiss, query[np.newaxis, :], fetch)
            pairs = [
                (int(p), float(s)) for p, s in zip(positions[0], scores[0], strict=True) if p >= 0
            ]
            if fetch >= count or len(pairs) < fetch:
                break
            if pairs[fetch - 1][1] < pairs[wanted - 1][1]:  # the last fetched is strictly worse
                break
            fetch = min(count, fetch * 2)
        pairs.sort(key=lambda pair: (-pair[1], pair[0]))
        return [SearchHit(position=p, score=s) for p, s in pairs[:wanted]]

    def _validated_query(self, query_vector: object) -> np.ndarray:
        if not isinstance(query_vector, np.ndarray):
            raise SearchInputError(
                f"query vector must be a numpy array, got {type(query_vector).__name__}"
            )
        if query_vector.dtype.kind != "f":
            raise SearchInputError(f"query vector must be floating point, got {query_vector.dtype}")
        if query_vector.ndim != 1:
            raise SearchInputError(
                f"query vector must be 1-D (dimension,), got shape {query_vector.shape}"
            )
        if query_vector.shape[0] != self.dimension:
            raise SearchInputError(
                f"query dimension {query_vector.shape[0]} does not match the index "
                f"({self.dimension})"
            )
        query = np.ascontiguousarray(query_vector, dtype=np.float32)
        if not np.isfinite(query).all():
            raise SearchInputError("query vector contains NaN or infinite values")
        deviation = abs(float(np.linalg.norm(query)) - 1.0)
        if deviation > NORM_TOLERANCE:
            raise SearchInputError(
                f"query vector is not unit length (|norm - 1| = {deviation:.3g}); "
                "cosine similarity would be wrong"
            )
        return query

    def _search_positions(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Raw ``(scores, positions)`` for already-embedded ``queries`` (serialisation tests).

        Low-level primitive kept for the persistence tests: no validation, no ``-1`` filtering, no
        tie-breaking. Use :meth:`search`.
        """
        return backend.search(self._faiss, queries, k)
