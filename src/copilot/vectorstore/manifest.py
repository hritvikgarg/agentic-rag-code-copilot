"""The index identity (``IndexSpec``), its deterministic id, and the persisted manifest.

Two layers, kept apart on purpose:

* ``IndexSpec``: the *compatibility-relevant* facts. Two indexes with the same spec are
  interchangeable; a different spec means the vectors/mapping cannot be trusted for the same
  question. The index id is a hash of the spec, nothing else.
* ``IndexManifest``: the spec plus provenance (when it was built, with which runtime), counts and
  artifact checksums. Provenance never affects identity.

**Index id contract** (``INDEX_ID_SCHEMA = "index-id/1"``)::

    payload = {"schema": INDEX_ID_SCHEMA, "spec": IndexSpec as JSON-compatible dict}
    text    = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    index_id = SHA-256(text as UTF-8).hexdigest()[:16]

Same repository state + chunking + embedding model + representation gives the same id; changing any
``IndexSpec`` field changes it. No timestamps, absolute paths or random values are involved.
Adding a field to ``IndexSpec`` deliberately changes every id (a new identity dimension).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MANIFEST_SCHEMA = "vector-index-manifest/1"  # layout/keys of manifest.json
INDEX_ID_SCHEMA = "index-id/1"  # how the id is computed
INDEX_TYPE = "IndexFlatIP"  # exact search, inner product
METRIC = "inner_product"  # == cosine similarity because every vector has unit length

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"

TextStyle = Literal["prefixed", "raw"]


class IndexSpec(BaseModel):
    """Everything that decides whether an index is compatible with a configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repository_name: str = Field(min_length=1)
    repository_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunking_strategy: str = Field(min_length=1)
    chunking_strategy_version: int = Field(ge=1)
    chunk_params: dict[str, int | str]  # the chunker's parameter mapping (size, overlap, cap)
    chunk_id_schema: str = Field(min_length=1)  # formula version of the chunk ids in the mapping
    embedding_model_id: str = Field(min_length=1)
    embedding_dimension: int = Field(ge=1)
    embedding_normalized: bool
    text_style: TextStyle  # "prefixed" or "raw" embedding representation
    representation_version: int = Field(ge=1)
    index_type: str = Field(min_length=1)
    metric: str = Field(min_length=1)


def compute_index_id(spec: IndexSpec) -> str:
    """Deterministic 16-hex-character id of ``spec`` (see the module docstring)."""
    payload = {"schema": INDEX_ID_SCHEMA, "spec": spec.model_dump(mode="json")}
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class Provenance(BaseModel):
    """How/when the index was built. Informational: never part of the identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    created_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")  # UTC
    source_file_count: int = Field(ge=0)  # accepted files that were ingested
    ingestion_truncated: bool  # True if a repository limit stopped ingestion early
    embedding_runtime: str  # e.g. "fastembed-onnx"
    embedding_runtime_version: str | None = None
    embedding_pooling: str | None = None
    faiss_version: str


class ArtifactInfo(BaseModel):
    """Integrity data for one stored file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class IndexManifest(BaseModel):
    """The compatibility/build record stored as ``manifest.json``.

    Contains no vectors, no source text, no secrets and no absolute paths.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    index_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    spec: IndexSpec
    provenance: Provenance
    vector_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    chunk_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")  # digest of the ordered chunk ids
    artifacts: dict[str, ArtifactInfo]  # file name -> size and SHA-256 (manifest excluded)


@dataclass(frozen=True)
class IndexExpectation:
    """What the caller requires of an index. ``None`` means "do not check this field".

    ``IndexExpectation()`` (everything ``None``) therefore means "self-consistency only".
    """

    repository_name: str | None = None
    repository_fingerprint: str | None = None
    chunking_strategy: str | None = None
    chunking_strategy_version: int | None = None
    chunk_params: dict[str, int | str] | None = None
    chunk_id_schema: str | None = None
    embedding_model_id: str | None = None
    embedding_dimension: int | None = None
    embedding_normalized: bool | None = None
    text_style: str | None = None
    representation_version: int | None = None
    index_type: str | None = None
    metric: str | None = None

    @classmethod
    def from_spec(cls, spec: IndexSpec) -> IndexExpectation:
        """Expect exactly ``spec`` (every field checked)."""
        return cls(**spec.model_dump())


@dataclass(frozen=True)
class Mismatch:
    """One field where the stored spec differs from the expectation."""

    field: str
    expected: object
    actual: object

    def __str__(self) -> str:
        return f"{self.field}: expected {self.expected!r}, index has {self.actual!r}"


def find_mismatches(spec: IndexSpec, expected: IndexExpectation) -> list[Mismatch]:
    """All differences between a stored ``spec`` and ``expected`` (empty list: compatible)."""
    found: list[Mismatch] = []
    for item in fields(expected):
        wanted = getattr(expected, item.name)
        if wanted is None:
            continue
        actual = getattr(spec, item.name)
        if actual != wanted:
            found.append(Mismatch(item.name, wanted, actual))
    return found
