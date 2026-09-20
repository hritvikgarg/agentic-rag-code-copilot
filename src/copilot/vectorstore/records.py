"""The chunk sidecar (``chunks.jsonl``): vector position <-> chunk id <-> citation metadata.

One JSON object per line; **line ``i`` (0-based) describes FAISS vector ``i``**. A record carries
what is needed to cite a result and to re-materialise the chunk text later (file path, line range,
``source_sha256`` and ``content_sha256``) but **never the chunk text**: the source stays in the
repository, and a future retriever re-ingests/re-chunks (deterministic) and looks the chunk up by
``chunk_id``, verifying ``content_sha256``.

Repository name and chunking strategy/parameters are identical for every record, so they live once
in the manifest rather than being repeated per line.

Serialisation is canonical (sorted keys, ASCII-only JSON, ``\\n`` line ends, UTF-8 bytes) so the
same index always produces byte-identical files.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from copilot.models.chunk import Chunk, ChunkType
from copilot.utils.paths import validate_relative_posix
from copilot.vectorstore.errors import IndexCorruptError


class ChunkRecord(BaseModel):
    """Citation metadata of one indexed chunk (no source text)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    position: int = Field(ge=0)  # row in the FAISS index
    chunk_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    file_path: str
    language: str
    chunk_type: ChunkType
    chunk_index: int = Field(ge=0)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    token_estimate: int = Field(ge=0)
    fragment_index: int | None = None
    fragment_count: int | None = None
    symbol_name: str | None = None
    qualified_name: str | None = None
    parent_class: str | None = None

    @field_validator("file_path")
    @classmethod
    def _canonical_file_path(cls, value: str) -> str:
        return validate_relative_posix(value)


def record_from_chunk(chunk: Chunk, position: int) -> ChunkRecord:
    """Sidecar record for ``chunk`` stored at vector ``position``."""
    return ChunkRecord(
        position=position,
        chunk_id=chunk.chunk_id,
        file_path=chunk.file_path,
        language=chunk.language,
        chunk_type=chunk.chunk_type,
        chunk_index=chunk.chunk_index,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        source_sha256=chunk.source_sha256,
        content_sha256=chunk.content_sha256,
        token_estimate=chunk.token_estimate,
        fragment_index=chunk.fragment_index,
        fragment_count=chunk.fragment_count,
        symbol_name=chunk.symbol_name,
        qualified_name=chunk.qualified_name,
        parent_class=chunk.parent_class,
    )


def encode_records(records: Sequence[ChunkRecord]) -> bytes:
    """Canonical JSONL bytes: one compact, key-sorted JSON object per line."""
    lines = (
        json.dumps(
            r.model_dump(mode="json"), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        for r in records
    )
    return "".join(line + "\n" for line in lines).encode("utf-8")


def decode_records(data: bytes) -> list[ChunkRecord]:
    """Parse ``chunks.jsonl`` bytes; any malformed line raises ``IndexCorruptError``."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IndexCorruptError("chunks.jsonl is not valid UTF-8") from exc
    lines = text.split("\n")
    if lines and lines[-1] == "":  # the canonical file ends with a newline
        lines.pop()
    records: list[ChunkRecord] = []
    for number, line in enumerate(lines, 1):
        try:
            records.append(ChunkRecord.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise IndexCorruptError(
                f"chunks.jsonl line {number} is invalid: {type(exc).__name__}"
            ) from exc
    return records


def chunk_ids_digest(chunk_ids: Sequence[str]) -> str:
    """SHA-256 over the ordered chunk ids (detects reordering as well as substitution)."""
    text = json.dumps(list(chunk_ids), separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
