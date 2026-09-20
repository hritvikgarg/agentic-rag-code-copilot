"""Source re-materialisation: turning index records back into verified source text.

The index stores metadata only (Milestone 5a), so to show a chunk we

1. re-ingest the repository with the same ignore rules and compare its **fingerprint** with the one
   in the index manifest (any added/removed/edited accepted file is detected here);
2. re-chunk it with the chunking configuration **recorded in the manifest** (not the current
   settings) and build a ``chunk_id -> Chunk`` map;
3. for each retrieved record, locate the chunk and verify id, location, source hash and
   ``content_sha256`` against the sidecar record before its text is released.

Step 1 makes step 3 nearly redundant for a healthy repository; step 3 is the second, independent
line of defence (it also catches a changed ingestion/chunking implementation whose output differs
although the fingerprint matches). Nothing stale is ever returned silently.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

from copilot.chunking import ChunkingError, chunk_repository, create_chunker_from_params
from copilot.config.settings import Settings, get_settings
from copilot.ingestion import IngestionPolicy, ingest_repository
from copilot.models.chunk import CHUNK_ID_SCHEMA, Chunk
from copilot.retrieval.errors import (
    ChunkContentMismatchError,
    ChunkNotFoundError,
    MaterializationError,
    StaleRepositoryError,
)
from copilot.vectorstore import ChunkRecord, IndexSpec, repository_fingerprint


class ChunkSource:
    """A verified ``chunk_id -> Chunk`` view of a repository, for one index."""

    def __init__(self, chunks_by_id: Mapping[str, Chunk], *, repository_name: str) -> None:
        self._chunks = dict(chunks_by_id)
        self.repository_name = repository_name

    def __len__(self) -> int:
        return len(self._chunks)

    @classmethod
    def from_repository(
        cls,
        root: str | os.PathLike[str],
        spec: IndexSpec,
        *,
        settings: Settings | None = None,
        ignore_directories: Sequence[str] = (),
    ) -> ChunkSource:
        """Re-ingest and re-chunk ``root`` as described by the index ``spec``.

        Raises ``StaleRepositoryError`` when the repository's fingerprint differs from the index's,
        and ``MaterializationError`` when the recorded chunking cannot be reproduced. Ingestion
        errors (e.g. ``InvalidRepositoryError`` for a missing directory) propagate unchanged.
        """
        settings = settings or get_settings()
        policy = IngestionPolicy.from_settings(settings).with_extra_ignored_directories(
            *ignore_directories
        )
        ingestion = ingest_repository(root, policy, repository_name=spec.repository_name)
        if ingestion.stats.truncated:
            raise StaleRepositoryError(
                f"ingestion stopped early ({ingestion.stats.truncation_reason}); the repository "
                "cannot be compared with the index"
            )
        if repository_fingerprint(ingestion.files) != spec.repository_fingerprint:
            raise StaleRepositoryError(
                "the repository does not match the index (its fingerprint differs): a file was "
                "added, removed or edited since the index was built, or a different --ignore-dir "
                "set or repository name is in use. Rebuild the index or use the indexed tree."
            )
        if spec.chunk_id_schema != CHUNK_ID_SCHEMA:
            raise MaterializationError(
                f"the index uses chunk id schema {spec.chunk_id_schema!r}, this code uses "
                f"{CHUNK_ID_SCHEMA!r}; rebuild the index"
            )
        try:
            chunker = create_chunker_from_params(
                spec.chunking_strategy, spec.chunking_strategy_version, spec.chunk_params
            )
            chunking = chunk_repository(ingestion, chunker)
        except ChunkingError as exc:
            raise MaterializationError(
                f"the chunking recorded in the index cannot be reproduced: {exc}"
            ) from exc
        return cls({c.chunk_id: c for c in chunking.chunks}, repository_name=spec.repository_name)

    def get(self, record: ChunkRecord) -> Chunk:
        """The chunk for ``record``, verified against it; never returns anything unverified."""
        chunk = self._chunks.get(record.chunk_id)
        if chunk is None:
            raise ChunkNotFoundError(
                f"chunk {record.chunk_id} ({record.file_path}:{record.start_line}-"
                f"{record.end_line}) is listed in the index but was not found when the repository "
                "was re-chunked; rebuild the index"
            )
        if (
            chunk.file_path != record.file_path
            or chunk.start_line != record.start_line
            or chunk.end_line != record.end_line
            or chunk.source_sha256 != record.source_sha256
            or chunk.content_sha256 != record.content_sha256
        ):
            raise ChunkContentMismatchError(
                f"chunk {record.chunk_id} ({record.file_path}:{record.start_line}-"
                f"{record.end_line}) no longer matches the index (location or content hash "
                "differs); rebuild the index"
            )
        return chunk
