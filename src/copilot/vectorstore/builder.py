"""The index build pipeline (no LLM, no network apart from a first-time model download).

    ingest -> baseline chunking -> canonical ordering -> embedding representation -> embeddings
           -> consistency checks -> FAISS index -> atomic save (manifest last)

Empty inputs fail *before* the embedding model is loaded (a 0.6 GB load is pointless for nothing):
no accepted files and zero chunks raise ``EmptyIndexError``.

Reports contain counts, ids and timings only: never vectors, source text or absolute paths beyond
the directory the caller asked us to write.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from copilot.chunking import chunk_repository, create_chunker
from copilot.chunking.base import Chunker
from copilot.config.settings import Settings, get_settings
from copilot.embeddings.base import Embedder
from copilot.embeddings.chunk_embeddings import embed_chunks
from copilot.embeddings.representation import REPRESENTATION_VERSION, TextStyle
from copilot.ingestion import IngestionPolicy, ingest_repository
from copilot.models.chunk import CHUNK_ID_SCHEMA
from copilot.models.embedding import EmbeddingModelInfo
from copilot.models.ingestion import IngestionResult, SourceFile
from copilot.vectorstore.errors import EmptyIndexError, IndexBuildError
from copilot.vectorstore.faiss_backend import FAISS_VERSION
from copilot.vectorstore.fingerprint import repository_fingerprint
from copilot.vectorstore.index import VectorIndex
from copilot.vectorstore.manifest import (
    INDEX_TYPE,
    MANIFEST_FILE,
    METRIC,
    IndexExpectation,
    IndexSpec,
    Provenance,
)
from copilot.vectorstore.validation import sort_chunks_canonically

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildReport:
    """What one build did. Safe to print and log."""

    repository_name: str
    files: int
    chunks: int
    vectors: int
    dimension: int
    embedding_model_id: str
    text_style: str
    representation_version: int
    index_type: str
    index_id: str
    index_path: Path
    ingestion_truncated: bool
    seconds_ingest: float
    seconds_chunk: float
    seconds_embed: float
    seconds_index_and_save: float
    seconds_total: float
    artifact_sizes: dict[str, int]  # file name -> bytes (including manifest.json)


def utc_timestamp() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_spec(
    *,
    repository_name: str,
    files: Sequence[SourceFile],
    chunker: Chunker,
    model: EmbeddingModelInfo,
    text_style: TextStyle,
) -> IndexSpec:
    """The compatibility-relevant description of an index built from these parts."""
    return IndexSpec(
        repository_name=repository_name,
        repository_fingerprint=repository_fingerprint(files),
        chunking_strategy=chunker.name,
        chunking_strategy_version=chunker.version,
        chunk_params=dict(chunker.params()),
        chunk_id_schema=CHUNK_ID_SCHEMA,
        embedding_model_id=model.model_id,
        embedding_dimension=model.dimension,
        embedding_normalized=model.normalized,
        text_style=text_style,
        representation_version=REPRESENTATION_VERSION,
        index_type=INDEX_TYPE,
        metric=METRIC,
    )


def _ingest(
    root: str | os.PathLike[str],
    settings: Settings,
    repository_name: str | None,
    ignore_directories: Sequence[str],
) -> IngestionResult:
    policy = IngestionPolicy.from_settings(settings).with_extra_ignored_directories(
        *ignore_directories
    )
    return ingest_repository(root, policy, repository_name=repository_name)


def expectation_for_repository(
    root: str | os.PathLike[str],
    *,
    settings: Settings | None = None,
    repository_name: str | None = None,
    ignore_directories: Sequence[str] = (),
    text_style: TextStyle | None = None,
) -> IndexExpectation:
    """What an index of ``root`` should look like under the current configuration.

    Ingests the repository (to fingerprint it) but embeds nothing and loads no model, so the
    embedding *dimension* is left unchecked (``None``); the model id comes from the settings.
    """
    settings = settings or get_settings()
    ingestion = _ingest(root, settings, repository_name, ignore_directories)
    chunker = create_chunker(settings=settings)
    return IndexExpectation(
        repository_name=ingestion.repository_name,
        repository_fingerprint=repository_fingerprint(ingestion.files),
        chunking_strategy=chunker.name,
        chunking_strategy_version=chunker.version,
        chunk_params=dict(chunker.params()),
        chunk_id_schema=CHUNK_ID_SCHEMA,
        embedding_model_id=settings.embedding_model,
        text_style=text_style or settings.embedding_text_style,
        representation_version=REPRESENTATION_VERSION,
        index_type=INDEX_TYPE,
        metric=METRIC,
    )


def build_repository_index(
    root: str | os.PathLike[str],
    *,
    embedder: Embedder,
    settings: Settings | None = None,
    indexes_dir: str | os.PathLike[str] | None = None,
    repository_name: str | None = None,
    ignore_directories: Sequence[str] = (),
    text_style: TextStyle | None = None,
    overwrite: bool = False,
    allow_truncated: bool = False,
) -> BuildReport:
    """Ingest, chunk, embed and index ``root``, then atomically save the index.

    Raises:
        InvalidRepositoryError: ``root`` is not a scannable directory.
        EmptyIndexError: no accepted files, or they produced no chunks.
        IndexBuildError: ingestion hit a repository limit (partial index) and ``allow_truncated``
            is false, or the embeddings/chunks are inconsistent.
        IndexExistsError: the same index already exists and ``overwrite`` is false.
    """
    settings = settings or get_settings()
    style: TextStyle = text_style or settings.embedding_text_style
    started = time.perf_counter()

    ingestion = _ingest(root, settings, repository_name, ignore_directories)
    after_ingest = time.perf_counter()
    if ingestion.stats.truncated and not allow_truncated:
        raise IndexBuildError(
            f"ingestion stopped early ({ingestion.stats.truncation_reason}); the index would be "
            "incomplete. Raise the limit or pass allow_truncated=True."
        )
    if not ingestion.files:
        raise EmptyIndexError("the repository has no accepted files to index")

    chunker = create_chunker(settings=settings)
    chunking = chunk_repository(ingestion, chunker)
    chunks = sort_chunks_canonically(chunking.chunks)
    after_chunk = time.perf_counter()
    if not chunks:
        raise EmptyIndexError(
            f"{len(ingestion.files)} accepted files produced zero chunks (all blank?)"
        )

    logger.info("Embedding %d chunks with %s", len(chunks), settings.embedding_model)
    embeddings = embed_chunks(embedder, chunks, style=style)  # loads the model on first use
    after_embed = time.perf_counter()
    if list(embeddings.chunk_ids) != [c.chunk_id for c in chunks]:
        raise IndexBuildError("embedding order does not match the chunk order")

    spec = make_spec(
        repository_name=ingestion.repository_name,
        files=ingestion.files,
        chunker=chunker,
        model=embeddings.model,
        text_style=style,
    )
    provenance = Provenance(
        created_at=utc_timestamp(),
        source_file_count=len(ingestion.files),
        ingestion_truncated=ingestion.stats.truncated,
        embedding_runtime=embeddings.model.runtime,
        embedding_runtime_version=embeddings.model.runtime_version,
        embedding_pooling=embeddings.model.pooling,
        faiss_version=FAISS_VERSION,
    )
    index = VectorIndex.build(
        spec=spec, provenance=provenance, chunks=chunks, vectors=embeddings.vectors
    )
    target_root = Path(indexes_dir) if indexes_dir is not None else settings.indexes_dir
    path = index.save(target_root, overwrite=overwrite)
    finished = time.perf_counter()

    sizes = {name: info.size_bytes for name, info in index.manifest.artifacts.items()}
    sizes[MANIFEST_FILE] = (path / MANIFEST_FILE).stat().st_size
    logger.info(
        "Built index %s: files=%d chunks=%d dimension=%d in %.1fs",
        index.index_id,
        len(ingestion.files),
        len(chunks),
        index.dimension,
        finished - started,
    )
    return BuildReport(
        repository_name=ingestion.repository_name,
        files=len(ingestion.files),
        chunks=len(chunks),
        vectors=index.count,
        dimension=index.dimension,
        embedding_model_id=spec.embedding_model_id,
        text_style=style,
        representation_version=REPRESENTATION_VERSION,
        index_type=spec.index_type,
        index_id=index.index_id,
        index_path=path,
        ingestion_truncated=ingestion.stats.truncated,
        seconds_ingest=after_ingest - started,
        seconds_chunk=after_chunk - after_ingest,
        seconds_embed=after_embed - after_chunk,
        seconds_index_and_save=finished - after_embed,
        seconds_total=finished - started,
        artifact_sizes=sizes,
    )
