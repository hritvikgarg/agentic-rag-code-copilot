"""``Retriever``: natural-language query -> ranked, verified ``RetrievalResult`` objects.

Flow (no LLM anywhere)::

    query --validate--> text --embedder.embed_query--> unit vector
          --VectorIndex.search--> (position, cosine score)*
          --index.record_at--> ChunkRecord --ChunkSource.get (verify hashes)--> Chunk
          --> RetrievalResult(rank, score, metadata, verified text)

**Query embedding.** The query is embedded with the *same model* the index was built with; the
constructor refuses an embedder whose model id, dimension or normalisation differs from the index
manifest. The Jina code model is symmetric (no query prefix), and the query is deliberately **not**
given the ``prefixed`` metadata header even for a ``prefixed`` index: the header describes a stored
chunk, and a question has no file or line range.

**Scores** are cosine similarities (inner products of unit vectors). They order results for one
query; they are not calibrated probabilities and are not comparable across queries or models.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from copilot.config.settings import Settings, get_settings
from copilot.embeddings.base import Embedder
from copilot.retrieval.errors import EmbedderMismatchError, QueryError, RetrievalError
from copilot.retrieval.materialize import ChunkSource
from copilot.retrieval.results import RetrievalResult
from copilot.vectorstore import IndexExpectation, SearchInputError, VectorIndex

MAX_TOP_K = 50  # also the upper bound of Settings.retrieval_top_k
MAX_QUERY_CHARS = 2000  # a question, not a document; far below the model's 8192-token limit


def validate_query(query: object) -> str:
    """The query text, stripped; raises ``QueryError`` unless it is a non-blank ``str``."""
    if not isinstance(query, str):
        raise QueryError(f"query must be a str, got {type(query).__name__}")
    text = query.strip()
    if not text:
        raise QueryError("query is empty or whitespace-only")
    if len(text) > MAX_QUERY_CHARS:
        raise QueryError(
            f"query is too long ({len(text)} characters; the limit is {MAX_QUERY_CHARS})"
        )
    return text


def validate_top_k(top_k: object) -> int:
    """``top_k`` as an int in ``1..MAX_TOP_K``; raises ``QueryError`` otherwise (``bool`` too)."""
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise QueryError(f"top_k must be an integer, got {type(top_k).__name__}")
    if not 1 <= top_k <= MAX_TOP_K:
        raise QueryError(f"top_k must be between 1 and {MAX_TOP_K}, got {top_k}")
    return top_k


class Retriever:
    """Searches one index and returns verified chunks; build it with ``Retriever.open``."""

    def __init__(
        self,
        *,
        index: VectorIndex,
        embedder: Embedder,
        source: ChunkSource,
        default_top_k: int = 5,
    ) -> None:
        self._check_embedder(index, embedder)
        if source.repository_name != index.spec.repository_name:
            raise RetrievalError(
                "the chunk source belongs to a different repository than the index"
            )
        self._index = index
        self._embedder = embedder
        self._source = source
        self._default_top_k = validate_top_k(default_top_k)

    @staticmethod
    def _check_embedder(index: VectorIndex, embedder: Embedder) -> None:
        info, spec = embedder.info, index.spec
        problems = []
        if info.model_id != spec.embedding_model_id:
            problems.append(f"model {info.model_id!r} (index: {spec.embedding_model_id!r})")
        if info.dimension != spec.embedding_dimension:
            problems.append(f"dimension {info.dimension} (index: {spec.embedding_dimension})")
        if info.normalized != spec.embedding_normalized:
            problems.append(f"normalised={info.normalized} (index: {spec.embedding_normalized})")
        if problems:
            raise EmbedderMismatchError(
                "the query embedder is incompatible with the index: " + "; ".join(problems)
            )

    @classmethod
    def open(
        cls,
        index: VectorIndex | str | os.PathLike[str],
        repo_path: str | os.PathLike[str],
        *,
        embedder: Embedder,
        settings: Settings | None = None,
        ignore_directories: Sequence[str] = (),
    ) -> Retriever:
        """Load the index (if given as a path), re-ingest the repository and verify they match.

        ``ignore_directories`` must be the same set that was used at build time. Raises
        ``StaleRepositoryError`` when the repository no longer matches the index,
        ``EmbedderMismatchError`` when ``embedder`` is not the index's model, and the vector-store
        errors (``IndexCorruptError``/``IndexCompatibilityError``) for a damaged index.
        """
        settings = settings or get_settings()
        if not isinstance(index, VectorIndex):
            index = VectorIndex.load(Path(index), expected=IndexExpectation())
        source = ChunkSource.from_repository(
            repo_path, index.spec, settings=settings, ignore_directories=ignore_directories
        )
        return cls(
            index=index,
            embedder=embedder,
            source=source,
            default_top_k=settings.retrieval_top_k,
        )

    @property
    def index(self) -> VectorIndex:
        return self._index

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """The ``top_k`` chunks most similar to ``query``, best first, with verified source text.

        Returns fewer than ``top_k`` results when the index is smaller. Raises ``QueryError`` for a
        bad query/``top_k`` (nothing is embedded first) and a ``MaterializationError`` if a hit's
        source cannot be verified.
        """
        text = validate_query(query)
        k = validate_top_k(self._default_top_k if top_k is None else top_k)
        vector = self._embedder.embed_query(text)
        try:
            hits = self._index.search(vector, k)
        except SearchInputError as exc:
            raise EmbedderMismatchError(
                f"the embedder produced an unusable query vector: {exc}"
            ) from exc
        results: list[RetrievalResult] = []
        for rank, hit in enumerate(hits, start=1):
            record = self._index.record_at(hit.position)
            chunk = self._source.get(record)
            results.append(
                RetrievalResult(
                    rank=rank,
                    score=hit.score,
                    position=hit.position,
                    chunk_id=chunk.chunk_id,
                    repository_name=chunk.repository_name,
                    file_path=chunk.file_path,
                    language=chunk.language,
                    chunk_type=chunk.chunk_type,
                    chunk_index=chunk.chunk_index,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    fragment_index=chunk.fragment_index,
                    fragment_count=chunk.fragment_count,
                    symbol_name=chunk.symbol_name,
                    qualified_name=chunk.qualified_name,
                    parent_class=chunk.parent_class,
                    token_estimate=chunk.token_estimate,
                    content_sha256=chunk.content_sha256,
                    text=chunk.content,
                )
            )
        return results


def retrieve(
    query: str,
    top_k: int,
    repo_path: str | os.PathLike[str],
    index: VectorIndex | str | os.PathLike[str],
    *,
    embedder: Embedder,
    settings: Settings | None = None,
    ignore_directories: Sequence[str] = (),
) -> list[RetrievalResult]:
    """One-shot retrieval. Re-ingests the repository on every call: for many queries build a
    ``Retriever`` once with ``Retriever.open`` and call ``retrieve`` on it."""
    validate_query(query)  # fail on a bad query before any expensive work
    validate_top_k(top_k)
    retriever = Retriever.open(
        index,
        repo_path,
        embedder=embedder,
        settings=settings,
        ignore_directories=ignore_directories,
    )
    return retriever.retrieve(query, top_k)
