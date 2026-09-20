"""Semantic retrieval (Milestone 5b): query -> embedding -> FAISS -> verified source chunks.

No LLM is involved anywhere in this package. The index stores metadata only, so returning source
text means re-ingesting and re-chunking the repository and *verifying* every chunk against the
index (repository fingerprint, chunk id, content hash) before it is returned.

Typical use::

    retriever = Retriever.open(index_dir, repo_path, embedder=create_embedder())
    for result in retriever.retrieve("Where is repository ingestion implemented?", top_k=5):
        print(result.rank, result.score, result.location)
"""

from copilot.retrieval.errors import (
    ChunkContentMismatchError,
    ChunkNotFoundError,
    EmbedderMismatchError,
    MaterializationError,
    QueryError,
    RetrievalError,
    StaleRepositoryError,
)
from copilot.retrieval.materialize import ChunkSource
from copilot.retrieval.results import RetrievalResult
from copilot.retrieval.retriever import (
    MAX_QUERY_CHARS,
    MAX_TOP_K,
    Retriever,
    retrieve,
    validate_query,
    validate_top_k,
)

__all__ = [
    "MAX_QUERY_CHARS",
    "MAX_TOP_K",
    "ChunkContentMismatchError",
    "ChunkNotFoundError",
    "ChunkSource",
    "EmbedderMismatchError",
    "MaterializationError",
    "QueryError",
    "RetrievalError",
    "RetrievalResult",
    "Retriever",
    "StaleRepositoryError",
    "retrieve",
    "validate_query",
    "validate_top_k",
]
