"""Persistent exact vector index (FAISS ``IndexFlatIP``) with a verified chunk mapping.

Milestone 5: building, saving, loading and validating an index. There is deliberately **no**
retrieval/ranking API here yet (semantic retrieval is the next milestone). Nothing outside this
package imports FAISS.
"""

from copilot.vectorstore.builder import (
    BuildReport,
    build_repository_index,
    expectation_for_repository,
)
from copilot.vectorstore.errors import (
    EmptyIndexError,
    IndexBuildError,
    IndexCompatibilityError,
    IndexCorruptError,
    IndexExistsError,
    IndexStorageError,
    VectorStoreError,
)
from copilot.vectorstore.fingerprint import repository_fingerprint
from copilot.vectorstore.index import VectorCheck, VectorIndex
from copilot.vectorstore.manifest import (
    INDEX_ID_SCHEMA,
    INDEX_TYPE,
    MANIFEST_SCHEMA,
    IndexExpectation,
    IndexManifest,
    IndexSpec,
    Mismatch,
    compute_index_id,
    find_mismatches,
)
from copilot.vectorstore.records import ChunkRecord

__all__ = [
    "INDEX_ID_SCHEMA",
    "INDEX_TYPE",
    "MANIFEST_SCHEMA",
    "BuildReport",
    "ChunkRecord",
    "EmptyIndexError",
    "IndexBuildError",
    "IndexCompatibilityError",
    "IndexCorruptError",
    "IndexExistsError",
    "IndexExpectation",
    "IndexManifest",
    "IndexSpec",
    "IndexStorageError",
    "Mismatch",
    "VectorCheck",
    "VectorIndex",
    "VectorStoreError",
    "build_repository_index",
    "compute_index_id",
    "expectation_for_repository",
    "find_mismatches",
    "repository_fingerprint",
]
