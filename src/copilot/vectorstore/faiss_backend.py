"""The only module that imports FAISS.

Everything else talks to ``VectorIndex``. The rest of the project therefore does not depend on
FAISS types, and swapping the backend (or adding an approximate index) touches this file.

Why ``IndexFlatIP``: it is *exact* brute-force search by inner product. For vectors of unit length
inner product equals cosine similarity, ``cos(a, b) = a.b / (|a||b|) = a.b``. Exact search makes the
baseline defensible (no recall lost to approximation) and is fast at repository scale (tens of
thousands of vectors). HNSW/IVF are deliberately not used.

Serialisation uses ``faiss.serialize_index`` to bytes so that *we* control file writing (atomic
staging, checksums, non-ASCII Windows paths) instead of ``faiss.write_index(path)``.
"""

from __future__ import annotations

import faiss
import numpy as np

from copilot.vectorstore.errors import IndexCorruptError

FAISS_VERSION: str = faiss.__version__
_INNER_PRODUCT = faiss.METRIC_INNER_PRODUCT


def build_flat_ip(vectors: np.ndarray) -> faiss.Index:
    """An exact inner-product index holding ``vectors`` (float32, shape ``(n, d)``)."""
    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return index


def serialize(index: faiss.Index) -> bytes:
    """Deterministic byte representation of ``index``."""
    return faiss.serialize_index(index).tobytes()


def deserialize(data: bytes) -> faiss.Index:
    """Rebuild an index from ``serialize`` output; damaged data raises ``IndexCorruptError``."""
    try:
        return faiss.deserialize_index(np.frombuffer(data, dtype=np.uint8))
    except (RuntimeError, ValueError) as exc:
        raise IndexCorruptError("index.faiss could not be deserialised (damaged file?)") from exc


def index_type_name(index: faiss.Index) -> str:
    """FAISS class name, e.g. ``"IndexFlatIP"``."""
    return type(index).__name__


def is_inner_product(index: faiss.Index) -> bool:
    return bool(index.metric_type == _INNER_PRODUCT)


def total(index: faiss.Index) -> int:
    return int(index.ntotal)


def dimension(index: faiss.Index) -> int:
    return int(index.d)


def reconstruct_all(index: faiss.Index, batch: int = 4096) -> np.ndarray:
    """All stored vectors, ``(ntotal, d)`` float32 (flat indexes store them verbatim)."""
    n, d = total(index), dimension(index)
    out = np.empty((n, d), dtype=np.float32)
    for start in range(0, n, batch):
        count = min(batch, n - start)
        out[start : start + count] = index.reconstruct_n(start, count)
    return out


def search(index: faiss.Index, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Raw ``(scores, positions)`` for ``queries`` (a serialisation-check primitive)."""
    scores, positions = index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
    return scores, positions
