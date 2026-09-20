"""Helpers shared by the vector-index tests: tiny files, chunks, specs and ready-made indexes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from copilot.chunking import LineChunker
from copilot.embeddings import HashEmbedder, embed_chunks
from copilot.models import Chunk, SourceFile
from copilot.vectorstore import IndexSpec, VectorIndex
from copilot.vectorstore.builder import make_spec
from copilot.vectorstore.manifest import (
    CHUNKS_FILE,
    INDEX_FILE,
    MANIFEST_FILE,
    Provenance,
)
from copilot.vectorstore.validation import sort_chunks_canonically
from tests.chunking_helpers import make_file

CREATED_AT = "2026-01-01T00:00:00Z"

DEFAULT_FILES: dict[str, str] = {
    "src/a.py": "def a():\n    return 1\n\n\ndef b():\n    return 2\n",
    "src/b.py": "class B:\n    pass\n\nx = 1\ny = 2\nz = 3\n",
    "README.md": "# Title\n\nsome text\nmore text\nlast line\n",
}


def make_files(contents: Mapping[str, str] | None = None) -> list[SourceFile]:
    """SourceFiles (repository name ``"repo"``) for ``{path: content}``."""
    return [
        make_file(text, path, language="python" if path.endswith(".py") else "markdown")
        for path, text in (contents or DEFAULT_FILES).items()
    ]


def make_chunker(size: int = 4, overlap: int = 1, max_tokens: int = 512) -> LineChunker:
    return LineChunker(size_lines=size, overlap_lines=overlap, max_tokens=max_tokens)


def make_chunks(files: list[SourceFile], chunker: LineChunker | None = None) -> list[Chunk]:
    """All chunks of ``files`` in canonical order."""
    chunker = chunker or make_chunker()
    return sort_chunks_canonically([c for f in files for c in chunker.chunk_file(f)])


def make_provenance(**overrides: object) -> Provenance:
    values: dict[str, object] = {
        "created_at": CREATED_AT,
        "source_file_count": 3,
        "ingestion_truncated": False,
        "embedding_runtime": "python-hash",
        "embedding_runtime_version": None,
        "embedding_pooling": "bag-of-words",
        "faiss_version": "test",
    }
    values.update(overrides)
    return Provenance(**values)  # type: ignore[arg-type]


def build_test_index(
    contents: Mapping[str, str] | None = None,
    *,
    chunker: LineChunker | None = None,
    embedder: HashEmbedder | None = None,
    style: str = "prefixed",
) -> tuple[VectorIndex, list[Chunk], HashEmbedder]:
    """Build a real FAISS index from tiny files with the fake hash embedder."""
    files = make_files(contents)
    chunker = chunker or make_chunker()
    embedder = embedder or HashEmbedder(dimension=32)
    chunks = make_chunks(files, chunker)
    embeddings = embed_chunks(embedder, chunks, style=style)  # type: ignore[arg-type]
    spec = make_spec(
        repository_name="repo",
        files=files,
        chunker=chunker,
        model=embeddings.model,
        text_style=style,  # type: ignore[arg-type]
    )
    index = VectorIndex.build(
        spec=spec, provenance=make_provenance(), chunks=chunks, vectors=embeddings.vectors
    )
    return index, chunks, embedder


def spec_of(**changes: object) -> IndexSpec:
    """The default test spec with some fields replaced."""
    index, _, _ = build_test_index()
    return index.spec.model_copy(update=changes)


def read_manifest_json(directory: Path) -> dict:
    return json.loads((directory / MANIFEST_FILE).read_bytes())


def rewrite_artifact(directory: Path, name: str, data: bytes) -> None:
    """Replace an artifact AND update its manifest checksum, so only deeper checks can object."""
    import hashlib

    (directory / name).write_bytes(data)
    manifest = read_manifest_json(directory)
    manifest["artifacts"][name] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def edit_manifest(directory: Path, mutate) -> None:
    manifest = read_manifest_json(directory)
    mutate(manifest)
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


ARTIFACT_NAMES = (MANIFEST_FILE, INDEX_FILE, CHUNKS_FILE)
