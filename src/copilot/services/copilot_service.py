"""The one object the UI (and later the CLI/graph) talks to: index a repository, open it, ask.

``CopilotService`` is a thin façade over the existing, separately tested components - it adds no
retrieval or prompting logic of its own:

    build_index  -> copilot.vectorstore.builder.build_repository_index   (local, loads the embedder)
    open_index   -> copilot.retrieval.Retriever.open                    (verifies the repo still
                                                                         matches the index)
    ask          -> copilot.rag.RagService.answer                        (secret gate -> hosted LLM)
    ask_plain    -> copilot.rag.answer_plain                             (no repository context)

Why a façade: the Streamlit app must contain no retrieval or LLM logic (it would be untestable and
impossible to reuse), and a later FastAPI layer or the LangGraph workflow can wrap this same object.
Heavy resources (the ONNX embedding model, the LLM client) are created lazily and once, so opening
the app or listing indexes never downloads a model or needs an API key. Both factories are
injectable, which is how the tests run the whole flow with a fake embedder and a fake LLM.

Error messages raised below this layer are fixed and secret-free by construction (see
``copilot.llm.errors`` and ``copilot.security``), so the UI may show them verbatim.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from copilot.config import Settings, get_settings
from copilot.embeddings.base import Embedder
from copilot.embeddings.factory import create_embedder
from copilot.llm import LLMClient, create_llm_client
from copilot.rag.citations import Citation
from copilot.rag.models import PlainAnswer, RagAnswer
from copilot.rag.service import RagService, answer_plain
from copilot.retrieval import Retriever
from copilot.utils.paths import UnsafePathError, safe_join
from copilot.vectorstore.builder import BuildReport, build_repository_index
from copilot.vectorstore.errors import VectorStoreError
from copilot.vectorstore.index import read_manifest

logger = logging.getLogger(__name__)

MAX_EVIDENCE_LINES = 150  # a UI preview never shows more than this many lines of one source


class ServiceStateError(RuntimeError):
    """The action needs a prepared repository (``open_index``) or a different argument."""


@dataclass(frozen=True)
class IndexSummary:
    """What the UI needs to show about one built index (read from its manifest only)."""

    index_id: str
    path: Path
    repository_name: str
    chunk_count: int
    source_file_count: int
    created_at: str
    embedding_model_id: str
    chunking_strategy: str
    text_style: str


@dataclass(frozen=True)
class OpenIndex:
    """An index that was verified against the repository on disk and is ready to answer."""

    summary: IndexSummary
    repo_path: Path
    ignore_directories: tuple[str, ...]


def summarize_index(index_dir: str | Path) -> IndexSummary:
    """Summary of the index in ``index_dir`` (reads ``manifest.json`` only; no model)."""
    path = Path(index_dir)
    manifest = read_manifest(path)
    spec, prov = manifest.spec, manifest.provenance
    return IndexSummary(
        index_id=manifest.index_id,
        path=path,
        repository_name=spec.repository_name,
        chunk_count=manifest.chunk_count,
        source_file_count=prov.source_file_count,
        created_at=prov.created_at,
        embedding_model_id=spec.embedding_model_id,
        chunking_strategy=f"{spec.chunking_strategy} v{spec.chunking_strategy_version}",
        text_style=spec.text_style,
    )


class CopilotService:
    """Index a repository, open an index, ask repository-aware (or plain) questions."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        embedder_factory: Callable[[Settings], Embedder] = create_embedder,
        llm_factory: Callable[[Settings], LLMClient] = create_llm_client,
    ) -> None:
        self.settings = settings or get_settings()
        self._embedder_factory = embedder_factory
        self._llm_factory = llm_factory
        self._embedder: Embedder | None = None
        self._llm: LLMClient | None = None
        self._retriever: Retriever | None = None
        self._opened: OpenIndex | None = None

    # -- lazy resources -------------------------------------------------------------------
    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = self._embedder_factory(self.settings)
        return self._embedder

    def _get_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = self._llm_factory(self.settings)  # fails early if model/key are unset
        return self._llm

    # -- indexes --------------------------------------------------------------------------
    def list_indexes(self, indexes_dir: str | Path | None = None) -> list[IndexSummary]:
        """Every readable index under ``indexes_dir`` (default: ``<data_dir>/indexes``).

        Hidden staging directories and unreadable/corrupt ones are skipped (and logged); a
        listing must never fail because of one broken folder. Newest first.
        """
        root = Path(indexes_dir) if indexes_dir is not None else self.settings.indexes_dir
        if not root.is_dir():
            return []
        found: list[IndexSummary] = []
        for child in sorted(root.iterdir()):
            if child.name.startswith(".") or not child.is_dir():
                continue
            try:
                found.append(summarize_index(child))
            except (VectorStoreError, OSError, ValueError) as exc:
                logger.warning("skipping unreadable index folder %r: %s", child.name, exc)
        return sorted(found, key=lambda s: s.created_at, reverse=True)

    def build_index(
        self,
        repo_path: str | Path,
        *,
        ignore_directories: Sequence[str] = (),
        repository_name: str | None = None,
        indexes_dir: str | Path | None = None,
        overwrite: bool = False,
    ) -> BuildReport:
        """Ingest, chunk, embed and save an index. Local only: nothing leaves the machine."""
        return build_repository_index(
            repo_path,
            embedder=self._get_embedder(),
            settings=self.settings,
            indexes_dir=indexes_dir,
            repository_name=repository_name,
            ignore_directories=ignore_directories,
            overwrite=overwrite,
        )

    def open_index(
        self,
        index_dir: str | Path,
        repo_path: str | Path,
        *,
        ignore_directories: Sequence[str] = (),
    ) -> OpenIndex:
        """Load the index and verify the repository on disk still matches it.

        Raises the retrieval/vector-store errors (``StaleRepositoryError`` when a file changed
        since the index was built, ``EmbedderMismatchError`` for a different embedding model).
        """
        retriever = Retriever.open(
            index_dir,
            repo_path,
            embedder=self._get_embedder(),
            settings=self.settings,
            ignore_directories=ignore_directories,
        )
        self._retriever = retriever
        self._opened = OpenIndex(
            summary=summarize_index(index_dir),
            repo_path=Path(repo_path).resolve(),
            ignore_directories=tuple(ignore_directories),
        )
        return self._opened

    @property
    def opened(self) -> OpenIndex | None:
        return self._opened

    # -- answering ------------------------------------------------------------------------
    def ask(self, question: str, top_k: int | None = None) -> RagAnswer:
        """Repository-aware answer. Sends the question and retrieved chunks to the hosted model
        after the secret gate; when evidence is insufficient the model is not called at all."""
        if self._retriever is None:
            raise ServiceStateError("no index is open: build or open an index first")
        return RagService(self._retriever, self._get_llm(), settings=self.settings).answer(
            question, top_k
        )

    def ask_plain(self, question: str) -> PlainAnswer:
        """Baseline answer with NO repository context (not grounded; for comparison only)."""
        return answer_plain(question, self._get_llm(), settings=self.settings)

    # -- evidence preview -----------------------------------------------------------------
    def evidence_text(self, citation: Citation) -> str:
        """The cited lines, read from the verified repository (for display next to an answer).

        The path is re-validated against the repository root, so a citation can never make the UI
        read outside it. Long ranges are truncated with a visible marker.
        """
        if self._opened is None:
            raise ServiceStateError("no index is open")
        try:
            file_path = safe_join(self._opened.repo_path, citation.file_path)
        except UnsafePathError as exc:
            raise ServiceStateError(f"refusing unsafe citation path: {exc}") from exc
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = lines[citation.start_line - 1 : citation.end_line]
        if len(selected) > MAX_EVIDENCE_LINES:
            hidden = len(selected) - MAX_EVIDENCE_LINES
            selected = [*selected[:MAX_EVIDENCE_LINES], f"... ({hidden} more lines not shown)"]
        return "\n".join(selected)
