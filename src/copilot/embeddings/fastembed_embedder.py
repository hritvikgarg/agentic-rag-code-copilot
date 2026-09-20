"""Local embeddings through fastembed (ONNX Runtime). No network calls except model download.

The only module that touches fastembed. fastembed is imported lazily so importing
``copilot.embeddings`` stays cheap and fails with a clear message if the library is missing.

Facts verified against fastembed 0.8.0 source (see docs/embeddings.md):

* ``jinaai/jina-embeddings-v2-base-code`` is supported: 768 dimensions, ONNX file
  ``onnx/model.onnx`` from the Hugging Face repo of the same name, ~0.64 GB, Apache-2.0.
* It is handled by fastembed's ``PooledNormalizedEmbedding``: mean pooling over tokens weighted by
  the attention mask, then L2 normalisation.
* fastembed enables truncation in its tokenizer at ``min(model_max_length, max_length)`` read from
  the model's ``tokenizer_config.json``, so over-long input is silently truncated, not rejected.
* fastembed's default cache is the OS temp directory; we always pass an explicit ``cache_dir``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from copilot.embeddings.errors import (
    EmbeddingConfigError,
    EmbeddingModelUnavailableError,
    EmbeddingRuntimeError,
)
from copilot.embeddings.vectors import finalize_vectors, validate_text, validate_texts
from copilot.models.embedding import EmbeddingModelInfo

logger = logging.getLogger(__name__)

RUNTIME_NAME = "fastembed-onnx"


def _default_factory() -> Any:
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise EmbeddingConfigError(
            "fastembed is not installed; run `uv sync` (it is a declared dependency)"
        ) from exc
    return TextEmbedding


class FastEmbedEmbedder:
    """Embedder backed by a fastembed ``TextEmbedding`` model (loaded on first use)."""

    def __init__(
        self,
        model_id: str,
        *,
        cache_dir: Path,
        batch_size: int = 32,
        threads: int | None = None,
        model_factory: Callable[[], Any] | None = None,
    ) -> None:
        """
        Args:
            model_id: fastembed model name, e.g. ``"jinaai/jina-embeddings-v2-base-code"``.
            cache_dir: Directory for downloaded model files.
            batch_size: Texts per ONNX forward pass.
            threads: ONNX intra-op threads (``None``: let onnxruntime decide).
            model_factory: Returns the ``TextEmbedding`` class; injectable for tests.
        """
        if batch_size < 1:
            raise EmbeddingConfigError("batch_size must be >= 1")
        self._model_id = model_id
        self._cache_dir = Path(cache_dir)
        self._batch_size = batch_size
        self._threads = threads
        self._factory = model_factory or _default_factory
        self._model: Any = None
        self._dimension: int | None = None
        self._max_input_tokens: int | None = None
        self._count_tokenizer: Any = None

    # ------------------------------------------------------------------ loading
    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        text_embedding = self._factory()

        supported = {m["model"]: m for m in text_embedding.list_supported_models()}
        if self._model_id not in supported:
            similar = sorted(m for m in supported if self._model_id.split("/")[-1][:6] in m)[:5]
            hint = f" Similar supported models: {', '.join(similar)}." if similar else ""
            raise EmbeddingConfigError(
                f"model {self._model_id!r} is not supported by this fastembed version.{hint}"
            )
        self._dimension = int(supported[self._model_id]["dim"])

        logger.info(
            "Loading embedding model %s (cache: %s); the first run downloads ~%s GB",
            self._model_id,
            self._cache_dir,
            supported[self._model_id].get("size_in_GB", "?"),
        )
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = text_embedding(
                model_name=self._model_id,
                cache_dir=str(self._cache_dir),
                threads=self._threads,
            )
        except Exception as exc:  # boundary: wrap with context, keep the cause
            raise EmbeddingModelUnavailableError(
                f"could not load embedding model {self._model_id!r} "
                f"({type(exc).__name__}: {exc}). Check network access to huggingface.co for the "
                f"first download, free disk space, and the cache directory {self._cache_dir}."
            ) from exc

        tokenizer = self._tokenizer()
        if tokenizer is not None and tokenizer.truncation:
            self._max_input_tokens = int(tokenizer.truncation["max_length"])
        return self._model

    def _tokenizer(self) -> Any:
        """fastembed's own tokenizer (truncation and padding enabled). Do not mutate it."""
        inner = getattr(self._model, "model", None)
        return getattr(inner, "tokenizer", None)

    # ------------------------------------------------------------------ metadata
    @property
    def info(self) -> EmbeddingModelInfo:
        self._load()
        assert self._dimension is not None
        try:
            from importlib.metadata import version

            runtime_version: str | None = version("fastembed")
        except Exception:  # pragma: no cover - metadata missing
            runtime_version = None
        return EmbeddingModelInfo(
            model_id=self._model_id,
            dimension=self._dimension,
            runtime=RUNTIME_NAME,
            runtime_version=runtime_version,
            normalized=True,  # guaranteed by finalize_vectors regardless of the library
            pooling="mean",
            max_input_tokens=self._max_input_tokens,
            batch_size=self._batch_size,
            asymmetric=False,
        )

    def runtime_limits(self) -> dict[str, int | None]:
        """Input limits as stated by the model's own files (``None`` when not available)."""
        self._load()
        limits: dict[str, int | None] = {"tokenizer_truncation_max_length": self._max_input_tokens}
        model_dir = getattr(getattr(self._model, "model", None), "_model_dir", None)
        for file_name, key, out in (
            ("config.json", "max_position_embeddings", "config_max_position_embeddings"),
            ("tokenizer_config.json", "model_max_length", "tokenizer_config_model_max_length"),
        ):
            value = None
            if model_dir is not None:
                try:
                    data = json.loads((Path(model_dir) / file_name).read_text(encoding="utf-8"))
                    value = data.get(key)
                except (OSError, ValueError):
                    value = None
            limits[out] = int(value) if isinstance(value, int) else None
        return limits

    # ------------------------------------------------------------------ embedding
    def _run(self, texts: list[str], *, query: bool) -> np.ndarray:
        model = self._load()
        assert self._dimension is not None
        try:
            if query:
                raw = list(model.query_embed(texts, batch_size=self._batch_size))
            else:
                raw = list(model.embed(texts, batch_size=self._batch_size))
        except Exception as exc:
            raise EmbeddingRuntimeError(
                f"embedding runtime failed for {len(texts)} text(s) with {self._model_id!r} "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        return finalize_vectors(raw, count=len(texts), dimension=self._dimension)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        items = validate_texts(texts)
        if not items:
            self._load()
            assert self._dimension is not None
            return finalize_vectors([], count=0, dimension=self._dimension)
        return self._run(items, query=False)

    def embed_query(self, text: str) -> np.ndarray:
        validate_text(text)
        return self._run([text], query=True)[0]

    # ------------------------------------------------------------------ tokens
    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Tokens per text as the model's tokenizer sees them, **without** truncation.

        Includes the special tokens the model adds. Uses a private copy of fastembed's tokenizer
        with truncation and padding switched off, so fastembed's own (truncating) tokenizer, and
        therefore the embeddings it produces, are untouched.
        """
        items = validate_texts(texts)
        self._load()
        if self._count_tokenizer is None:
            tokenizer = self._tokenizer()
            if tokenizer is None:
                raise EmbeddingRuntimeError(
                    "this fastembed version does not expose the tokenizer; cannot count tokens"
                )
            from tokenizers import Tokenizer

            clone = Tokenizer.from_str(tokenizer.to_str())
            clone.no_truncation()
            clone.no_padding()
            self._count_tokenizer = clone
        return [len(e.ids) for e in self._count_tokenizer.encode_batch(items)]
