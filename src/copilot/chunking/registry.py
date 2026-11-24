"""Chunking strategy registry: strategy name (from ``Settings``) -> ``Chunker`` instance.

Two strategies are implemented: ``"line"`` (baseline, the default) and ``"ast"`` (structure-aware
Python chunking, Milestone 9). Selecting one never silently falls back to the other: an unknown or
unimplemented name fails loudly, which keeps the two-strategy experiment honest. ``_RESERVED`` is
the (now empty) mechanism for announcing a configuration value before its implementation exists.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from copilot.chunking.ast_chunker import AstChunker
from copilot.chunking.base import Chunker
from copilot.chunking.errors import (
    ChunkingError,
    StrategyNotImplementedError,
    UnknownChunkingStrategyError,
)
from copilot.chunking.line_chunker import LineChunker
from copilot.config.settings import Settings, get_settings

_FACTORIES: dict[str, Callable[[Settings], Chunker]] = {
    "line": lambda s: LineChunker(
        size_lines=s.chunk_size_lines,
        overlap_lines=s.chunk_overlap_lines,
        max_tokens=s.chunk_max_tokens,
    ),
    # The fallback line splitter of the "ast" strategy uses the same size/overlap/token settings as
    # the baseline, so non-Python files are cut identically by both strategies.
    "ast": lambda s: AstChunker(
        max_tokens=s.chunk_max_tokens,
        fallback_size_lines=s.chunk_size_lines,
        fallback_overlap_lines=s.chunk_overlap_lines,
    ),
}

# Valid configuration values whose implementation is a later milestone -> where it arrives.
_RESERVED: dict[str, str] = {}


def available_strategies() -> list[str]:
    """Names of implemented strategies."""
    return sorted(_FACTORIES)


def create_chunker(name: str | None = None, *, settings: Settings | None = None) -> Chunker:
    """Create the chunker for ``name`` (default: ``settings.chunking_strategy``)."""
    settings = settings or get_settings()
    strategy = name or settings.chunking_strategy
    factory = _FACTORIES.get(strategy)
    if factory is not None:
        return factory(settings)
    if strategy in _RESERVED:
        raise StrategyNotImplementedError(
            f"chunking strategy {strategy!r} is reserved for {_RESERVED[strategy]} and is not "
            f"implemented yet; available: {', '.join(available_strategies())}"
        )
    raise UnknownChunkingStrategyError(
        f"unknown chunking strategy {strategy!r}; available: {', '.join(available_strategies())}"
    )


def create_chunker_from_params(name: str, version: int, params: Mapping[str, int | str]) -> Chunker:
    """Recreate the chunker an *existing index* was built with (from its manifest).

    ``create_chunker`` reads the current ``Settings``; retrieval must instead reproduce the exact
    chunking recorded in the index, whatever the settings are now. Raises ``ChunkingError`` if a
    parameter is missing or the strategy's algorithm version differs from the recorded one (the
    chunk ids would then differ), and the usual unknown/not-implemented errors for the name.
    """
    if name == "line":
        try:
            chunker: Chunker = LineChunker(
                size_lines=int(params["size_lines"]),
                overlap_lines=int(params["overlap_lines"]),
                max_tokens=int(params["max_tokens"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ChunkingError(
                f"invalid chunking parameters for {name!r}: {dict(params)}"
            ) from exc
    elif name == "ast":
        try:
            chunker = AstChunker(
                max_tokens=int(params["max_tokens"]),
                fallback_size_lines=int(params["fallback_size_lines"]),
                fallback_overlap_lines=int(params["fallback_overlap_lines"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ChunkingError(
                f"invalid chunking parameters for {name!r}: {dict(params)}"
            ) from exc
        if dict(params) != chunker.params():  # e.g. the line splitter version changed
            raise ChunkingError(
                f"the index was built with {name!r} parameters {dict(params)}, this code uses "
                f"{chunker.params()}: chunk ids would differ; rebuild the index"
            )
    elif name in _RESERVED:
        raise StrategyNotImplementedError(
            f"chunking strategy {name!r} is reserved for {_RESERVED[name]} and is not "
            "implemented yet"
        )
    else:
        raise UnknownChunkingStrategyError(
            f"unknown chunking strategy {name!r}; available: {', '.join(available_strategies())}"
        )
    if chunker.version != version:
        raise ChunkingError(
            f"the index was built with {name!r} chunker version {version}, this code has "
            f"version {chunker.version}: chunk ids would differ; rebuild the index"
        )
    return chunker
