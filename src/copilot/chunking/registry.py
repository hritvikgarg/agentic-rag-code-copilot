"""Chunking strategy registry: strategy name (from ``Settings``) -> ``Chunker`` instance.

Only implemented strategies are registered. ``"ast"`` is a *reserved* name (it is a valid value
of ``Settings.chunking_strategy``) whose implementation arrives in Milestone 9; asking for it now
fails loudly instead of silently falling back to the baseline, which would corrupt the planned
two-strategy experiment.
"""

from __future__ import annotations

from collections.abc import Callable

from copilot.chunking.base import Chunker
from copilot.chunking.errors import StrategyNotImplementedError, UnknownChunkingStrategyError
from copilot.chunking.line_chunker import LineChunker
from copilot.config.settings import Settings, get_settings

_FACTORIES: dict[str, Callable[[Settings], Chunker]] = {
    "line": lambda s: LineChunker(
        size_lines=s.chunk_size_lines,
        overlap_lines=s.chunk_overlap_lines,
        max_tokens=s.chunk_max_tokens,
    ),
}

# Valid configuration values whose implementation is a later milestone -> where it arrives.
_RESERVED: dict[str, str] = {"ast": "Milestone 9 (structure-aware chunking)"}


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
