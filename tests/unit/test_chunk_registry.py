import pytest

from copilot.chunking import (
    LineChunker,
    StrategyNotImplementedError,
    UnknownChunkingStrategyError,
    available_strategies,
    create_chunker,
)
from copilot.config.settings import Settings


def test_only_the_baseline_strategy_is_registered():
    assert available_strategies() == ["line"]


def test_default_strategy_comes_from_settings_and_uses_its_parameters():
    settings = Settings(
        _env_file=None, chunk_size_lines=20, chunk_overlap_lines=4, chunk_max_tokens=99
    )
    chunker = create_chunker(settings=settings)
    assert isinstance(chunker, LineChunker)
    assert chunker.params() == {"size_lines": 20, "overlap_lines": 4, "max_tokens": 99}


def test_strategy_is_selectable_by_name():
    assert create_chunker("line", settings=Settings(_env_file=None)).name == "line"


def test_ast_strategy_is_reserved_and_fails_loudly_instead_of_falling_back():
    with pytest.raises(StrategyNotImplementedError, match="Milestone 9"):
        create_chunker("ast", settings=Settings(_env_file=None))


def test_configured_ast_strategy_also_fails_loudly(monkeypatch):
    monkeypatch.setenv("COPILOT_CHUNKING_STRATEGY", "ast")
    with pytest.raises(StrategyNotImplementedError):
        create_chunker(settings=Settings(_env_file=None))


def test_unknown_strategy_lists_available_ones():
    with pytest.raises(UnknownChunkingStrategyError, match="line"):
        create_chunker("semantic", settings=Settings(_env_file=None))
