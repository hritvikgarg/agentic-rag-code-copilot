import pytest

from copilot.chunking import (
    LineChunker,
    StrategyNotImplementedError,
    UnknownChunkingStrategyError,
    available_strategies,
    create_chunker,
    registry,
)
from copilot.config.settings import Settings


def test_baseline_and_ast_strategies_are_registered():
    assert available_strategies() == ["ast", "line"]


def test_default_strategy_comes_from_settings_and_uses_its_parameters():
    settings = Settings(
        _env_file=None, chunk_size_lines=20, chunk_overlap_lines=4, chunk_max_tokens=99
    )
    chunker = create_chunker(settings=settings)
    assert isinstance(chunker, LineChunker)
    assert chunker.params() == {"size_lines": 20, "overlap_lines": 4, "max_tokens": 99}


def test_strategy_is_selectable_by_name():
    assert create_chunker("line", settings=Settings(_env_file=None)).name == "line"


def test_ast_strategy_is_selectable_by_name_and_uses_the_baseline_settings_as_fallback():
    settings = Settings(
        _env_file=None, chunk_size_lines=20, chunk_overlap_lines=4, chunk_max_tokens=99
    )
    chunker = create_chunker("ast", settings=settings)
    assert (chunker.name, chunker.version) == ("ast", 1)
    assert chunker.params() == {
        "max_tokens": 99,
        "fallback_size_lines": 20,
        "fallback_overlap_lines": 4,
        "fallback": "line@1",
    }


def test_configured_ast_strategy_is_used_when_not_named(monkeypatch):
    monkeypatch.setenv("COPILOT_CHUNKING_STRATEGY", "ast")
    assert create_chunker(settings=Settings(_env_file=None)).name == "ast"


def test_baseline_remains_the_default_strategy():
    assert Settings(_env_file=None).chunking_strategy == "line"
    assert create_chunker(settings=Settings(_env_file=None)).name == "line"


def test_reserved_name_fails_loudly_instead_of_falling_back(monkeypatch):
    monkeypatch.setitem(registry._RESERVED, "future", "a later milestone")
    with pytest.raises(StrategyNotImplementedError, match="a later milestone"):
        create_chunker("future", settings=Settings(_env_file=None))


def test_unknown_strategy_lists_available_ones():
    with pytest.raises(UnknownChunkingStrategyError, match="line"):
        create_chunker("semantic", settings=Settings(_env_file=None))
