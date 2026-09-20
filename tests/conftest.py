"""Shared pytest fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from copilot.config.settings import get_settings
from tests.fixtures.synthetic_repo import build_synthetic_repo


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch):
    """Make every test independent of the developer's real environment and .env file."""
    for name in list(os.environ):
        if name.upper().startswith("COPILOT_") or name.upper() == "GEMINI_API_KEY":
            monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def synthetic_repo(tmp_path: Path) -> Path:
    """A fresh synthetic repository (see tests/fixtures/synthetic_repo.py) in a temp directory."""
    return build_synthetic_repo(tmp_path / "synthetic-repo")
