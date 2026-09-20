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


def pytest_collection_modifyitems(config, items):
    """Skip tests marked ``live`` (they need the real embedding model) unless explicitly enabled.

    Enable with ``COPILOT_RUN_LIVE=1`` (the first run downloads the model, ~0.64 GB).
    """
    if os.environ.get("COPILOT_RUN_LIVE") == "1":
        return
    skip = pytest.mark.skip(reason="live test: set COPILOT_RUN_LIVE=1 to run (downloads the model)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
