"""Shared pytest fixtures."""

from __future__ import annotations

import os

import pytest

from copilot.config.settings import get_settings


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch):
    """Make every test independent of the developer's real environment and .env file."""
    for name in list(os.environ):
        if name.upper().startswith("COPILOT_") or name.upper() == "GEMINI_API_KEY":
            monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
