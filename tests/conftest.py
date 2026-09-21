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
    """Skip opt-in tests unless explicitly enabled.

    ``live`` (real embedding model): ``COPILOT_RUN_LIVE=1`` (the first run downloads ~0.64 GB).
    ``llm_live`` (hosted LLM call): ``COPILOT_RUN_LLM_LIVE=1``; the test itself also needs
    ``COPILOT_LLM_MODEL`` and ``GEMINI_API_KEY`` and skips with a clear reason without them.
    """
    skip_live = pytest.mark.skip(
        reason="live test: set COPILOT_RUN_LIVE=1 to run (downloads the model)"
    )
    skip_llm = pytest.mark.skip(
        reason="hosted-LLM test: set COPILOT_RUN_LLM_LIVE=1 (plus COPILOT_LLM_MODEL and "
        "GEMINI_API_KEY) to run"
    )
    for item in items:
        if "live" in item.keywords and os.environ.get("COPILOT_RUN_LIVE") != "1":
            item.add_marker(skip_live)
        if "llm_live" in item.keywords and os.environ.get("COPILOT_RUN_LLM_LIVE") != "1":
            item.add_marker(skip_llm)
