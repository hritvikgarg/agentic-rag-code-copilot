"""Hosted-Gemini smoke test. Opt-in only; never runs in normal test runs or CI.

Run (PowerShell)::

    $env:COPILOT_RUN_LLM_LIVE = "1"
    uv run pytest tests/integration/test_llm_live.py -v -s

Requires ``COPILOT_LLM_MODEL`` and ``GEMINI_API_KEY`` (environment or ``.env``); otherwise the tests
skip with a clear reason. The prompts contain only synthetic, non-sensitive text. These tests
check mechanics (a non-empty reply, metadata, gate refusal), not answer quality.
"""

import os

import pytest

from copilot.config.settings import Settings
from copilot.embeddings import HashEmbedder
from copilot.llm import create_llm_client
from copilot.rag import answer_plain, answer_with_rag
from copilot.security import RepositorySecretRiskError
from copilot.vectorstore import build_repository_index
from tests.retrieval_helpers import write_repo
from tests.security import secret_helpers as h

pytestmark = pytest.mark.llm_live

# conftest clears COPILOT_* variables for every test, so capture them at import time.
_CAPTURED = {
    key: os.environ[key]
    for key in ("COPILOT_LLM_MODEL", "GEMINI_API_KEY", "COPILOT_LLM_TIMEOUT_SECONDS")
    if key in os.environ
}


@pytest.fixture
def settings() -> Settings:
    values: dict = {}
    if "COPILOT_LLM_MODEL" in _CAPTURED:
        values["llm_model"] = _CAPTURED["COPILOT_LLM_MODEL"]
    if "GEMINI_API_KEY" in _CAPTURED:
        values["gemini_api_key"] = _CAPTURED["GEMINI_API_KEY"]
    settings = Settings(**values)  # falls back to .env for anything not in the environment
    if not settings.llm_model:
        pytest.skip("COPILOT_LLM_MODEL is not set (choose a model id; see docs/llm.md)")
    if settings.gemini_api_key is None:
        pytest.skip("GEMINI_API_KEY is not set")
    return settings


def test_plain_call_returns_text_and_metadata(settings):
    answer = answer_plain(
        "Reply with one short sentence about Python.",
        create_llm_client(settings),
        settings=settings,
    )
    assert answer.answer.strip()
    assert answer.generation.provider == "gemini" and answer.generation.model == settings.llm_model


def test_the_gate_still_refuses_before_the_hosted_call(settings):
    secret = h.github_token("live")
    with pytest.raises(RepositorySecretRiskError):
        answer_plain(f"Is {secret} valid?", create_llm_client(settings), settings=settings)


def test_rag_call_over_a_tiny_repository(settings, tmp_path):
    repo = write_repo(tmp_path / "demo repo")
    embedder = HashEmbedder(dimension=256)
    index = build_repository_index(
        repo,
        embedder=embedder,
        settings=settings,
        indexes_dir=tmp_path / "indexes",
        text_style="raw",
    ).index_path
    answer = answer_with_rag(
        "How is the chunk id computed?", repo, index, 3,
        llm=create_llm_client(settings), embedder=embedder, settings=settings,
    )  # fmt: skip
    assert answer.answer and answer.sources
    assert answer.sources[0].file_path.endswith(".py") or answer.sources[0].file_path.endswith(
        ".md"
    )
