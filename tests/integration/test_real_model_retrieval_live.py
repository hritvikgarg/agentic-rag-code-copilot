"""Real Jina query embedding -> FAISS search -> verified source. Skipped unless COPILOT_RUN_LIVE=1.

Run:

    COPILOT_RUN_LIVE=1 uv run pytest tests/integration/test_real_model_retrieval_live.py -v -s

PowerShell: ``$env:COPILOT_RUN_LIVE = "1"`` first. The model must already be cached (or be
downloadable). This test checks the *mechanics* (dimension, finite scores, ordering, source
verification). It deliberately does not assert retrieval quality; measured quality lives in
``docs/evaluation.md`` and comes from ``python -m copilot.evaluation``.
"""

import math

import pytest

from copilot.config.settings import Settings
from copilot.embeddings import create_embedder
from copilot.retrieval import Retriever
from copilot.vectorstore import build_repository_index
from tests.fixtures.synthetic_repo import SYNTHETIC_MAX_FILE_SIZE

pytestmark = pytest.mark.live


def test_real_model_retrieval_mechanics(synthetic_repo, tmp_path):
    settings = Settings(_env_file=None, max_file_size_bytes=SYNTHETIC_MAX_FILE_SIZE)
    embedder = create_embedder(settings)
    report = build_repository_index(
        synthetic_repo, embedder=embedder, settings=settings, indexes_dir=tmp_path / "indexes"
    )
    retriever = Retriever.open(
        report.index_path, synthetic_repo, embedder=embedder, settings=settings
    )
    results = retriever.retrieve("Where is the main function defined?", top_k=5)

    assert 1 <= len(results) <= 5
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(math.isfinite(s) and -1.001 <= s <= 1.001 for s in scores)
    for r in results:
        assert r.text.strip()
        assert not r.file_path.startswith("/") and ":" not in r.file_path.split("/")[0]
