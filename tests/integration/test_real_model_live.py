"""Tests against the REAL embedding model. Skipped unless ``COPILOT_RUN_LIVE=1``.

The first run downloads ~0.64 GB into the configured model cache directory
(``data/cache/models`` by default). Run:

    COPILOT_RUN_LIVE=1 uv run pytest tests/integration/test_real_model_live.py -v -s

PowerShell: set ``$env:COPILOT_RUN_LIVE = "1"`` first, then run the same pytest command.
"""

import numpy as np
import pytest

from copilot.config.settings import Settings
from copilot.embeddings import create_embedder
from copilot.utils.tokens import estimate_tokens

pytestmark = pytest.mark.live

TEXTS = [
    "def connect(url):\n    return create_engine(url)",
    "class UserRepository:\n    def get(self, user_id): ...",
    "# Installation\nRun `uv sync` to install the dependencies.",
]


@pytest.fixture(scope="module")
def embedder():
    return create_embedder(Settings())  # real settings: real cache dir, real model


def test_model_loads_with_the_verified_identity_and_dimension(embedder):
    info = embedder.info
    assert info.model_id == "jinaai/jina-embeddings-v2-base-code"
    assert info.dimension == 768
    assert info.max_input_tokens is not None and info.max_input_tokens >= 512


def test_real_vectors_are_finite_unit_length_and_deterministic(embedder):
    first = embedder.embed_documents(TEXTS)
    second = embedder.embed_documents(TEXTS)
    assert first.shape == (3, 768) and np.isfinite(first).all()
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-4)
    assert np.allclose(first, second, atol=1e-5)
    assert embedder.embed_query("how do I connect to the database?").shape == (768,)


def test_real_tokenizer_counts_are_untruncated_and_positive(embedder):
    long_text = "def f(x):\n    return x + 1\n" * 400
    count = embedder.count_tokens([long_text])[0]
    assert count > 512  # counted in full, not clipped
    assert all(c > 0 for c in embedder.count_tokens(TEXTS))
    assert estimate_tokens(long_text) > 0  # the estimator we are validating still works
