"""Real Jina embeddings -> FAISS -> save -> reload -> validate. Skipped unless COPILOT_RUN_LIVE=1.

Indexes the small synthetic repository (about 20 chunks), so it embeds in about a minute. The
first run downloads the model (~0.64 GB) into the configured model cache. Run:

    COPILOT_RUN_LIVE=1 uv run pytest tests/integration/test_real_model_index_live.py -v -s

PowerShell: ``$env:COPILOT_RUN_LIVE = "1"`` first, then the same pytest command.
No retrieval quality is measured or asserted here; this proves indexing and persistence only.
"""

import numpy as np
import pytest

from copilot.config.settings import Settings
from copilot.embeddings import create_embedder
from copilot.vectorstore import VectorIndex, build_repository_index, expectation_for_repository
from tests.fixtures.synthetic_repo import SYNTHETIC_MAX_FILE_SIZE

pytestmark = pytest.mark.live


def test_real_model_index_round_trip(synthetic_repo, tmp_path):
    settings = Settings(_env_file=None, max_file_size_bytes=SYNTHETIC_MAX_FILE_SIZE)
    report = build_repository_index(
        synthetic_repo,
        embedder=create_embedder(settings),
        settings=settings,
        indexes_dir=tmp_path / "indexes",
    )
    assert report.dimension == 768
    assert report.embedding_model_id == "jinaai/jina-embeddings-v2-base-code"
    assert report.index_type == "IndexFlatIP" and report.chunks == report.vectors > 0

    expected = expectation_for_repository(synthetic_repo, settings=settings)
    loaded = VectorIndex.load(report.index_path, expected=expected)  # a fresh object from disk
    assert loaded.index_id == report.index_id
    check = loaded.verify_vectors()
    assert check.max_norm_deviation < 1e-3 and check.min_self_score > 0.999

    vectors = np.stack([loaded.reconstruct(i) for i in range(loaded.count)])
    scores, _ = loaded._search_positions(vectors, 1)  # primitive self-search, not retrieval
    assert np.allclose(scores[:, 0], 1.0, atol=1e-3)  # every vector finds itself
