"""Streamlit smoke tests with ``AppTest`` and a pre-seeded fake service (no model, no network)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from copilot.config import Settings  # noqa: E402
from copilot.embeddings import HashEmbedder  # noqa: E402
from copilot.llm import FakeLLMClient  # noqa: E402
from copilot.services.copilot_service import CopilotService  # noqa: E402
from tests.retrieval_helpers import write_repo  # noqa: E402

APP = str(Path(__file__).resolve().parents[2] / "ui" / "streamlit_app.py")


def fake_service(tmp_path) -> CopilotService:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    return CopilotService(
        settings=settings,
        embedder_factory=lambda s: HashEmbedder(dimension=64),
        llm_factory=lambda s: FakeLLMClient("It hashes the payload [Source 1]."),
    )


def test_the_app_starts_and_asks_for_an_index(tmp_path):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["service"] = fake_service(tmp_path)
    at.run()
    assert not at.exception
    assert any("Open or build an index" in i.value for i in at.info)


def test_build_open_and_ask_end_to_end(tmp_path):
    repo = write_repo(tmp_path / "repo")
    service = fake_service(tmp_path)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["service"] = service
    at.run()
    at.sidebar.text_input[0].set_value(str(repo)).run()
    build = next(b for b in at.sidebar.button if b.label == "Build index")
    build.click().run()
    assert not at.exception, at.exception
    assert any("Indexed" in s.value for s in at.sidebar.success)
    assert any("Ready:" in s.value for s in at.success)  # the new index was opened for us
    at.text_area[0].set_value("How is the chunk id made?").run()
    next(b for b in at.button if b.label == "Ask").click().run()
    assert not at.exception, at.exception
    assert any("chunk_ids.py" in e.label for e in at.expander)
    assert any("hashes the payload" in m.value for m in at.markdown)


def test_a_missing_folder_is_reported_not_crashed(tmp_path):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["service"] = fake_service(tmp_path)
    at.run()
    at.sidebar.text_input[0].set_value(str(tmp_path / "nope")).run()
    next(b for b in at.sidebar.button if b.label == "Build index").click().run()
    assert not at.exception
    assert any("does not exist" in e.value for e in at.error)
