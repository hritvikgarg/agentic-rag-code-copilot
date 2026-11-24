"""Streamlit smoke tests with ``AppTest`` and a pre-seeded fake service (no model, no network)."""

import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from copilot.config import Settings  # noqa: E402
from copilot.embeddings import HashEmbedder  # noqa: E402
from copilot.llm import FakeLLMClient  # noqa: E402
from copilot.services.copilot_service import CopilotService  # noqa: E402
from tests.retrieval_helpers import FILES, write_repo  # noqa: E402

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


def test_source_filenames_and_repo_names_are_shown_literally(tmp_path):
    """``__init__.py`` must not be rendered as bold "init.py" (a Markdown double underscore)."""
    files = {**FILES, "pkg/__init__.py": "def zebra_unique_marker():\n    return 'zebra'\n"}
    repo = write_repo(tmp_path / "my__repo", files)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["service"] = fake_service(tmp_path)
    at.run()
    at.sidebar.text_input[0].set_value(str(repo)).run()
    next(b for b in at.sidebar.button if b.label == "Build index").click().run()
    at.text_area[0].set_value("zebra_unique_marker").run()
    next(b for b in at.button if b.label == "Ask").click().run()
    assert not at.exception, at.exception

    labels = [e.label for e in at.expander]
    init_labels = [label for label in labels if "__init__.py" in label]
    assert init_labels, labels
    for label in labels:
        # every occurrence of a path is inside a code span, never bare Markdown
        assert "__init__" not in re.sub(r"`[^`]*`", "", label), label
    assert any("`pkg/__init__.py:" in label for label in init_labels)

    ready = next(s.value for s in at.success if "Ready:" in s.value)
    assert "`my__repo`" in ready  # a repository name with underscores is literal too
