"""CopilotService: build -> open -> ask with a fake embedder and fake LLM (no model/network)."""

import pytest

from copilot.config import Settings
from copilot.embeddings import HashEmbedder
from copilot.llm import FakeLLMClient
from copilot.rag.citations import Citation
from copilot.rag.models import AnswerStatus
from copilot.retrieval.errors import StaleRepositoryError
from copilot.services.copilot_service import CopilotService, ServiceStateError
from tests.retrieval_helpers import write_repo

SETTINGS = Settings(_env_file=None)


def make_service(llm=None):
    return CopilotService(
        settings=SETTINGS,
        embedder_factory=lambda s: HashEmbedder(dimension=64),
        llm_factory=lambda s: llm or FakeLLMClient("The chunk id is a sha256 hash [Source 1]."),
    )


@pytest.fixture
def prepared(tmp_path):
    repo = write_repo(tmp_path / "repo")
    indexes = tmp_path / "indexes"
    service = make_service()
    report = service.build_index(repo, indexes_dir=indexes)
    return service, repo, indexes, report


def test_build_then_list_and_summarize(prepared):
    service, _repo, indexes, report = prepared
    listed = service.list_indexes(indexes)
    assert [i.index_id for i in listed] == [report.index_id]
    assert listed[0].chunk_count == report.chunks and listed[0].repository_name == "repo"


def test_list_indexes_ignores_missing_dirs_staging_and_broken_folders(tmp_path):
    service = make_service()
    assert service.list_indexes(tmp_path / "nope") == []
    root = tmp_path / "idx"
    (root / ".staging-x").mkdir(parents=True)
    (root / "broken").mkdir()
    (root / "broken" / "manifest.json").write_text("{not json", encoding="utf-8")
    assert service.list_indexes(root) == []


def test_ask_requires_an_open_index():
    with pytest.raises(ServiceStateError, match="no index is open"):
        make_service().ask("anything?")


def test_open_ask_returns_grounded_answer_with_authoritative_sources(prepared):
    service, repo, indexes, report = prepared
    opened = service.open_index(report.index_path, repo)
    assert opened.summary.index_id == report.index_id and service.opened is opened
    answer = service.ask("How is the chunk id made?", top_k=3)
    assert answer.status is AnswerStatus.ANSWERED and answer.sources
    assert answer.cited_source_numbers == (1,)
    assert answer.sources[0].file_path == "src/pkg/chunk_ids.py"


def test_open_refuses_a_repository_that_changed_since_the_index(prepared):
    service, repo, _indexes, report = prepared
    (repo / "src/pkg/ingest.py").write_text("def changed():\n    pass\n", encoding="utf-8")
    with pytest.raises(StaleRepositoryError):
        service.open_index(report.index_path, repo)
    assert service.opened is None


def test_evidence_text_returns_the_cited_lines_and_truncates_long_ranges(prepared):
    service, repo, _i, report = prepared
    service.open_index(report.index_path, repo)
    cite = Citation(
        source_number=1, file_path="src/pkg/chunk_ids.py", start_line=1, end_line=2,
        chunk_id="0123456789abcdef", score=0.5, rank=1,
    )  # fmt: skip
    text = service.evidence_text(cite)
    assert text.splitlines()[0].startswith("def make_chunk_id")
    big = cite.model_copy(update={"end_line": 10_000})
    assert len(service.evidence_text(big).splitlines()) <= 4  # the file only has 4 lines


def test_evidence_text_refuses_a_path_that_escapes_the_repository(prepared):
    service, repo, _i, report = prepared
    service.open_index(report.index_path, repo)
    bad = Citation.model_construct(  # bypass validation to simulate a tampered citation
        source_number=1, file_path="../outside.txt", start_line=1, end_line=2,
        chunk_id="0123456789abcdef", score=0.1, rank=1,
    )  # fmt: skip
    with pytest.raises(ServiceStateError, match="unsafe"):
        service.evidence_text(bad)


def test_ask_plain_has_no_repository_context(prepared):
    service, *_ = prepared
    plain = service.ask_plain("What is RAG?")
    assert plain.answer and plain.status is AnswerStatus.ANSWERED


def test_resources_are_created_lazily_and_once(tmp_path):
    calls = {"embedder": 0, "llm": 0}

    def embedder_factory(settings):
        calls["embedder"] += 1
        return HashEmbedder(dimension=64)

    def llm_factory(settings):
        calls["llm"] += 1
        return FakeLLMClient("ok")

    service = CopilotService(
        settings=SETTINGS, embedder_factory=embedder_factory, llm_factory=llm_factory
    )
    assert calls == {"embedder": 0, "llm": 0}  # constructing and listing load nothing
    service.list_indexes(tmp_path)
    assert calls == {"embedder": 0, "llm": 0}
    repo = write_repo(tmp_path / "repo")
    service.build_index(repo, indexes_dir=tmp_path / "ix")
    service.build_index(repo, indexes_dir=tmp_path / "ix2")
    assert calls["embedder"] == 1 and calls["llm"] == 0
