"""The agentic layer leaves the baseline, evaluation artifacts and settings contract alone."""

import ast
import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

import copilot
from copilot.agents import AgentConfig
from copilot.config import Settings
from copilot.embeddings import HashEmbedder
from copilot.llm import FakeLLMClient
from copilot.rag import RAG_PROMPT_VERSION, AnswerStatus, RagService
from copilot.retrieval import Retriever
from copilot.vectorstore import build_repository_index
from tests.agent_helpers import SETTINGS, ScriptedRetriever, chunk, workflow
from tests.rag_helpers import FakeRetriever
from tests.retrieval_helpers import write_repo

SRC = Path(copilot.__file__).parent


def test_the_baseline_service_gives_the_same_answer_through_answer_and_answer_from_evidence():
    results = [chunk("def make_chunk_id(): ...", path="src/ids.py", start=3, rank=1),
               chunk("def other(): ...", path="src/o.py", start=1, rank=2)]  # fmt: skip
    one, two = FakeLLMClient("a [Source 1]"), FakeLLMClient("a [Source 1]")
    a = RagService(FakeRetriever(results), one, settings=SETTINGS).answer("How are ids made?", 5)
    b = RagService(FakeRetriever([]), two, settings=SETTINGS).answer_from_evidence(
        "How are ids made?", results
    )
    assert one.requests[0] == two.requests[0]
    for field in ("status", "answer", "sources", "cited_source_numbers", "prompt_version",
                  "retrieved_count", "context_estimated_tokens", "dropped_evidence"):  # fmt: skip
        assert getattr(a, field) == getattr(b, field)
    assert a.prompt_version == RAG_PROMPT_VERSION == "rag-v1"


def test_baseline_abstention_is_unchanged():
    llm = FakeLLMClient()
    answer = RagService(FakeRetriever([]), llm, settings=SETTINGS).answer("How are ids made?")
    assert answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE and llm.call_count == 0
    assert answer.insufficient_reason == "no_results"


def test_no_baseline_package_imports_the_agent_package():
    for package in ("rag", "retrieval", "llm", "embeddings", "vectorstore", "evaluation",
                    "security", "ingestion", "chunking", "services"):  # fmt: skip
        for path in (SRC / package).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith("copilot.agents"), path
                elif isinstance(node, ast.Import):
                    assert not any(a.name.startswith("copilot.agents") for a in node.names), path


def test_the_agent_package_never_touches_evaluation_snapshots_or_indexes_by_name():
    for path in (SRC / "agents").glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for forbidden in ("data/evaluation", "snapshots", "comparison", "build_repository_index",
                          "submission"):  # fmt: skip
            assert forbidden not in text, (path.name, forbidden)


def test_a_run_writes_nothing_to_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wf, _ = workflow(ScriptedRetriever([[chunk("authentication code", path="src/a.py")]]))
    wf.run("Explain how authentication works")
    assert list(tmp_path.iterdir()) == []


def test_agent_settings_defaults_and_validation(monkeypatch):
    s = Settings(_env_file=None)
    assert (s.agent_max_retrieval_rounds, s.agent_max_evidence_chunks) == (3, 15)
    assert (s.agent_min_term_coverage, s.agent_architecture_min_files) == (0.5, 2)
    config = AgentConfig.from_settings(s)
    assert (config.max_rounds, config.top_k) == (3, s.retrieval_top_k)
    assert AgentConfig.from_settings(s, max_rounds=1).max_rounds == 1
    for name, bad in (("MAX_RETRIEVAL_ROUNDS", "0"), ("MAX_RETRIEVAL_ROUNDS", "6"),
                      ("MIN_TERM_COVERAGE", "1.5"), ("MAX_EVIDENCE_CHUNKS", "0")):  # fmt: skip
        monkeypatch.setenv(f"COPILOT_AGENT_{name}", bad)
        with pytest.raises(ValidationError):
            Settings(_env_file=None)
        monkeypatch.delenv(f"COPILOT_AGENT_{name}")
    monkeypatch.setenv("COPILOT_AGENT_MAX_RETRIEVAL_ROUNDS", "2")
    assert Settings(_env_file=None).agent_max_retrieval_rounds == 2


def test_the_workflow_over_a_real_index_cites_real_repository_locations(tmp_path):
    repo = write_repo(tmp_path / "repo")
    index = build_repository_index(
        repo, embedder=HashEmbedder(dimension=64), settings=Settings(_env_file=None),
        indexes_dir=tmp_path / "indexes", text_style="raw",
    ).index_path  # fmt: skip
    retriever = Retriever.open(index, repo, embedder=HashEmbedder(dimension=64),
                               settings=Settings(_env_file=None))  # fmt: skip
    wf, llm = workflow(retriever)
    result = wf.run("deterministic chunk id sha256 hash payload")
    assert result.generated and result.citations
    for citation in result.citations:
        source = repo / citation.file_path
        assert source.is_file() and citation.file_path == source.relative_to(repo).as_posix()
        lines = source.read_text(encoding="utf-8").splitlines()
        assert 1 <= citation.start_line <= citation.end_line <= len(lines)
    # the cited chunk text really is what the repository contains (verified retrieval)
    first = result.citations[0]
    chunk_text = "\n".join(
        (repo / first.file_path)
        .read_text(encoding="utf-8")
        .splitlines()[first.start_line - 1 : first.end_line]
    )
    assert chunk_text.strip() in llm.requests[0].user_prompt


def test_a_run_leaves_the_index_and_repository_byte_identical(tmp_path):
    repo = write_repo(tmp_path / "repo")
    index = build_repository_index(
        repo, embedder=HashEmbedder(dimension=64), settings=Settings(_env_file=None),
        indexes_dir=tmp_path / "indexes", text_style="raw",
    ).index_path  # fmt: skip

    def digest():
        h = hashlib.sha256()
        for root in (repo, index):
            for path in sorted(Path(root).rglob("*")):
                if path.is_file():
                    h.update(path.relative_to(root).as_posix().encode() + path.read_bytes())
        return h.hexdigest()

    before = digest()
    retriever = Retriever.open(index, repo, embedder=HashEmbedder(dimension=64),
                               settings=Settings(_env_file=None))  # fmt: skip
    workflow(retriever)[0].run("Explain how deterministic chunk ids are hashed", max_rounds=3)
    assert digest() == before


def test_the_trace_is_json_serialisable_and_bounded():
    import json

    wf, _ = workflow(ScriptedRetriever([[chunk("authentication code")]]))
    result = wf.run("Explain how authentication works")
    json.dumps([e.model_dump() for e in result.trace])
    assert len(result.trace) <= 3 * result.max_rounds + 2
