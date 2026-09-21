"""End to end: real ingestion/chunking/FAISS + hash embeddings -> retrieval -> RAG -> fake LLM.

Only the embedding model and the LLM are fakes. Nothing here says anything about answer quality.
"""

import pytest

from copilot.config import Settings
from copilot.embeddings import HashEmbedder
from copilot.llm import FakeLLMClient
from copilot.rag import AnswerStatus, answer_plain, answer_with_rag
from copilot.retrieval import StaleRepositoryError
from copilot.security import RepositorySecretRiskError
from copilot.vectorstore import build_repository_index
from tests.retrieval_helpers import FILES, write_repo
from tests.security import secret_helpers as h

SETTINGS = Settings(_env_file=None)


@pytest.fixture
def embedder():
    return HashEmbedder(dimension=256)


def build(repo, tmp_path, embedder):
    return build_repository_index(
        repo, embedder=embedder, settings=SETTINGS, indexes_dir=tmp_path / "indexes",
        text_style="raw", overwrite=True,
    ).index_path  # fmt: skip


@pytest.fixture
def repo(tmp_path):
    return write_repo(tmp_path / "demo repo")  # a space in the path on purpose


def test_the_expected_evidence_reaches_the_context_and_the_citations(repo, tmp_path, embedder):
    index = build(repo, tmp_path, embedder)
    fake = FakeLLMClient("Chunk ids are sha256 hashes [Source 1].")
    answer = answer_with_rag(
        "deterministic chunk id sha256 hash payload", repo, index, 3,
        llm=fake, embedder=embedder, settings=SETTINGS,
    )  # fmt: skip
    assert answer.status is AnswerStatus.ANSWERED
    assert answer.sources[0].file_path == "src/pkg/chunk_ids.py"
    assert answer.sources[0].location == "src/pkg/chunk_ids.py:1-4"
    (request,) = fake.requests
    assert FILES["src/pkg/chunk_ids.py"].splitlines()[1] in request.user_prompt
    assert "file: src/pkg/chunk_ids.py" in request.user_prompt
    # only repository-relative POSIX paths leave the machine; never the host path
    for text in (request.outbound_text(), answer.model_dump_json()):
        assert (
            str(repo) not in text
            and str(repo.parent) not in text
            and "\\" not in "".join(s.file_path for s in answer.sources)
        )
    assert answer.cited_source_numbers == (1,)


def test_a_stale_index_is_refused_and_nothing_is_sent(repo, tmp_path, embedder):
    index = build(repo, tmp_path, embedder)
    (repo / "src/pkg/chunk_ids.py").write_text("x = 1\n", encoding="utf-8")  # stale index
    fake = FakeLLMClient()
    with pytest.raises(StaleRepositoryError):
        answer_with_rag("chunk id", repo, index, 3, llm=fake, embedder=embedder, settings=SETTINGS)
    assert fake.call_count == 0  # stale evidence is never sent


def test_a_secret_in_the_repository_is_refused_before_the_client(repo, tmp_path, embedder):
    secret = h.github_token("pipeline")
    (repo / "src/pkg/config.py").write_text(
        f"# chunk id configuration for the deterministic sha256 hash payload\nTOKEN = '{secret}'\n",
        encoding="utf-8",
    )
    index = build(repo, tmp_path, embedder)
    fake = FakeLLMClient()
    with pytest.raises(RepositorySecretRiskError) as info:
        answer_with_rag(
            "deterministic chunk id sha256 hash payload configuration", repo, index, 5,
            llm=fake, embedder=embedder, settings=SETTINGS,
        )  # fmt: skip
    assert fake.call_count == 0
    assert "src/pkg/config.py" in str(info.value) and not h.contains_secret(str(info.value), secret)


def test_the_secret_free_question_still_works_when_the_secret_chunk_is_not_retrieved(
    repo, tmp_path, embedder
):
    secret = h.github_token("elsewhere")
    (repo / "src/pkg/unrelated.py").write_text(f"TOKEN = '{secret}'\n", encoding="utf-8")
    index = build(repo, tmp_path, embedder)
    fake = FakeLLMClient("ok")
    # top_k=1: only the best chunk is retrieved, and it is not the secret file
    answer = answer_with_rag(
        "configure logging handlers formatter log level", repo, index, 1,
        llm=fake, embedder=embedder, settings=SETTINGS,
    )  # fmt: skip
    assert answer.sources[0].file_path == "src/pkg/logging_setup.py" and fake.call_count == 1
    assert secret not in fake.requests[0].outbound_text()


def test_plain_and_rag_use_the_same_client_with_different_requests(repo, tmp_path, embedder):
    index = build(repo, tmp_path, embedder)
    fake = FakeLLMClient()
    answer_plain("How is logging configured?", fake, settings=SETTINGS)
    answer_with_rag(
        "How is logging configured?", repo, index, 2, llm=fake, embedder=embedder, settings=SETTINGS
    )
    plain_request, rag_request = fake.requests
    assert "SOURCE" not in plain_request.user_prompt and "BEGIN SOURCE" in rag_request.user_prompt
    assert plain_request.prompt_version != rag_request.prompt_version
