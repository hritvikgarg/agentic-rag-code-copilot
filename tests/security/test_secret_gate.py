"""The fail-closed external-LLM gate and the scanner's repository integration."""

import hashlib
import inspect
import logging

import pytest

from copilot.ingestion import ingest_repository
from copilot.models.chunk import Chunk, ChunkType
from copilot.retrieval import RetrievalResult
from copilot.security import (
    RepositorySecretRiskError,
    ScanTarget,
    SecurityError,
    as_scan_target,
    assert_safe_for_external_llm,
    scan_repository,
    scan_source_file,
)
from copilot.security.errors import MAX_LISTED_FINDINGS
from tests.security import secret_helpers as h

SHA = "a" * 64


def make_chunk(content: str, *, path: str = "src/app.py", start: int = 10) -> Chunk:
    lines = content.count("\n") + 1
    return Chunk(
        chunk_id="0123456789abcdef", repository_name="demo", file_path=path, source_sha256=SHA,
        language="python", chunk_type=ChunkType.LINE_WINDOW, chunking_strategy="line",
        chunking_version=1, chunk_index=0, start_line=start, end_line=start + lines - 1,
        content=content, content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        token_estimate=5,
    )  # fmt: skip


def make_result(text: str, *, path: str = "src/app.py", start: int = 10) -> RetrievalResult:
    lines = text.count("\n") + 1
    return RetrievalResult(
        rank=1, score=0.5, position=0, chunk_id="0123456789abcdef", repository_name="demo",
        file_path=path, language="python", chunk_type=ChunkType.LINE_WINDOW, chunk_index=0,
        start_line=start, end_line=start + lines - 1, token_estimate=5, content_sha256=SHA,
        text=text,
    )  # fmt: skip


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "my repo ü"  # a space and a non-ASCII character on purpose
    (root / "src").mkdir(parents=True)
    (root / "src" / "clean.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (root / "src" / "leaky.py").write_text(
        f'A = 1\nGITHUB = "{h.github_token("repo")}"\nkey = "{h.aws_access_key("repo")}"\n',
        encoding="utf-8",
    )
    (root / "notes.md").write_text("Set API_KEY to your-api-key-here.\n", encoding="utf-8")
    # a sensitive NAME: ingestion skips it, so its content is never scanned or indexed
    (root / ".env").write_text(f"TOKEN={h.github_token('dotenv')}\n", encoding="utf-8")
    return root


# --------------------------------------------------------------------------------------------
# scan_repository / scan_source_file
# --------------------------------------------------------------------------------------------


def test_scan_repository_counts_and_relative_paths(repo):
    ingestion = ingest_repository(repo)
    report = scan_repository(ingestion.files)
    assert report.files_scanned == 3  # .env was skipped by name before content was read
    assert report.findings_count == 2
    assert report.counts_by_rule == {"aws-access-key-id": 1, "github-token": 1}
    assert report.files_with_findings == ["src/leaky.py"]
    assert [(f.line, f.rule_id) for f in report.findings] == [
        (2, "github-token"),
        (3, "aws-access-key-id"),
    ]
    text = repr(report) + report.model_dump_json()
    assert str(repo) not in text and "my repo" not in text


def test_scan_source_file_uses_the_ingested_relative_path(repo):
    files = {f.relative_path: f for f in ingest_repository(repo).files}
    assert scan_source_file(files["src/clean.py"]) == []
    assert {f.file_path for f in scan_source_file(files["src/leaky.py"])} == {"src/leaky.py"}


def test_scanning_a_repository_is_deterministic(repo):
    first = scan_repository(ingest_repository(repo).files)
    second = scan_repository(reversed(ingest_repository(repo).files))
    assert first == second


# --------------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------------


def test_gate_passes_clean_context_and_returns_the_report():
    report = assert_safe_for_external_llm(
        [make_chunk("def f():\n    return 1\n"), make_result("x = os.getenv('API_KEY')")]
    )
    assert report.files_scanned == 1 and not report.blocking


def test_gate_with_no_items_is_clean():
    assert not assert_safe_for_external_llm([]).blocking


def test_gate_refuses_and_the_message_is_safe():
    token = h.github_token("gate")
    context = [make_chunk("ok = 1"), make_result(f'a = 1\nbad = "{token}"\n', start=40)]
    with pytest.raises(RepositorySecretRiskError) as caught:
        assert_safe_for_external_llm(context)
    message = str(caught.value)
    assert "REFUSED for external LLM use" in message
    assert "src/app.py:41 [github-token]" in message  # chunk start line + offset
    assert "There is no override" in message
    assert "1 potential secret(s) in 1 file(s)" in message
    assert not h.contains_secret(message + repr(caught.value) + repr(caught.value.report), token)
    assert isinstance(caught.value, SecurityError)
    assert caught.value.report.blocking


def test_gate_message_lists_at_most_ten_findings():
    text = "\n".join(f'k{i} = "{h.github_token(f"many{i}")}"' for i in range(15))
    with pytest.raises(RepositorySecretRiskError) as caught:
        assert_safe_for_external_llm([make_chunk(text, start=1)])
    assert "15 potential secret(s)" in str(caught.value)
    assert f"... and {15 - MAX_LISTED_FINDINGS} more" in str(caught.value)
    assert str(caught.value).count("[github-token]") == MAX_LISTED_FINDINGS


def test_gate_refuses_a_split_private_key_header_in_a_chunk():
    header = "-----BEGIN " + "RSA PRIVATE KEY-----"
    with pytest.raises(RepositorySecretRiskError, match="private-key-header"):
        assert_safe_for_external_llm([make_chunk(f"x = 1\n{header}\n")])


def test_gate_does_not_log_the_secret(caplog):
    token = h.github_token("logs")
    with caplog.at_level(logging.DEBUG), pytest.raises(RepositorySecretRiskError):
        assert_safe_for_external_llm([make_chunk(f'k = "{token}"')])
    assert not h.contains_secret(caplog.text, token)


def test_gate_has_no_bypass_argument():
    assert list(inspect.signature(assert_safe_for_external_llm).parameters) == ["items"]
    with pytest.raises(TypeError):
        assert_safe_for_external_llm([], force=True)  # type: ignore[call-arg]


def test_gate_scans_user_supplied_text_given_as_a_target():
    error_log = ScanTarget("user-input.txt", f"request failed with token {h.github_token('q')}")
    with pytest.raises(RepositorySecretRiskError):
        assert_safe_for_external_llm([make_chunk("ok = 1"), error_log])


def test_gate_refuses_items_it_cannot_scan():
    with pytest.raises(TypeError, match="cannot scan an object of type 'str'"):
        assert_safe_for_external_llm(["just a string"])
    with pytest.raises(TypeError):
        assert_safe_for_external_llm([object()])


def test_adapters_map_types_to_targets(repo):
    source_file = ingest_repository(repo).files[0]
    assert as_scan_target(source_file).path == source_file.relative_path
    chunk_target = as_scan_target(make_chunk("a", start=7))
    assert (chunk_target.path, chunk_target.first_line) == ("src/app.py", 7)
    assert as_scan_target(make_result("a", start=9)).first_line == 9
    target = ScanTarget("a.py", "x")
    assert as_scan_target(target) is target


def test_gate_covers_whole_repository_context(repo):
    with pytest.raises(RepositorySecretRiskError, match="src/leaky.py:2 .github-token"):
        assert_safe_for_external_llm(ingest_repository(repo).files)
