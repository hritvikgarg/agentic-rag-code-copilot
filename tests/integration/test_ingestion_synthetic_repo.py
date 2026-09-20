"""End-to-end ingestion of the synthetic repository against hand-written expectations."""

import hashlib
import logging
import os

import pytest

from copilot.ingestion import IngestionPolicy, ingest_repository
from copilot.models import SkipReason
from tests.fixtures.synthetic_repo import (
    EXPECTED_ACCEPTED,
    EXPECTED_LANGUAGE_COUNTS,
    EXPECTED_PRUNED,
    EXPECTED_SKIPPED,
    FILES,
    MAIN_PY_MARKER,
    PRUNED_PREFIXES,
    SYNTHETIC_MAX_FILE_SIZE,
    UTF16_TEXT,
)

POLICY = IngestionPolicy(max_file_size_bytes=SYNTHETIC_MAX_FILE_SIZE)


@pytest.fixture
def result(synthetic_repo):
    return ingest_repository(synthetic_repo, POLICY)


def test_accepts_exactly_the_expected_files_in_sorted_order(result):
    assert result.relative_paths == list(EXPECTED_ACCEPTED)
    assert result.relative_paths == sorted(result.relative_paths)


def test_nested_files_are_found(result):
    assert "src/app/utils/helpers.py" in result.relative_paths
    assert "UPPER/SCRIPT.PY" in result.relative_paths  # upper-case extension


def test_skipped_files_and_reasons_match_expectations(result):
    assert {s.relative_path: s.reason for s in result.skipped} == EXPECTED_SKIPPED
    assert [s.relative_path for s in result.skipped] == sorted(EXPECTED_SKIPPED)


def test_pruned_directories_are_counted_and_never_enumerated(result):
    assert result.stats.pruned_directory_names == EXPECTED_PRUNED
    assert result.stats.directories_pruned == sum(EXPECTED_PRUNED.values())
    seen = result.relative_paths + [s.relative_path for s in result.skipped]
    assert not [p for p in seen if p.startswith(PRUNED_PREFIXES)]


def test_pruned_directories_are_never_even_listed(synthetic_repo, monkeypatch):
    """Pruning must happen during traversal, not by scanning and discarding afterwards."""
    scanned: list[str] = []
    real_scandir = os.scandir

    def spy(path):
        scanned.append(os.fspath(path))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", spy)
    ingest_repository(synthetic_repo, POLICY)

    pruned_names = set(EXPECTED_PRUNED)
    assert scanned, "scandir was not used"
    assert not [p for p in scanned if os.path.basename(p) in pruned_names]


def test_statistics_are_consistent(result):
    stats = result.stats
    assert stats.files_accepted == len(EXPECTED_ACCEPTED) == len(result.files)
    assert stats.files_skipped == len(EXPECTED_SKIPPED) == len(result.skipped)
    assert stats.files_discovered == stats.files_accepted + stats.files_skipped
    assert sum(stats.skip_reasons.values()) == stats.files_skipped
    assert stats.skip_reasons == {
        "binary": 1,
        "encoding_error": 1,
        "ignored_file": 2,
        "oversized": 1,
        "sensitive": 7,
        "unsupported_extension": 3,
    }
    assert stats.languages == EXPECTED_LANGUAGE_COUNTS
    assert stats.total_bytes_accepted == sum(f.size_bytes for f in result.files)
    assert stats.directories_unreadable == 0
    assert stats.truncated is False and stats.truncation_reason is None
    assert stats.elapsed_seconds >= 0


def test_language_and_extension_fields(result):
    by_path = {f.relative_path: f for f in result.files}
    assert (by_path["web/App.jsx"].language, by_path["web/App.jsx"].extension) == (
        "javascript",
        ".jsx",
    )
    assert by_path["UPPER/SCRIPT.PY"].extension == ".py"  # extension lower-cased
    assert by_path["config/ci.yml"].language == "yaml"


def test_names_that_merely_look_sensitive_are_kept(result):
    assert "src/app/secrets.py" in result.relative_paths  # source module
    assert "models/tokenizer.json" in result.relative_paths


def test_content_normalisation_and_metadata(result, synthetic_repo):
    by_path = {f.relative_path: f for f in result.files}
    crlf = by_path["src/app/crlf_module.py"]
    assert "\r" not in crlf.content
    assert crlf.content.count("\n") == 6  # same number of lines as the CRLF original
    assert by_path["src/app/utf8_bom.py"].content == "VALUE = 1\n"  # BOM stripped
    assert by_path["src/app/utf16_module.py"].content == UTF16_TEXT
    assert by_path["src/app/__init__.py"].content == ""  # empty files are valid

    for f in result.files:
        raw = (synthetic_repo.joinpath(*f.relative_path.split("/"))).read_bytes()
        assert f.size_bytes == len(raw)
        assert f.sha256 == hashlib.sha256(raw).hexdigest()
        assert f.repository_name == "synthetic-repo"


def test_ingestion_is_deterministic(synthetic_repo):
    first = ingest_repository(synthetic_repo, POLICY)
    second = ingest_repository(synthetic_repo, POLICY)
    assert first.files == second.files
    assert first.skipped == second.skipped
    assert first.stats.model_copy(update={"elapsed_seconds": 0}) == second.stats.model_copy(
        update={"elapsed_seconds": 0}
    )


def test_result_contains_no_absolute_host_paths(result, synthetic_repo):
    dumped = result.model_dump_json()
    assert str(synthetic_repo) not in dumped
    assert os.fspath(synthetic_repo.parent) not in dumped
    for f in result.files:
        assert not f.relative_path.startswith(("/", "\\")) and ":" not in f.relative_path


def test_repository_name_default_and_override(synthetic_repo):
    assert ingest_repository(synthetic_repo, POLICY).repository_name == "synthetic-repo"
    named = ingest_repository(synthetic_repo, POLICY, repository_name="  my-repo ")
    assert named.repository_name == "my-repo"
    assert {f.repository_name for f in named.files} == {"my-repo"}


def test_default_policy_comes_from_settings(synthetic_repo, monkeypatch):
    default = ingest_repository(synthetic_repo)  # 500,000-byte default limit
    assert "big/huge_module.py" in default.relative_paths
    monkeypatch.setenv("COPILOT_MAX_FILE_SIZE_BYTES", str(SYNTHETIC_MAX_FILE_SIZE))
    from copilot.config import get_settings

    get_settings.cache_clear()
    limited = ingest_repository(synthetic_repo)
    assert "big/huge_module.py" not in limited.relative_paths
    assert {s.relative_path: s.reason for s in limited.skipped}["big/huge_module.py"] == (
        SkipReason.OVERSIZED
    )


def test_extra_ignored_directory_is_pruned(synthetic_repo):
    result = ingest_repository(synthetic_repo, POLICY.with_extra_ignored_directories("docs"))
    assert "docs/guide.md" not in result.relative_paths
    assert result.stats.pruned_directory_names["docs"] == 1


def test_logging_reports_a_summary_but_never_contents_or_host_paths(
    synthetic_repo, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.DEBUG, logger="copilot"):
        ingest_repository(synthetic_repo, POLICY)
    text = "\n".join(r.getMessage() for r in caplog.records)
    info = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(info) == 1 and "Ingested repository" in info[0].getMessage()
    assert MAIN_PY_MARKER not in text
    assert str(synthetic_repo) not in text
    assert FILES["config/settings.json"].strip() not in text  # type: ignore[union-attr]
    assert any("(sensitive)" in r.getMessage() for r in caplog.records)  # per-skip detail exists
    assert all(
        r.levelno == logging.DEBUG for r in caplog.records if "(sensitive)" in r.getMessage()
    )
