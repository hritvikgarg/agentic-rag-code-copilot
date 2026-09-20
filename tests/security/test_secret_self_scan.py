"""The repository must not contain secrets - and the scanner must not flag its own sources."""

from pathlib import Path

import pytest

from copilot.ingestion import IngestionPolicy, ingest_repository
from copilot.security import scan_repository

ROOT = Path(__file__).resolve().parents[2]
TRACKED_DIRS = ("src", "tests", "docs", "benchmarks")


@pytest.mark.skipif(not (ROOT / "src").is_dir(), reason="needs a source checkout")
def test_project_sources_tests_and_docs_contain_no_potential_secrets():
    others = [
        entry.name
        for entry in ROOT.iterdir()
        if entry.is_dir() and entry.name not in TRACKED_DIRS and not entry.name.startswith(".")
    ]
    policy = IngestionPolicy().with_extra_ignored_directories(*others)
    files = [
        f
        for f in ingest_repository(ROOT, policy).files
        if f.relative_path.split("/")[0] in TRACKED_DIRS
    ]
    assert files, "no files were scanned"
    report = scan_repository(files)
    assert report.findings_count == 0, [(f.location, f.rule_id) for f in report.findings]
