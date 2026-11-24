"""Frozen evaluation snapshots: the evaluated tree stays valid while the application moves on.

Everything is local: a throw-away git repository, a hash-based fake embedder, no network.
"""

import io
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from copilot.config import Settings
from copilot.embeddings import HashEmbedder
from copilot.evaluation import __main__ as cli
from copilot.evaluation.snapshot import (
    METADATA_FILENAME,
    SnapshotError,
    create_snapshot,
    export_commit,
    resolve_commit,
    snapshot_paths,
    verify_tree_against_index,
)
from copilot.llm import FakeLLMClient
from copilot.retrieval import StaleRepositoryError
from copilot.services.copilot_service import CopilotService
from tests.retrieval_helpers import FILES

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
         "-c", "commit.gpgsign=false", "-c", "init.defaultBranch=main", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return result.stdout.strip()


def commit_all(repo: Path, message: str) -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def live_repo(tmp_path) -> Path:
    """The "application under development": a git repository with one commit."""
    repo = tmp_path / "live"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / ".gitattributes").write_text("* text=auto eol=lf\n", encoding="utf-8", newline="\n")
    for relative, text in {**FILES, "tests/test_x.py": "def test_x():\n    pass\n"}.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    commit_all(repo, "first")
    return repo


@pytest.fixture
def service(tmp_path) -> CopilotService:
    return CopilotService(
        settings=Settings(_env_file=None, data_dir=tmp_path / "data"),
        embedder_factory=lambda s: HashEmbedder(dimension=64),
        llm_factory=lambda s: FakeLLMClient("x"),
    )


def build(service, repo, tmp_path, ignore=("tests",)) -> Path:
    report = service.build_index(repo, ignore_directories=ignore, indexes_dir=tmp_path / "ix")
    return Path(report.index_path)


def test_a_snapshot_stays_valid_for_the_index_after_the_live_tree_changes(
    tmp_path, live_repo, service
):
    index = build(service, live_repo, tmp_path)
    tree, check = create_snapshot(
        live_repo, "HEAD", tmp_path / "snaps", index_dir=index, ignore_directories=["tests"]
    )
    assert check is not None and check.matches and check.file_count == len(FILES)

    # the application keeps being developed: an indexed file changes and is committed
    (live_repo / "src/pkg/ingest.py").write_text("def ingest_repository(root):\n    pass\n")
    commit_all(live_repo, "second")

    with pytest.raises(StaleRepositoryError):  # the live tree no longer matches the index...
        service.open_index(index, live_repo, ignore_directories=["tests"])
    opened = service.open_index(index, tree, ignore_directories=["tests"])  # ...the snapshot does
    assert opened.summary.index_id == check.index_id
    assert "discover_files(root)" in (tree / "src/pkg/ingest.py").read_text(encoding="utf-8")


def test_a_snapshot_that_does_not_match_the_index_installs_nothing(tmp_path, live_repo, service):
    index = build(service, live_repo, tmp_path)
    (live_repo / "docs/guide.md").write_text("# Guide\n\nChanged.\n", encoding="utf-8")
    commit_all(live_repo, "second")  # HEAD no longer reproduces the index

    snaps = tmp_path / "snaps"
    with pytest.raises(SnapshotError, match="does not match the index"):
        create_snapshot(live_repo, "HEAD", snaps, index_dir=index, ignore_directories=["tests"])
    assert list(snaps.iterdir()) == []  # no snapshot and no staging leftovers


def test_creating_twice_returns_the_same_tree_and_never_overwrites_it(tmp_path, live_repo, service):
    index = build(service, live_repo, tmp_path)
    snaps = tmp_path / "snaps"
    first, _ = create_snapshot(
        live_repo, "HEAD", snaps, index_dir=index, ignore_directories=["tests"]
    )
    second, check = create_snapshot(
        live_repo, "HEAD", snaps, index_dir=index, ignore_directories=["tests"]
    )
    assert first == second and check is not None and check.matches

    # a hand-edit of the frozen tree is detected by the re-check, and is not silently "repaired"
    target = first / "docs/guide.md"
    target.write_text("# tampered\n", encoding="utf-8", newline="\n")
    with pytest.raises(SnapshotError, match="does not match the index"):
        create_snapshot(live_repo, "HEAD", snaps, index_dir=index, ignore_directories=["tests"])
    assert target.read_text(encoding="utf-8") == "# tampered\n"


def test_the_metadata_sits_next_to_the_tree_and_never_changes_the_fingerprint(
    tmp_path, live_repo, service
):
    index = build(service, live_repo, tmp_path)
    tree, _ = create_snapshot(
        live_repo, "HEAD", tmp_path / "snaps", index_dir=index, ignore_directories=["tests"]
    )
    meta = tree.parent / METADATA_FILENAME
    assert meta.is_file() and not (tree / METADATA_FILENAME).exists()
    text = meta.read_text(encoding="utf-8")
    assert git(live_repo, "rev-parse", "HEAD") in text
    assert str(tmp_path) not in text  # no absolute host paths in the record
    assert verify_tree_against_index(tree, index, ignore_directories=["tests"]).matches


def test_verification_uses_the_same_ignore_directories_as_the_index(tmp_path, live_repo, service):
    index = build(service, live_repo, tmp_path, ignore=("tests",))
    tree, _ = create_snapshot(live_repo, "HEAD", tmp_path / "snaps")  # unverified install
    assert verify_tree_against_index(tree, index, ignore_directories=["tests"]).matches
    wrong = verify_tree_against_index(tree, index)  # tests/ is now accepted -> different set
    assert not wrong.matches and wrong.actual_fingerprint != wrong.expected_fingerprint


def test_a_missing_or_corrupt_manifest_is_a_clear_error(tmp_path, live_repo):
    tree, _ = create_snapshot(live_repo, "HEAD", tmp_path / "snaps")
    empty = tmp_path / "empty-index"
    empty.mkdir()
    with pytest.raises(SnapshotError, match="cannot read the index manifest"):
        verify_tree_against_index(tree, empty)


def test_revisions_are_resolved_safely(live_repo):
    full = git(live_repo, "rev-parse", "HEAD")
    assert resolve_commit(live_repo, "HEAD") == full
    assert resolve_commit(live_repo, full[:8]) == full
    with pytest.raises(SnapshotError):
        resolve_commit(live_repo, "--output=evil")  # option-looking input is refused
    with pytest.raises(SnapshotError):
        resolve_commit(live_repo, "no-such-branch")


def test_a_hostile_archive_cannot_write_outside_the_destination(tmp_path, live_repo, monkeypatch):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        info = tarfile.TarInfo("../escaped.txt")
        payload = b"owned"
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    monkeypatch.setattr("copilot.evaluation.snapshot._git", lambda *a, **k: buffer.getvalue())
    destination = tmp_path / "dest"
    with pytest.raises(SnapshotError, match="cannot extract"):
        export_commit(live_repo, "a" * 40, destination)
    assert not (tmp_path / "escaped.txt").exists()


def test_snapshot_paths_are_keyed_by_the_full_commit(tmp_path):
    tree, meta = snapshot_paths(tmp_path, "a" * 40)
    assert tree == tmp_path / ("a" * 40) / "tree" and meta.name == "snapshot.json"


def test_the_command_line_creates_verifies_and_reports_a_mismatch(
    monkeypatch, tmp_path, live_repo, service, capsys
):
    index = build(service, live_repo, tmp_path)
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    snaps = tmp_path / "snaps"

    code = cli.main(
        ["snapshot", "create", "--commit", "HEAD", "--repo-root", str(live_repo),
         "--dest", str(snaps), "--index", str(index), "--ignore-dir", "tests"]
    )  # fmt: skip
    out = capsys.readouterr().out
    assert code == 0 and "MATCH" in out and "use it as: compare" in out
    tree = next(snaps.iterdir()) / "tree"

    assert cli.main(
        ["snapshot", "verify", "--snapshot", str(tree), "--index", str(index),
         "--ignore-dir", "tests"]
    ) == 0  # fmt: skip
    assert "MATCH" in capsys.readouterr().out

    assert cli.main(["snapshot", "verify", "--snapshot", str(tree), "--index", str(index)]) == 2
    assert "MISMATCH" in capsys.readouterr().out  # wrong ignore set


def test_resume_refuses_when_the_benchmark_changed_since_the_file_was_written():
    """The benchmark is anchored to one snapshot; a changed question must not be mixed in."""
    from copilot.evaluation.comparison import (
        ComparisonEntry,
        ComparisonMismatchError,
        ComparisonRun,
        PlainRecord,
        RagRecord,
        run_comparison,
    )
    from tests.unit.test_eval_circuit_breaker import bench, retriever

    q = bench(1)[0]
    stale = ComparisonEntry(
        question_id=q.id,
        question="an older wording of the question?",
        expected_regions=("src/ids.py:7-9",),
        plain=PlainRecord(answer="old"),
        rag=RagRecord(status="answered", answer="old"),
    )
    from copilot.rag.prompts import PLAIN_PROMPT_VERSION, RAG_PROMPT_VERSION

    settings = Settings(_env_file=None)
    prior = ComparisonRun(
        benchmark_name="b",
        llm_provider="fake",
        llm_model="fake-model",
        plain_prompt_version=PLAIN_PROMPT_VERSION,
        rag_prompt_version=RAG_PROMPT_VERSION,
        top_k=3,
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_output_tokens,
        entries=[stale],
    )
    llm = FakeLLMClient()
    with pytest.raises(ComparisonMismatchError, match=r"question text or expected regions"):
        run_comparison(
            [q], retriever(), llm, top_k=3, settings=settings, benchmark_name="b", resume_from=prior
        )
    assert llm.call_count == 0  # refused before any provider call
