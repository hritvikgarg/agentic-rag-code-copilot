"""Security-focused ingestion tests: path traversal, symlinks, special files, limits, failures."""

import os
import stat
import sys

import pytest

from copilot.ingestion import (
    IngestionPolicy,
    InvalidRepositoryError,
    ingest_repository,
    loader,
    reader,
    resolve_repository_root,
)
from copilot.ingestion.discovery import Candidate
from copilot.models import SkipReason
from copilot.utils.paths import UnsafePathError, is_within_root, safe_join

OUTSIDE_MARKER = "OUTSIDE_FILE_MARKER_c91d"


def make_symlink(link, target, *, is_dir=False):
    try:
        link.symlink_to(target, target_is_directory=is_dir)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported/permitted on this platform")


@pytest.fixture
def repo_and_outside(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "ok.py").write_text("x = 1\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "leak.py").write_text(f"LEAK = '{OUTSIDE_MARKER}'\n")
    (outside / "nested").mkdir()
    (outside / "nested" / "deep.py").write_text("y = 2\n")
    return repo, outside


def skipped(result):
    return {s.relative_path: s.reason for s in result.skipped}


# ----------------------------------------------------------------------------- symlink policy
def test_symlink_to_outside_file_is_rejected_and_never_read(repo_and_outside):
    repo, outside = repo_and_outside
    make_symlink(repo / "src" / "link.py", outside / "leak.py")
    result = ingest_repository(repo)
    assert result.relative_paths == ["src/ok.py"]
    assert skipped(result)["src/link.py"] == SkipReason.UNSAFE_PATH
    assert OUTSIDE_MARKER not in result.model_dump_json()


def test_symlink_to_outside_directory_is_not_traversed(repo_and_outside):
    repo, outside = repo_and_outside
    make_symlink(repo / "escape", outside, is_dir=True)
    result = ingest_repository(repo)
    assert result.relative_paths == ["src/ok.py"]
    assert skipped(result)["escape"] == SkipReason.UNSAFE_PATH
    assert OUTSIDE_MARKER not in result.model_dump_json()


def test_symlinks_inside_the_root_are_also_skipped_by_policy(repo_and_outside):
    repo, _ = repo_and_outside
    make_symlink(repo / "alias.py", repo / "src" / "ok.py")
    make_symlink(repo / "srclink", repo / "src", is_dir=True)
    result = ingest_repository(repo)
    assert result.relative_paths == ["src/ok.py"]  # no duplicates via links
    assert skipped(result)["alias.py"] == SkipReason.SYMLINK
    assert skipped(result)["srclink"] == SkipReason.SYMLINK


def test_broken_and_self_referential_symlinks_do_not_crash(repo_and_outside):
    repo, _ = repo_and_outside
    make_symlink(repo / "broken.py", repo / "does-not-exist.py")
    make_symlink(repo / "loop.py", repo / "loop.py")
    result = ingest_repository(repo)
    assert result.relative_paths == ["src/ok.py"]
    assert skipped(result)["broken.py"] in (SkipReason.SYMLINK, SkipReason.UNSAFE_PATH)
    assert skipped(result)["loop.py"] in (SkipReason.SYMLINK, SkipReason.UNSAFE_PATH)


def test_user_chosen_root_may_itself_be_a_symlink(repo_and_outside, tmp_path):
    repo, _ = repo_and_outside
    link = tmp_path / "root-link"
    make_symlink(link, repo, is_dir=True)
    result = ingest_repository(link)
    assert result.relative_paths == ["src/ok.py"]
    assert result.repository_name == "repo"  # named after the resolved directory


def test_load_step_rechecks_containment(repo_and_outside):
    """Defence in depth: even if discovery were fooled, loading refuses a path outside the root."""
    repo, outside = repo_and_outside
    candidate = Candidate(outside / "leak.py", "leak.py", ".py", "python")
    outcome = loader._load_candidate(
        candidate, resolve_repository_root(repo), "r", IngestionPolicy()
    )
    assert outcome == SkipReason.UNSAFE_PATH


# ------------------------------------------------------------------------------ special files
@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs os.mkfifo (POSIX)")
def test_fifo_named_like_source_is_rejected_without_blocking(repo_and_outside):
    repo, _ = repo_and_outside
    os.mkfifo(repo / "pipe.py")
    assert stat.S_ISFIFO((repo / "pipe.py").stat().st_mode)
    result = ingest_repository(repo)
    assert result.relative_paths == ["src/ok.py"]
    assert skipped(result)["pipe.py"] == SkipReason.SPECIAL_FILE


@pytest.mark.skipif(sys.platform == "win32", reason="backslash is a separator on Windows")
def test_backslash_in_a_linux_file_name_is_rejected(repo_and_outside):
    repo, _ = repo_and_outside
    (repo / "evil\\name.py").write_text("x = 1\n")
    result = ingest_repository(repo)
    assert result.relative_paths == ["src/ok.py"]
    assert skipped(result)["evil\\name.py"] == SkipReason.UNSAFE_PATH


# ------------------------------------------------------------------ sensitive files never opened
def test_sensitive_and_pruned_files_are_never_opened(synthetic_repo, monkeypatch):
    opened: list[str] = []
    real_open = os.open

    def spy(path, *args, **kwargs):
        opened.append(os.fspath(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", spy)
    ingest_repository(synthetic_repo, IngestionPolicy(max_file_size_bytes=4096))
    forbidden = (".env", "server.pem", "private.key", "id_rsa", "secrets.yaml", "credentials.json")
    names = [os.path.basename(p) for p in opened]
    assert opened, "os.open was not used"
    assert not [n for n in names if n in forbidden or n.startswith(".env")]
    assert not [p for p in opened if "node_modules" in p or ".ssh" in p or "/.git/" in p]


def test_oversized_files_are_rejected_from_stat_without_being_opened(synthetic_repo, monkeypatch):
    opened: list[str] = []
    real_open = os.open

    def spy(path, *args, **kwargs):
        opened.append(os.path.basename(os.fspath(path)))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", spy)
    ingest_repository(synthetic_repo, IngestionPolicy(max_file_size_bytes=4096))
    assert "huge_module.py" not in opened


# ------------------------------------------------------------------------------ path traversal
@pytest.mark.parametrize(
    "value",
    [
        "../secret.py",
        "..\\secret.py",
        "a/../../secret.py",
        "a\\..\\..\\secret.py",
        "/etc/passwd",
        "\\\\server\\share\\x.py",
        "C:\\Windows\\system.ini",
        "C:relative.py",
        "c:/x.py",
        "",
        ".",
        "bad\x00.py",
    ],
)
def test_safe_join_rejects_traversal_and_absolute_paths(repo_and_outside, value):
    repo, _ = repo_and_outside
    with pytest.raises(UnsafePathError):
        safe_join(resolve_repository_root(repo), value)


def test_safe_join_accepts_either_separator_style(repo_and_outside):
    repo, _ = repo_and_outside
    root = resolve_repository_root(repo)
    expected = root / "src" / "ok.py"
    assert safe_join(root, "src/ok.py") == expected
    assert safe_join(root, "src\\ok.py") == expected
    assert safe_join(root, "./src//ok.py") == expected


def test_safe_join_rejects_symlink_escape(repo_and_outside):
    repo, outside = repo_and_outside
    make_symlink(repo / "escape", outside, is_dir=True)
    with pytest.raises(UnsafePathError):
        safe_join(resolve_repository_root(repo), "escape/leak.py")


def test_is_within_root_rejects_sibling_with_common_prefix(tmp_path):
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo-evil").mkdir()
    root = resolve_repository_root(tmp_path / "repo")
    assert is_within_root(tmp_path / "repo" / "a.py", root)
    assert not is_within_root(tmp_path / "repo-evil" / "a.py", root)


# ------------------------------------------------------------------------------ invalid roots
@pytest.mark.parametrize("bad", ["", "\x00", "definitely/not/a/real/dir"])
def test_invalid_roots_are_rejected(bad):
    with pytest.raises(InvalidRepositoryError):
        ingest_repository(bad)


def test_file_as_root_is_rejected(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x")
    with pytest.raises(InvalidRepositoryError):
        ingest_repository(f)


def test_filesystem_root_is_refused():
    with pytest.raises(InvalidRepositoryError):
        ingest_repository(os.path.abspath(os.sep))


# ------------------------------------------------------------------------------------ limits
def _write_many(repo, count, size=10):
    repo.mkdir(exist_ok=True)
    for i in range(count):
        (repo / f"f{i:02d}.py").write_text("x" * size)


def test_file_count_limit_truncates_and_says_so(tmp_path):
    _write_many(tmp_path / "r", 6)
    result = ingest_repository(tmp_path / "r", IngestionPolicy(max_files=3))
    assert result.relative_paths == ["f00.py", "f01.py", "f02.py"]
    assert result.stats.truncated is True
    assert result.stats.truncation_reason == "max_files=3"
    assert result.stats.skip_reasons == {"limit_exceeded": 1}
    assert result.stats.files_discovered == 4  # scan stopped at the first file over the limit


def test_exactly_at_the_file_limit_is_not_truncated(tmp_path):
    _write_many(tmp_path / "r", 3)
    result = ingest_repository(tmp_path / "r", IngestionPolicy(max_files=3))
    assert len(result.files) == 3 and result.stats.truncated is False


def test_total_size_limit_truncates(tmp_path):
    _write_many(tmp_path / "r", 5, size=100)
    result = ingest_repository(tmp_path / "r", IngestionPolicy(max_total_bytes=250))
    assert result.relative_paths == ["f00.py", "f01.py"]
    assert result.stats.total_bytes_accepted == 200
    assert result.stats.truncated and result.stats.truncation_reason == "max_total_bytes=250"


def test_single_file_size_boundary(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "exact.py").write_bytes(b"x" * 100)
    (repo / "over.py").write_bytes(b"x" * 101)
    result = ingest_repository(repo, IngestionPolicy(max_file_size_bytes=100))
    assert result.relative_paths == ["exact.py"]
    assert skipped(result) == {"over.py": SkipReason.OVERSIZED}


# --------------------------------------------------------------------- unreadable / vanishing
def _fail_open_for(monkeypatch, name, error):
    real_open = os.open

    def flaky(path, *args, **kwargs):
        if os.path.basename(os.fspath(path)) == name:
            raise error
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(reader.os, "open", flaky)


def test_permission_error_on_one_file_does_not_stop_the_scan(repo_and_outside, monkeypatch):
    repo, _ = repo_and_outside
    (repo / "locked.py").write_text("x = 1\n")
    _fail_open_for(monkeypatch, "locked.py", PermissionError(13, "denied"))
    result = ingest_repository(repo)
    assert result.relative_paths == ["src/ok.py"]
    assert skipped(result)["locked.py"] == SkipReason.UNREADABLE
    assert result.stats.skip_reasons == {"unreadable": 1}


def test_file_deleted_between_discovery_and_read_is_unreadable(repo_and_outside, monkeypatch):
    repo, _ = repo_and_outside
    (repo / "vanishing.py").write_text("x = 1\n")
    _fail_open_for(monkeypatch, "vanishing.py", FileNotFoundError(2, "gone"))
    assert skipped(ingest_repository(repo))["vanishing.py"] == SkipReason.UNREADABLE


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="permission bits are not enforced for root/Windows",
)
def test_unlistable_directory_is_counted_and_scan_continues(repo_and_outside):
    repo, _ = repo_and_outside
    locked = repo / "locked_dir"
    locked.mkdir()
    (locked / "hidden.py").write_text("x = 1\n")
    locked.chmod(0)
    try:
        result = ingest_repository(repo)
    finally:
        locked.chmod(0o700)
    assert result.relative_paths == ["src/ok.py"]
    assert result.stats.directories_unreadable == 1
