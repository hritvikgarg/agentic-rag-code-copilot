"""Path helpers, including Windows path *semantics* exercised via PureWindowsPath.

These tests run on Linux. ``PureWindowsPath`` lets us check Windows rules (drive letters,
case-insensitive comparison, backslash separators) without pretending the host is Windows.
"""

from pathlib import PurePosixPath, PureWindowsPath

import pytest
from pydantic import ValidationError

from copilot.models import SourceFile
from copilot.utils.paths import UnsafePathError, relative_posix, validate_relative_posix


def test_relative_posix_posix_paths():
    assert relative_posix(PurePosixPath("/repo/src/app/main.py"), PurePosixPath("/repo")) == (
        "src/app/main.py"
    )


def test_relative_posix_windows_paths_use_forward_slashes():
    path = PureWindowsPath(r"C:\repo\src\app\main.py")
    assert relative_posix(path, PureWindowsPath(r"C:\repo")) == "src/app/main.py"


def test_relative_posix_windows_is_case_insensitive():
    path = PureWindowsPath(r"C:\Repo\SRC\Main.py")
    assert relative_posix(path, PureWindowsPath(r"c:\repo")) == "SRC/Main.py"


def test_relative_posix_root_itself_is_dot():
    assert relative_posix(PurePosixPath("/repo"), PurePosixPath("/repo")) == "."


@pytest.mark.parametrize(
    ("path", "root"),
    [
        (PureWindowsPath(r"D:\repo\a.py"), PureWindowsPath(r"C:\repo")),  # different drive
        (PureWindowsPath(r"C:\other\a.py"), PureWindowsPath(r"C:\repo")),
        (PurePosixPath("/etc/passwd"), PurePosixPath("/repo")),
        (PurePosixPath("/repository/a.py"), PurePosixPath("/repo")),  # prefix, not parent
    ],
)
def test_relative_posix_rejects_paths_outside_root(path, root):
    with pytest.raises(UnsafePathError):
        relative_posix(path, root)


@pytest.mark.parametrize(
    "value",
    ["a.py", "src/app/main.py", ".hidden/x.md", "dir with spaces/f.py", "ünï/cödé.py", "a..b/c.py"],
)
def test_validate_relative_posix_accepts_canonical_paths(value):
    assert validate_relative_posix(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        "/abs/path.py",
        "C:/x.py",
        "c:x.py",
        r"src\app.py",
        "../x.py",
        "a/../b.py",
        "a/./b.py",
        "a//b.py",
        "a/",
        "bad\x00name.py",
    ],
)
def test_validate_relative_posix_rejects_unsafe_values(value):
    with pytest.raises(UnsafePathError):
        validate_relative_posix(value)


def _source_file(**overrides):
    values = {
        "repository_name": "repo",
        "relative_path": "src/a.py",
        "extension": ".py",
        "language": "python",
        "size_bytes": 3,
        "content": "SUPER-SECRET-BODY",
        "sha256": "0" * 64,
    }
    values.update(overrides)
    return SourceFile(**values)


def test_source_file_rejects_non_canonical_paths():
    with pytest.raises(ValidationError):
        _source_file(relative_path=r"src\a.py")
    with pytest.raises(ValidationError):
        _source_file(relative_path="../a.py")


def test_source_file_repr_never_contains_content():
    assert "SUPER-SECRET-BODY" not in repr(_source_file())
    assert "SUPER-SECRET-BODY" not in str(_source_file())


def test_source_file_is_immutable():
    with pytest.raises(ValidationError):
        _source_file().content = "changed"
