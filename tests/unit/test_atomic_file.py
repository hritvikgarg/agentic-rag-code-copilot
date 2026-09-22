"""Single-file atomic writes: never a partial file, old content survives a failed write."""

import os

import pytest

from copilot.utils.atomic_file import write_text_atomically


def test_writes_the_file_and_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "out.json"
    write_text_atomically(path, "hello\n")
    assert path.read_text(encoding="utf-8") == "hello\n"


def test_overwrites_existing_content(tmp_path):
    path = tmp_path / "out.json"
    write_text_atomically(path, "first\n")
    write_text_atomically(path, "second\n")
    assert path.read_text(encoding="utf-8") == "second\n"


def test_no_leftover_temp_files_after_a_successful_write(tmp_path):
    path = tmp_path / "out.json"
    write_text_atomically(path, "content\n")
    leftovers = [p for p in tmp_path.iterdir() if p != path]
    assert leftovers == []


def test_a_failure_during_the_rename_leaves_the_previous_file_intact_and_cleans_up(
    tmp_path, monkeypatch
):
    path = tmp_path / "out.json"
    write_text_atomically(path, "original\n")

    def boom(*args, **kwargs):
        raise OSError("simulated failure during replace")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="simulated failure"):
        write_text_atomically(path, "new content that must not appear\n")

    assert path.read_text(encoding="utf-8") == "original\n"
    leftovers = [p for p in tmp_path.iterdir() if p != path]
    assert leftovers == []  # the temp file was removed, not left behind


def test_a_failure_while_writing_the_temp_file_leaves_the_previous_file_intact(
    tmp_path, monkeypatch
):
    path = tmp_path / "out.json"
    write_text_atomically(path, "original\n")

    real_fdopen = os.fdopen

    def boom_fdopen(fd, *args, **kwargs):
        handle = real_fdopen(fd, *args, **kwargs)
        handle.write = lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full"))
        return handle

    monkeypatch.setattr(os, "fdopen", boom_fdopen)
    with pytest.raises(OSError, match="disk full"):
        write_text_atomically(path, "new content\n")

    assert path.read_text(encoding="utf-8") == "original\n"
    leftovers = [p for p in tmp_path.iterdir() if p != path]
    assert leftovers == []
