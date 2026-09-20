"""Crash-safe writing: a failed or interrupted save must never look like a valid index."""

import os

import pytest

from copilot.vectorstore import (
    IndexCorruptError,
    IndexExistsError,
    IndexExpectation,
    IndexStorageError,
    VectorIndex,
    atomic,
)
from copilot.vectorstore.atomic import STAGING_PREFIX, write_directory_atomically
from copilot.vectorstore.manifest import CHUNKS_FILE, INDEX_FILE, MANIFEST_FILE
from tests.vectorstore_helpers import build_test_index


def leftovers(parent):
    return sorted(p.name for p in parent.iterdir() if p.name.startswith("."))


def test_successful_write_creates_exactly_the_files(tmp_path):
    target = tmp_path / "idx"
    write_directory_atomically(target, {"a.bin": b"1", "b.bin": b"22"})
    assert sorted(p.name for p in target.iterdir()) == ["a.bin", "b.bin"]
    assert (target / "b.bin").read_bytes() == b"22"
    assert leftovers(tmp_path) == []


def test_bytes_are_written_verbatim_without_newline_translation(tmp_path):
    payload = b"line1\nline2\r\nline3\n\x00\xff"
    write_directory_atomically(tmp_path / "idx", {"f": payload})
    assert (tmp_path / "idx" / "f").read_bytes() == payload


def test_existing_target_is_protected_unless_overwriting(tmp_path):
    target = tmp_path / "idx"
    write_directory_atomically(target, {"f": b"old"})
    with pytest.raises(IndexExistsError):
        write_directory_atomically(target, {"f": b"new"})
    assert (target / "f").read_bytes() == b"old"
    write_directory_atomically(target, {"f": b"new"}, overwrite=True)
    assert (target / "f").read_bytes() == b"new"
    assert leftovers(tmp_path) == []


@pytest.mark.parametrize("name", ["../evil", "sub/f", "", ".", ".."])
def test_unsafe_file_names_are_rejected(tmp_path, name):
    with pytest.raises(IndexStorageError, match="invalid artifact file name"):
        write_directory_atomically(tmp_path / "idx", {name: b"x"})
    assert not (tmp_path / "idx").exists()


def test_failure_while_writing_leaves_no_index_and_cleans_staging(tmp_path, monkeypatch):
    index, _, _ = build_test_index()
    real = atomic._write_file
    calls = []

    def fail_on_second(path, data):
        calls.append(path.name)
        if len(calls) == 2:
            raise OSError("simulated disk failure")
        real(path, data)

    monkeypatch.setattr(atomic, "_write_file", fail_on_second)
    with pytest.raises(OSError, match="simulated disk failure"):
        index.save(tmp_path / "indexes")

    root = tmp_path / "indexes"
    assert calls == [INDEX_FILE, CHUNKS_FILE]  # died before the manifest was reached
    assert not (root / index.index_id).exists()  # no directory posing as an index
    assert leftovers(root) == []  # staging directory removed


def test_a_crash_that_cannot_clean_up_still_leaves_no_valid_index(tmp_path, monkeypatch):
    """Simulate a hard kill: the write fails AND cleanup never runs."""
    index, _, _ = build_test_index()
    real = atomic._write_file

    def fail_on_manifest(path, data):
        if path.name == MANIFEST_FILE:
            raise OSError("killed while writing the manifest")
        real(path, data)

    monkeypatch.setattr(atomic, "_write_file", fail_on_manifest)
    monkeypatch.setattr(atomic.shutil, "rmtree", lambda *a, **k: None)  # no cleanup happens
    with pytest.raises(OSError):
        index.save(tmp_path / "indexes")

    root = tmp_path / "indexes"
    assert not (root / index.index_id).exists()  # the final path never existed
    staged = [p for p in root.iterdir() if p.name.startswith(STAGING_PREFIX)]
    assert len(staged) == 1
    assert sorted(p.name for p in staged[0].iterdir()) == [CHUNKS_FILE, INDEX_FILE]  # no manifest
    for candidate in (root / index.index_id, staged[0]):
        with pytest.raises(IndexCorruptError):  # neither can be loaded as an index
            VectorIndex.load(candidate, expected=IndexExpectation())


def test_failure_of_the_final_rename_leaves_no_index(tmp_path, monkeypatch):
    index, _, _ = build_test_index()

    def broken_replace(src, dst):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(atomic.os, "replace", broken_replace)
    with pytest.raises(IndexStorageError, match="could not move"):
        index.save(tmp_path / "indexes")
    assert not (tmp_path / "indexes" / index.index_id).exists()
    assert leftovers(tmp_path / "indexes") == []


def test_failed_replacement_restores_the_previous_index(tmp_path, monkeypatch):
    index, _, _ = build_test_index()
    path = index.save(tmp_path / "indexes")
    before = {p.name: p.read_bytes() for p in path.iterdir()}

    real = os.replace
    calls = {"n": 0}

    def fail_second(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:  # 1st: old -> backup (ok), 2nd: staging -> target (fails)
            raise OSError("simulated rename failure")
        real(src, dst)

    monkeypatch.setattr(atomic.os, "replace", fail_second)
    with pytest.raises(IndexStorageError):
        index.save(tmp_path / "indexes", overwrite=True)
    monkeypatch.undo()

    assert {p.name: p.read_bytes() for p in path.iterdir()} == before  # old index intact
    assert VectorIndex.load(path, expected=IndexExpectation()).index_id == index.index_id
    assert leftovers(tmp_path / "indexes") == []


def test_keyboard_interrupt_also_cleans_up(tmp_path, monkeypatch):
    def interrupted(path, data):
        raise KeyboardInterrupt

    monkeypatch.setattr(atomic, "_write_file", interrupted)
    with pytest.raises(KeyboardInterrupt):
        write_directory_atomically(tmp_path / "idx", {"f": b"x"})
    assert leftovers(tmp_path) == [] and not (tmp_path / "idx").exists()
