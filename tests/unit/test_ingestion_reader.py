"""Bounded reading and text decoding."""

import codecs
import os

import pytest

from copilot.ingestion.reader import decode_source, normalize_newlines, read_bytes_safely
from copilot.models import SkipReason


def test_utf8_decodes():
    assert decode_source("héllo\n".encode()) == "héllo\n"


def test_utf8_bom_is_stripped():
    assert decode_source(codecs.BOM_UTF8 + b"x = 1\n") == "x = 1\n"


@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-16-be"])
def test_utf16_with_bom_decodes(encoding):
    text = "größe = 1\n"
    data = text.encode(encoding)
    if encoding != "utf-16":  # the explicit-endian codecs do not write a BOM; add one
        data = (codecs.BOM_UTF16_LE if encoding.endswith("le") else codecs.BOM_UTF16_BE) + data
    assert decode_source(data) == text


def test_utf16_without_bom_is_rejected_as_binary():
    assert decode_source("abc".encode("utf-16-le")) == SkipReason.BINARY


def test_utf32_is_not_mistaken_for_text():
    assert decode_source("abc".encode("utf-32")) == SkipReason.BINARY


def test_nul_byte_means_binary():
    assert decode_source(b"MZ\x00\x01\x02") == SkipReason.BINARY


def test_nul_byte_after_the_sniff_window_is_still_binary():
    assert decode_source(b"a" * 9000 + b"\x00") == SkipReason.BINARY


def test_invalid_utf8_is_an_encoding_error_not_silently_accepted():
    assert decode_source(b"name = '\xe9'\n") == SkipReason.ENCODING_ERROR


def test_truncated_utf16_is_an_encoding_error():
    data = codecs.BOM_UTF16_LE + "é".encode("utf-16-le") + b"\x3d"  # odd trailing byte
    assert decode_source(data) in (SkipReason.ENCODING_ERROR, SkipReason.BINARY)


def test_empty_file_is_valid_text():
    assert decode_source(b"") == ""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a\r\nb\r\n", "a\nb\n"),
        ("a\rb\r", "a\nb\n"),
        ("a\nb\n", "a\nb\n"),
        ("a\r\n\r\nb", "a\n\nb"),
    ],
)
def test_newlines_are_normalised_without_changing_line_count(raw, expected):
    assert normalize_newlines(raw) == expected
    assert raw.count("\n") + raw.count("\r") - raw.count("\r\n") == expected.count("\n")


def test_read_regular_file(tmp_path):
    f = tmp_path / "a.py"
    f.write_bytes(b"abc")
    assert read_bytes_safely(f, 10) == b"abc"


def test_read_size_boundary(tmp_path):
    f = tmp_path / "a.py"
    f.write_bytes(b"x" * 10)
    assert read_bytes_safely(f, 10) == b"x" * 10  # exactly at the limit is fine
    assert read_bytes_safely(f, 9) == SkipReason.OVERSIZED


def test_missing_file_is_unreadable_not_a_crash(tmp_path):
    assert read_bytes_safely(tmp_path / "gone.py", 10) == SkipReason.UNREADABLE


def test_directory_is_not_read_as_a_file(tmp_path):
    assert read_bytes_safely(tmp_path, 10) in (SkipReason.SPECIAL_FILE, SkipReason.UNREADABLE)


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="needs O_NOFOLLOW (POSIX)")
def test_symlink_is_refused_at_open_time(tmp_path):
    target = tmp_path / "real.py"
    target.write_bytes(b"x = 1\n")
    link = tmp_path / "link.py"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported here")
    assert read_bytes_safely(link, 100) == SkipReason.UNSAFE_PATH
