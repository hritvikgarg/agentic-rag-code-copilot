"""The repository fingerprint: deterministic, content-addressed, order- and location-independent."""

import hashlib
import json
import random

import pytest

from copilot.models.chunk import sha256_text
from copilot.vectorstore import IndexBuildError, repository_fingerprint
from tests.chunking_helpers import make_file
from tests.vectorstore_helpers import make_files

# Locked value: the fingerprint of make_files() (see tests/vectorstore_helpers.py). If this changes,
# the fingerprint formula changed and every stored index is (correctly) stale: bump
# FINGERPRINT_SCHEMA in the same commit and update this value deliberately.
GOLDEN = "b24e3c152160dd2602287eb5d889e453ff1412c6efdb25a7f157435f59809594"


def test_golden_value_is_locked():
    assert repository_fingerprint(make_files()) == GOLDEN


def test_matches_the_documented_formula_written_out_by_hand():
    file = make_file("x = 1\n", "pkg/mod.py")
    expected_entry = [
        "pkg/mod.py",
        hashlib.sha256(b"x = 1\n").hexdigest(),
        sha256_text("x = 1\n"),
        "python",
    ]
    text = json.dumps(
        ["repo-fingerprint/1", [expected_entry]], ensure_ascii=True, separators=(",", ":")
    )
    assert repository_fingerprint([file]) == hashlib.sha256(text.encode()).hexdigest()


def test_same_input_same_fingerprint():
    assert repository_fingerprint(make_files()) == repository_fingerprint(make_files())


def test_input_order_does_not_matter():
    files = make_files()
    shuffled = list(files)
    random.Random(7).shuffle(shuffled)
    assert repository_fingerprint(shuffled) == repository_fingerprint(files)
    assert repository_fingerprint(list(reversed(files))) == repository_fingerprint(files)


def test_is_64_hex_characters():
    fp = repository_fingerprint(make_files())
    assert len(fp) == 64 and set(fp) <= set("0123456789abcdef")


def test_content_change_changes_it():
    changed = {**{p: f.content for p, f in ((f.relative_path, f) for f in make_files())}}
    changed["src/a.py"] += "# edit\n"
    assert repository_fingerprint(make_files(changed)) != GOLDEN


def test_whitespace_only_change_changes_it():
    contents = {f.relative_path: f.content for f in make_files()}
    contents["README.md"] = contents["README.md"] + "\n"
    assert repository_fingerprint(make_files(contents)) != GOLDEN


def test_adding_or_removing_a_file_changes_it():
    contents = {f.relative_path: f.content for f in make_files()}
    added = {**contents, "src/new.py": "pass\n"}
    removed = {k: v for k, v in contents.items() if k != "README.md"}
    prints = {
        GOLDEN,
        repository_fingerprint(make_files(added)),
        repository_fingerprint(make_files(removed)),
    }
    assert len(prints) == 3


def test_renaming_a_file_changes_it():
    contents = {f.relative_path: f.content for f in make_files()}
    renamed = {("src/renamed.py" if k == "src/a.py" else k): v for k, v in contents.items()}
    assert repository_fingerprint(make_files(renamed)) != GOLDEN


def test_language_change_changes_it():
    a = make_file("x = 1\n", "f.py", language="python")
    b = make_file("x = 1\n", "f.py", language="text")
    assert repository_fingerprint([a]) != repository_fingerprint([b])


def test_repository_name_is_not_part_of_it():
    files = make_files()
    renamed = [f.model_copy(update={"repository_name": "another-name"}) for f in files]
    assert repository_fingerprint(renamed) == repository_fingerprint(files)


def test_normalised_text_change_with_same_bytes_is_detected():
    same_bytes = make_file("x = 1\n", "f.py")
    other_text = same_bytes.model_copy(update={"content": "x = 1\r\n"})  # decoder behaviour changed
    assert repository_fingerprint([same_bytes]) != repository_fingerprint([other_text])


def test_duplicate_paths_are_rejected():
    with pytest.raises(IndexBuildError, match="duplicate"):
        repository_fingerprint([make_file("a\n", "same.py"), make_file("b\n", "same.py")])


def test_empty_repository_has_a_defined_fingerprint():
    assert len(repository_fingerprint([])) == 64
