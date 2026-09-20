"""Extension -> language mapping and its extensibility."""

import pytest
from pydantic import ValidationError

from copilot.ingestion import DEFAULT_LANGUAGE_BY_EXTENSION, IngestionPolicy, detect_language


@pytest.mark.parametrize(
    ("name", "language"),
    [
        ("a.py", "python"),
        ("a.js", "javascript"),
        ("a.jsx", "javascript"),
        ("a.ts", "typescript"),
        ("a.tsx", "typescript"),
        ("A.java", "java"),
        ("a.c", "c"),
        ("a.cpp", "cpp"),
        ("README.md", "markdown"),
        ("a.json", "json"),
        ("a.yaml", "yaml"),
        ("a.yml", "yaml"),
    ],
)
def test_supported_extensions(name, language):
    assert detect_language(name) == language


def test_matching_is_case_insensitive_and_uses_last_suffix():
    assert detect_language("SCRIPT.PY") == "python"
    assert detect_language("component.test.ts") == "typescript"
    assert detect_language("archive.py.txt") is None


@pytest.mark.parametrize("name", ["notes.txt", "Makefile", "a.h", "a.hpp", "a.go", ".py", "a."])
def test_unsupported_names(name):
    assert detect_language(name) is None


def test_default_mapping_is_exactly_the_v1_set():
    assert set(DEFAULT_LANGUAGE_BY_EXTENSION) == {
        ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cpp", ".md", ".json", ".yaml", ".yml"
    }  # fmt: skip


def test_central_mapping_is_immutable():
    with pytest.raises(TypeError):
        DEFAULT_LANGUAGE_BY_EXTENSION[".go"] = "go"  # type: ignore[index]


def test_policy_can_be_extended_without_touching_defaults():
    policy = IngestionPolicy().with_languages({".GO": "go", ".h": "c"})
    assert policy.language_for("main.go") == "go"
    assert policy.language_for("util.h") == "c"
    assert policy.language_for("a.py") == "python"  # existing entries preserved
    assert IngestionPolicy().language_for("main.go") is None  # original policy unchanged
    assert ".go" not in DEFAULT_LANGUAGE_BY_EXTENSION


@pytest.mark.parametrize("bad", ["py", ".", "a/b", ".a/b", ".a\\b"])
def test_policy_rejects_malformed_extensions(bad):
    with pytest.raises(ValidationError):
        IngestionPolicy(language_by_extension={bad: "x"})
