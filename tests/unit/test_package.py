"""The package must import and expose a version string."""

import importlib

import pytest

import copilot

SUBPACKAGES = [
    "config", "models", "ingestion", "parsing", "chunking", "embeddings", "vectorstore",
    "retrieval", "llm", "rag", "agents", "evaluation", "services", "utils",
]  # fmt: skip


def test_package_imports_and_has_version():
    assert isinstance(copilot.__version__, str)
    assert copilot.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str):
    assert importlib.import_module(f"copilot.{name}").__doc__
