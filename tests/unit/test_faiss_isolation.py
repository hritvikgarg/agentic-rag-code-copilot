"""Architecture guard: only ``vectorstore/faiss_backend.py`` may import FAISS."""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "copilot"
IMPORT = re.compile(r"^\s*(import|from)\s+faiss\b", re.MULTILINE)


def test_only_the_backend_module_imports_faiss():
    offenders = [
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if IMPORT.search(path.read_text()) and path.name != "faiss_backend.py"
    ]
    assert offenders == []


def test_the_backend_module_does_import_faiss():
    assert IMPORT.search((SRC / "vectorstore" / "faiss_backend.py").read_text())
