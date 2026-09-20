"""Guard the repository's security-critical ignore rules with `git check-ignore`."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(),
    reason="needs git and a git checkout",
)


def is_ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", path], cwd=ROOT, capture_output=True, check=False
    )
    return result.returncode == 0


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        ".venv/lib/x.py",
        "src/copilot/__pycache__/x.pyc",
        ".pytest_cache/v",
        ".ruff_cache/v",
        "data/indexes/repo/index.faiss",
        "data/indexes/0123456789abcdef/index.faiss",
        "data/indexes/0123456789abcdef/manifest.json",
        "data/indexes/0123456789abcdef/chunks.jsonl",
        "data/indexes/.staging-0123456789abcdef-x1/chunks.jsonl",
        "data/indexes/.replaced-0123456789abcdef-x1/manifest.json",
        "data/cache/models/models--jinaai--x/snapshots/1/onnx/model.onnx",
        "some/where/else/index.faiss",
        "data/repos/some-repo/app.py",
        "data/uploads/upload.zip",
        "results/runs/2026/out.json",
        "server.pem",
        "id_rsa",
    ],
)
def test_sensitive_or_generated_paths_are_ignored(path: str):
    assert is_ignored(path), f"{path} must be git-ignored"


@pytest.mark.parametrize(
    "path",
    [
        ".env.example",
        "data/.gitkeep",
        "uv.lock",
        ".python-version",
        "src/copilot/models/__init__.py",  # regression guard: `models/` must not be ignored
        "src/copilot/config/settings.py",
        "tests/unit/test_gitignore.py",
    ],
)
def test_project_files_are_not_ignored(path: str):
    assert not is_ignored(path), f"{path} must NOT be git-ignored"
