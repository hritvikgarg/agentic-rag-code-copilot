"""A tiny topical repository shared by the retrieval and evaluation tests."""

from __future__ import annotations

from pathlib import Path

FILES = {
    "src/pkg/ingest.py": (
        "def ingest_repository(root):\n"
        '    """Discover repository files and read their text."""\n'
        "    files = discover_files(root)\n"
        "    return [read_text(path) for path in files]\n"
    ),
    "src/pkg/chunk_ids.py": (
        "def make_chunk_id(path, start, end):\n"
        '    """Deterministic chunk id: a sha256 hash of the canonical payload."""\n'
        "    payload = f'{path}:{start}:{end}'\n"
        "    return sha256(payload).hexdigest()[:16]\n"
    ),
    "src/pkg/logging_setup.py": (
        "def setup_logging(level):\n"
        '    """Configure logging: handlers, formatter and the log level."""\n'
        "    handler = make_handler()\n"
        "    logging.getLogger().setLevel(level)\n"
    ),
    "docs/guide.md": "# Guide\n\nThis guide explains the tutorial and installation steps.\n",
}


def write_repo(root: Path, files=FILES) -> Path:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    return root
