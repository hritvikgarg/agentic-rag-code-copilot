"""Central extension -> language mapping (the single source of truth for supported file types).

To support another type, add it here or pass an extended mapping through
``IngestionPolicy(language_by_extension=...)``; nothing else needs to change. No parsers live here:
Milestone 2 only *identifies* languages.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePath
from types import MappingProxyType

DEFAULT_LANGUAGE_BY_EXTENSION: Mapping[str, str] = MappingProxyType(
    {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".md": "markdown",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
    }
)

# Languages that are configuration/data rather than code or prose. Used by the sensitive-name
# heuristics: a data file called ``secrets.yaml`` is suspicious; a source module
# ``secrets.py`` is not.
DATA_LANGUAGES = frozenset({"json", "yaml"})


def extension_of(file_name: str) -> str:
    """Lower-cased extension including the dot (``""`` if none). ``Foo.PY`` -> ``.py``."""
    return PurePath(file_name).suffix.lower()


def detect_language(
    file_name: str,
    mapping: Mapping[str, str] = DEFAULT_LANGUAGE_BY_EXTENSION,
) -> str | None:
    """Return the language for ``file_name`` or ``None`` if its extension is not supported."""
    return mapping.get(extension_of(file_name))
