"""Agentic RAG-based software engineering copilot.

A repository-aware assistant that answers questions about a codebase using retrieved
repository evidence. See README.md and docs/ARCHITECTURE_PLAN.md for the roadmap.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agentic-rag-code-copilot")
except PackageNotFoundError:  # running from a source tree that has not been installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
