"""Ingestion policy: what to look at, what to prune, what counts as sensitive, and the limits.

All rules are data on a frozen ``IngestionPolicy`` so they can be extended without touching the
traversal code, e.g. ``policy.with_extra_ignored_directories("data")``.

Matching rules are deliberately conservative and *name based*. They are a first line of defence,
not a secret scanner: file contents are never inspected here (see docs/ingestion.md).
"""

from __future__ import annotations

from collections.abc import Mapping
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from copilot.ingestion.languages import (
    DATA_LANGUAGES,
    DEFAULT_LANGUAGE_BY_EXTENSION,
    extension_of,
)

if TYPE_CHECKING:  # avoid importing settings machinery at runtime
    from copilot.config.settings import Settings

# Directories that are never traversed (pruned during the walk, compared case-insensitively).
DEFAULT_IGNORED_DIRECTORY_NAMES = frozenset(
    {
        ".git", ".hg", ".svn",
        "node_modules", "bower_components",
        "dist", "build", "out",
        "venv", ".venv", "site-packages",
        "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".tox", ".nox", ".eggs",
        "coverage", "htmlcov", ".coverage_html",
        ".idea", ".vscode", ".vs",
        ".gradle", ".next", ".nuxt", ".cache", ".terraform",
    }
)  # fmt: skip
DEFAULT_IGNORED_DIRECTORY_PATTERNS = ("*.egg-info", "cmake-build-*")

# Directories that hold credentials by convention: pruned and never opened.
DEFAULT_SENSITIVE_DIRECTORY_NAMES = frozenset({".ssh", ".aws", ".gnupg", ".kube"})

# File names that are sensitive regardless of extension (matched case-insensitively).
DEFAULT_SENSITIVE_FILE_PATTERNS = (
    ".env", ".env.*", "*.env",
    "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.keystore", "*.ppk", "*.gpg", "*.pgp",
    "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*",
    ".npmrc", ".pypirc", ".netrc", "_netrc", ".htpasswd", ".git-credentials", ".pgpass",
    ".dockercfg",
    "*.tfstate", "*.tfstate.*", "*.tfvars", "*.tfvars.json",
    "kubeconfig", "*.kubeconfig",
)  # fmt: skip

# File *stems* (name before the first dot) that indicate secrets, applied only to data-like or
# unknown extensions. ``secrets.yaml`` is rejected; ``secrets.py`` (a source module) and
# ``tokenizer.json`` (stem "tokenizer", not "token") are kept.
DEFAULT_SENSITIVE_STEMS = frozenset(
    {"credentials", "credential", "secret", "secrets", "token", "tokens", "password", "passwords"}
)
DEFAULT_SENSITIVE_STEM_PATTERNS = ("client_secret*", "service-account*", "service_account*")

# Generated or vendored files that add noise, not evidence.
DEFAULT_GENERATED_FILE_PATTERNS = (
    "package-lock.json", "npm-shrinkwrap.json", "pnpm-lock.yaml", "yarn.lock",
    "*.min.js", "*.bundle.js", "*.chunk.js", "*_pb2.py", "*_pb2_grpc.py",
)  # fmt: skip


def _lower_all(values: object) -> object:
    if isinstance(values, (set, frozenset, list, tuple)):
        return frozenset(str(v).casefold() for v in values)
    return values


class IngestionPolicy(BaseModel):
    """Immutable set of rules and limits for one ingestion run."""

    model_config = ConfigDict(frozen=True)

    # --- limits ---
    max_file_size_bytes: int = Field(default=500_000, ge=1)
    max_files: int = Field(default=10_000, ge=1)  # accepted files
    max_total_bytes: int = Field(default=200 * 1024 * 1024, ge=1)  # accepted bytes

    # --- rules ---
    language_by_extension: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_LANGUAGE_BY_EXTENSION)
    )
    ignored_directory_names: frozenset[str] = DEFAULT_IGNORED_DIRECTORY_NAMES
    ignored_directory_patterns: tuple[str, ...] = DEFAULT_IGNORED_DIRECTORY_PATTERNS
    sensitive_directory_names: frozenset[str] = DEFAULT_SENSITIVE_DIRECTORY_NAMES
    sensitive_file_patterns: tuple[str, ...] = DEFAULT_SENSITIVE_FILE_PATTERNS
    sensitive_stems: frozenset[str] = DEFAULT_SENSITIVE_STEMS
    sensitive_stem_patterns: tuple[str, ...] = DEFAULT_SENSITIVE_STEM_PATTERNS
    generated_file_patterns: tuple[str, ...] = DEFAULT_GENERATED_FILE_PATTERNS

    @field_validator("language_by_extension")
    @classmethod
    def _normalise_extensions(cls, value: dict[str, str]) -> dict[str, str]:
        normalised: dict[str, str] = {}
        for ext, language in value.items():
            ext = ext.lower()
            if not ext.startswith(".") or len(ext) < 2 or "/" in ext or "\\" in ext:
                raise ValueError(f"invalid extension {ext!r}; expected e.g. '.py'")
            if not language:
                raise ValueError(f"empty language for {ext!r}")
            normalised[ext] = language
        return normalised

    @field_validator("ignored_directory_names", "sensitive_directory_names", "sensitive_stems",
                     mode="before")  # fmt: skip
    @classmethod
    def _casefold_sets(cls, value: object) -> object:
        return _lower_all(value)

    # ------------------------------------------------------------------ construction helpers
    @classmethod
    def from_settings(cls, settings: Settings, **overrides: object) -> Self:
        """Build a policy from application ``Settings`` (limits only; rules keep their defaults)."""
        values: dict[str, object] = {
            "max_file_size_bytes": settings.max_file_size_bytes,
            "max_files": settings.max_repo_files,
            "max_total_bytes": settings.max_repo_total_mb * 1024 * 1024,
        }
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]

    def with_extra_ignored_directories(self, *names: str) -> Self:
        """Return a copy that also prunes the given directory names."""
        extra = frozenset(n.casefold() for n in names)
        return self.model_copy(
            update={"ignored_directory_names": self.ignored_directory_names | extra}
        )

    def with_languages(self, mapping: Mapping[str, str]) -> Self:
        """Return a copy that also supports the given ``{".ext": "language"}`` entries."""
        merged = {**self.language_by_extension, **mapping}
        return type(self)(**{**self.model_dump(), "language_by_extension": merged})

    # ------------------------------------------------------------------ rules
    def language_for(self, file_name: str) -> str | None:
        """Language for a file name, or ``None`` if the extension is unsupported."""
        return self.language_by_extension.get(extension_of(file_name))

    def is_ignored_directory(self, name: str) -> bool:
        folded = name.casefold()
        return folded in self.ignored_directory_names or any(
            fnmatchcase(folded, pattern) for pattern in self.ignored_directory_patterns
        )

    def is_sensitive_directory(self, name: str) -> bool:
        return name.casefold() in self.sensitive_directory_names

    def is_generated_file(self, name: str) -> bool:
        folded = name.casefold()
        return any(fnmatchcase(folded, pattern) for pattern in self.generated_file_patterns)

    def is_sensitive_file(self, name: str) -> bool:
        """Name-based sensitivity check. Never looks at file contents."""
        folded = name.casefold()
        if any(fnmatchcase(folded, pattern) for pattern in self.sensitive_file_patterns):
            return True
        # Stem rules apply only to data-like or unknown extensions, so normal source files and
        # documentation named ``secrets.py`` / ``token.md`` are not rejected on their name alone.
        language = self.language_for(folded)
        if language is not None and language not in DATA_LANGUAGES:
            return False
        stem = folded.partition(".")[0]
        return stem in self.sensitive_stems or any(
            fnmatchcase(stem, pattern) for pattern in self.sensitive_stem_patterns
        )
