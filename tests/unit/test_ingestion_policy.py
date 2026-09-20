"""Name-based rules: ignored directories, sensitive files, generated files, limits."""

import pytest
from pydantic import ValidationError

from copilot.config import Settings
from copilot.ingestion import IngestionPolicy

POLICY = IngestionPolicy()


@pytest.mark.parametrize(
    "name",
    [
        ".git", "node_modules", "dist", "build", "venv", ".venv", "__pycache__", ".pytest_cache",
        ".ruff_cache", ".mypy_cache", "coverage", "htmlcov", ".idea", ".vscode", ".tox",
        "Node_Modules", "BUILD", "mypkg.egg-info", "cmake-build-debug",
    ],
)  # fmt: skip
def test_ignored_directories(name):
    assert POLICY.is_ignored_directory(name)


@pytest.mark.parametrize("name", ["src", "app", "tests", "docs", "builder", "distribution", "env"])
def test_ordinary_directories_are_not_ignored(name):
    assert not POLICY.is_ignored_directory(name)


@pytest.mark.parametrize("name", [".ssh", ".aws", ".gnupg", ".kube", ".SSH"])
def test_sensitive_directories(name):
    assert POLICY.is_sensitive_directory(name)


@pytest.mark.parametrize(
    "name",
    [
        ".env", ".env.local", ".env.production", ".env.example", "prod.env", ".ENV",
        "server.pem", "cert.PEM", "private.key", "keystore.jks", "id_rsa", "id_rsa.pub",
        "id_ed25519", ".npmrc", ".pypirc", ".netrc", ".htpasswd", ".git-credentials",
        "terraform.tfstate", "prod.tfvars", "kubeconfig",
        "credentials", "credentials.json", "Credentials.JSON", "secrets.yaml", "secrets.yml",
        "secrets.prod.yaml", "secret.json", "token.json", "tokens.yaml", "passwords.txt",
        "client_secret_123.apps.example.json", "service-account-prod.json",
        "service_account.json",
    ],
)  # fmt: skip
def test_sensitive_file_names(name):
    assert POLICY.is_sensitive_file(name)


@pytest.mark.parametrize(
    "name",
    [
        # normal source / docs must not be rejected just because of a suggestive name
        "secrets.py", "secret.py", "token.py", "tokens.ts", "credentials.py", "passwords.md",
        "credentials.md", "auth_token.py", "token_utils.js",
        # names that only *contain* or *start with* a sensitive word
        "tokenizer.json", "tokenizer_config.json", "secretary.yaml", "environment.yaml",
        "envelope.py", "keys.py", "keyboard.js", "key_bindings.json", "package.json",
        "settings.json", "config.yaml", "README.md", "main.py",
    ],
)  # fmt: skip
def test_normal_files_are_not_flagged_by_name(name):
    assert not POLICY.is_sensitive_file(name)


@pytest.mark.parametrize(
    "name",
    ["package-lock.json", "pnpm-lock.yaml", "app.min.js", "vendor.bundle.js", "x_pb2.py"],
)
def test_generated_files(name):
    assert POLICY.is_generated_file(name)


@pytest.mark.parametrize("name", ["package.json", "app.js", "minimal.js", "lock.py", "pb2.py"])
def test_hand_written_files_are_not_generated(name):
    assert not POLICY.is_generated_file(name)


def test_extra_ignored_directories_are_added_case_insensitively():
    policy = POLICY.with_extra_ignored_directories("Data", "tmp_repos")
    assert policy.is_ignored_directory("data")
    assert policy.is_ignored_directory("TMP_REPOS")
    assert policy.is_ignored_directory("node_modules")  # defaults kept
    assert not POLICY.is_ignored_directory("data")  # original untouched


def test_policy_is_frozen():
    with pytest.raises(ValidationError):
        POLICY.max_files = 1


@pytest.mark.parametrize("field", ["max_file_size_bytes", "max_files", "max_total_bytes"])
def test_limits_must_be_positive(field):
    with pytest.raises(ValidationError):
        IngestionPolicy(**{field: 0})


def test_from_settings_maps_limits():
    settings = Settings(
        _env_file=None, max_file_size_bytes=2000, max_repo_files=7, max_repo_total_mb=3
    )
    policy = IngestionPolicy.from_settings(settings)
    assert policy.max_file_size_bytes == 2000
    assert policy.max_files == 7
    assert policy.max_total_bytes == 3 * 1024 * 1024


def test_from_settings_allows_overrides():
    policy = IngestionPolicy.from_settings(Settings(_env_file=None), max_files=2)
    assert policy.max_files == 2
