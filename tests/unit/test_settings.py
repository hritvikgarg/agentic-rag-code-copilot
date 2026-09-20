"""Configuration defaults, environment overrides, validation and secret handling."""

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from copilot.config import Settings, get_settings

ROOT = Path(__file__).resolve().parents[2]


def make(**kwargs) -> Settings:
    """Build Settings ignoring any real .env file."""
    return Settings(_env_file=None, **kwargs)


def test_defaults_are_sensible():
    s = make()
    assert s.llm_provider == "gemini"
    assert s.chunking_strategy == "line"
    assert s.chunk_size_lines > s.chunk_overlap_lines >= 0
    assert 1 <= s.retrieval_top_k <= 50
    assert s.log_level == "INFO"
    assert s.gemini_api_key is None


def test_llm_model_is_unset_by_default_and_validated():
    """No unverified model ID is shipped as a default; it is chosen in Milestone 6."""
    assert make().llm_model is None
    assert make(llm_model="some-model-id").llm_model == "some-model-id"
    with pytest.raises(ValidationError):
        make(llm_model="")


def test_environment_overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COPILOT_RETRIEVAL_TOP_K", "9")
    monkeypatch.setenv("COPILOT_CHUNKING_STRATEGY", "ast")
    monkeypatch.setenv("COPILOT_LOG_LEVEL", "DEBUG")
    s = make()
    assert (s.retrieval_top_k, s.chunking_strategy, s.log_level) == (9, "ast", "DEBUG")


def test_derived_paths_live_under_data_dir():
    s = make(data_dir=Path("somewhere"))
    for p in (s.indexes_dir, s.repos_dir, s.uploads_dir, s.cache_dir):
        assert p.parent == Path("somewhere")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"retrieval_top_k": 0},
        {"chunk_size_lines": 10, "chunk_overlap_lines": 10},
        {"chunk_size_lines": 3},
        {"llm_provider": "made-up"},
        {"log_level": "LOUD"},
    ],
)
def test_invalid_values_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        make(**kwargs)


def test_api_key_comes_from_environment_and_is_never_printed(monkeypatch: pytest.MonkeyPatch):
    fake_key = "unit-test-not-a-real-key-123456"
    monkeypatch.setenv("GEMINI_API_KEY", fake_key)
    s = make()
    assert s.gemini_api_key is not None
    assert s.gemini_api_key.get_secret_value() == fake_key
    assert fake_key not in repr(s)
    assert fake_key not in str(s)
    assert fake_key not in s.model_dump_json()
    assert s.secret_values() == [fake_key]


def test_env_file_is_read(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("COPILOT_RETRIEVAL_TOP_K=7\nUNRELATED_VARIABLE=ignored\n")
    assert Settings(_env_file=env_file).retrieval_top_k == 7


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


def test_env_example_is_loadable_and_contains_no_real_key():
    example = ROOT / ".env.example"
    s = Settings(_env_file=example)  # every documented variable must be valid
    assert s.gemini_api_key is not None
    assert s.gemini_api_key.get_secret_value() == "your-gemini-api-key-here"
    text = example.read_text(encoding="utf-8")
    assert not re.search(r"AIza[0-9A-Za-z_\-]{30,}", text)  # Google-style key shape
    assert not re.search(r"(sk|ghp|gho|github_pat)[-_][0-9A-Za-z_\-]{20,}", text)


def test_chunk_max_tokens_default_and_bounds():
    assert make().chunk_max_tokens == 512
    assert make(chunk_max_tokens=16).chunk_max_tokens == 16
    for bad in (15, 8193):
        with pytest.raises(ValidationError):
            make(chunk_max_tokens=bad)
