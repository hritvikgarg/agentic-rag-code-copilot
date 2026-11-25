"""Milestone 11A: plain-v2 is a separately versioned prompt and the older prompts are untouched."""

from __future__ import annotations

import hashlib

import pytest

from copilot.config import Settings
from copilot.llm import FakeLLMClient
from copilot.rag import (
    PLAIN_PROMPT_VERSION,
    PLAIN_PROMPT_VERSION_V2,
    PLAIN_SYSTEM_PROMPT,
    PLAIN_SYSTEM_PROMPT_V2,
    RAG_PROMPT_VERSION,
    RAG_SYSTEM_PROMPT,
    answer_plain,
)
from copilot.security import RepositorySecretRiskError
from tests.security import secret_helpers as h

SETTINGS = Settings(_env_file=None)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_historical_prompts_are_byte_for_byte_unchanged():
    # Digests of the texts committed at c18a436 (before Milestone 11A). If one of these fails, a
    # historical result would no longer correspond to the prompt that produced it: bump a version
    # instead of editing the text.
    assert PLAIN_PROMPT_VERSION == "plain-v1" and RAG_PROMPT_VERSION == "rag-v1"
    assert _sha(PLAIN_SYSTEM_PROMPT) == (
        "b506a7a90e9e61b8d51ecef68fda2fa406ae8106e76c4098fa9a9751672a494e"
    )
    assert _sha(RAG_SYSTEM_PROMPT) == (
        "ec7e1fa7558537ed46a976febcd6e77706afc161e8366bf311feae7d96a0f34a"
    )


def test_plain_v2_has_its_own_identifiable_version():
    assert PLAIN_PROMPT_VERSION_V2 == "plain-v2"
    assert len({PLAIN_PROMPT_VERSION, PLAIN_PROMPT_VERSION_V2, RAG_PROMPT_VERSION}) == 3
    assert PLAIN_SYSTEM_PROMPT_V2 not in (PLAIN_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT)


def test_default_callers_still_get_plain_v1():
    fake = FakeLLMClient()
    answer = answer_plain("What does the loader do?", fake, settings=SETTINGS)
    request = fake.requests[0]
    assert request.system_instruction == PLAIN_SYSTEM_PROMPT
    assert request.prompt_version == answer.prompt_version == "plain-v1"


def test_plain_v2_is_selected_explicitly_and_recorded():
    fake = FakeLLMClient()
    answer = answer_plain(
        "What does the loader do?",
        fake,
        settings=SETTINGS,
        system_prompt=PLAIN_SYSTEM_PROMPT_V2,
        prompt_version=PLAIN_PROMPT_VERSION_V2,
    )
    request = fake.requests[0]
    assert request.system_instruction == PLAIN_SYSTEM_PROMPT_V2
    assert request.prompt_version == answer.prompt_version == "plain-v2"


def test_the_plain_arm_never_receives_repository_evidence():
    fake = FakeLLMClient()
    answer_plain(
        "What does the loader do?",
        fake,
        settings=SETTINGS,
        system_prompt=PLAIN_SYSTEM_PROMPT_V2,
        prompt_version=PLAIN_PROMPT_VERSION_V2,
    )
    request = fake.requests[0]
    assert request.user_prompt == "What does the loader do?"
    assert "BEGIN SOURCE" not in request.outbound_text()


@pytest.mark.parametrize(
    "phrase",
    ["Repository evidence", "General knowledge, not from this repository", "Do not invent"],
)
def test_plain_v2_and_rag_v1_share_the_honesty_rules(phrase):
    assert phrase in PLAIN_SYSTEM_PROMPT_V2 and phrase in RAG_SYSTEM_PROMPT


def test_both_prompts_instruct_abstention_when_the_repository_cannot_be_known():
    assert "say so explicitly" in PLAIN_SYSTEM_PROMPT_V2
    assert "say so explicitly" in RAG_SYSTEM_PROMPT
    # plain-v1 has no abstention instruction: that is why plain-v2 exists.
    assert "say so" not in PLAIN_SYSTEM_PROMPT


def test_the_secret_gate_still_guards_the_plain_v2_request():
    secret = h.github_token("v2")
    fake = FakeLLMClient()
    with pytest.raises(RepositorySecretRiskError):
        answer_plain(
            f"Why does {secret} fail?",
            fake,
            settings=SETTINGS,
            system_prompt=PLAIN_SYSTEM_PROMPT_V2,
            prompt_version=PLAIN_PROMPT_VERSION_V2,
        )
    assert fake.call_count == 0
