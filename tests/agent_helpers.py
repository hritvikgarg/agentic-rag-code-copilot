"""Shared builders for the Milestone 8 tests (no model, no network, no embedding model)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from copilot.agents import AgentConfig, AgentWorkflow
from copilot.config import Settings
from copilot.llm import FakeLLMClient
from copilot.retrieval import RetrievalResult
from tests.rag_helpers import make_result

SETTINGS = Settings(_env_file=None, rag_context_max_tokens=5000)


class ScriptedRetriever:
    """Returns a scripted list of results per call (the last list repeats) and records the calls.

    ``script`` may also be a callable ``(query, top_k) -> results``.
    """

    def __init__(
        self,
        script: Sequence[Sequence[RetrievalResult]] | Callable[[str, int | None], list],
        error: Exception | None = None,
    ) -> None:
        self._script = script
        self._error = error
        self.calls: list[tuple[str, int | None]] = []

    @property
    def queries(self) -> list[str]:
        return [q for q, _ in self.calls]

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        if self._error is not None:
            raise self._error
        if callable(self._script):
            return list(self._script(query, top_k))
        index = min(len(self.calls) - 1, len(self._script) - 1)
        return list(self._script[index])


def chunk(text: str, *, path: str = "src/a.py", start: int = 1, rank: int = 1, **kw):
    return make_result(text, rank=rank, path=path, start=start, **kw)


def workflow(
    retriever,
    llm: FakeLLMClient | None = None,
    *,
    settings: Settings = SETTINGS,
    **config: object,
) -> tuple[AgentWorkflow, FakeLLMClient]:
    llm = llm or FakeLLMClient("answer [Source 1]")
    cfg = AgentConfig.from_settings(settings).model_copy(update=config)
    return AgentWorkflow(retriever, llm, settings=settings, config=cfg), llm


# A five-term question whose evidence can be supplied one term at a time.
Q_MULTI = "Explain how authentication, session tokens and password hashing work"
MULTI_ROUNDS = [
    [chunk("authentication handler", path="src/auth.py", start=1)],
    [chunk("session store", path="src/session.py", start=1)],
    [chunk("tokens issued here", path="src/tokens.py", start=1)],
]
