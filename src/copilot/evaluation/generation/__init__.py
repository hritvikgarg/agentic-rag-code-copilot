"""Plain LLM vs repository-aware RAG: answer-quality evaluation infrastructure (Milestone 11A).

Everything here is offline: no module calls a hosted provider unless a caller passes a real
``LLMClient`` and explicitly confirms the model. See ``docs/generation-evaluation.md``.
"""
