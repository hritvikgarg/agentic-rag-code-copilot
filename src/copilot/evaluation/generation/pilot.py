"""The bounded pilot: four predefined questions run end to end in an isolated namespace.

A pilot answers "does the pipeline work against the real provider, and what does it cost?" before
the full experiment is paid for. It is deliberately *not* a small experiment:

* **Selection is fixed in advance and uses no outcome.** For each repository, one answerable and
  one unanswerable question are chosen by the smallest SHA-256 of
  ``seed|repository|answerability|question id``. Only identifiers and the answerability flag are
  read - never a historical answer's quality, retrieval success, or any M10 result. The chosen ids
  are written to ``generation-eval/protocol/pilot-v1.json`` and committed *before* any live call.
* **Same configuration as the primary run:** the same resolved protocol (model, prompts, top-k,
  temperature, max output tokens), judge and rubric.
* **Separate namespace:** a different output directory, runs stamped ``scope="pilot"`` (the primary
  commands refuse them and vice versa), no historical reuse, no import into the primary analysis.
* **Bounded:** at most 4 questions x 2 arms = 8 generation calls and 8 judge calls (16 provider
  calls before retries), with the same bounded retries, rate-limit breaker and model confirmation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from copilot.evaluation.generation.execution import RepositoryContext
from copilot.evaluation.generation.schema import GenerationQuestion

PILOT_SCHEMA = "generation-pilot/1"
PILOT_SPEC_PATH = "generation-eval/protocol/pilot-v1.json"
DEFAULT_PILOT_OUTPUT_DIR = "data/generation-eval/pilot"
PILOT_SELECTION_SEED = "generation-pilot-v1"
PILOT_SCOPE = "pilot"

SELECTION_RULE = (
    "For each repository and each answerability (answerable, unanswerable) choose the question "
    "whose SHA-256 of '<seed>|<repository_id>|<answerability>|<question_id>' is smallest. Only "
    "question ids and the answerability flag are read. No historical answer, retrieval result, "
    "judge score or M10 outcome is used."
)


class PilotSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = PILOT_SCHEMA
    name: str
    protocol_name: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")  # of the *unconfirmed* protocol
    selection_rule: str
    selection_seed: str
    questions: dict[str, tuple[str, ...]]  # repository id -> question ids
    benchmark_digests: dict[str, str]
    recorded_before_any_live_execution: bool = True
    output_namespace: str = DEFAULT_PILOT_OUTPUT_DIR
    bounds: dict[str, int]
    notes: tuple[str, ...] = ()


def _rank(seed: str, repository_id: str, answerable: bool, question_id: str) -> str:
    kind = "answerable" if answerable else "unanswerable"
    return hashlib.sha256(f"{seed}|{repository_id}|{kind}|{question_id}".encode()).hexdigest()


def select_pilot_questions(
    repository_id: str, questions: Sequence[GenerationQuestion], *, seed: str = PILOT_SELECTION_SEED
) -> tuple[str, ...]:
    """``(answerable id, unanswerable id)``; reads only ids and the answerability flag."""
    chosen: list[str] = []
    for answerable in (True, False):
        pool = [q.id for q in questions if q.is_answerable is answerable]
        if not pool:
            raise ValueError(
                f"{repository_id}: no {'answerable' if answerable else 'unanswerable'} question"
            )
        chosen.append(min(pool, key=lambda qid: _rank(seed, repository_id, answerable, qid)))
    return tuple(chosen)


def build_pilot_spec(
    protocol_name: str,
    protocol_sha256: str,
    benchmarks: Mapping[str, Sequence[GenerationQuestion]],
    digests: Mapping[str, str],
) -> PilotSpec:
    selected = {
        repo: select_pilot_questions(repo, questions)
        for repo, questions in sorted(benchmarks.items())
    }
    count = sum(len(ids) for ids in selected.values())
    return PilotSpec(
        name="plain-vs-rag-pilot-v1",
        protocol_name=protocol_name,
        protocol_sha256=protocol_sha256,
        selection_rule=SELECTION_RULE,
        selection_seed=PILOT_SELECTION_SEED,
        questions=selected,
        benchmark_digests=dict(sorted(digests.items())),
        bounds={
            "questions": count,
            "generation_calls_max": 2 * count,
            "judge_calls_max": 2 * count,
            "provider_calls_max_before_retries": 4 * count,
        },
        notes=(
            "Same resolved protocol, prompts, judge and rubric as the primary run.",
            "Separate artifact namespace; runs carry scope=pilot; no historical reuse.",
            "Not imported into the primary analysis; far too small for inference.",
        ),
    )


def dump_pilot_spec(spec: PilotSpec) -> str:
    return spec.model_dump_json(indent=2) + "\n"


def load_pilot_spec(path: str | Path) -> PilotSpec:
    return PilotSpec.model_validate_json(Path(path).read_text(encoding="utf-8"))


class PilotMismatchError(ValueError):
    """The pilot specification does not belong to the current protocol or benchmarks."""


def verify_pilot_spec(spec: PilotSpec, protocol_sha256: str, digests: Mapping[str, str]) -> None:
    if spec.protocol_sha256 != protocol_sha256:
        raise PilotMismatchError(
            "the pilot specification was written for a different protocol; it must not be reused"
        )
    for repo, ids in spec.questions.items():
        if digests.get(repo) != spec.benchmark_digests.get(repo):
            raise PilotMismatchError(
                f"{repo}: the benchmark differs from the one the pilot recorded"
            )
        if not ids:
            raise PilotMismatchError(f"{repo}: the pilot lists no question")


def restrict_context(ctx: RepositoryContext, spec: PilotSpec) -> RepositoryContext:
    """``ctx`` limited to the pilot's questions and stamped ``scope="pilot"``."""
    wanted = spec.questions.get(ctx.spec.repository_id)
    if wanted is None:
        raise PilotMismatchError(f"{ctx.spec.repository_id}: not part of the pilot")
    by_id = {q.id: q for q in ctx.questions}
    missing = [qid for qid in wanted if qid not in by_id]
    if missing:
        raise PilotMismatchError(f"{ctx.spec.repository_id}: unknown pilot question(s) {missing}")
    return replace(ctx, questions=tuple(by_id[qid] for qid in wanted), scope=PILOT_SCOPE)


def check_namespaces(pilot_dir: str | Path, primary_dir: str | Path) -> None:
    """Refuse a pilot whose output directory is (or sits inside) the primary one."""
    pilot, primary = Path(pilot_dir).resolve(), Path(primary_dir).resolve()
    if pilot == primary or primary in pilot.parents or pilot in primary.parents:
        raise PilotMismatchError(
            "the pilot output directory must be separate from the primary one "
            f"({pilot_dir} vs {primary_dir})"
        )
