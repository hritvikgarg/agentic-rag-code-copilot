"""Arm definitions and the frozen experiment configuration.

The configuration is a JSON file written *before* any retrieval is run
(``experiments/chunking-comparison/design/frozen-config.json``)
and hashed; every later step refuses to run if the file's hash does not match the one recorded in
the design document's freeze record. Arms are described by strategy name, version and parameters,
and the chunker is rebuilt through ``create_chunker_from_params`` - the same path the retriever
uses to re-materialise an index - so the experiment cannot drift from what an index manifest says.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from copilot.chunking.base import Chunker
from copilot.chunking.registry import create_chunker, create_chunker_from_params
from copilot.config.settings import Settings
from copilot.ingestion import IngestionPolicy, ingest_repository
from copilot.models.ingestion import IngestionResult

SCHEMA = "chunking-experiment-config/1"
ARM_IDS = ("A", "B", "C", "D")


class ExperimentConfigError(ValueError):
    """The experiment configuration is missing, malformed or does not match its recorded hash."""


@dataclass(frozen=True)
class ArmConfig:
    """One experimental arm: which chunker (by registry identity) and why."""

    arm_id: str
    label: str
    strategy: str
    strategy_version: int
    params: Mapping[str, int | str]

    def chunker(self) -> Chunker:
        return create_chunker_from_params(self.strategy, self.strategy_version, self.params)

    def to_json(self) -> dict[str, object]:
        return {
            "arm_id": self.arm_id,
            "label": self.label,
            "strategy": self.strategy,
            "strategy_version": self.strategy_version,
            "params": dict(self.params),
        }


def production_arm(arm_id: str, label: str, strategy: str, settings: Settings) -> ArmConfig:
    """An arm that uses a production strategy with the current production settings."""
    chunker = create_chunker(strategy, settings=settings)
    return ArmConfig(arm_id, label, chunker.name, chunker.version, dict(chunker.params()))


def ingest_snapshot(
    tree: str | Path,
    settings: Settings,
    *,
    repository_name: str,
    ignore_directories: tuple[str, ...],
) -> IngestionResult:
    """Ingest the frozen snapshot tree with the production ingestion rules."""
    policy = IngestionPolicy.from_settings(settings).with_extra_ignored_directories(
        *ignore_directories
    )
    return ingest_repository(tree, policy, repository_name=repository_name)


def canonical_json(data: object) -> str:
    """Deterministic JSON text (sorted keys, 2-space indent, trailing newline)."""
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen_config(
    path: str | Path, *, expected_sha256: str | None = None
) -> dict[str, object]:
    """Load the frozen configuration; verify its hash if ``expected_sha256`` is given."""
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise ExperimentConfigError(f"cannot read the frozen configuration {p}: {exc}") from exc
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ExperimentConfigError(
            f"{p} does not match its recorded SHA-256; the design was modified after freezing"
        )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExperimentConfigError(f"{p} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ExperimentConfigError(f"{p} is not a {SCHEMA} file")
    return data


def arms_from_config(config: Mapping[str, object]) -> list[ArmConfig]:
    """The four arms of a loaded configuration, in A, B, C, D order."""
    try:
        raw = config["arms"]
        assert isinstance(raw, list)
        arms = [
            ArmConfig(
                arm_id=str(a["arm_id"]),
                label=str(a["label"]),
                strategy=str(a["strategy"]),
                strategy_version=int(a["strategy_version"]),
                params=dict(a["params"]),
            )
            for a in raw
        ]
    except (KeyError, TypeError, ValueError, AssertionError) as exc:
        raise ExperimentConfigError("the configuration's 'arms' section is malformed") from exc
    if [a.arm_id for a in arms] != list(ARM_IDS):
        raise ExperimentConfigError(f"arms must be exactly {ARM_IDS} in order")
    return arms
