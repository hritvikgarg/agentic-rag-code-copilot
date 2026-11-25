"""Can an answer from an earlier (Milestone 7.x) comparison run be reused in the primary run?

The historical files (``answer-comparison/1``) were produced with plain-v1 and rag-v1 prompts and a
model/budget that may differ from the primary protocol. Mixing them silently would compare answers
made under different conditions. This module checks, **side by side and before any use**, every
condition for reuse and reports the reasons a side is *not* reusable. It only reads: historical
files are never modified, and nothing here calls a provider.

Reuse conditions (all must hold; a side with any failed condition is *not* reusable):

* the file's schema, repository name and index id are those of the target protocol;
* provider, model, temperature, ``max_output_tokens`` and ``top_k`` equal the target's;
* the side's prompt version equals the target arm's prompt version **and** that version's prompt
  text still hashes to its pinned SHA-256 (so the text cannot have drifted under the same label);
* the question text equals the generation benchmark's text for that id;
* the side was successful (an answer, no error, not a placeholder);
* RAG only: the index on disk has the frozen retriever spec (the manifest), and every recorded
  source's chunk id exists in that index with the same file and line range (request-level evidence
  provenance), and the record holds the evidence it claims (an answered record without sources is
  rejected).

Plain answers from ``plain-v1`` are never reusable in a ``plain-v2`` experiment: the system prompts
differ, so they are a separate earlier experiment.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from copilot.evaluation.comparison import (
    COMPARISON_SCHEMA,
    ComparisonRun,
    PlainRecord,
    RagRecord,
    side_state,
)
from copilot.vectorstore.index import read_manifest
from copilot.vectorstore.manifest import IndexManifest


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReuseTarget:
    """The frozen protocol parameters a historical side must match."""

    repository_name: str
    index_id: str
    llm_provider: str
    llm_model: str
    temperature: float
    max_output_tokens: int
    top_k: int
    plain_prompt_version: str
    rag_prompt_version: str
    questions: Mapping[str, str]  # question id -> text from the generation benchmark
    prompt_digests: Mapping[str, str]  # prompt version -> pinned sha256 of the text now in code
    prompt_texts: Mapping[str, str]  # prompt version -> text now in code
    retriever: Mapping[str, object]  # required manifest spec fields (strategy, params, model, ...)


@dataclass(frozen=True)
class SideVerdict:
    question_id: str
    arm: str  # "plain" | "rag"
    reusable: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompatReport:
    source_name: str
    run_level_reasons: tuple[str, ...]
    verdicts: tuple[SideVerdict, ...]
    successful_sides: Mapping[str, int] = field(default_factory=dict)

    def reusable(self, arm: str) -> list[str]:
        return [v.question_id for v in self.verdicts if v.arm == arm and v.reusable]

    def summary(self) -> dict[str, object]:
        return {
            "source": self.source_name,
            "run_level_reasons": list(self.run_level_reasons),
            "successful_sides": dict(self.successful_sides),
            "reusable": {arm: len(self.reusable(arm)) for arm in ("plain", "rag")},
        }


def _run_level(run: ComparisonRun, target: ReuseTarget) -> list[str]:
    reasons: list[str] = []

    def check(label: str, old: object, new: object) -> None:
        if old != new:
            reasons.append(f"{label}: file has {old!r}, protocol needs {new!r}")

    check("schema_version", run.schema_version, COMPARISON_SCHEMA)
    check("repository_name", run.repository_name, target.repository_name)
    check("index_id", run.index_id, target.index_id)
    check("llm_provider", run.llm_provider, target.llm_provider)
    check("llm_model", run.llm_model, target.llm_model)
    check("temperature", run.temperature, target.temperature)
    check("max_output_tokens", run.max_output_tokens, target.max_output_tokens)
    check("top_k", run.top_k, target.top_k)
    return reasons


def _prompt_reasons(version: str, wanted: str, target: ReuseTarget) -> list[str]:
    if version != wanted:
        return [f"prompt version {version!r} differs from the protocol's {wanted!r}"]
    pinned = target.prompt_digests.get(version)
    text = target.prompt_texts.get(version)
    if pinned is None or text is None:
        return [f"no pinned digest for prompt version {version!r}"]
    if sha256_text(text) != pinned:
        return [f"prompt text for {version!r} no longer matches its pinned SHA-256"]
    return []


def _retriever_reasons(manifest: IndexManifest, target: ReuseTarget) -> list[str]:
    spec = manifest.spec.model_dump()
    return [
        f"index spec {key}: {spec.get(key)!r} != required {want!r}"
        for key, want in target.retriever.items()
        if spec.get(key) != want
    ]


def _load_chunk_index(index_dir: Path) -> dict[str, tuple[str, int, int]]:
    out: dict[str, tuple[str, int, int]] = {}
    path = index_dir / "chunks.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["chunk_id"]] = (row["file_path"], row["start_line"], row["end_line"])
    return out


def _rag_provenance(
    record: RagRecord, chunks: Mapping[str, tuple[str, int, int]], top_k: int
) -> list[str]:
    reasons: list[str] = []
    if record.status == "answered":
        if not record.sources:
            return ["answered record holds no retrieved sources (no request-level evidence)"]
        if len(record.sources) != record.retrieved_count:
            reasons.append("retrieved_count does not match the number of recorded sources")
    if len(record.sources) > top_k:
        reasons.append(f"more than top_k={top_k} recorded sources")
    for source in record.sources:
        known = chunks.get(source.chunk_id)
        if known is None:
            reasons.append(f"source chunk {source.chunk_id} is not in the index")
        elif known != (source.file_path, source.start_line, source.end_line):
            reasons.append(f"source chunk {source.chunk_id} lines differ from the index")
    return reasons


def check_historical_run(
    run: ComparisonRun, target: ReuseTarget, *, source_name: str, index_dir: Path | None
) -> CompatReport:
    """Per-side reuse verdicts for one historical file (read-only)."""
    run_reasons = _run_level(run, target)
    plain_prompt = _prompt_reasons(run.plain_prompt_version, target.plain_prompt_version, target)
    rag_prompt = _prompt_reasons(run.rag_prompt_version, target.rag_prompt_version, target)

    index_reasons: list[str] = []
    chunks: dict[str, tuple[str, int, int]] = {}
    if index_dir is None or not index_dir.is_dir():
        index_reasons.append("the index of the historical run is not available to verify against")
    else:
        try:
            manifest = read_manifest(index_dir)
            if manifest.index_id != run.index_id:
                index_reasons.append("index manifest id differs from the file's index_id")
            index_reasons += _retriever_reasons(manifest, target)
            chunks = _load_chunk_index(index_dir)
        except Exception as exc:  # unreadable index: not verifiable, so not reusable
            index_reasons.append(f"index could not be verified ({type(exc).__name__})")

    verdicts: list[SideVerdict] = []
    successful = {"plain": 0, "rag": 0}
    for entry in run.entries:
        wanted = target.questions.get(entry.question_id)
        text_reasons = (
            ["question id is not in the generation benchmark"]
            if wanted is None
            else ([] if entry.question == wanted else ["question text differs from the benchmark"])
        )
        for arm, record in (("plain", entry.plain), ("rag", entry.rag)):
            state = side_state(record)
            reasons = [*run_reasons, *text_reasons]
            reasons += plain_prompt if arm == "plain" else [*rag_prompt, *index_reasons]
            if state != "successful":
                reasons.append(f"side state is {state}")
            else:
                successful[arm] += 1
                if arm == "plain":
                    assert isinstance(record, PlainRecord)
                    if record.answer is None or not record.answer.strip():
                        reasons.append("plain record has no answer")
                else:
                    assert isinstance(record, RagRecord)
                    reasons += _rag_provenance(record, chunks, target.top_k) if chunks else []
            verdicts.append(SideVerdict(entry.question_id, arm, not reasons, tuple(reasons)))
    return CompatReport(source_name, tuple(run_reasons), tuple(verdicts), successful)


#: SHA-256 of each prompt text as committed. A change to a text under the same version label would
#: invalidate every result produced with it, so reuse checks (and tests) compare against these.
PINNED_PROMPT_DIGESTS: dict[str, str] = {
    "plain-v1": "b506a7a90e9e61b8d51ecef68fda2fa406ae8106e76c4098fa9a9751672a494e",
    "plain-v2": "ac411a779f81ae3fa26304620d2fceb88649bed935d2d33a1545367a27aeac56",
    "rag-v1": "ec7e1fa7558537ed46a976febcd6e77706afc161e8366bf311feae7d96a0f34a",
}

#: The retriever of the original reference index (Milestones 5b/10). Checked field by field.
ORIGINAL_RETRIEVER: dict[str, object] = {
    "chunking_strategy": "line",
    "chunk_params": {"size_lines": 60, "overlap_lines": 10, "max_tokens": 512},
    "embedding_model_id": "jinaai/jina-embeddings-v2-base-code",
    "text_style": "prefixed",
    "index_type": "IndexFlatIP",
    "embedding_normalized": True,
}


def frozen_target(
    *,
    questions: Mapping[str, str],
    repository_name: str,
    index_id: str,
    llm_provider: str,
    llm_model: str,
    max_output_tokens: int,
    temperature: float = 0.0,
    top_k: int = 5,
    plain_prompt_version: str = "plain-v2",
    rag_prompt_version: str = "rag-v1",
) -> ReuseTarget:
    """The reuse target for a primary run, with prompt texts taken from the code as it is now."""
    from copilot.rag import prompts

    texts = {
        "plain-v1": prompts.PLAIN_SYSTEM_PROMPT,
        "plain-v2": prompts.PLAIN_SYSTEM_PROMPT_V2,
        "rag-v1": prompts.RAG_SYSTEM_PROMPT,
    }
    return ReuseTarget(
        repository_name=repository_name,
        index_id=index_id,
        llm_provider=llm_provider,
        llm_model=llm_model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_k=top_k,
        plain_prompt_version=plain_prompt_version,
        rag_prompt_version=rag_prompt_version,
        questions=dict(questions),
        prompt_digests=dict(PINNED_PROMPT_DIGESTS),
        prompt_texts=texts,
        retriever=dict(ORIGINAL_RETRIEVER),
    )
