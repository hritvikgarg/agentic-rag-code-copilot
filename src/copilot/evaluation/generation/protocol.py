"""The frozen primary protocol of the plain-LLM vs repository-aware-RAG answer experiment.

Everything here is decided **before any generation call**: which arms are compared, what is held
equal, how correctness is scored, what the primary endpoint is and which secondary endpoints are
reported. The protocol is a versioned JSON file (``generation-eval/protocol/primary-protocol.json``)
whose SHA-256 is recorded in every run file, so a run can be proven to belong to it and a changed
protocol can never be resumed as if it were the same experiment.

The generation model is deliberately **not** fixed in the file (``llm_provider``/``llm_model`` are
``null``): the person running the experiment must confirm it explicitly (``--confirm-model``), and
the confirmed identity becomes part of the resolved protocol hash. Nothing here calls a provider.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from copilot.evaluation.generation.compat import ORIGINAL_RETRIEVER, PINNED_PROMPT_DIGESTS

PROTOCOL_SCHEMA = "generation-protocol/1"
PROTOCOL_PATH = "generation-eval/protocol/primary-protocol.json"


class Endpoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    definition: str


class RepositorySpec(BaseModel):
    """One repository of the experiment (paths are relative to the project root)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repository_id: str
    kind: str  # "existing-snapshot" | "synthetic-authored"
    repository_name: str  # the name its index is built under
    source_root: str
    benchmark_path: str
    benchmark_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    repository_fingerprint: str | None = None
    index_dir: str | None = None  # an existing index to use; None = build locally (see docs)
    index_id: str | None = None
    ignore_directories: tuple[str, ...] = ()
    independent: bool = False


class JudgeSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_version: str
    rubric_version: str
    temperature: float = 0.0
    max_output_tokens: int = 4096
    note: str = ""


class PrimaryProtocol(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = PROTOCOL_SCHEMA
    name: str
    arm_a: str  # plain LLM
    arm_b: str  # repository-aware RAG
    plain_prompt_version: str
    rag_prompt_version: str
    prompt_digests: dict[str, str]
    llm_provider: str | None = None  # confirmed by the person running the experiment
    llm_model: str | None = None
    temperature: float = 0.0
    max_output_tokens: int
    top_k: int
    retriever: dict[str, object]
    retriever_rationale: str
    agentic_arm: str
    held_equal: tuple[str, ...]
    prompt_fairness: str
    repositories: tuple[RepositorySpec, ...]
    primary_endpoint: Endpoint
    secondary_endpoints: tuple[Endpoint, ...]
    judge: JudgeSettings
    audit_fraction: float = 0.20
    audit_seed: int = 20260
    bootstrap_resamples: int = 10000
    bootstrap_seed: int = 11
    confidence: float = 0.95
    limitations: tuple[str, ...] = ()

    @property
    def model_confirmed(self) -> bool:
        return bool(self.llm_provider and self.llm_model)

    def with_model(self, provider: str, model: str) -> PrimaryProtocol:
        """The same protocol with the generation model confirmed (the hash then changes)."""
        if not provider.strip() or not model.strip():
            raise ValueError("provider and model must both be given")
        return self.model_copy(update={"llm_provider": provider, "llm_model": model})

    def repository(self, repository_id: str) -> RepositorySpec:
        for spec in self.repositories:
            if spec.repository_id == repository_id:
                return spec
        raise KeyError(repository_id)


def canonical_json(protocol: PrimaryProtocol) -> str:
    return json.dumps(protocol.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def protocol_hash(protocol: PrimaryProtocol) -> str:
    """SHA-256 of the canonical protocol (including the confirmed model, when there is one)."""
    return hashlib.sha256(canonical_json(protocol).encode("utf-8")).hexdigest()


def dump_protocol(protocol: PrimaryProtocol) -> str:
    return json.dumps(protocol.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"


def load_protocol(path: str | Path) -> PrimaryProtocol:
    return PrimaryProtocol.model_validate_json(Path(path).read_text(encoding="utf-8"))


PRIMARY_ENDPOINT = Endpoint(
    name="paired difference in mean key-fact correctness (RAG - plain), answerable questions",
    definition=(
        "For every answerable question both arms answered successfully and that were judged "
        "successfully, d = score(RAG) - score(plain) where score is the 0-2 key-fact correctness "
        "score of rubric gen-rubric-1. The estimate is the mean of d, with a percentile bootstrap "
        "95% CI over questions (shared resample indices, fixed seed). Reported per repository and "
        "combined. Questions with a missing or failed side are listed and counted separately, "
        "never dropped silently and never imputed. A claim of improvement requires the CI to "
        "exclude 0; the synthetic and the self-referential repository are not pooled into a "
        "single headline without also showing each one."
    ),
)

SECONDARY_ENDPOINTS: tuple[Endpoint, ...] = (
    Endpoint(name="groundedness", definition="Judge groundedness grade 0-2 (answerable and not)."),
    Endpoint(name="completeness", definition="Fraction of key facts judged supported."),
    Endpoint(
        name="citation correctness",
        definition=(
            "Programmatic: fraction of the answer's citations that resolve to a real file and "
            "line range (valid_fraction); applicable to RAG, and to plain answers only when they "
            "write explicit file:line locations. Never imputed for answers without citations."
        ),
    ),
    Endpoint(
        name="hallucinated-reference indicators",
        definition=(
            "Programmatic and conservative: count of repository files/symbols named in an answer "
            "that do not exist in the frozen repository (not_found), per answer; "
            "ambiguous/cannot-evaluate reported separately."
        ),
    ),
    Endpoint(
        name="appropriate abstention",
        definition="Unanswerable questions: share of responses that abstain (system or text).",
    ),
    Endpoint(
        name="false refusal",
        definition="Answerable questions: share of responses that refuse or say evidence is "
        "insufficient.",
    ),
    Endpoint(
        name="retrieval relevance",
        definition="RAG only: whether a retrieved source overlaps a key-fact region (hit@k).",
    ),
    Endpoint(
        name="latency",
        definition="Wall-clock seconds of generated sides only (cache hits and reused historical "
        "responses are excluded and counted).",
    ),
    Endpoint(
        name="tokens",
        definition="Input, output and reasoning tokens reported by the provider (when reported).",
    ),
    Endpoint(
        name="end-to-end reliability",
        definition="Share of requested sides that ended successful / failed / not attempted.",
    ),
)

DEFAULT_LIMITATIONS: tuple[str, ...] = (
    "The copilot repository was authored by the project team and its documentation is in the "
    "retrieval index: this arm measures retrieval-augmented use of a repository the model has "
    "probably never seen, but the question author knew the code (author bias).",
    "The second repository is synthetic and written by the project team: it tests mechanics on "
    "unseen code, not performance on real-world independent software.",
    "A single hosted model generates both arms; results do not transfer to other models.",
    "The rubric judge is an LLM and is validated only through a human audit of a ~20% sample; "
    "judge self-preference toward the generator's model family is possible.",
    "Plain-v2 receives the same honesty rules as RAG but cannot cite [Source n]; citation "
    "metrics are therefore RAG-only unless the plain answer writes explicit locations.",
)


def default_protocol(repositories: tuple[RepositorySpec, ...]) -> PrimaryProtocol:
    """The frozen protocol for the given repositories (model left unconfirmed)."""
    return PrimaryProtocol(
        name="plain-vs-rag-primary-v1",
        arm_a="plain LLM (plain-v2 prompt, no repository evidence)",
        arm_b="repository-aware RAG (rag-v1 prompt, original line retriever, top-k 5)",
        plain_prompt_version="plain-v2",
        rag_prompt_version="rag-v1",
        prompt_digests=dict(PINNED_PROMPT_DIGESTS),
        max_output_tokens=8192,
        top_k=5,
        retriever=dict(ORIGINAL_RETRIEVER),
        retriever_rationale=(
            "The original line-window retriever (60 lines, 10 overlap, 512 tokens; Jina v2 base "
            "code, prefixed text, FAISS IndexFlatIP, top-k 5) is the one the project's earlier "
            "answers and the 8192-token historical run used. Milestones 5b and 10 compared "
            "retrieval variants on retrieval metrics; they did not establish that any variant "
            "yields better *answers*, so no superiority is claimed here and no variant was tuned "
            "on this benchmark. Using the original retriever also keeps the eight verified "
            "historical RAG responses comparable."
        ),
        agentic_arm="deferred (not part of this experiment)",
        held_equal=(
            "generation model and provider",
            "temperature 0",
            "max_output_tokens",
            "the question text (one field, sent verbatim to both arms)",
            "the frozen repository snapshot",
        ),
        prompt_fairness=(
            "plain-v1 had no abstention or labelling instruction while rag-v1 does, which would "
            "confound the comparison; plain-v2 adds the same honesty rules (do not state "
            "repository specifics you cannot know, label repository evidence vs general "
            "knowledge, say explicitly when you cannot answer) without any evidence. "
            "Residual asymmetry: only RAG is told to cite [Source n]. plain-v1 and rag-v1 are "
            "unchanged and pinned by SHA-256."
        ),
        repositories=repositories,
        primary_endpoint=PRIMARY_ENDPOINT,
        secondary_endpoints=SECONDARY_ENDPOINTS,
        judge=JudgeSettings(
            prompt_version="gen-judge-1",
            rubric_version="gen-rubric-1",
            note="Blind to the arm; a different model family from the generator is recommended.",
        ),
        limitations=DEFAULT_LIMITATIONS,
    )
