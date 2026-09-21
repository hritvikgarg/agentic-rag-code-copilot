# Agentic RAG Software Engineering Copilot

An academic project: a repository-aware AI assistant that answers questions about a codebase
using evidence retrieved from that codebase, and cites the files, functions and line ranges
it used.

> **Status: Milestone 5c (content-secret scanner and external-LLM gate).** The foundation,
> repository ingestion, baseline chunking, a local embedding service, a persistent vector index,
> semantic retrieval (no LLM), a retrieval benchmark harness and a content-based secret scanner
> with a fail-closed gate exist. The RAG pipeline, LLM
> answers, agents and the UI described below are **planned and do not exist yet.** See
> [Current implementation status](#current-implementation-status).

## Academic context

University B.Tech CSE Generative AI course project (Track B, Applied GenAI). It is built in
small, verified milestones and evaluated with measurable experiments, not just demonstrated.

## Problem statement

Developers and students spend a lot of time understanding unfamiliar repositories: finding the
relevant files, tracing how features work, and locating the cause of bugs. A general-purpose LLM
does not know a specific repository, so it may invent files, functions or architecture and give
generic advice. This project retrieves real repository evidence first and answers only from it,
stating clearly when the evidence is insufficient.

## Target users

Software developers, computer science students, developers joining an unfamiliar project, and
small teams working with medium-sized repositories.

## Project goals

1. Ingest a repository **safely** (no secrets, binaries, path traversal or oversized files).
2. Chunk, embed and index source code and documentation with useful metadata.
3. Retrieve relevant code semantically (later hybrid with keyword search).
4. Answer repository-specific questions with citations (file, function/class, line range).
5. Refuse to fabricate: say so when repository evidence is insufficient.
6. Provide evidence-based debugging help, clearly separating evidence from hypotheses.
7. Measure it: plain LLM vs repository-aware RAG, and baseline vs structure-aware chunking.

## High-level architecture (planned)

```
Indexing (offline)
  Repository -> Safe file discovery -> Parsing -> Chunking -> Metadata
             -> Embeddings -> Vector store (FAISS + sidecar metadata)

Querying (online)
  Question -> LangGraph router -> Retrieval -> Evidence check -> LLM answer
           -> Citation validation -> Answer + files/functions/line ranges -> Streamlit UI
```

The full design, technology decisions, milestones and evaluation plan are in
[`docs/ARCHITECTURE_PLAN.md`](docs/ARCHITECTURE_PLAN.md).

## Technology stack

| Layer | Choice | Status |
|---|---|---|
| Language / packaging | Python 3.12, `uv`, `pyproject.toml`, `src/` layout | **Implemented now** |
| Configuration | `pydantic-settings` (`.env`, environment variables) | **Implemented now** |
| Logging | Standard library `logging` with secret redaction | **Implemented now** |
| Testing / lint | `pytest`, `pytest-cov`, `ruff` | **Implemented now** |
| Embeddings | `fastembed` (ONNX), `jina-embeddings-v2-base-code`, fallback `bge-small-en-v1.5` (unused) | **Implemented and validated on Windows (Milestone 4)** |
| Vector index | FAISS `IndexFlatIP` (exact) + JSONL sidecar + manifest, atomic save | **Implemented and validated on Windows with the real model (Milestone 5a)** |
| Semantic retrieval | Query embedding -> exact FAISS search -> verified source text (no LLM) | **Implemented (Milestone 5b)** |
| Retrieval evaluation | Hit@k, MRR, mean lines per hit; JSONL benchmark; cap x representation matrix | **Implemented (Milestone 5b); benchmark is not independent** |
| Secret scanner | Content scan (private keys, provider tokens, secret-like assignments) + fail-closed gate | **Implemented (Milestone 5c); not yet wired: no LLM path exists** |
| LLM | Gemini via `google-genai`; Ollama optional fallback | Planned (Milestone 6) |
| Interface | Streamlit | Planned (Milestone 7) |
| Orchestration | LangGraph | Planned (Milestone 8) |
| Parsing | Python `ast` for structure-aware chunking | Planned (Milestone 9) |

## Repository structure

```
agentic-rag-code-copilot/
├── pyproject.toml          # project metadata, dependencies, tool configuration
├── LICENSE                 # MIT
├── .env.example            # variable names and placeholders only (never real secrets)
├── src/copilot/
│   ├── config/             # IMPLEMENTED: settings.py, logging_setup.py
│   ├── ingestion/          # IMPLEMENTED: policy, discovery, reader, loader, CLI
│   ├── chunking/           # IMPLEMENTED (baseline only): line windows, registry, stats, CLI
│   ├── models/             # IMPLEMENTED: ingestion and chunk models
│   ├── embeddings/         # IMPLEMENTED: Embedder interface, fastembed adapter, token validation
│   ├── utils/              # IMPLEMENTED: safe path helpers, token estimate
│   ├── vectorstore/        # IMPLEMENTED: FAISS index, manifest, fingerprint, atomic save, CLI
│   ├── retrieval/          # IMPLEMENTED: semantic retrieval, hash-verified source, CLI
│   ├── evaluation/         # IMPLEMENTED: retrieval benchmark/metrics, plain-vs-RAG comparison
│   ├── security/           # IMPLEMENTED: content-secret scanner, external-LLM gate, CLI
│   ├── llm/                # IMPLEMENTED: LLMClient, Gemini, fake, guarded client (Milestone 6)
│   ├── rag/                # IMPLEMENTED: plain baseline, context, citations, RagService, CLI
│   ├── parsing/ agents/ services/
│   │                       # PLANNED: currently docstring-only packages
├── tests/                  # unit/ integration/ security/ + fixtures/ (synthetic repository builder)
├── docs/                   # ARCHITECTURE_PLAN.md, ingestion.md, chunking.md, embeddings.md, vector-index.md,
│                           # retrieval.md, evaluation.md, security.md, llm.md, rag.md
└── data/                   # local runtime data; contents are git-ignored
```

Directories added when first needed: `ui/` (Milestone 7), `scripts/`, `benchmarks/`,
`results/` (Milestones 5 and 10).

## Current implementation status

**Implemented now (Milestones 1-6)**

- Project packaging and dependency configuration (`pyproject.toml`).
- Typed configuration with validation and safe defaults (`copilot.config.Settings`).
- Logging setup that masks known secret values (`copilot.config.setup_logging`).
- `.gitignore` protecting secrets, environments, caches, indexes and runtime data.
- **Safe repository ingestion** (`copilot.ingestion.ingest_repository`): prunes ignored and
  sensitive directories during traversal, rejects sensitive, binary, oversized, unsupported and
  undecodable files, never follows symlinks, enforces file-count/size limits, and returns typed
  `SourceFile` objects plus statistics. Supported types: `.py .js .jsx .ts .tsx .java .c .cpp .md
  .json .yaml .yml`. Details, policies and limitations: [`docs/ingestion.md`](docs/ingestion.md).
- **Baseline chunking** (`copilot.chunking`, strategy `line`): a structure-blind baseline that cuts
  *all* supported text (code, Markdown, JSON, YAML) into overlapping line windows, with an
  estimated-token safety cap, a documented deterministic chunk-id contract, citation metadata (file,
  language, 1-based line range) and chunk/line statistics. Only the baseline
  strategy exists; the structure-aware `ast` strategy is reserved for Milestone 9 and fails loudly
  if selected. Details and limitations: [`docs/chunking.md`](docs/chunking.md).
- **Local embedding service** (`copilot.embeddings`): an `Embedder` interface (`embed_documents`,
  `embed_query`, `count_tokens`), a fastembed/ONNX adapter for `jinaai/jina-embeddings-v2-base-code`
  (768-d, unit-length vectors), a deterministic fake for tests, batching, a documented model cache,
  a chunk-to-embedding-text representation (metadata prefix, raw source untouched), `chunk_id <->
  vector` pairing, and tooling that measures the token estimator against the real tokenizer.
  Validated on native Windows with the real model (768-d unit vectors, 8192-token limit). On this
  repository the heuristic estimator **systematically underestimates** the real Jina tokenizer
  (89% of chunks; about 14.5% in total), which is safe (largest embedded chunk 736 of 8192 tokens);
  chunk defaults are unchanged. The Milestone 5b retrieval matrix kept the 512 cap and metadata-prefixed
  text as defaults (measured on one self-authored benchmark; see below). Measurements and decisions:
  [`docs/embeddings.md`](docs/embeddings.md).
- **Persistent vector index** (`copilot.vectorstore`): builds an exact FAISS `IndexFlatIP` (cosine
  similarity on unit vectors) from repository -> chunks -> Jina embeddings, with a deterministic
  repository fingerprint, a deterministic index id, a verified vector-position <-> chunk-id mapping
  (`chunks.jsonl`, citation metadata only, no source text), a `manifest.json` recording every
  compatibility-critical setting and artifact checksums, strict loading that **refuses**
  incompatible, stale or corrupt indexes, and atomic saves. CLI: `python -m copilot.vectorstore
  {build,info,validate}` (`info` and `validate` do not load the model). This is indexing and
  persistence, plus the exact `VectorIndex.search` primitive. Details, contracts and
  limitations: [`docs/vector-index.md`](docs/vector-index.md).
- **Semantic retrieval** (`copilot.retrieval`, Milestone 5b): `retrieve(query, top_k, repo_path,
  index)` embeds the query locally with the index's model, runs exact FAISS search (deterministic tie
  order, FAISS `-1` padding filtered), re-materialises each chunk's source text from the repository
  and verifies it against the index (stale repositories are refused), and returns typed
  `RetrievalResult` objects with score, file, line range and text. **No LLM is involved and it
  returns chunks, not answers.** CLI: `python -m copilot.retrieval search INDEX_DIR --repo PATH
  "query"`. Details: [`docs/retrieval.md`](docs/retrieval.md).
- **Retrieval evaluation** (`copilot.evaluation`): a transparent JSONL benchmark format (questions
  with hand-verified source regions, pinned by text hashes), Hit@1/3/5/10, MRR and mean lines per
  hit, and a runner for the chunk-cap x representation matrix. The included benchmark covers this
  repository at commit `055a8d5` and is **not independent** (same author wrote code and questions).
  The 6-configuration matrix (cap 512/768/1024 x prefixed/raw) was run on Windows with the real
  model: **512 + prefixed** had the best Hit@1 (47.1%), Hit@3 (70.6%) and MRR (0.603) and stays the
  default; `raw` had higher Hit@10 (85.3% vs 82.4%). Larger chunks have a line-overlap advantage,
  the differences are one to three questions of 34, and no significance or generalisation is
  claimed. Full table and limits: [`docs/evaluation.md`](docs/evaluation.md).
- **Content-secret scanner and external-LLM gate** (`copilot.security`, Milestone 5c): scans the
  *content* of repository text for private keys, provider token formats (GitHub, AWS, Google, Slack,
  Stripe, GitLab, Hugging Face, `sk-` keys...), bearer values, URL credentials and non-trivial literals
  assigned to secret-like names, with placeholder/environment-lookup/hash/UUID false-positive handling.
  Findings carry only safe metadata (rule, relative path, line, column, masked preview); the value is
  never stored, logged or raised. `assert_safe_for_external_llm(...)` raises
  `RepositorySecretRiskError` on any finding and has **no bypass**. CLI: `python -m copilot.security
  scan PATH` (exit 0 clean, 1 findings, 2 error/incomplete). Since Milestone 6 the gate sits
  immediately before every hosted-LLM request. It reduces risk but cannot guarantee that every
  secret is found. Details, rules and limitations: [`docs/security.md`](docs/security.md).
- **LLM layer** (`copilot.llm`, Milestone 6): a small `LLMClient` interface, a deterministic
  `FakeLLMClient` for tests, a Google Gemini client (official `google-genai` SDK; the model id comes
  from `COPILOT_LLM_MODEL` and **no default model is shipped**), typed secret-safe errors, and
  `GuardedLLMClient`, the single choke point that runs the secret gate on the exact outbound text
  before any provider call (no bypass flag). Details: [`docs/llm.md`](docs/llm.md).
- **Basic RAG and plain-LLM baseline** (`copilot.rag`, Milestone 6): `answer_plain` (question only,
  no repository) and `answer_with_rag` (semantic retrieval -> deterministic bounded context ->
  secret gate -> grounded prompt -> answer with retrieval-derived `file:start-end` citations, and an
  explicit `insufficient_evidence` status without calling the model when nothing usable was
  retrieved). CLI: `python -m copilot.rag ask INDEX_DIR --repo PATH "question"` and
  `python -m copilot.rag plain "question"`. **Not measured yet: no claim is made that RAG answers
  are better than plain ones.** Details: [`docs/rag.md`](docs/rag.md).
- **Plain-vs-RAG comparison framework** (`copilot.evaluation.comparison`): `python -m
  copilot.evaluation compare` collects paired answers, evidence, citations, status and latency for a
  subset of the benchmark, and `summarize` aggregates *manual* rubric ratings (correctness,
  citation correctness, hallucinated claims, completeness, abstention). The LLM is not used as a
  judge. No results exist yet. Plan and limits: [`docs/evaluation.md`](docs/evaluation.md).
- Unit, integration and security tests for the above (run against synthetic repositories).

**Planned (not implemented; do not expect these to work)**

Measured plain-vs-RAG results (the framework exists, the ratings do not), lexical/hybrid retrieval,
Streamlit UI, LangGraph workflow, structure-aware chunking, debugging assistance, and an evaluation
on an independent repository.

## Setup (current milestone)

Requires [uv](https://docs.astral.sh/uv/) (it will download Python 3.12 if needed).

```bash
git clone <this repository>
cd agentic-rag-code-copilot
uv sync                      # creates .venv with Python 3.12 and installs dependencies
```

Optional configuration (nothing needs a key yet; the LLM model ID is intentionally unset until
Milestone 6):

```bash
cp .env.example .env         # Windows PowerShell: Copy-Item .env.example .env
# edit .env; it is git-ignored and must never be committed
```

Inspect what ingestion would accept from any local repository (prints statistics and relative
paths only, never file contents):

```bash
uv run python -m copilot.ingestion path/to/repo --skipped
```

Inspect embeddings (the model downloads ~0.64 GB into `data/cache/models` on first load; `info` and
`sizes` need no download):

```bash
uv run python -m copilot.embeddings info
uv run python -m copilot.embeddings sizes path/to/repo --ignore-dir data
uv run python -m copilot.embeddings smoke path/to/repo -n 8
uv run python -m copilot.embeddings tokens path/to/repo
```

Chunk a repository with the configured strategy and print chunk statistics (metadata only):

```bash
uv run python -m copilot.chunking path/to/repo --samples 5
```

Build and inspect a vector index (loads the embedding model; the first run downloads ~0.64 GB;
`info` and `validate` never load it):

```bash
uv run python -m copilot.vectorstore build path/to/repo --ignore-dir data
uv run python -m copilot.vectorstore info data/indexes/<index-id>
uv run python -m copilot.vectorstore validate data/indexes/<index-id> --repo path/to/repo --ignore-dir data
```

Search a built index (loads the embedding model; use the same `--ignore-dir` values as when
building) and run the retrieval benchmark:

```bash
uv run python -m copilot.retrieval search data/indexes/<index-id> --repo path/to/repo --ignore-dir data "How is logging configured?"
uv run python -m copilot.evaluation verify benchmarks/copilot_self_055a8d5.jsonl --repo <export of commit 055a8d5>
```

Scan a repository's content for secrets (uses the safe ingestion; exit status 1 if found):

```bash
uv run python -m copilot.security scan path/to/repo --ignore-dir data
```

Run the checks:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Development roadmap

Priority order: safe ingestion, baseline chunking, embeddings, semantic retrieval, basic RAG,
citations/grounding, Streamlit UI, evaluation, then LangGraph and advanced retrieval.
The 19-milestone plan is a framework, not a promise that every optional feature will be built.

| Milestone | Topic | Status |
|---|---|---|
| 0 | Architecture and planning | Done |
| 1 | Project foundation | Done |
| 2 | Safe repository ingestion | Done |
| 3 | Baseline chunking | Done |
| 4 | Embedding service and tokenizer validation | **Done (validated on Windows)** |
| 5a | FAISS vector index | **Done (validated on Windows)** |
| 5b | Semantic retrieval and retrieval evaluation | **Done (matrix run on Windows; results in `docs/evaluation.md`)** |
| 5c | Content-secret scanner and external-LLM gate | **Done** |
| 6 | Basic RAG + plain-LLM baseline + comparison framework | **Done (unit-tested with fakes; live verification is manual; results not yet measured)** |
| 7 | Streamlit MVP | Planned |
| 8-11 | LangGraph, structure-aware chunking, two experiments | Planned |
| 12-18 | Debugging, security hardening, testing, docs, deployment, viva prep | Planned (optional tail) |

## Security note

- Secrets come only from environment variables or a local `.env` file. `.env` is git-ignored;
  `.env.example` contains placeholders only.
- Never commit API keys, tokens, private keys or credentials. Logging masks known secret values.
- Ingestion skips sensitive file *names* (`.env*`, keys, credential files), binaries, oversized
  files and unsafe paths, and never follows symlinks. Filename filtering alone cannot see a key
  pasted into an ordinary source file, so a **content scanner** (`copilot.security`, Milestone 5c)
  inspects file text, and its fail-closed gate `assert_safe_for_external_llm` runs immediately before
  every hosted-LLM request (`GuardedLLMClient`, no bypass). It cannot guarantee that every secret
  is found.
- `python -m copilot.rag ask` and `plain`, and `python -m copilot.evaluation compare`, send the
  question (and, for RAG, retrieved repository chunks) to the configured hosted provider. Only
  index public or your own repositories when using a hosted provider. The API key is read from
  `GEMINI_API_KEY`, is never printed or logged, and provider error text is never shown.
- The project never executes code from an indexed repository.

## License

Released under the [MIT License](LICENSE).
