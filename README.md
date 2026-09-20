# Agentic RAG Software Engineering Copilot

An academic project: a repository-aware AI assistant that answers questions about a codebase
using evidence retrieved from that codebase, and cites the files, functions and line ranges
it used.

> **Status: Milestone 1 (project foundation) only.** The RAG pipeline, embeddings, retrieval,
> agents and UI described below are **planned and do not exist yet.** See
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
| Embeddings | `fastembed` (ONNX), `jina-embeddings-v2-base-code`, fallback `bge-small-en-v1.5` | Planned (Milestone 4) |
| Vector store | FAISS `IndexFlatIP` + JSON sidecar + manifest | Planned (Milestone 4) |
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
│   ├── models/ ingestion/ parsing/ chunking/ embeddings/ vectorstore/
│   ├── retrieval/ llm/ rag/ agents/ evaluation/ services/ utils/
│   │                       # PLANNED: currently docstring-only packages
├── tests/                  # unit/ integration/ security/ (unit tests exist for the foundation)
├── docs/                   # ARCHITECTURE_PLAN.md
└── data/                   # local runtime data; contents are git-ignored
```

Directories added when first needed: `ui/` (Milestone 7), `scripts/`, `benchmarks/`,
`results/` (Milestones 5 and 10).

## Current implementation status

**Implemented now (Milestone 1: foundation only)**

- Project packaging and dependency configuration (`pyproject.toml`).
- Typed configuration with validation and safe defaults (`copilot.config.Settings`).
- Logging setup that masks known secret values (`copilot.config.setup_logging`).
- `.gitignore` protecting secrets, environments, caches, indexes and runtime data.
- Unit tests for the package, configuration, logging and ignore rules.

**Planned (not implemented; do not expect these to work)**

Safe repository ingestion, chunking, embeddings, vector search, RAG answers, citations,
Streamlit UI, LangGraph workflow, structure-aware chunking, debugging assistance, evaluation.

## Setup (current milestone)

Requires [uv](https://docs.astral.sh/uv/) (it will download Python 3.12 if needed).

```bash
git clone <this repository>
cd agentic-rag-code-copilot
uv sync                      # creates .venv with Python 3.12 and installs dependencies
```

Optional configuration (nothing needs a key yet; the LLM model ID is intentionally unset until Milestone 6):

```bash
cp .env.example .env         # Windows PowerShell: Copy-Item .env.example .env
# edit .env; it is git-ignored and must never be committed
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
| 1 | Project foundation | **Done (this commit)** |
| 2 | Safe repository ingestion | Next |
| 3-7 | Chunking, embeddings, retrieval, basic RAG, Streamlit MVP | Planned |
| 8-11 | LangGraph, structure-aware chunking, two experiments | Planned |
| 12-18 | Debugging, security hardening, testing, docs, deployment, viva prep | Planned (optional tail) |

## Security note

- Secrets come only from environment variables or a local `.env` file. `.env` is git-ignored;
  `.env.example` contains placeholders only.
- Never commit API keys, tokens, private keys or credentials. Logging masks known secret values.
- Ingestion will skip secrets, binaries, oversized files and unsafe paths (Milestone 2, planned).
- When the LLM is used, retrieved repository text is sent to an external API. Only index public
  or your own repositories when using a hosted provider.
- The project never executes code from an indexed repository.

## License

Released under the [MIT License](LICENSE).
