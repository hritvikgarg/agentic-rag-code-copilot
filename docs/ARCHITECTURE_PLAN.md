# Agentic RAG Software Engineering Copilot — Greenfield Architecture Plan (Milestone 0)

Status: APPROVED 2026-09-20 with the scope amendments listed in Section 12. Written 2026-09-20.

Confidence tags used throughout: **[Certain]** verified from a primary source or definitional, **[Likely]** strong inference, **[Guessing]** filling a gap — flagged so you can challenge it.

---

## 0. Uncomfortable truths first

1. **Embedding-only retrieval will fail on some of your own example questions.** "Where is this function used?" and "Which files implement user registration?" depend on exact identifiers and call sites. Dense embeddings blur identifiers. This plan therefore adds a lexical retriever and a deterministic AST usage finder, and measures them as ablations. [Likely]
2. **Your Experiment 1 can be invalidated by benchmark choice.** If you benchmark on a famous repo (Flask, Django, requests), the plain LLM has probably memorised it, and "RAG beats plain LLM" shrinks or vanishes. Benchmark on at least one repo the model cannot have seen (this project itself, or a small unpopular repo) and report both. [Likely]
3. **With 30–60 benchmark questions, differences under roughly 10 percentage points are noise.** We report raw counts, not just percentages, and never claim significance we did not test. [Certain — small-sample arithmetic]
4. **19 milestones is more than a semester of casual work if all are done deeply.** This plan defines a cut line (Section 9) so the core MVP plus both experiments are protected and the tail (deployment polish, extra languages) is what slips. [Likely]
5. **Structure-aware chunking will look better or worse for reasons unrelated to structure** unless we control chunk size, embedding model, k and dedup. Experiment 2 is only meaningful with those controls (Section 7). [Certain]

---

## 1. Verified facts used for decisions (fetched 2026-09-20)

| Fact | Source | Confidence |
|---|---|---|
| LangGraph latest 1.2.11 (2026-08-11), needs Python >= 3.10 | PyPI langgraph | [Certain] |
| faiss-cpu 1.15.1 (2026-09-16); Python 3.10–3.14; Windows win_amd64 wheels exist | PyPI faiss-cpu | [Certain] |
| fastembed 0.8.0 (2026-03-23), ONNX Runtime, no PyTorch, Python >= 3.10 | PyPI fastembed | [Certain] |
| fastembed supports `jinaai/jina-embeddings-v2-base-code` (768-d, ~0.64 GB, Apache-2.0) and `BAAI/bge-small-en-v1.5` (384-d, ~0.07 GB, MIT) | fastembed docs | [Certain] |
| Jina code model loaded via sentence-transformers needs `trust_remote_code=True` (runs downloaded code) | Hugging Face model card | [Certain] |
| sentence-transformers 6.1.0 needs PyTorch 2.2+ | PyPI | [Certain] |
| chromadb 1.5.9 (2026-05-05) | PyPI | [Certain] |
| Gemini docs list stable text models incl. `gemini-3.8-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite`; embedding `gemini-embedding-001`; Gemini 2.0 models deprecated | ai.google.dev/gemini-api/docs/models | [Likely] (page summarised by a tool) |
| Gemini free-tier request limits are NOT published on the docs page; only visible in your Google AI Studio account | ai.google.dev rate-limits page | [Certain] |

**Not verified, must be checked in Milestone 1/4/6:** whether fastembed runs cleanly on the target Windows machine, the actual Gemini quota, available GPU/RAM, and whether Ollama models are pulled locally.

---

## 2. Final architecture

### 2.1 Design principles
- **Two independent phases.** *Indexing* (offline, deterministic, no LLM) and *Querying* (online). Each is testable alone.
- **Every stage is an interface plus one concrete implementation** where we expect to swap or compare (chunker, embedder, vector store, LLM, retriever).
- **Grounding by construction, not by prompting.** The LLM is shown chunks labelled `[C1]..[Cn]` and may only cite those IDs. File paths and line ranges shown to the user are looked up from chunk metadata by our code. The LLM never types a path or line number. This makes fabricated file names structurally impossible in the citation section. (It does not prove the cited chunk supports the claim; that is measured separately as *faithfulness*.)
- **Deterministic where possible.** Routing starts rule-based; LLM only for ambiguity. Citation validation, path safety, secret scanning, and traceback parsing are all plain code.
- **Untrusted input everywhere.** Repository content is data, never instructions (prompt-injection defence, Section 11).

### 2.2 Data-flow diagram

```
                              INDEXING (offline, no LLM)
 ┌────────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────┐   ┌───────────┐
 │ Repo input │──▶│ Safe file    │──▶│ Parsing  │──▶│ Chunking │──▶│ Metadata  │
 │ dir / zip  │   │ discovery    │   │ AST(py)  │   │ A:lines  │   │ attached  │
 │ (URL later)│   │ allow/deny,  │   │ Markdown │   │ B:struct │   │ per chunk │
 └────────────┘   │ size, binary,│   └──────────┘   └──────────┘   └─────┬─────┘
                  │ secrets,path │                                       │
                  └──────────────┘                                       ▼
                                                 ┌────────────┐   ┌─────────────┐
                                                 │ Embeddings │──▶│ Vector store│
                                                 │ (fastembed)│   │ FAISS flat  │
                                                 └────────────┘   │ + chunks.jsonl
                                                                  │ + manifest  │
                                                                  └──────┬──────┘
                                                                         │
                              QUERYING (online)                          │
 ┌──────────┐   ┌───────────────────────────────┐                        │
 │ User Q + │──▶│ LangGraph: router (intent)     │                        │
 │ optional │   └───────┬───────────┬───────────┘                        │
 │ error    │           │           │                                    │
 │ text     │   repo_search   code_explain   doc_arch    debugging       │
 └──────────┘           └─────┬─────┴───────────┴────────────┘           │
                              ▼                                          │
                   ┌─────────────────────┐   dense + lexical + symbol    │
                   │ Retrieval           │◀──────────────────────────────┘
                   └──────────┬──────────┘
                              ▼
                   ┌─────────────────────┐  insufficient? ─▶ rewrite query ─┐
                   │ Evidence grader     │◀──────────────────────────────────┘
                   └──────────┬──────────┘  (max 1 retry, deterministic gate first)
                              ▼ sufficient / or explicit "insufficient"
                   ┌─────────────────────┐
                   │ Generator (LLM)     │  cites [C#] only, structured JSON
                   └──────────┬──────────┘
                              ▼
                   ┌─────────────────────┐
                   │ Citation validator  │  drop unknown IDs, resolve → file/symbol/lines
                   └──────────┬──────────┘
                              ▼
                   Answer + Evidence + Files + Symbols + Line ranges  ──▶ Streamlit
```

### 2.3 LangGraph workflow (Milestone 8)

```
START ─▶ classify_intent ─▶ [route] ─▶ retrieve_<intent> ─▶ grade_evidence
                                                  ▲               │
                                                  │      insufficient & retries<1
                                                  └── rewrite_query ◀┘
                             grade_evidence ─▶ generate_<intent> ─▶ validate_citations ─▶ END
                             grade_evidence ─▶ (insufficient after retry) ─▶ refuse_with_explanation ─▶ END
```

Why each node exists (no decorative agents):
- `classify_intent`: rule-based keywords/regex first (traceback present ⇒ debugging; "where is/defined/used" ⇒ repo_search; "explain/what does" + symbol ⇒ code_explain; "architecture/overview/README" ⇒ doc_arch); LLM classifier only when rules are ambiguous. Different intents change retrieval parameters and prompt, which is the only justification for routing.
- `retrieve_*`: same retrieval core, different config. Debugging additionally parses the traceback (deterministic) to fetch named files/functions directly.
- `grade_evidence`: deterministic gate (min chunks, score threshold tuned on the dev set, identifier presence) then generator may also declare insufficiency. **The retry loop is the one genuinely agentic behaviour.**
- `validate_citations`: pure code. Removes any citation not in retrieved set; flags answers with zero valid citations as ungrounded.
- Not LLM agents: router (mostly), grader (mostly), validator (never).

---

## 3. Technology decisions (Version 1)

| Area | Decision | Why | What would go wrong otherwise |
|---|---|---|---|
| Python | **3.12** | All key wheels (faiss, onnxruntime, streamlit) support it; mature; avoids 3.13/3.14 wheel gaps | 3.14 risks a missing wheel mid-project. [Likely] |
| Dependency mgmt | **uv** with `pyproject.toml` + committed `uv.lock`, `.python-version` | One-command reproducible env on Windows, installs Python itself; `pip install -e .` still works from pyproject as fallback | "Works on my machine" env drift; slow installs |
| Layout | **`src/copilot/`** package (src layout) instead of a top-level `app/` | Prevents accidental imports of un-installed code; tests run against the installed package; "app" is too generic | Import-path bugs that only show in CI |
| LangGraph | **Yes, `langgraph` (1.x)** for the router/workflow only, from Milestone 8 | Explicit typed state, conditional edges, and a retry loop are exactly its use case; gives a diagram for the viva | A hand-rolled if/else chain hides the workflow you must defend |
| LangChain | **Not used directly in V1.** (`langgraph` pulls `langchain-core` transitively; we do not import LangChain chat/vectorstore/splitter wrappers) | Our chunkers must keep exact line numbers (LangChain splitters do not); our own 60-line `LLMClient`/`Embedder`/`VectorStore` interfaces are simpler to explain and test. Reconsider only if a specific integration is needed | Extra abstraction layers you must explain in a viva without benefit |
| LLM (primary) | **Gemini API via `google-genai` SDK**, model ID is **config, not code**. Provisional: `gemini-3.8-flash` for answers, `gemini-3.5-flash-lite` for routing/judging fallback; final pick in M6 after you check quotas in AI Studio | Free/cheap, strong at code, no local GPU needed | Model IDs churn (Gemini 2.0 already deprecated); hard-coding one breaks the project |
| LLM (fallback) | **Ollama** provider behind the same `LLMClient` protocol, plus a `FakeLLM` for tests | Offline, no quota, reproducible; protects against rate limits during evaluation runs | Free-tier 429s mid-experiment |
| LLM response cache | On-disk cache keyed by (provider, model, prompt hash, temperature) — gitignored | Cheap, reproducible eval reruns | Re-paying and re-varying every rerun |
| Embeddings | **`fastembed` (ONNX) with `jinaai/jina-embeddings-v2-base-code`** (768-d, code-specialised, Apache-2.0). Alternative in config: `BAAI/bge-small-en-v1.5` (384-d, tiny, faster). `FakeEmbedder` for tests | No PyTorch (multi-hundred-MB dependency), no `trust_remote_code`, local, free, modular | Remote-code execution risk and heavy install with sentence-transformers route. **Unverified on your Windows box; smoke-test in M4, fallback bge-small** |
| Vector store | **FAISS `IndexFlatIP` on L2-normalised vectors** (= exact cosine) + `chunks.jsonl` sidecar + `manifest.json` | Exact search ⇒ deterministic experiments (Chroma's HNSW is approximate); repos are small (10^3–10^5 chunks) so brute force is fast; cosine similarity is visible and explainable | ANN approximation noise contaminating Hit@k comparisons |
| Chroma | Not used | Would add a persistence layer and approximate search we do not need | — |
| Lexical retrieval | **`rank_bm25`** over chunk text with a code-aware tokenizer (split snake_case/camelCase) | Fixes identifier queries; cheap; measurable ablation | Embedding-only misses exact names |
| UI | **Streamlit** | Fast, adequate, in your spec | — |
| FastAPI | **Not in MVP.** UI calls a `CopilotService` façade directly | A separate API server adds deployment/auth/CORS surface for no MVP benefit; the façade makes adding FastAPI later a thin wrapper | Time spent on plumbing instead of RAG quality |
| Testing | **pytest** + `pytest-cov`; markers `slow`, `live` (real models/API, skipped by default); **ruff** for lint+format | Standard; CI runs only fast deterministic tests with fakes | Flaky tests depending on network/model downloads |
| Config | **`pydantic-settings`**: typed `Settings` from environment / `.env`; secrets only via env; `.env.example` committed | Validation at startup; one source of truth | Scattered `os.getenv` and hard-coded paths |
| Logging | **stdlib `logging`**, one `setup_logging()`, module-level loggers, secrets never logged | No extra dependency; standard | Print-debugging and leaked keys |
| Parsing | **Python: stdlib `ast`.** Markdown: heading-based splitter. Other languages in V1: line-based fallback only. **tree-sitter** for JS/TS/Java = optional stretch after Milestone 10 | `ast` is exact for Python, zero dependency, does not execute code; multi-language AST is the biggest scope trap | Half-working multi-language parsers |
| Evaluation | Own harness: benchmark files (YAML), programmatic retrieval metrics, programmatic citation checks, rubric-based correctness (key-fact checklist) + LLM judge as a secondary signal + manual audit of a sample | LLM-as-judge alone is biased and non-reproducible | Numbers nobody can defend |
| CI | GitHub Actions: ruff + pytest (fakes only) | Catches regressions; visible in the repo | Silent breakage |

---

## 4. Final repository structure

```
agentic-rag-code-copilot/
├── pyproject.toml            # deps, ruff/pytest config, entry points
├── uv.lock                   # committed for reproducibility
├── .python-version           # 3.12
├── .env.example              # placeholders only, committed
├── .gitignore                # see Section 11
├── README.md
├── LICENSE                   # you choose (MIT suggested)
├── .github/workflows/ci.yml
│
├── src/copilot/
│   ├── config/               # settings.py (pydantic-settings), logging_setup.py
│   ├── models/               # schemas: RepoFile, Chunk, RetrievedChunk, Citation, Answer, GraphState
│   ├── ingestion/            # discovery.py, filters.py, secrets.py, archive.py (zip-safe), loader.py, git_source.py (optional, later)
│   ├── parsing/              # python_ast.py (symbol table), markdown.py, languages.py
│   ├── chunking/             # base.py (Chunker protocol), line_chunker.py (A), ast_chunker.py (B), markdown_sections.py (isolated utility, not used by A), registry.py
│   ├── embeddings/           # base.py, fastembed_embedder.py, fake.py
│   ├── vectorstore/          # base.py, faiss_store.py, manifest.py (model/dim/chunker/repo-hash guard)
│   ├── retrieval/            # semantic.py, lexical.py, hybrid.py (RRF fusion), symbol_lookup.py (AST usages)
│   ├── llm/                  # base.py (LLMClient protocol), gemini_client.py, ollama_client.py, fake.py, cache.py
│   ├── rag/                  # prompts.py, context_builder.py, generator.py, citations.py, pipeline.py, plain_baseline.py
│   ├── agents/               # state.py, router.py, nodes.py, graph.py, traceback_parser.py
│   ├── evaluation/           # benchmark.py, metrics.py, exp_chunking.py, exp_rag_vs_plain.py, judge.py, report.py
│   ├── services/             # copilot_service.py — the ONLY thing UI/CLI/scripts call
│   └── utils/                # paths.py (safe join), hashing.py, tokens.py, timing.py
│
├── ui/                       # streamlit_app.py, components/ (thin; no business logic)
├── scripts/                  # index_repo.py, run_eval.py, fetch_benchmark_repos.py
├── benchmarks/               # repos.yaml (pinned commit SHAs), questions/<repo>.yaml — committed
├── tests/
│   ├── unit/  integration/  security/
│   └── fixtures/sample_repo/ # tiny purpose-built repo incl. CRLF file, secret-bait files, symlink case
├── docs/
│   ├── architecture.md  security.md  evaluation.md
│   ├── decisions/        # ADRs (short: context, decision, consequences)
│   └── viva/             # concept notes, Q&A
├── results/              # curated final results (CSV/JSON/PNG) — committed; results/runs/ is gitignored
└── data/                 # gitignored contents: repos/ indexes/ uploads/ cache/ (with .gitkeep)
```

Deviations from your starting sketch, with reasons: `app/` → `src/copilot/` (src layout); `retrieval/` split from new `vectorstore/` (storing vs. searching are different responsibilities); new `parsing/` (shared by chunker, symbol lookup, and evaluation ground truth — avoids duplicated AST code); new `services/` (single entry point, makes FastAPI optional); new `benchmarks/` (ground truth is a first-class, versioned artefact); `security` lives inside `ingestion/` and `utils/paths.py` rather than a separate top-level package.

### Responsibility of each major component
- **ingestion**: turns a directory/zip into a list of *safe* `RepoFile` objects (relative POSIX path, language, text, sha256). Never reads a file it should skip; never follows a path out of the root.
- **parsing**: extracts the symbol table (functions, classes, methods, line ranges) from Python via `ast`; splits Markdown by headings. Pure functions, no I/O.
- **chunking**: `Chunker` protocol → `list[Chunk]`. Strategy A (line windows with overlap) and B (AST units with size-capped sub-splitting and file-header context) are both permanent and selectable by config, so Experiment 2 is a one-flag switch.
- **embeddings**: `Embedder.embed_documents / embed_query` returning float32 arrays; records model name and dimension.
- **vectorstore**: build/save/load/search. The manifest stores embedding model, dimension, chunker name+params, repo content hash; loading refuses a mismatch (prevents silently mixing incompatible vectors).
- **retrieval**: dense, BM25, hybrid (reciprocal-rank fusion), symbol lookup; all return `RetrievedChunk` with score and source retriever. Testable with no LLM.
- **llm**: `LLMClient.generate(messages, json_schema?) -> str`; providers are interchangeable; cache wraps any provider.
- **rag**: builds the labelled context within a token budget, prompts for JSON `{answer, citations:[C#], evidence_sufficient}`, validates and resolves citations.
- **agents**: LangGraph graph; nodes are thin wrappers over `rag` and `retrieval`.
- **evaluation**: loads benchmarks, runs configs, computes metrics, writes reproducible result files with config + git SHA + model IDs.
- **services**: `CopilotService.index(repo_path, strategy) / ask(question, error_text=None)`.

### Chunk metadata schema (the contract everything depends on)
`chunk_id, repo_id, file_path (relative, POSIX), language, chunk_type (baseline: line_window; structure-aware types such as function|method|class|module_header are added in Milestone 9), symbol_name, qualified_name (Class.method), parent_class, start_line, end_line (1-based, inclusive), content, content_sha256, token_estimate, chunking_strategy`

---

## 5. Milestone plan

Rules that apply to every milestone: implement only that milestone; run its tests; report with the agreed response format; suggest a commit message; STOP. Nothing is pushed without your explicit go-ahead. Improvements over your original list: benchmark ground truth and a retrieval Hit@k harness start in **M5** (not M10) so retrieval is measured from the first day; the lexical/hybrid retriever and AST usage finder are added (M5, M8) because of Truth #1; a plain-LLM baseline pipeline is built in M6 so Experiment 1 needs no rework.

### Milestone 0 — Architecture and planning
- **OBJECTIVE:** Agree what we build, with what, in what order.
- **WHAT WILL BE BUILT:** This document only.
- **KEY FILES:** `ARCHITECTURE_PLAN.md` (later copied to `docs/architecture.md`).
- **TESTS:** None.
- **ACCEPTANCE:** Approved on 2026-09-20 with the amendments in Section 12.

### Milestone 1 — Project/repository foundation
- **OBJECTIVE:** A clean, reproducible, secret-safe project skeleton.
- **WHAT WILL BE BUILT:** Project folder and approved structure; `git init`; `.gitignore`; `.env.example`; `pyproject.toml` + `uv.lock` + `.python-version`; README skeleton; `Settings` (pydantic-settings); `setup_logging()`; pytest + ruff config; CI workflow; one smoke test per infrastructure piece. Empty packages contain only `__init__.py` (no placeholder logic).
- **KEY FILES:** `pyproject.toml`, `.gitignore`, `.env.example`, `src/copilot/config/{settings,logging_setup}.py`, `tests/unit/test_settings.py`, `.github/workflows/ci.yml`.
- **TESTS:** Settings load defaults; env override works; missing required secret fails only when the LLM is actually used; logging does not print secret values; `.gitignore` audit test asserting `.env`, `data/`, `.venv` are ignored (`git check-ignore`).
- **ACCEPTANCE:** From a fresh clone, `uv sync` then `uv run pytest` and `uv run ruff check .` pass; `git ls-files` contains no `.env`, no `data/` content, no `.venv`; a secret-pattern scan of staged files is clean. GitHub step (only after local verification and explicit approval; repository will be public): create the repo, add remote via normal authenticated `gh`/git credentials (never a token in the URL), push; CI is green on GitHub.

### Milestone 2 — Safe repository ingestion
- **OBJECTIVE:** Turn a directory or zip into a list of safe, clean `RepoFile`s.
- **WHAT WILL BE BUILT:** Extension allowlist (`.py .js .jsx .ts .tsx .java .cpp .c .md .json .yaml .yml` + `.h .txt` optional); directory denylist (`.git node_modules dist build venv .venv __pycache__ .idea .vscode`); filename/pattern denylist (`.env*`, `*.pem`, `*.key`, `id_rsa*`, `*.p12`, `credentials*`, `.npmrc`, `.pypirc`, `*.sqlite`); size cap (default 500 KB, configurable); binary detection (NUL byte / UTF-8 decode failure); symlink handling (resolve and require `is_relative_to(root)`, never follow out); content-based secret scan (AWS/GitHub/Google key shapes, PEM headers, generic `api_key = "…"`) that skips the file and records the *reason only*, never the value; safe zip extraction (reject `..`/absolute paths, cap file count, total uncompressed size and compression ratio); `IngestionReport` (files seen/kept/skipped by reason). Deterministic sorted output; LF/CRLF normalised for line counting.
- **KEY FILES:** `ingestion/{discovery,filters,secrets,archive,loader}.py`, `models/schemas.py`, `utils/paths.py`.
- **TESTS:** `tests/security/`: fixture repo with bait `.env`, private-key file, oversized file, binary with `.py` extension, symlink escaping root, zip with `../evil`, zip bomb, secret embedded in an otherwise valid `.py`. `tests/unit/`: language detection, ordering determinism, CRLF.
- **ACCEPTANCE:** Every bait item is excluded with the correct reason; no secret string appears in logs or the report (asserted); path-traversal attempts raise `UnsafePathError`; running on a real repo prints a sane report; running twice yields identical output.

### Milestone 3 — Baseline chunking (Strategy A)
- **STATUS:** Implemented (baseline only) and corrected after review: Strategy A is structure-blind for *all* file types (Markdown included), so the Markdown heading chunker listed under WHAT WILL BE BUILT is **not** part of the baseline; it survives only as an isolated utility (`chunking/markdown_sections.py`) for a future structure-aware strategy. The chunk-id formula, line semantics, fragment and statistics contracts are in `docs/chunking.md`; `chunk_max_tokens` setting added; `ast` is reserved and unregistered (amendment 11).
- **OBJECTIVE:** A transparent, working baseline chunker with correct metadata.
- **WHAT WILL BE BUILT:** `Chunk` schema; line-window chunker (target N lines/tokens, overlap M, hard token cap); (Markdown heading chunker dropped from the baseline after review, see STATUS); deterministic `chunk_id`; `Chunker` protocol and registry keyed by strategy name; chunk statistics (count, mean/median/p95 tokens).
- **KEY FILES:** `chunking/{base,windows,line_chunker,registry,stats}.py`, `chunking/markdown_sections.py` (isolated), `models/chunk.py`, `utils/tokens.py`.
- **TESTS:** Invariant tests: for every chunk, `content == source_lines[start-1:end]`; no chunk exceeds the token cap; overlap correct; empty and one-line files; very long single line; unicode; CRLF; determinism; golden chunk list for the fixture repo.
- **ACCEPTANCE:** Invariants hold on the fixture repo and on one real repo; stats printed; strategy selectable by name from `Settings`.

### Milestone 4 — Embeddings and vector store
- **STATUS (rescoped after Milestone 3 review):** Milestone 4 was narrowed to *embedding model integration and tokenizer validation*. Implemented: `Embedder` interface, fastembed adapter, fake and stub-ONNX test paths, embedding-text representation, chunk-id/vector pairing, token-validation tooling (`docs/embeddings.md`). The real-model smoke test and estimated-vs-actual measurement were completed on native Windows (results in `docs/embeddings.md`): 768-d unit vectors, 8192-token limit, heuristic underestimates real Jina tokens (89.1% of chunks), no chunk near the limit (max 736). Chunk defaults unchanged; retrieval evaluation must compare estimated caps 512/768/1024 and raw vs prefixed text. The FAISS store, sidecar and manifest listed below move to the next milestone, before semantic retrieval.
- **OBJECTIVE:** Turn chunks into a persistent, searchable, *safe-to-reload* index.
- **WHAT WILL BE BUILT:** `Embedder` protocol; `FastEmbedEmbedder` (Jina code model default, bge-small alternative); `FakeEmbedder` (deterministic hash-based vectors for tests); batching and progress; `FaissStore` (`IndexFlatIP`, normalised vectors); `chunks.jsonl` sidecar; `manifest.json` (embedding model+dim, chunker name+params, repo content hash, created-at, library versions); index directory naming `data/indexes/<repo>__<strategy>__<embed>/`; timing of embed and build (feeds the indexing-time metric).
- **KEY FILES:** `embeddings/*`, `vectorstore/{base,faiss_store,manifest}.py`.
- **TESTS:** Save→load→search identical results; manifest mismatch (different model/dim/chunker) is refused; known-geometry test (hand-made unit vectors give expected cosine ordering); empty index; `live`-marked smoke test that downloads the real model, embeds 20 chunks, and checks dimension = 768.
- **ACCEPTANCE:** Index a real repo end-to-end from a script; reload in a fresh process; results match. If the Jina model fails on your Windows machine, fall back to bge-small and record why in an ADR.

### Milestone 5 (as delivered) — FAISS vector index and safe persistence
- **STATUS:** Implemented and tested with the deterministic fake embedder and real FAISS; the real-model (Jina) index smoke test is run by the user on Windows (commands in `docs/vector-index.md`). Not in scope and not implemented: semantic search/ranking API, BM25, LLM, RAG, the cap/representation experiment.
- **OBJECTIVE:** A deterministic, reloadable, refuse-if-incompatible vector index with an explicit vector <-> chunk mapping.
- **WHAT WAS BUILT:** `copilot.vectorstore`: `VectorIndex` (build/save/load/verify), FAISS confined to `faiss_backend.py`, `IndexFlatIP` on unit vectors, repository fingerprint, deterministic index id (hash of `IndexSpec`), `manifest.json` (schema, spec, provenance, counts, artifact checksums), `chunks.jsonl` sidecar (citation metadata, no source text), strict `IndexExpectation` compatibility checks, atomic staging-directory saves, build pipeline and `build/info/validate` CLI. Layout: `data/indexes/<index_id>/{manifest.json,index.faiss,chunks.jsonl}`.
- **KEY FILES:** `vectorstore/{index,manifest,records,fingerprint,validation,atomic,faiss_backend,builder,errors,__main__}.py`; contracts in `docs/vector-index.md`.
- **DEVIATION FROM THE ORIGINAL SKETCH (recorded, see amendment 14):** index directories are named by the deterministic `index_id`, not `<repo>__<strategy>__<embed>`; the `Embedder`/`VectorStore` protocol pair was reduced to one concrete `VectorIndex` behind a single FAISS module, since there is one backend and no second implementation to justify a protocol.

### Milestone 5b — Semantic Retrieval & Evaluation (as delivered)
- **STATUS:** Implemented and tested. Real-model measurements are in `docs/evaluation.md`. Delivered: `VectorIndex.search` (deterministic ordering, FAISS `-1` filtering), `copilot.retrieval` (`Retriever`, `retrieve`, `RetrievalResult`, source re-materialisation with fingerprint and per-chunk hash verification, CLI), `copilot.evaluation` (JSONL benchmark with hash-pinned regions, Hit@k/MRR/mean-lines-per-hit, cap x representation matrix runner, CLI). Design in `docs/retrieval.md` and `docs/evaluation.md`.
- **DEVIATIONS (recorded):** the index stores metadata only, so source text is re-materialised at search time and verified by hash; the benchmark is this repository at a pinned commit and is explicitly **not independent**; benchmark ground truth is file + line regions, never chunk ids. **Deferred, not implemented:** `LexicalRetriever` (BM25), `HybridRetriever` (RRF), language/path filters, an independent benchmark repository.
- **NEXT:** Milestone 6 (RAG) may start only after the content-based secret scanner gate (amendment 10).

### Milestone 5 (original numbering) — Semantic retrieval + benchmark v0 (partly superseded by 5b above)
- **STATUS:** Semantic part delivered as Milestone 5b; the lexical and hybrid retrievers below remain not started. Later milestone numbers are not renumbered here; the LLM gate (amendment 10) is attached to "the first external-LLM call", whatever its number.
- **OBJECTIVE:** Retrieval that is independently measurable before any LLM exists.
- **WHAT WILL BE BUILT:** `SemanticRetriever` (top-k, scores, optional language/path-prefix filter); `LexicalRetriever` (BM25, code-aware tokenizer); `HybridRetriever` (reciprocal-rank fusion); `RetrievedChunk`; benchmark file format (`question, category, expected_files, expected_symbols, answerable`); pinned benchmark repos (`benchmarks/repos.yaml` with commit SHAs); first 15–20 hand-written questions; retrieval evaluation harness computing Hit@1/3/5 and MRR.
- **KEY FILES:** `retrieval/{semantic,lexical,hybrid}.py`, `evaluation/{benchmark,metrics}.py`, `benchmarks/questions/*.yaml`, `scripts/run_eval.py`.
- **TESTS:** Retrieval on fixture repo returns the planted file for planted queries; scores sorted; k larger than corpus; BM25 tokenizer splits `getUserById` and `get_user_by_id`; metric functions on hand-computed toy cases; RRF on hand-computed ranks.
- **ACCEPTANCE:** One command prints Hit@k/MRR for dense, BM25 and hybrid on the v0 benchmark, with question counts. **These are real measurements, whatever they turn out to be.**

### Milestone 6 — Basic repository-aware RAG (+ plain-LLM baseline)
- **OBJECTIVE:** Grounded answers with resolvable citations, and the baseline we will compare against.
- **WHAT WILL BE BUILT:** `LLMClient` protocol; `GeminiClient` (google-genai), `OllamaClient`, `FakeLLM`; response cache; context builder (chunks labelled `[C1]..`, token budget, dedup/merge of adjacent chunks); prompt requiring JSON `{answer, citations, evidence_sufficient, missing_information}`; citation resolver (chunk ID → file, symbol, lines); insufficient-evidence path; `plain_baseline.py` (same LLM, no retrieval, same output schema); CLI `scripts/ask.py`. Final Gemini model chosen here after you check AI Studio quotas.
- **KEY FILES:** `llm/*`, `rag/{prompts,context_builder,generator,citations,pipeline,plain_baseline}.py`.
- **TESTS:** With `FakeLLM`: unknown citation ID is dropped and flagged; zero valid citations ⇒ marked ungrounded; malformed JSON handled with one repair retry then a clean error; context never exceeds token budget; cache hit avoids a second call; API key never in logs. `live` test: real answer on fixture question has ≥1 valid citation.
- **GATE (mandatory, amendment 10):** the content-based secret scanner must exist, be tested and be wired in *before* any retrieved repository text is sent to a hosted LLM. `FakeLLM` and Ollama-local development may proceed without it; the first `GeminiClient` call with repository context may not.
- **ACCEPTANCE:** Ask 5 fixture questions and 2 unanswerable ones: answerable ones cite real files/lines; unanswerable ones return "insufficient repository evidence" instead of inventing; a test proves no citation can reference a chunk that was not retrieved.

### Milestone 7 — Streamlit MVP
- **OBJECTIVE:** A usable interface over the service layer.
- **WHAT WILL BE BUILT:** `CopilotService` façade; Streamlit app: repo path/zip upload → Index (progress, files processed/skipped, chunks, index status) → question box → answer, evidence snippets, files, functions/classes, line ranges, insufficient-evidence banner; sidebar (top-k, chunking strategy, model info). Indexing cached in session state; errors shown, never stack traces with paths/secrets.
- **KEY FILES:** `services/copilot_service.py`, `ui/streamlit_app.py`, `ui/components/*`.
- **TESTS:** Service tests with fakes; Streamlit `AppTest` smoke (loads, indexes fixture, answers with `FakeLLM`); invalid path and oversized upload rejected in UI.
- **ACCEPTANCE:** Manual walkthrough on a sample repo works end-to-end; `ui/` contains no retrieval/LLM logic.

### Milestone 8 — LangGraph agentic workflow
- **OBJECTIVE:** Make routing, state, retry and refusal explicit and justified.
- **WHAT WILL BE BUILT:** `GraphState`; rule-based intent router with LLM fallback; per-intent retrieval configs; deterministic-first evidence grader; one-shot query rewrite retry; refusal node; citation validation node; AST `find_usages` tool for "where is X used" (Python); route trace exposed to UI; graph diagram exported.
- **KEY FILES:** `agents/{state,router,nodes,graph}.py`, `retrieval/symbol_lookup.py`, `docs/diagrams/`.
- **TESTS:** ≥25 labelled routing queries with a confusion table; graph path tests with `FakeLLM` (retry fires exactly once; refusal path; max-iteration guard); usage finder on fixture with method calls, aliases, and false-positive cases (same name, different symbol).
- **ACCEPTANCE:** Routing accuracy measured and reported on the labelled set (no invented target); every node's necessity is written in an ADR; UI shows route and retry count. If the graph does not beat the linear RAG pipeline on the benchmark, we report that honestly.

### Milestone 9 — Structure-aware chunking (Strategy B)
- **OBJECTIVE:** AST-based chunking that keeps functions/classes whole, without breaking Strategy A.
- **WHAT WILL BE BUILT:** `ast_chunker` for Python: one chunk per function/method (decorators included in the range), class chunks as signature + docstring + method index (bodies live in method chunks — no duplicate giant class chunk), a module-header chunk (imports, module docstring, constants), oversize units split at statement boundaries while keeping `symbol_name`; syntax-error files fall back to Strategy A with a logged reason. **Control:** both strategies embed the same text-header format (`file path` line) so path information is not a hidden advantage; header on/off is a flag for an ablation.
- **KEY FILES:** `parsing/python_ast.py`, `chunking/ast_chunker.py`.
- **TESTS:** Golden outputs; exact start/end lines; nested functions, async functions, decorators, properties, lambdas, `if __name__` blocks, very long functions, unicode, syntax-error fallback; coverage invariant (every code line is in some chunk); same `Chunker` protocol so all retrieval tests run against both strategies.
- **ACCEPTANCE:** Both strategies produce indexes from one flag; chunk-size distribution comparison printed (needed for Experiment 2 controls).

### Milestone 10 — Chunking experiment (Experiment 2)
- **OBJECTIVE:** A defensible measured comparison of Strategy A vs B.
- **WHAT WILL BE BUILT:** `exp_chunking.py`; runs both strategies on ≥2 repos with identical embedder, k, filters, and question set; metrics: Hit@1/3/5, MRR, expected-file recall, expected-symbol hit (line-overlap with AST ground truth, so it is fair to baseline chunks that carry no symbol metadata), indexing time, retrieval latency (median and p95 over repeated warm runs), chunk-count and size stats; per-category breakdown; header ablation; results written with config + git SHA.
- **KEY FILES:** `evaluation/exp_chunking.py`, `scripts/run_eval.py`, `results/`.
- **TESTS:** Metric unit tests; harness runs on fixture repo with fakes and produces the expected file layout; determinism test (same inputs ⇒ identical metrics).
- **ACCEPTANCE:** One command regenerates every table/figure from raw result files; report lists sample sizes and limitations; if A ≈ B, we say so.

### Milestone 11 — Plain LLM vs RAG (Experiment 1)
- **OBJECTIVE:** Measure whether repository awareness improves answers.
- **WHAT WILL BE BUILT:** `exp_rag_vs_plain.py`; configurations: plain LLM, RAG (best retriever from M5/M10), optional LangGraph agentic; benchmark expanded to ≥30 questions per repo incl. ~20% unanswerable; metrics per Section 7; judge module with a documented judge model, plus manual audit of a random 20% sample and reported agreement; cached LLM calls, temperature 0.
- **KEY FILES:** `evaluation/{exp_rag_vs_plain,judge,report}.py`.
- **TESTS:** Programmatic citation checker and hallucinated-path detector on crafted answers; refusal metrics on hand-made cases; judge parsing with `FakeLLM`; report generation.
- **ACCEPTANCE:** Every number in the report traces to a raw result file; contamination note included; at least one repo the model cannot plausibly have memorised.

### Milestone 12 — Evidence-based debugging
- **OBJECTIVE:** Debugging help that separates proof from guesses.
- **WHAT WILL BE BUILT:** Python traceback parser (multi-frame, chained exceptions, Windows paths); frame→repo file/line mapping (suffix matching, unmatched frames reported as such); enclosing-function chunk fetch; callers via usage finder; prompt with fixed sections **REPOSITORY EVIDENCE** and **POSSIBLE CAUSES (HYPOTHESES)**; "which files should I investigate" mode using retrieval only; explicit statement that code was not executed.
- **KEY FILES:** `agents/traceback_parser.py`, debugging prompt/workflow nodes, `benchmarks/questions/debug_*.yaml`.
- **TESTS:** Parser fixtures (nested, chained, syntax errors, non-Python text); frames pointing at non-existent files are flagged, not invented; both sections always present; seeded-bug benchmark (≥8 bugs injected into a fixture or own repo).
- **ACCEPTANCE:** Reports the honest rate at which the true buggy file appears in the investigated set; never claims a fix is verified.

### Milestone 13 — Security and robustness
- **OBJECTIVE:** Close the gaps Section 11 lists and prove it with tests.
- **WHAT WILL BE BUILT:** Written threat model; secret redaction in displayed snippets; prompt-injection hardening (delimiting untrusted context, instruction hierarchy, output-schema enforcement); timeouts, retries with backoff, quota handling; upload/repo size limits; dependency audit (`pip-audit`); log-leak tests; `.gitignore` audit.
- **KEY FILES:** `docs/security.md`, `tests/security/*`, hardening in `rag/` and `ingestion/`.
- **TESTS:** Injection fixture files ("ignore previous instructions…") cannot change output schema or trigger tool use (prompt-construction tests are deterministic; a `live` sample documents actual model behaviour); huge file/repo; malformed archive; unicode tricks in paths; 429/timeout simulation.
- **ACCEPTANCE:** Security suite green; residual risks documented honestly (LLM injection cannot be fully eliminated).

### Milestone 14 — Final UI
- **OBJECTIVE:** A demo-quality interface.
- **WHAT WILL BE BUILT:** Tabs (Ask, Evidence, Index/Stats, Route trace, Evaluation results); code viewer with highlighted line ranges; insufficient-evidence badge; history; markdown export; results page reading `results/` files only.
- **TESTS:** Extended `AppTest` flows; empty/invalid states.
- **ACCEPTANCE:** A scripted 8-question demo runs cleanly; nothing on screen is hard-coded or fabricated.

### Milestone 15 — Complete testing
- **OBJECTIVE:** Confidence and a regression net.
- **WHAT WILL BE BUILT:** Coverage report; gap-filling tests; end-to-end tests with fakes; documented `live` suite; regression test per bug found; manual test checklist.
- **ACCEPTANCE:** CI green; coverage on core packages (ingestion, chunking, retrieval, rag, agents) at or above a threshold you choose (I suggest 80% [Guessing]); all security tests pass.

### Milestone 16 — Documentation
- **OBJECTIVE:** Anyone can run, understand and evaluate the project.
- **WHAT WILL BE BUILT:** README (setup, usage, architecture, results); `docs/architecture.md`; ADRs; `docs/evaluation.md`; `docs/security.md`; limitations; diagrams.
- **ACCEPTANCE:** A clean-machine setup following only the README works; every quoted number links to a file in `results/`.

### Milestone 17 — Deployment
- **OBJECTIVE:** One documented, reproducible way to run it beyond your laptop.
- **WHAT WILL BE BUILT:** Primary: documented local run. Optional: Dockerfile; hosted demo on Streamlit Community Cloud or similar using Gemini + a **public sample repo only**. Ollama and a 0.6 GB embedding model may not fit a free host [Guessing], so the hosted path may need the smaller embedder.
- **ACCEPTANCE:** Chosen path works from a clean environment; secrets only via the platform's secret store; image/repo scanned for secrets.

### Milestone 18 — Viva/demo preparation
- **OBJECTIVE:** You can defend every decision live.
- **WHAT WILL BE BUILT:** Demo script; offline demo mode from cached LLM responses (in case Wi-Fi or quota fails); backup screen recording; viva Q&A; one-page architecture; results summary; limitations slide.
- **ACCEPTANCE:** You explain each component unaided; timed dry run completed.

---

## 6. Testing strategy

- **Pyramid.** Many fast unit tests (chunk invariants, metrics, path safety, citation validation, traceback parsing); fewer integration tests (index fixture repo → retrieve → answer with fakes); a small `live` suite (real embedder, real LLM) that is skipped in CI and run manually.
- **Fakes make the LLM path deterministic:** `FakeEmbedder`, `FakeLLM` (scripted responses), so CI needs no API key, no model download, no network.
- **Invariant/property tests** for chunking (content equals source slice; no chunk exceeds cap; coverage) because line-number correctness is the foundation of every citation.
- **Security tests** are a first-class directory (`tests/security/`), built on deliberately hostile fixtures.
- **Golden tests** for chunker output and prompt construction; changes must be reviewed intentionally.
- **Evaluation is not testing.** Benchmarks measure quality; pytest checks correctness of code. Metric functions are unit-tested against hand-computed examples.
- **LLM non-determinism:** tests never assert on free-form model text; they assert on structure (valid JSON, valid citations, refusal flag) and on prompt contents.
- **Windows specifics** tested: CRLF fixtures, backslash paths in tracebacks, path normalisation to POSIX in metadata.

## 7. Evaluation strategy (overview)

### Benchmark construction
- **Repos:** (a) this project itself (cannot be memorised by the LLM), (b) one small/medium Python repo the model is unlikely to know, (c) optionally one popular repo, reported separately as a contamination check. Pin each to a commit SHA in `benchmarks/repos.yaml`.
- **Questions:** ≥30 per repo across categories (search, explain, architecture/docs, usage, debugging) plus ~20% **unanswerable** (feature that does not exist). Each stores expected files, expected symbols (where applicable), key facts for correctness, and `answerable`.
- **Labelling discipline:** write questions and ground truth *before* seeing retrieval output; have a second person spot-check a sample; version the files.

### Experiment 1 — Plain LLM vs RAG
Same LLM, temperature 0, same questions.
- **Correctness:** rubric key-fact checklist (0 / partial / full), scored programmatically where key facts are keyword-checkable, otherwise by judge; manual audit of 20% with reported agreement.
- **Groundedness/faithfulness:** does the cited chunk actually support each claim (judge + audit).
- **Citation accuracy:** programmatic — cited file exists, line range within file, range overlaps an expected/relevant chunk.
- **Hallucination:** programmatic detection of file paths, symbols, and dependency names in the answer that do not exist in the repo; judge as secondary.
- **Insufficient-evidence handling:** refusal rate on unanswerable questions; false-refusal rate on answerable ones.
- Also latency and tokens. The plain baseline has no chunk IDs, so its file/function mentions are checked directly against the repo.

### Experiment 2 — Baseline vs structure-aware chunking
Retrieval only, no LLM.
- **Metrics:** Hit@1/3/5, MRR, expected-file recall, expected-symbol hit (line-overlap with AST ground truth), indexing time, retrieval latency (median/p95, warm, repeated), chunk count and size distribution.
- **Controls:** identical embedder, k, question set, dedup rule, and header format; report the chunk-size distributions because structure-aware chunks are not uniform. Chunks over the embedder's effective length are split, not truncated.
- **Reporting:** raw counts with percentages, per-category breakdown, no significance claims unless a test was actually run (e.g., paired bootstrap over questions).

### Reproducibility
Every run writes config, git SHA, model IDs, library versions, seed, and timestamp next to its metrics; tables and figures are generated from those files by script, never typed by hand.

---

## 8. Five major technical risks and mitigations

| # | Risk | Why it is real | Mitigation |
|---|---|---|---|
| 1 | **Dense retrieval misses identifier, usage and flow questions** ("where is `get_db` used", "endpoint → database flow") | Embeddings capture topic, not exact names or call graphs [Likely] | BM25 + RRF hybrid (M5); AST usage finder (M8); multi-hop flow questions are answered from *retrieved* evidence only and the answer states which links in the chain are missing; measure each retriever as an ablation |
| 2 | **Evaluation validity** (memorised popular repos, tiny benchmark, LLM-judge bias, non-determinism) | The two experiments are the academic core; weak methodology sinks them | Own/unknown repo included; ≥30 questions per repo with ~20% unanswerable; ground truth written before viewing outputs; programmatic metrics wherever possible; judge ≠ answerer where feasible; 20% manual audit; temperature 0 + response cache; raw counts and stated sample sizes; no significance claims without a test |
| 3 | **Chunking experiment confounds** (size, header text, embedder context limit, dedup) | Structure-aware chunks are variable-size; an unfair setup produces a meaningless winner | Same embedder/k/header for both strategies; oversize units split not truncated; report size distributions; header on/off ablation; symbol-hit measured by line overlap with AST ground truth so baseline chunks are judged fairly |
| 4 | **LLM provider instability, quota and privacy** | Gemini model IDs already churn (2.0 deprecated); free-tier quotas are only visible in AI Studio; free-tier data may be used by the provider to improve products [Likely — check current terms before indexing anything you do not own] | Model IDs in config; `LLMClient` abstraction with Gemini + Ollama + Fake; on-disk response cache; backoff/retry; index only public or your own repos; offline demo mode from cache (M18) |
| 5 | **Scope and schedule** (19 milestones, multi-language temptation, deployment surprises) | Solo student project with a viva | Python-first; tree-sitter multi-language only as a stretch after M10; cut line in Section 9; each milestone independently demonstrable; UI kept thin |

Additional watch-items: Windows path/CRLF handling (dedicated tests); embedding model download size (~0.64 GB) and speed on CPU (measured in M4, fallback bge-small); prompt injection from repo content (M13).

---

## 9. Recommended development order and cut line

What exists: nothing. What to keep: nothing (only your project instructions and this plan). What to change: nothing. What to create: everything below.

Order: **M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7 → M8 → M9 → M10 → M11 → M12 → M13 → M14 → M15 → M16 → M17 → M18.** I kept your order. One dependency note: Experiment 2 (M10) needs only M2–M5 and M9, not the LLM or the graph, so if the LangGraph work (M8) runs long, M9→M10 can be pulled ahead of it without rework. [Likely]

**Cut line (protect in this order):**
1. **Must-have core (M1–M7):** safe ingestion, chunking, embeddings, retrieval, grounded RAG, Streamlit MVP.
2. **Must-have academic (M8–M11):** LangGraph workflow, structure-aware chunking, both experiments.
3. **Should-have (M12, M13, M15, M16):** debugging, security hardening, testing, docs.
4. **Nice-to-have (M14, M17, extra languages):** final UI polish, hosted deployment, tree-sitter.
M18 (viva prep) is non-negotiable and needs its own reserved time; do not let it be squeezed to zero.

---

## 10. What you need to understand for the viva

For each: *what it is / why we need it / how we use it / what happens without it.*
- **RAG:** retrieve evidence first, then generate from it. We use it so answers come from the repository; without it the LLM guesses from general knowledge.
- **Embeddings and cosine similarity:** text → vector; similar meaning → nearby vectors. We normalise vectors and use inner product, which equals cosine; you should be able to compute it by hand for two 3-d vectors.
- **Vector database vs FAISS:** FAISS is a similarity-search library (exact flat index here); it stores no metadata, so our `chunks.jsonl` sidecar and manifest do that job. Why exact search: deterministic experiments.
- **Chunking:** why whole files are too big and too vague; overlap; token limits; why line-window chunks cut functions in half and why AST chunks do not.
- **AST:** the parse tree of source code; Python's `ast` gives function/class nodes with `lineno`/`end_lineno`; parsing does not execute code.
- **Metadata and citations:** why file/symbol/lines are stored per chunk, and the design where the LLM cites chunk IDs and our code resolves them.
- **Top-k, scores, thresholds:** why cosine scores are not calibrated probabilities and why the insufficiency threshold is tuned on data.
- **Hybrid retrieval / BM25 / RRF:** lexical vs semantic matching; why fusion by rank rather than by raw score.
- **Context window and token budget:** what happens when retrieved text exceeds it and how we trim.
- **Grounding vs hallucination:** difference between *citation exists*, *citation supports claim* (faithfulness), and *answer is correct*.
- **LangGraph:** state, nodes, conditional edges, the retry loop; why routing and a grader earn their place and what would be lost with a linear pipeline.
- **Agentic vs deterministic:** which parts are LLM decisions and which are plain code, and why.
- **Evaluation:** Hit@k, MRR, recall; controls; contamination; sample-size limits; LLM-judge bias; why unanswerable questions are included.
- **Security:** allowlist over blocklist, path traversal, zip-slip/zip-bomb, secret scanning, prompt injection from repository content, API-key handling.
- **Trade-offs you must be able to defend:** FAISS vs Chroma, fastembed vs sentence-transformers, no LangChain, Streamlit without FastAPI, Gemini vs local model.
- **Limitations you must state yourself before an examiner does:** Python-first structure awareness, small benchmark, no code execution, flows only as good as retrieval.

---

## 11. Security considerations

**Ingestion**
- Allowlist of extensions; directory and filename denylists; never index `.env*`, private keys, certificates, token files, credential files, or databases.
- Content-based secret scan; matching files are skipped and only the *reason* is logged, never the secret.
- Size cap per file and total; binary detection; UTF-8 decode failure ⇒ skip.
- Path safety: resolve every path, require it to stay inside the repo root; symlinks never followed outside the root; relative POSIX paths in metadata; no absolute host paths shown in the UI.
- Archives: reject `..`/absolute entries (zip-slip), cap entries/total size/compression ratio (zip bomb), extract to a fresh temp dir.
- Never execute repository code, run its tests, or install its dependencies. `ast.parse` is parse-only; parser errors and recursion limits are caught.
- GitHub URL ingestion (optional extension): HTTPS only, shallow clone into `data/repos/`, hooks/submodules disabled, treated exactly like a local folder.

**LLM boundary**
- Anything indexed may be sent to an external API. Rule: only public or your own repos with the Gemini free tier; Ollama for anything sensitive.
- Prompt injection: repository text is delimited as untrusted data, the system prompt states it may contain instructions to ignore, output must match a JSON schema, no tools with side effects are exposed to the model, citations are validated by code. Cannot be fully eliminated; residual risk documented.
- Secrets in retrieved snippets are redacted before display and before prompting.

**Secrets and Git**
- API keys only via environment/`.env`; `.env.example` holds placeholders; `Settings` never logs values.
- `.gitignore` will exclude: `.env`, `.env.*` (except `.env.example`), `.venv/`, `venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `.coverage`, `htmlcov/`, `data/*` (except `.gitkeep`) including `data/indexes/`, `data/repos/`, `data/uploads/`, `data/cache/`, `results/runs/`, `*.faiss`, `*.index`, IDE folders (`.idea/`, `.vscode/` except shared settings if any), `.DS_Store`, `Thumbs.db`, `*.log`, model caches (`.fastembed_cache/`, `models/`). Curated final results in `results/` are committed.
- Never put a token in a remote URL; use authenticated `gh` or the credential manager.
- Pre-commit scan of staged files for secret patterns in M1; `git ls-files` audit before the first push.

---

## 12. Approved scope amendments (2026-09-20)

1. **Hybrid retrieval (dense + BM25)** stays in the architecture but is implemented incrementally; basic semantic retrieval must work first.
2. **AST usage finder** ("where is this function used?") stays planned but is not an MVP blocker and must not delay the core RAG pipeline.
3. **Ollama** stays behind the LLM abstraction as an optional/fallback provider; **Gemini** is the primary provider initially.
4. **Approved stack:** Python 3.12, uv + pyproject.toml, `src/copilot/` layout, LangGraph, no unnecessary direct LangChain dependency in V1, Gemini via `google-genai`, modular LLM interface, fastembed with `jina-embeddings-v2-base-code` (subject to the Milestone 4 compatibility test) and `bge-small` fallback, FAISS `IndexFlatIP`, sidecar metadata + manifest, Streamlit, no FastAPI for the MVP, pytest, ruff, pydantic-settings, standard logging, Python `ast` for later structure-aware parsing.
5. **Benchmarks:** this project's own repository once sufficiently developed, plus one additional unfamiliar public repository selected later. A famous framework (Django, Flask) is not used as the primary benchmark.
6. **Evaluation stays scientifically modest:** raw results and metrics are reported; no claim of statistical significance unless a test was actually run.
7. **The 19-milestone roadmap is a planning framework**, not a requirement that every optional feature is completed. Priority order: safe ingestion, baseline chunking, embeddings, semantic retrieval, basic RAG, citations/grounding, Streamlit, evaluation, LangGraph/advanced retrieval, additional improvements.
8. **GitHub repository** `agentic-rag-code-copilot` will be **public**; nothing is created or pushed until the local foundation is verified and explicitly approved.
9. **No unverified model IDs as defaults (added after Milestone 1 review).** Gemini model names in Sections 1 and 3 are research notes, not configuration. `COPILOT_LLM_MODEL` is unset by default; the concrete model is chosen and verified in Milestone 6 against the provider's current docs and the account's quota.
10. **Mandatory pre-Milestone-6 security gate (added after Milestone 2 review).** A lightweight *content-based* secret scanner (API-key patterns, private-key blocks, high-entropy assignments in otherwise-accepted files) is **deferred from Milestone 2 but mandatory before any repository context is sent to an external LLM**. Milestone 6 must not call a hosted LLM with retrieved repository text until the scanner exists, is tested, and is wired into either ingestion or context building. Ingestion scope is not expanded before then unless Milestone 3 reveals an actual defect.
11. **Chunking strategies stay independent (added after Milestone 2 review).** The baseline line-window strategy (Milestone 3) and the structure-aware AST strategy (Milestone 9) are separate `Chunker` implementations behind a registry; Milestone 3 implements only the baseline, and the `ast` strategy name is reserved but deliberately unregistered until Milestone 9 so the experiment compares two genuinely independent implementations.
12. **Milestone 4 rescoped; vector-store persistence deferred (added after Milestone 3 review).** Embedding integration and tokenizer validation are done first; FAISS persistence (`IndexFlatIP`, `chunks.jsonl`, `manifest.json`) follows. Vectors are L2-normalised in the embedding layer. Chunk-size defaults are unchanged (decision D: defer tuning to retrieval evaluation; compare at least cap 512 vs 1024 in Milestone 5).
13. **Chunk cap and embedding representation are decided by retrieval evaluation (added after Milestone 4 validation).** The 512-estimated-token cap stays because there is no safety issue (largest real embedded chunk 736 of 8192 tokens) and model capacity is not a reason to change it. The retrieval/evaluation milestone evaluates caps 512, 768 and 1024 (estimated tokens) and `prefixed` vs `raw` embedding text; the prefixed text is the current default with unproven benefit.
14. **Vector index delivered as its own milestone (added at Milestone 5).** FAISS `IndexFlatIP` persistence, the repository fingerprint, the deterministic index id, the manifest, strict compatibility checks and atomic saves are implemented (`docs/vector-index.md`). Indexes are stored by `index_id`, so the planned cap (512/768/1024) and representation (`prefixed`/`raw`) variants coexist. Semantic retrieval, BM25 and the evaluation harness remain the next task; the cap/representation experiment is recorded, not run.
