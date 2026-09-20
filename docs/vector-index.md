# Vector index (Milestone 5): FAISS index and safe persistence

Status: implemented and tested with a deterministic fake embedder and real FAISS. The real-model
(Jina) index smoke test is run by the user on Windows (commands at the end).

**Scope.** This milestone builds, saves, loads and validates an index. It does **not** rank results
for a question: there is no query-embedding-plus-search service, no top-k API, no BM25, no LLM.
`VectorIndex` has one private search primitive (`_search_positions`) that exists only to prove that
serialisation preserved the vectors.

## 1. What a vector index is, and why we need one

An embedding model turns text into a vector (768 numbers for our Jina model) so that texts with
similar meaning get vectors pointing in similar directions. A **vector index** stores one vector
per chunk and can find the stored vectors closest to a query vector without comparing text.

* **In our implementation:** one vector per chunk, in a FAISS index, plus a mapping from vector
  position to chunk id and citation metadata.
* **Without it:** every question would re-embed the whole repository (minutes on a CPU) or scan
  files with keyword search only, which cannot match "where is the database connection configured?"
  to code that never uses those words.

## 2. Why FAISS, and why `IndexFlatIP`

FAISS is a similarity-search *library*: it stores vectors and searches them. It stores no metadata
(no file names, no line numbers), which is why we keep our own mapping (section 4).

`IndexFlatIP` is **exact** (brute-force) search by inner product. We chose it because:

* **Exactness makes experiments defensible.** Approximate indexes (HNSW, IVF) can miss the true
  nearest neighbour; that noise would contaminate the planned Hit@k comparisons between chunking
  strategies. With an exact index any retrieval difference comes from chunking or embeddings.
* **Repositories are small.** Tens of thousands of 768-d vectors are searched in milliseconds by
  brute force; approximation would buy speed we do not need.
* **It is explainable.** Ranking by inner product on unit vectors is ranking by cosine similarity.

Approximate indexes are deliberately not used. They would be reconsidered only if a repository were
large enough for exact search to be too slow (not the case for this project).

### Cosine similarity vs inner product

```
cosine(a, b) = (a . b) / (|a| * |b|)          # angle between the vectors, ignores length
inner(a, b)  =  a . b                          # what IndexFlatIP computes
```

When `|a| = |b| = 1` the denominator is 1, so `cosine(a, b) = inner(a, b)`. The embedding layer
(Milestone 4) L2-normalises every vector, and Windows validation confirmed norms of 1.000000.
The index therefore needs no extra step, and `build` **refuses** vectors whose norm deviates from 1
by more than 1e-3, because on un-normalised vectors inner product is *not* cosine (long vectors would
win regardless of meaning). Without that check a silently un-normalised model would corrupt every ranking.

## 3. Build pipeline

```
Repository -> safe ingestion -> baseline chunking -> canonical order
           -> embedding text (prefixed or raw) -> embeddings (Jina, 768-d, unit length)
           -> consistency checks -> FAISS IndexFlatIP -> atomic save (staging dir, manifest last)
```

Implemented in `copilot.vectorstore.build_repository_index`. Nothing leaves the machine. The only
network access is the first-time download of the embedding model. Empty inputs fail *before* the model is loaded.

Default configuration (unchanged by this milestone): 60-line windows, 10 overlap, 512
estimated-token cap, `prefixed` representation, `jinaai/jina-embeddings-v2-base-code`.

## 4. Vector position <-> chunk id <-> chunk metadata

FAISS knows only "vector number `i`". The sidecar `chunks.jsonl` makes the link explicit:
**line `i` (0-based) of `chunks.jsonl` describes FAISS vector `i`.**

Each line is a `ChunkRecord`:

| Field | Meaning |
|---|---|
| `position` | row in the FAISS index (equals the line number; verified on load) |
| `chunk_id` | deterministic chunk id (`docs/chunking.md`) |
| `file_path`, `language` | citation metadata (repository-relative POSIX path) |
| `chunk_type`, `chunk_index` | kind of chunk and its 0-based position within the file |
| `start_line`, `end_line` | 1-based inclusive line range |
| `source_sha256`, `content_sha256` | hashes of the source file bytes and of the chunk text |
| `token_estimate` | heuristic token estimate |
| `fragment_index`, `fragment_count` | set only for a piece of an over-long line |
| `symbol_name`, `qualified_name`, `parent_class` | reserved for structure-aware chunks (null now) |

Repository name and chunking strategy/parameters are identical for all records, so they are stored
once in the manifest.

**No source text is stored in the sidecar.** The chunk text is recoverable because chunking is
deterministic: a future retriever re-ingests the repository, re-chunks it, looks the chunk up by
`chunk_id` and checks `content_sha256`. If the repository changed, the fingerprint check already
refused the index.

### Vector ordering contract

Vector `i` belongs to the `i`-th chunk in **canonical order**: ascending `(file_path, chunk_index)`.

* `file_path` is compared as a Python string (Unicode code points), identical on every platform
  (so `B.py` < `a.py` < `a/b.py`); `chunk_index` is compared numerically.
* The pair is unique per chunk, so the order is total and never depends on directory-traversal order.
* The pipeline sorts explicitly; `VectorIndex.build` *verifies* the order and raises `IndexBuildError`
  instead of silently reordering, so a caller cannot pair a vector with the wrong chunk.

## 5. Repository fingerprint

Answers "what accepted source state was indexed?". `FINGERPRINT_SCHEMA = "repo-fingerprint/1"`:

```
entries     = sorted( [relative_path, sha256(raw file bytes), sha256(normalised text), language]
                      for every accepted file )
fingerprint = SHA-256( json.dumps([schema, entries], ensure_ascii=True, separators=(",", ":")) )
```

* No absolute path, timestamp or random value: a copy of the repository elsewhere has the same fingerprint.
* Input order is irrelevant (entries are sorted).
* Both hashes are included: the raw-bytes hash feeds chunk ids; the normalised-text hash is what the
  chunkers actually read, so a change in decoding or newline handling is detected even when bytes are unchanged.
* `language` is included because the default embedding text contains it.
* The repository *name* is not part of it (it is a separate field of the index identity).
* Only accepted files count. Skipped files (secrets, binaries...) never influence it.
* Any edit to any accepted file, an added, removed or renamed file, or a language-mapping change gives a new fingerprint.

## 6. Deterministic index id

`INDEX_ID_SCHEMA = "index-id/1"`:

```
payload  = {"schema": "index-id/1", "spec": IndexSpec as a JSON dict}
index_id = SHA-256( json.dumps(payload, ensure_ascii=True, sort_keys=True,
                               separators=(",", ":")) )[:16]        # 16 hex characters
```

`IndexSpec` holds exactly the compatibility-relevant state: repository name, repository
fingerprint, chunking strategy + version + parameters, chunk-id schema, embedding model id,
dimension, normalisation flag, text style, representation version, index type and metric.
Same state gives the same id; changing any of them changes it. Timestamps, runtime versions, host
paths and randomness are excluded (they are provenance, stored in the manifest but not hashed).
Adding a field to `IndexSpec` deliberately changes every id.

Consequence: indexes for different chunk caps or text styles get **different ids and coexist** in
`data/indexes/`, which is what the planned 512/768/1024 and raw/prefixed experiment needs.

## 7. Persistence layout

```
data/indexes/<index_id>/
    manifest.json    compatibility and build record (JSON, human-readable)
    index.faiss      FAISS IndexFlatIP, produced by faiss.serialize_index
    chunks.jsonl     position -> chunk id and citation metadata (one JSON object per line)

data/cache/models/   downloaded embedding model files (reusable runtime dependency)
```

Model cache and indexes are separate on purpose: the model is a reusable dependency that costs
0.64 GB to fetch; an index is a generated artifact of one repository and can be rebuilt. Both live
under `data/`, which is git-ignored (`/data/*`, plus `*.faiss`). Nothing is pickled: only JSON,
JSONL and FAISS's own format. All files are written in binary mode with `\n` line ends and UTF-8, so bytes
(and checksums) are identical on Windows and Linux.

## 8. Manifest schema (`vector-index-manifest/1`)

| Key | Content |
|---|---|
| `schema_version` | manifest layout version |
| `index_id` | recomputed from `spec` on load; a mismatch means the manifest was edited |
| `spec` | the identity fields listed in section 6 |
| `provenance` | `created_at` (UTC), `source_file_count`, `ingestion_truncated`, `embedding_runtime`, `embedding_runtime_version`, `embedding_pooling`, `faiss_version` |
| `vector_count`, `chunk_count` | must equal FAISS `ntotal` and the sidecar length |
| `chunk_ids_sha256` | digest of the ordered chunk ids (detects reordering and substitution) |
| `artifacts` | size and SHA-256 of `index.faiss` and `chunks.jsonl` |

The manifest contains no vectors, no source text, no secrets and no absolute host paths.
Provenance is informational only: for example a different fastembed version does not invalidate an index
(same model id, same vectors up to numerical noise), while a different model id does.

## 9. Compatibility validation

`VectorIndex.load(directory, expected=IndexExpectation(...))` runs, in order:

1. directory and `manifest.json` exist and parse; `schema_version` is supported; `index_id` matches the spec;
2. the manifest matches `expected` (cheap, before any large read);
3. artifacts exist and match their recorded size and SHA-256;
4. `chunks.jsonl` parses; row count, positions `0..n-1`, unique ids and the id digest match;
5. the FAISS index deserialises, is an inner-product `IndexFlatIP` of the declared dimension, and
   `ntotal == vector_count == chunk_count == number of sidecar rows`.

| Situation | Error |
|---|---|
| Unsupported or missing `schema_version` | `IndexCompatibilityError` |
| Manifest differs from `expected` (model, dimension, normalisation, text style, representation version, chunking strategy/version/params, chunk-id schema, repository name/fingerprint, index type, metric) | `IndexCompatibilityError` listing **every** mismatched field |
| Missing/corrupt artifact, checksum mismatch, bad JSON, wrong counts, wrong FAISS type or dimension, undeserialisable FAISS bytes | `IndexCorruptError` |
| Empty input to `build`, inconsistent vectors/chunks | `EmptyIndexError`, `IndexBuildError` |
| Index already exists, write failure | `IndexExistsError`, `IndexStorageError` |

`load` **never rebuilds, repairs or silently accepts** a stale index; the caller decides what to do.
`expected` is a required argument: pass `IndexExpectation()` (all fields `None`) to check internal
consistency only, or `expectation_for_repository(path, settings=...)` to also require that the index still
matches that repository and the current configuration. `verify_vectors()` additionally checks that
every stored vector is finite and unit length and that a deterministic sample of vectors finds itself
with similarity ~1.

## 10. Consistency checks before indexing

All raise `IndexBuildError`: vectors are a 2-D `float32` array; dimension equals the model's; every
value finite; every norm within 1e-3 of 1; `len(chunks) == len(vectors)`; chunk ids unique; chunks in
canonical order; every chunk belongs to the spec's repository and chunking strategy/version;
`ntotal` equals the chunk count after building.

**Empty cases.** No accepted files, accepted files that produce zero chunks, and an empty vector list
each raise `EmptyIndexError` and persist nothing (an empty index would look valid and answer nothing).
Ingestion stopped by a repository limit is refused with `IndexBuildError` unless `--allow-truncated`,
because a partial index silently answers from part of the repository.

## 11. Atomic writes

A failed save must not leave a directory that looks like a complete index (`atomic.py`):

1. write all files into a hidden sibling `.staging-<id>-<random>/` directory (manifest last);
2. flush and `fsync` each file;
3. one `os.replace` renames the staging directory to `<index_id>/` (atomic on the same filesystem);
4. on any error, including `KeyboardInterrupt`, the staging directory is removed.

The final path therefore either does not exist or holds a complete index. Replacing an existing
index (`--force`) renames the old one aside, moves the new one in, then deletes the old; if the move fails the
old index is restored. Even if a hard kill prevents cleanup, the leftover staging directory has no manifest
and is not at an index path, so it cannot be loaded. Random suffixes name temporary directories only.
Tested: failure while writing, a simulated crash without cleanup, failure of the final rename, failed replacement
restoring the old index, and `KeyboardInterrupt`.

## 12. Commands

```powershell
uv run python -m copilot.vectorstore build PATH [--ignore-dir NAME] [--force] [--text-style raw|prefixed]
uv run python -m copilot.vectorstore info INDEX_DIR [--json]        # reads manifest.json only
uv run python -m copilot.vectorstore validate INDEX_DIR [--repo PATH] [--ignore-dir NAME]
```

`build` loads the embedding model (first run downloads about 0.64 GB into `data/cache/models`) and prints
repository, files, chunks, vectors, dimension, representation, index type, index id, timings and artifact
sizes; never vectors or source text. `info` and `validate` do **not** load the embedding model. `validate`
exits 0 when intact, 1 when invalid or incompatible, 2 for usage or other errors.

## 13. Planned retrieval experiment (not run in this milestone)

Recorded so the next milestone can execute it on this index layer: caps 512 / 768 / 1024 estimated
tokens (2048 optional) crossed with `prefixed` vs `raw`, model, top_k, benchmark and retrieval method held
constant; Hit@k and MRR plus chunk count, index size and embedding time; expected results matched by
file path and line overlap, never by chunk id. Each configuration is a separate index id, so they coexist.
Details: `docs/embeddings.md`. No number in this document is a retrieval result.

## 14. Security

* No repository content is sent anywhere; there are no LLM or API calls.
* Artifacts contain citation metadata (relative paths, line ranges, hashes), not source text, secrets or
  absolute paths (tested). Files rejected by ingestion (`.env`, keys, credentials) never reach the index.
* FAISS deserialisation is not hardened against maliciously crafted files, and the SHA-256 checks catch
  accidental damage rather than an attacker who can edit both a file and the manifest. Load only indexes
  this application built.
* The mandatory content-based secret scanner remains a hard gate before any repository context is sent
  to an external LLM.

## 15. Limitations

* **Chunk text is not stored.** Answering needs the repository to be re-ingested and re-chunked;
  a moved or edited repository makes the index unusable (by design, reported as a fingerprint mismatch).
* **Any change rebuilds everything.** There is no incremental update; one edited file changes the fingerprint and index id.
* **The fingerprint covers accepted files only**, and does not track the ingestion *code* version beyond
  the normalised-text hash; rebuild after upgrading ingestion or chunking logic (strategy version bumps are detected).
* **Runtime is provenance, not identity.** A different runtime for the same model id (for example quantised weights) would not be flagged.
* **Single writer.** No locking between two simultaneous builds; the atomic rename keeps the result consistent but the last writer wins.
* **Exact search only**, `float32` vectors, everything in memory. One measurement (Linux VM, single run, random unit vectors, embedding time excluded): 20,000 chunks of 768 dimensions built in 0.34 s, saved in 0.08 s and loaded in 0.28 s, with a 61 MB `index.faiss` and a 9 MB `chunks.jsonl`. Memory grows linearly (about 3 KB per vector); this design is not meant for hundreds of millions of vectors.
* **`validate --repo` re-ingests the repository** (fast, but it must be the same ignore options as the build).
* **Real-model behaviour on Windows** is verified by the user's commands below, not in the cloud test environment
  (the model host is unreachable there); the unit and integration suites use the deterministic fake embedder with real FAISS.

## 16. How this prepares semantic retrieval

Retrieval will embed a query with the same model, call the index's search primitive for the top-k
positions, map positions to `ChunkRecord`s (file, lines, hashes), re-materialise the chunk text by
`chunk_id`, and hand it to the LLM with citations. Everything it needs is here: a verified position <-> chunk
mapping, refusal to mix incompatible indexes, and separate index ids per configuration for the cap and representation experiment.

## 17. Verify on Windows (PowerShell)

```powershell
cd <your-clone>\agentic-rag-code-copilot
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python -c "import faiss; print(faiss.__version__)"

# Real Jina -> FAISS -> save (indexes src/ and docs/ only; embedding takes a few minutes on a CPU)
uv run python -m copilot.vectorstore build . --ignore-dir data --ignore-dir tests
# note the printed "index id", then:
uv run python -m copilot.vectorstore info data\indexes\<INDEX_ID>
uv run python -m copilot.vectorstore validate data\indexes\<INDEX_ID> --repo . --ignore-dir data --ignore-dir tests

# Determinism: rebuilding an unchanged repository must print the SAME index id
uv run python -m copilot.vectorstore build . --ignore-dir data --ignore-dir tests --force

# Real-model pytest round trip on the small synthetic repository
$env:COPILOT_RUN_LIVE = "1"
uv run pytest tests/integration/test_real_model_index_live.py -v -s
Remove-Item Env:COPILOT_RUN_LIVE
```

Expected: 768 dimensions, `IndexFlatIP`, `OK: index ... is intact`, `compatible with the repository`.
Editing any indexed file and re-running `validate --repo` should then report `INCOMPATIBLE ... repository_fingerprint` (revert the edit afterwards).
