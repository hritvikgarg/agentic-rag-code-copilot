# Semantic retrieval (Milestone 5b)

Status: **implemented and tested.** Retrieval quality on the project's own benchmark is reported in
[`evaluation.md`](evaluation.md); it is *measured*, not assumed. There is no LLM, no answer
generation and no lexical (BM25) retrieval in this milestone.

## What it does

```
natural-language query
  -> validate (non-empty string, top_k in 1..50)
  -> embed with the SAME local model the index was built with        (Embedder.embed_query)
  -> FAISS exact inner-product search                                (VectorIndex.search)
  -> ranked (vector position, score) pairs
  -> position -> chunk metadata row in chunks.jsonl
  -> re-materialise the chunk's source text from the repository and verify its hash
  -> RetrievalResult objects (rank, score, file, lines, symbol, verified text)
```

No component sends anything to an external service: the query is embedded locally.

## Concepts (what / why / how here / without it)

**Query embedding.** *What:* the query text is turned into a 768-dimensional unit vector by the same
model that embedded the chunks. *Why:* similarity is only meaningful inside one vector space.
*Here:* `Retriever` refuses an embedder whose model id, dimension or `normalized` flag differs from
the index manifest (`EmbedderMismatchError`). Jina code-v2 is symmetric, so queries get no special
prefix. *Without it:* a different model would still return numbers, but they would be meaningless.

**Scoring.** The index is `IndexFlatIP` over unit vectors, so the inner product equals cosine
similarity (range -1..1). The search is *exact* (brute force over every vector), which is fine at
this scale (hundreds to low thousands of chunks) and removes approximate-search error as a variable.

**Top-k.** The `k` best-scoring chunks are returned. Too small a `k` misses evidence; too large a `k`
adds noise and, later, LLM context cost. `top_k` is limited to `MAX_TOP_K = 50`.

## Vector search contract (`VectorIndex.search(query_vector, top_k)`)

- The query must be a finite float vector of shape `(dimension,)` with norm 1 (tolerance 1e-3);
  `top_k` must be an `int` >= 1 (not a `bool`). Otherwise `SearchInputError`.
- `top_k` may exceed the index size; the result is then shorter (at most `count` hits).
- Result: `list[SearchHit(position, score)]`, best first.
- **Deterministic ordering.** Sorted by `(-score, position)`. FAISS does not promise an order among
  exactly tied scores, so the search fetches one extra hit and widens the request while the last
  fetched score equals the k-th score, then sorts explicitly. Duplicate chunks therefore always come
  back in vector-position order.
- **FAISS `-1` padding.** When FAISS is asked for more neighbours than vectors it pads with position
  `-1`. `search` never asks for more than `count`, and also drops any `-1` it receives. Python
  indexing with `-1` would silently return the *last* sidecar row, i.e. a wrong chunk with a plausible
  score, so this is guarded and tested (including a backend that pads deliberately).

## Source re-materialisation and stale-index detection

The index stores **metadata only** (`chunks.jsonl`, no source text, by design; see
[`vector-index.md`](vector-index.md)). Text is therefore rebuilt at search time:

1. `Retriever.open(index, repo_path, embedder=...)` loads the index strictly, then re-ingests the
   repository and computes its fingerprint. A different fingerprint (edited, added or deleted file,
   changed ignore set, different ingestion policy) raises `StaleRepositoryError` before any search.
   The error tells you to rebuild the index or pass the same `--ignore-dir` values; it contains no
   host paths.
2. The repository is re-chunked with the **manifest's** chunking strategy and parameters
   (`create_chunker_from_params`), not the current settings. Changing `CHUNK_MAX_TOKENS` in `.env`
   cannot silently alter what an existing index means.
3. For every result the chunk is looked up by `chunk_id` and checked against the sidecar row: file
   path, line range, `source_sha256` and `content_sha256`. `ChunkNotFoundError` or
   `ChunkContentMismatchError` is raised on any difference; error messages never contain source text.

Two independent layers (repository fingerprint, then per-chunk hashes) mean stale text is never
returned as if it were current evidence. Copying the repository elsewhere is fine: only content
matters, not the absolute path. Line endings are normalised on read, so CRLF and LF checkouts of the
same commit produce the same fingerprint and chunk hashes.

## `RetrievalResult`

Frozen model: `rank`, `score`, `position`, `chunk_id`, `repository_name`, `file_path` (relative POSIX
path, validated), `language`, `chunk_type`, `chunk_index`, `start_line`, `end_line`, `fragment_index`,
`fragment_count`, `symbol_name`, `qualified_name`, `parent_class`, `token_estimate`,
`content_sha256`, `text` (excluded from `repr`), plus `line_count` and `location` (`path:start-end`).
There are no absolute host paths anywhere. With the line baseline, symbol fields are empty.

## API

```python
from copilot.embeddings import create_embedder
from copilot.retrieval import Retriever, retrieve

retriever = Retriever.open(
    "data/indexes/<id>",
    "path/to/repo",
    embedder=create_embedder(settings),
    settings=settings,
    ignore_directories=("data",),
)
results = retriever.retrieve("How are chunk IDs generated?", top_k=5)

# one-shot (opens, verifies, searches):
results = retrieve("How is logging configured?", 5, "path/to/repo", "data/indexes/<id>", embedder=e)
```

Query validation (`QueryError`): must be a `str`, non-blank after stripping, at most 2000 characters;
`top_k` an `int` in 1..50. The embedder is never called for invalid input. The 50 upper bound is a
sanity limit for this academic system, not a technical FAISS limit.

## CLI

```bash
uv run python -m copilot.retrieval search data/indexes/<index-id> --repo path/to/repo \
    --ignore-dir data --ignore-dir tests --top-k 5 "Where is repository ingestion implemented?"
```

Use the same `--ignore-dir` values as when the index was built. Output per result: `#rank score`,
`path:start-end` (and the qualified symbol when known), chunk id and type, and a short preview
(`--preview-lines`, default 6) with indentation kept, long lines truncated and no absolute paths.
Errors print `error: ...` and exit with status 2. Loading the model takes a few seconds.

## Limitations

- **The retriever returns chunks, not answers.** It can rank documentation above the code that
  implements something (see the measured results); nothing filters by file type yet.
- Line-window chunks ignore function boundaries, so a hit may contain the right code plus unrelated
  neighbours, or split it across two chunks. Structure-aware chunking is a later milestone.
- Search is dense-only. Exact identifier matches (for example a function name) are not favoured; a
  lexical retriever is deferred.
- Re-ingesting the repository on every `Retriever.open` costs time proportional to repository size;
  acceptable here, wasteful for very large repositories.
- Embedding the query needs the model loaded (seconds per process).
