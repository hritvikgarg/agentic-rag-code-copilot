# Embeddings (Milestone 4)

Local embedding service behind a small interface, validated against the real
`jinaai/jina-embeddings-v2-base-code` model on native Windows, including a measurement of the
Milestone 3 token estimator against the model's real tokenizer.

> **Provenance of numbers.** Numbers labelled *Windows* were measured by the project owner on
> native Windows (Python 3.12.5, fastembed 0.8.0, onnxruntime CPU) against the tree at commit
> `163bd34` (its chunk counts, 294 / 230 / 207 / 196 for the four caps below, reproduce exactly with
> `git archive 163bd34`). Numbers labelled *tokenizer-free* were computed in the Linux development
> sandbox, which cannot reach `huggingface.co` and therefore never ran the real model. Nothing in
> this document is estimated to fill a gap; a figure that was not reported is not shown.

## What embeddings are, and why we use them

**What.** An embedding model maps a piece of text to a fixed-length vector of numbers (768 for
our model) so that texts with similar *meaning* land near each other in vector space. Nearness is
measured by cosine similarity.

**Why.** Retrieval-Augmented Generation must find the repository code relevant to a natural-language
question such as "where is the database connection configured?". Keyword search misses code that
never uses those words (`create_engine(url)`); embeddings match by meaning, including across
natural language and code.

**How we use them.** Each chunk is embedded once at indexing time; each question is embedded at
query time; the nearest chunk vectors are the candidate evidence (Milestone 5).

**Without them.** Retrieval would be exact-word matching only, and the assistant would either
miss relevant code or fall back on the LLM's general knowledge, which is what this project exists
to avoid.

## Selected model and validation status

`jinaai/jina-embeddings-v2-base-code`, run locally through fastembed (ONNX Runtime). It was the
approved preferred model and is now **validated**; the BGE-small fallback was never needed. Nothing
is sent to an external API.

| Fact | Value | Status |
|---|---|---|
| Supported by fastembed 0.8.0 | yes, `PooledNormalizedEmbedding` | Verified (source, registry) and works on Windows |
| Model identifier | `jinaai/jina-embeddings-v2-base-code` | Verified (registry; live test asserts it) |
| Dimension | 768 | **Windows: measured** (vector and query shapes are `(768,)`) |
| Input limit | 8192 tokens | **Windows: measured**: `smoke` reports max input tokens 8192, the tokenizer truncation length fastembed reads from the model files |
| Output normalisation | unit length | **Windows: measured**: norm min/max 1.000000 / 1.000000 |
| Values finite | yes | **Windows: measured** |
| Determinism | identical repeat | **Windows: measured**: max absolute difference 0.000e+00 on repeat |
| Download size / license | ~0.64 GB / Apache-2.0 | Registry values |
| PyTorch required | no | Verified (not in `uv.lock`) |
| Live tests (`COPILOT_RUN_LIVE=1`) | 3 of 3 passed on Windows | Measured: model identity and dimension; finite, unit-length, deterministic vectors; untruncated positive token counts |

Windows runtime: `fastembed-onnx` 0.8.0. First load including download took 43.45 s; 8 vectors were
embedded in 2.41 s (about 3.3 texts/s; CPU model and thread count were not recorded, and the 8
chunks were typical ~500-token texts). Peak memory was not recorded.

History: the development sandbox could not download the model (`ProxyError: 403`,
`X-Proxy-Error: blocked-by-allowlist` for `huggingface.co`, an environment/network-policy issue,
not a fastembed, ONNX or model defect), which is why the real-model validation was run on Windows.

## Interface

`copilot.embeddings` exposes `Embedder` (a `Protocol`); nothing outside this package imports
fastembed.

```python
embedder.info  # EmbeddingModelInfo (model_id, dimension, runtime, normalized, ...)
embedder.embed_documents(texts)  # (n, dim) float32, rows in input order
embedder.embed_query(text)  # (dim,) float32, same space as the documents
embedder.count_tokens(texts)  # tokens per text, NOT truncated (TokenCountingEmbedder)
```

Implementations: `FastEmbedEmbedder` (real) and `HashEmbedder` (deterministic fake for tests; not
a quality model). `create_embedder(settings)` builds the configured one; the model loads lazily on
first use. `embed_chunks(embedder, chunks, style=...)` returns `ChunkEmbeddings`: `chunk_ids[i]`
pairs with `vectors[i]`, plus model provenance (`EmbeddingModelInfo`), text style and representation
version. It holds no chunk text, so the FAISS index in the next milestone can store `chunk_id <->
vector` directly.

Defined behaviour: an empty batch returns shape `(0, dim)`; a bare `str`, a non-`str` item, or an
empty/whitespace-only text raises `EmbeddingInputError`; output with the wrong shape, NaN/inf or an
all-zero vector raises `EmbeddingRuntimeError`; an unsupported model id raises
`EmbeddingConfigError`; a failed download/load raises `EmbeddingModelUnavailableError` with the
original exception preserved as `__cause__`.

`EmbeddingModelInfo` records: `model_id`, `dimension`, `runtime` (`fastembed-onnx`),
`runtime_version`, `normalized`, `pooling`, `max_input_tokens`, `batch_size`, `asymmetric`.

## Normalisation decision

**Vectors are L2-normalised in the embedding layer (`vectors.finalize_vectors`), not at index or
search time.** Cosine similarity `a.b / (|a||b|)` equals the plain dot product when both vectors
have length 1, so the vector index (FAISS `IndexFlatIP` next milestone) needs no extra step and can
never mix normalised and un-normalised vectors. The step is idempotent: the Jina model already
returns unit vectors through fastembed (confirmed on Windows), and the adapter guarantees it
regardless of library behaviour. `EmbeddingModelInfo.normalized` is therefore always `True`.

## Query vs document behaviour

The Jina code model is symmetric: no prefix, same encoder. `embed_query` still calls fastembed's
`query_embed` (so a future asymmetric model works without changing callers) and returns a vector
in the same 768-dimensional space.

## Batching

`embedding_batch_size` (default 32, env `COPILOT_EMBEDDING_BATCH_SIZE`) is the number of texts per
ONNX forward pass; fastembed pads each batch to its longest text. Order is preserved across
batches (tested on the real fastembed/onnxruntime stack with a synthetic model: batched results
equal one-at-a-time results). The default is **untuned**. The only real-model speed measurement is
about 3.3 texts/s on the owner's Windows machine for 8 chunks; extrapolating, indexing a few hundred
chunks takes on the order of a minute or two on that machine (an extrapolation, not a measurement).

## Model cache

fastembed's built-in default cache is the **OS temp directory**, which the OS may purge and which
would force a 0.64 GB re-download. We always pass an explicit directory:
`<data_dir>/cache/models` (default `data/cache/models`, covered by the `/data/*` rule in
`.gitignore`), overridable with `COPILOT_EMBEDDING_CACHE_DIR`. `.gitignore` also ignores `*.onnx`
and `*.safetensors`. Set `HF_HUB_OFFLINE=1` to forbid network access once the model is cached.
`python -m copilot.embeddings info` prints the cache path.

### Windows: Hugging Face cache symlink warning (non-fatal)

On the validated Windows machine, `huggingface_hub` printed a warning that its cache cannot use
symlinks and is running in degraded mode. **This is non-fatal:** the model downloaded, loaded and
cached, and all measurements and live tests succeeded. Windows only allows symlinks for
administrators or with Developer Mode enabled; the degraded mode copies files instead (this can use
more disk space when several model revisions are cached; with a single ~0.64 GB model it is
negligible). **Do not enable Administrator mode or Developer Mode just to silence it.** If the
message is a nuisance, `HF_HUB_DISABLE_SYMLINKS_WARNING=1` hides it without changing behaviour.

## Embedding text representation

Raw source (`Chunk.content`) is what we cite and never modified. The text that is embedded is a
separate string (`copilot.embeddings.representation`):

```text
File: src/auth/service.py
Language: python
Lines: 10-40

<chunk content>
```

Fragments of an over-long line are labelled `Lines: 12 (part 2 of 5)`. The repository name is not
included (an index covers one repository). `embedding_text_style` selects `prefixed` (default) or
`raw`. `REPRESENTATION_VERSION` (currently 1) belongs in index manifests: vectors from different
representations are not comparable.

**Windows measurements (20 evenly spaced chunks):**

| Measure | Value |
|---|---|
| cosine(raw vector, prefixed vector): mean / min / max | 0.921 / 0.809 / 0.988 |
| Prefix cost in real Jina tokens: median / max | 23.5 / 27 (all 294 chunks: 23.5 / 29) |

**Decision.** `prefixed` stays the default. **Its retrieval benefit is UNPROVEN.** The cosine
figures show only that the prefix moves each vector by a modest, non-trivial amount (the closest
pair is 0.988, the furthest 0.809); they say nothing about whether the moved vectors retrieve
better. Retrieval does not exist yet. The prefixed and raw representations must be compared on the
retrieval benchmark (see "Planned retrieval experiment") before the default is treated as justified.

## Token limits: heuristic estimate vs actual Jina tokenizer

Two different things are called "tokens" in this project; keep them apart:

* **Estimated tokens** (`copilot.utils.tokens.estimate_tokens`): a dependency-free *heuristic*
  (ASCII word runs plus every other non-space character). It drives `chunk_max_tokens` and
  `Chunk.token_estimate`.
* **Actual tokens**: the count produced by the real Jina tokenizer, including special tokens, via
  `Embedder.count_tokens` (fastembed's tokenizer, cloned with truncation and padding disabled so
  counts are true lengths; fastembed's own `token_count()` sums post-truncation lengths, so it is
  not used).

fastembed truncates silently at the model limit (8192 here) instead of raising, so an inaccurate
estimate can silently lose the end of a chunk if chunks ever approach the limit.

### Measured comparison (Windows, 294 chunks, raw content, real Jina tokenizer)

| Measure | Value |
|---|---|
| Total estimated / actual tokens | 124,190 / 142,230 (actual is 14.5% higher) |
| Absolute error: mean / median / p95 / max | 67.2 / 54.5 / 176 / 242 tokens |
| Mean signed error (estimated - actual) | -61.4 tokens |
| Underestimated / overestimated / exact | 89.1% / 8.5% / 2.4% |
| actual / estimated ratio: median / p95 / max | 1.12 / 1.41 / 1.73 |

**Finding: the heuristic systematically underestimates Jina tokenization on this repository** (89.1%
of chunks underestimated, mean error -61 tokens, real counts about 12% above the estimate at the
median and up to 73% above). This confirms the expectation that a real code tokenizer splits
identifiers more finely than the word-run heuristic. The finding is specific to this repository
(mostly Python and Markdown, 294 chunks), not a general property of the estimator. The estimator
is unchanged: `chunk_max_tokens` remains an *estimated*-token cap, and one estimated token is
roughly 1.14 real Jina tokens here. The largest under/over-estimation examples were not part of the
reported results and can be regenerated with `python -m copilot.embeddings tokens . --ignore-dir data`
(metadata only: file path and line range).

## The 512-estimated-token cap: assessment and decision

Windows results by candidate cap (same 60-line window and 10-line overlap; only the cap varies;
prefixed embedding text; model limit 8192 tokens). The cap-cut column is *tokenizer-free* (Linux,
same tree) and is the share of non-final chunks the cap shortened below 60 lines.

| Estimated cap | Chunks | Cap-cut | Actual raw tokens median / p95 / max | Max embedded (prefixed) | Chunks over 8192 |
|---|---|---|---|---|---|
| **512 (current)** | **294** | **86.6%** | **525 / 659 / 710** | **736** | **0** |
| 768 | 230 | 39.3% | 644 / 839 / 879 | 902 | 0 |
| 1024 | 207 | 17.2% | 610 / 1059 / 1137 | 1160 | 0 |
| 2048 | 196 | 1.8% | 599 / 1081 / 2150 | 2173 | 0 |

Answers:

1. **Is 512 required by the model?** No. Jina's limit is 8192; 512 was only ever our own cap (and
   the window of the BGE-small fallback, which was not used).
2. **Is it unsafe?** No. The largest real embedded text at the current cap is 736 tokens, under 9%
   of the limit; even at cap 2048 the largest is 2173. Zero chunks exceed the limit at any cap.
3. **Is it overly conservative?** Only in the trivial sense that the model could take much longer
   chunks. Whether shorter or longer chunks retrieve better is a retrieval question, not a capacity
   question, and it is not answered by these measurements.
4. **How many chunks approach or exceed the limit?** None approach it: the maximum is 736 of 8192
   tokens.
5. **Would another cap give more coherent chunks?** Unknown. At cap 512, 86.6% of non-final chunks
   are shortened by the cap, so the token estimate, not the 60-line window, is the effective primary
   boundary of the baseline; at 1024 that falls to 17.2%. Coherence and retrieval quality are not
   measured here.

**Decision: keep the current defaults.** `chunk_size_lines` (60), `chunk_overlap_lines` (10),
`chunk_max_tokens` (512), the baseline strategy version (1) and all existing chunk IDs are
unchanged. There is no safety issue against the 8192-token limit, and changing chunk size without
retrieval evidence would be premature. The final choice must be made on retrieval quality, not on
the model's context capacity.

## Planned retrieval experiment (Milestone 5 onward)

Not run yet. Record of the intended design so the decision above is revisited with evidence:

* **Variable A, cap:** `chunk_max_tokens` = **512** (current), **768**, **1024** (estimated
  tokens), at minimum. Keep `chunk_size_lines=60` and `chunk_overlap_lines=10` fixed so the cap is
  the only chunking variable (a later sweep may vary lines and overlap). Cap 2048 is optional.
* **Variable B, representation:** `prefixed` vs `raw`, run for each cap.
* **Held constant:** embedding model, `top_k`, benchmark questions, and the retrieval method.
* **Metrics:** Hit@k and MRR against expected files/symbols (Milestone 5 benchmark), plus chunk
  count, index size and embedding time as costs.
* **Matching:** chunk IDs change with `chunk_max_tokens` (it is part of the ID contract), so
  expected results must be matched by file path and line-range overlap, never by chunk ID.
* **Reporting:** raw numbers only, no significance claims unless a test is actually run; a benchmark
  of tens of questions cannot separate small differences.
* **Possible outcome:** if a different cap wins, change the default in a separate commit; the ID
  contract already includes the cap, and a change to the estimator itself (rather than the cap)
  would additionally require bumping the strategy version.

## Measure it yourself

```powershell
uv run python -m copilot.embeddings info
uv run python -m copilot.embeddings sizes . --ignore-dir data         # no model, no download
uv run python -m copilot.embeddings smoke . --ignore-dir data -n 8
uv run python -m copilot.embeddings tokens . --ignore-dir data
uv run python -m copilot.embeddings representations . --ignore-dir data -n 20
$env:COPILOT_RUN_LIVE = "1"; uv run pytest tests/integration/test_real_model_live.py -v
```

The first model-loading command downloads ~0.64 GB into `data/cache/models`. Output contains
metadata only: never vectors or source text. Results depend on the repository tree; the figures
above are for commit `163bd34`.

## Tests

* Unit tests use fakes: `HashEmbedder` and a fake fastembed `TextEmbedding` (validation, ordering,
  normalisation, error wrapping, lazy loading, tokenizer cloning).
* `tests/integration/test_fastembed_runtime_stub.py` runs the **real** fastembed, onnxruntime and
  tokenizers stack on a tiny *synthetic* 768-dimensional ONNX model (built by
  `tests/fixtures/stub_onnx_model.py`, needs the dev-only `onnx` package). It validates our adapter
  and fastembed's pooling/normalisation/truncation behaviour, offline.
* `tests/integration/test_real_model_live.py` needs the real model and is skipped unless
  `COPILOT_RUN_LIVE=1`; it passed on Windows (3 of 3).

## Security

Embedding is local: no repository text leaves the machine in this milestone, and the only network
access is the one-time model download. The content-secret scanner remains a **mandatory gate
before any repository context is sent to an external LLM (Milestone 6)**; this milestone does not
change that.

## Known limitations

* Findings come from one repository (mostly Python and Markdown, 294 chunks) and one Windows
  machine; other repositories or languages may show different estimator error.
* The heuristic estimator underestimates real tokens by about 14.5% in total here; it is left as is
  (see decision) and its cap should be read as "estimated".
* Throughput and memory are essentially unmeasured (one 8-text run; no memory figure); batch size
  and threads are untuned.
* Whether the metadata prefix helps retrieval, and which cap retrieves best, are open questions for
  the retrieval experiment.
* `count_tokens` and `runtime_limits` use fastembed internals (`model.model.tokenizer`,
  `_model_dir`); a fastembed upgrade could break them, which is why fastembed is pinned to
  `>=0.8,<0.9` and failures raise a clear error.
* Silent truncation applies to any text over the model limit; no current chunk comes near it.
* Symmetric model only; `asymmetric=True` handling exists in the interface but is untested.
* The Hugging Face symlink warning on Windows is cosmetic (see above).
