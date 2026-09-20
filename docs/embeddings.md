# Embeddings (Milestone 4)

Local embedding service behind a small interface, plus tooling to validate the Milestone 3
token estimator against the real embedding tokenizer.

> **Validation status (read this first).** Everything below marked **Verified** was checked by
> reading the installed fastembed 0.8.0 source, running the test suite, or running the CLI.
> The real Jina model could **not** be downloaded in the development sandbox (the network
> allowlist blocks `huggingface.co`), so the following are **not yet measured**: the real model's
> actual output, the real tokenizer's token counts, and therefore the estimated-vs-actual
> comparison. Those measurements are produced by commands you run once (see "Measure it
> yourself"); no number in this document was invented to fill that gap.

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

## Selected model

`jinaai/jina-embeddings-v2-base-code`, run locally through fastembed (ONNX Runtime). It was the
approved preferred model; it has **not** been replaced by the fallback, because it did not fail
for a technical reason (see below). Nothing is sent to an external API.

| Fact | Value | Status |
|---|---|---|
| Supported by fastembed 0.8.0 | yes, in `PooledNormalizedEmbedding` | **Verified** (source + `list_supported_models()`) |
| Model identifier | `jinaai/jina-embeddings-v2-base-code` (Hugging Face repo of the same name) | **Verified** (registry entry) |
| Dimension | 768 | **Verified** (registry); runtime shape is checked on every call |
| ONNX file | `onnx/model.onnx`, ~0.64 GB | **Verified** (registry) |
| License | Apache-2.0 | **Verified** (registry) |
| Pooling / normalisation inside fastembed | mean pooling over attention mask, then L2 normalisation | **Verified** (source of `PooledNormalizedEmbedding`) |
| Advertised input limit | "8192 input tokens truncation" | **Verified as fastembed's description only**; not yet read from the model's own files |
| Query/document prefixes needed | "not necessary" | **Verified** (registry description) |
| Wheels for Windows (cp312 win_amd64) | onnxruntime, tokenizers, numpy, pillow, ... all present in `uv.lock` | **Verified** (lock file) |
| PyTorch required | no (onnxruntime only) | **Verified** (not in the lock file) |
| Real model download and inference | - | **Not verified**: blocked in the sandbox |

### Why the real model was not run in the sandbox

Attempted: `python -m copilot.embeddings smoke` (loads the configured model). Exact result:

```text
error: could not load embedding model 'jinaai/jina-embeddings-v2-base-code'
(ProxyError: 403 Forbidden). Check network access to huggingface.co ...
```

Cause classification: **environment issue**, not a fastembed, ONNX or model issue. `huggingface.co`
(and `cdn-lfs.huggingface.co`, fastembed's other download hosts) answer HTTP 403 with
`X-Proxy-Error: blocked-by-allowlist` from both the cloud container and the desktop VM; PyPI is
reachable. The dependency install, model-registry lookup and the whole runtime path (with a tiny
synthetic ONNX model) work. Because the model itself never failed, the BGE-small fallback was
**not** adopted.

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
returns unit vectors through fastembed, and the adapter guarantees it regardless of library
behaviour (a test feeds it un-normalised output). `EmbeddingModelInfo.normalized` is therefore
always `True`.

## Query vs document behaviour

The Jina code model is symmetric: no prefix, same encoder. `embed_query` still calls fastembed's
`query_embed` (so a future asymmetric model works without changing callers) and returns a vector
in the same 768-dimensional space.

## Batching

`embedding_batch_size` (default 32, env `COPILOT_EMBEDDING_BATCH_SIZE`) is the number of texts per
ONNX forward pass; fastembed pads each batch to its longest text. Order is preserved across
batches (tested with a real fastembed/onnxruntime stub, including padded multi-batch runs that
must equal one-at-a-time results). The default is a conservative guess that is **not tuned**;
throughput and memory on the real model are unmeasured.

## Model cache

fastembed's built-in default cache is the **OS temp directory**, which the OS may purge and which
would force a 0.64 GB re-download. We always pass an explicit directory:
`<data_dir>/cache/models` (default `data/cache/models`, covered by the `/data/*` rule in
`.gitignore`), overridable with `COPILOT_EMBEDDING_CACHE_DIR`. `.gitignore` also ignores `*.onnx`
and `*.safetensors`. Set `HF_HUB_OFFLINE=1` to forbid network access once the model is cached.
`python -m copilot.embeddings info` prints the cache path.

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

**Not claimed:** that the prefix improves retrieval. Retrieval does not exist yet; Milestone 5
indexes one style and can compare the other. The prefix is three short lines (path, language, line range), i.e. tens of characters; its real token
cost needs the real tokenizer (`python -m copilot.embeddings representations` reports it).

## Token limits and the provisional estimator

The estimator behind `chunk_max_tokens` (`utils/tokens.py`) was a guess. What is known now:

* **Truncation, not rejection.** fastembed enables truncation in its tokenizer at
  `min(model_max_length, max_length)` from the model's `tokenizer_config.json`; over-long input is
  **silently cut**, so a wrong estimate means silent loss of the end of a chunk, not an error.
  (Verified from fastembed's `load_tokenizer`. The actual numeric value for the Jina model is read
  from the model's files by `python -m copilot.embeddings tokens`; it is advertised as 8192.)
* **fastembed exposes the tokenizer** (`TextEmbedding.model.tokenizer`, a Hugging Face `tokenizers`
  object, already a dependency). No new dependency is needed for counting.
  `FastEmbedEmbedder.count_tokens` clones it and switches truncation and padding off, so counts
  are true lengths (special tokens included) and the embedding tokenizer is untouched. fastembed's
  own `token_count()` sums *post-truncation* lengths, which is why it is not used.
* **512 is not a Jina requirement.** It is the size of the fallback model's window (BGE-small:
  "512 input tokens truncation"), and it was only ever our safety cap.

### Tokenizer-free measurements (real, reproducible)

`python -m copilot.embeddings sizes PATH` needs no model. Measured on this repository at commit
`9e3586f` (`git archive 9e3586f`), line size 60, overlap 10, prefixed text:

| cap (est. tokens) | chunks | non-final chunks cut short by the cap | est. median / p95 / max | embedded bytes median / p95 / max |
|---|---|---|---|---|
| 256 | 565 | 100.0% | 246 / 255 / 256 | 1058 / 1312 / 1646 |
| 384 | 319 | 99.6% | 368 / 384 / 384 | 1542 / 1919 / 2262 |
| **512 (current)** | **220** | **89.2%** | 490 / 511 / 512 | 2014 / 2386 / 2850 |
| 768 | 168 | 46.7% | 538 / 762 / 766 | 2356 / 3314 / 3939 |
| 1024 | 146 | 24.1% | 502 / 1010 / 1023 | 2237 / 4309 / 4948 |
| 2048 | 136 | 2.7% | 492 / 1446 / 2045 | 2152 / 6041 / 8722 |

Two conclusions that need no tokenizer:

1. **Safety against an 8192 limit is provable.** A WordPiece/BPE tokenizer emits at most one token
   per byte, so `bytes + 2` bounds the token count. At the current cap the largest embedded text is
   2850 bytes, so **no current chunk can exceed 8192 tokens**, whatever the tokenizer does
   (`bytes-bound>limit` is 0 for caps up to 1024). This assumes the model's limit really is 8192,
   which is fastembed's description but not yet confirmed from the model files.
2. **The baseline is currently token-cap-driven, not line-window-driven.** With the 512 cap, 89.2%
   of non-final chunks are shortened below 60 lines by the cap, so the "primary boundary" in
   practice is the token estimate. At cap 1024 that falls to 24.1%.

For the **fallback** BGE-small (512-token window) the byte bound proves nothing: with limit 512 the
proof fails for 207 of 220 chunks, so a real tokenizer would be mandatory if that fallback were
ever adopted.

### Estimated vs actual (NOT YET MEASURED)

The mean/median/p95/max absolute error, percent under/over-estimated and the largest examples
require the real tokenizer. Tooling is implemented and tested (with a synthetic tokenizer):
`python -m copilot.embeddings tokens PATH`. Until it has been run against the real model, treat the
estimator as unvalidated. My expectation, not a result: real tokenizers split long identifiers into
several tokens, so the estimator probably *under*-counts code; that direction is safe here only
because of the 8192-token headroom above.

### Assessment of the 512 cap and tuning decision

* Required by the model? Not for Jina (advertised 8192); it is a self-imposed cap.
* Unsafe? For Jina, provably no (bytes bound above). Overly conservative in tokens: probably, but
  unmeasured.
* Would another cap give more coherent chunks? Coherence is a retrieval-quality question and cannot
  be settled without retrieval. What can be said is that cap 512 makes 89% of windows token-cut
  mid-function and cap 1024 makes 24% so; whether that helps retrieval is unknown.

**Decision: D. Defer tuning until retrieval evaluation.** No safety problem exists, so defaults
(`chunk_size_lines=60`, `chunk_overlap_lines=10`, `chunk_max_tokens=512`) and the strategy
version (1) are **unchanged**, and chunk ids are unchanged. Milestone 5's retrieval benchmark should
compare at least cap 512 vs 1024 (same line size), because those two bracket the "cap-driven vs
line-driven" regimes. The goal is retrieval quality, not filling the model's window.

## Measure it yourself (produces the missing numbers)

```powershell
uv run python -m copilot.embeddings info
uv run python -m copilot.embeddings smoke . --ignore-dir data -n 8
uv run python -m copilot.embeddings tokens . --ignore-dir data
uv run python -m copilot.embeddings representations . --ignore-dir data -n 20
$env:COPILOT_RUN_LIVE = "1"; uv run pytest tests/integration/test_real_model_live.py -v
```

The first model-loading command downloads ~0.64 GB into `data/cache/models`. Output contains
metadata only: never vectors or source text.

## Tests

* Unit tests use fakes: `HashEmbedder` and a fake fastembed `TextEmbedding` (validation, ordering,
  normalisation, error wrapping, lazy loading, tokenizer cloning).
* `tests/integration/test_fastembed_runtime_stub.py` runs the **real** fastembed, onnxruntime and
  tokenizers stack on a tiny *synthetic* 768-dimensional ONNX model (built by
  `tests/fixtures/stub_onnx_model.py`, needs the dev-only `onnx` package). It validates our adapter
  and fastembed's pooling/normalisation/truncation behaviour; it is not the Jina model and says
  nothing about quality.
* `tests/integration/test_real_model_live.py` needs the real model and is skipped unless
  `COPILOT_RUN_LIVE=1`.

## Security

Embedding is local: no repository text leaves the machine in this milestone, and the only network
access is the one-time model download. The content-secret scanner remains a **mandatory gate
before any repository context is sent to an external LLM (Milestone 6)**; this milestone does not
change that.

## Known limitations

* The real model, real tokenizer counts and the estimated-vs-actual comparison are unmeasured (see
  above); `info.max_input_tokens` and `runtime_limits()` will report the true values on first run.
* `count_tokens` and `runtime_limits` use fastembed internals (`model.model.tokenizer`,
  `_model_dir`); a fastembed upgrade could break them, which is why fastembed is pinned to
  `>=0.8,<0.9` and failures raise a clear error.
* Silent truncation applies to any text over the model limit; the pipeline relies on the chunk cap
  to avoid it.
* Batch size, threads and speed are untuned; the Jina model is CPU-heavy (0.64 GB).
* Windows behaviour is unverified until you run the commands above.
* Symmetric model only; `asymmetric=True` handling exists in the interface but is untested.
