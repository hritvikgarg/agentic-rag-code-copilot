# The LLM layer (Milestone 6)

Package `copilot.llm`. It is the only code that can send text to a hosted model, and every request
passes the content-secret gate first. It contains no retrieval or RAG logic.

## Components

| Piece | Role |
|---|---|
| `LLMClient` (Protocol) | `provider`, `model`, `generate(request) -> LLMResponse`. Small on purpose; two providers exist (Gemini, Ollama), so there is still no registry or plug-in machinery. |
| `LLMRequest` | Frozen model of *exactly* what leaves the machine: `system_instruction`, `user_prompt`, `temperature`, `max_output_tokens`, `prompt_version`. `outbound_text()` returns the sent text; `scan_targets()` returns the same two strings for the gate. The texts are excluded from `repr`. |
| `LLMResponse` | `text` (never empty), `provider`, `model`, `finish_reason`, token usage (`prompt_tokens`, `completion_tokens`, `total_tokens`, `reasoning_tokens`), `latency_seconds`. No request text, no credentials. |
| `FakeLLMClient` | Deterministic, offline; records every request in `.requests`. Used by all normal tests. |
| `GeminiLLMClient` | The hosted client (official `google-genai` SDK). |
| `OllamaLLMClient` | The local client (Milestone 11B): REST over `httpx`, loopback only. See [Provider: local Ollama](#provider-local-ollama-milestone-11b). |
| `GuardedLLMClient` | Wraps any client; scans the exact outbound strings, then delegates. See below. |
| `create_llm_client(settings)` | Builds the configured client **already wrapped in the guard**. |

## Security-gate placement

```
LLMRequest ──► GuardedLLMClient.generate
                 1. assert_safe_for_external_llm(request.scan_targets())   # raises on any finding
                 2. only if it returned: inner.generate(request)           # the provider call
```

* The scan covers the exact strings that will be transmitted (system instruction and user prompt),
  so it covers the retrieved chunks, the user's question and any other text placed in the request.
* If the gate raises, the inner client's `generate` is never entered (tests assert the fake client's
  call count is exactly zero, for a secret in a chunk, in the question and in the system text).
* `answer_plain` and `RagService` always wrap the client they receive, and the factory returns a
  guarded client, so a caller cannot obtain an unguarded path by mistake. Wrapping an already
  guarded client does not nest. There is no `force`, `skip` or environment switch anywhere;
  `tests/unit/test_llm_isolation.py` fails if another module imports the SDK or defines `generate`.
* The RAG service additionally runs the gate one step earlier on the question plus the chunks that
  entered the context, purely to give a diagnostic with `file:line` (the LLM-request scan can only
  name the pseudo-file "user prompt"). Both scans must pass.
* The gate is pattern-based and cannot guarantee that every secret is found ([`security.md`](security.md)).
  Only index repositories you are willing to send to a hosted model.

## Provider: Google Gemini

Implemented with the official `google-genai` SDK (declared as `google-genai>=2.24.0`; developed
against 2.24.0). The calls used were verified against the installed SDK, not assumed:

* `genai.Client(api_key=..., http_options=types.HttpOptions(timeout=<milliseconds>))`
* `client.models.generate_content(model=..., contents=<user prompt>, config=GenerateContentConfig(system_instruction, temperature, max_output_tokens))`
* `response.text`, `response.usage_metadata`, `response.candidates[0].finish_reason`, `response.prompt_feedback`

The SDK is imported in `llm/gemini.py` only. `GeminiLLMClient(..., sdk_client=fake)` lets tests inject
a fake SDK client, so no test needs the network.

## Provider: local Ollama (Milestone 11B)

Purpose: an **independent local judge** for the generation evaluation (different vendor and weights
from the Gemini generator, no quota, no repository text leaving the machine). It is a provider on the
same `LLMClient` protocol; nothing else in the stack changes. It was written and tested offline
(an `httpx.MockTransport` stands in for the server) and then exercised once against a real Ollama
0.40.18 on Windows: the preflight (`/api/version`, `/api/tags`, `/api/show`, `/api/ps`, warm-up) and
18 judge calls behaved as designed - `done_reason` was `stop` on the warm-up, token counts were
returned on every call, no context-limit failure occurred, and the model was 100% GPU resident at
`num_ctx` 8192. That run is the local-judge validation, which the model **failed** on output quality,
not on the plumbing: [`docs/judge-validation-llama3.1-8b.md`](judge-validation-llama3.1-8b.md).

### Call chain

```
judge CLI --judge-provider ollama
  -> create_llm_client(settings)                      (factory builds OllamaLLMClient, wraps it)
  -> GuardedLLMClient.generate(request)               (secret gate on the exact outbound text)
  -> OllamaLLMClient.check_input_fits(request)        (conservative context pre-check)
  -> POST http://localhost:11434/api/chat             (stream=false, format="json", explicit options)
  -> _check_not_truncated(response)                   (done / done_reason / token-count validation)
  -> LLMResponse
```

In `LLMJudge` the order is: build request -> input hash -> **secret gate** -> runtime identity
(`/api/version`, `/api/tags`, `/api/show`) -> cache lookup (the fingerprint must match) -> guarded
generate -> parse. Nothing is cached or sent before the gate has passed.

### Endpoint security

* Only `http`/`https`; no credentials, path, query or fragment in the URL; port 1-65535.
* **Loopback only**: `localhost` or a loopback IP literal. Odd forms (`127.1`, decimal IPs) are
  rejected. A remote host needs `COPILOT_OLLAMA_ALLOW_REMOTE=true` **and** `https`.
* `httpx.Client(follow_redirects=False, trust_env=False)`: no redirect following and no proxy
  environment variables, so a redirect cannot move the request off the loopback host. Any 3xx is
  refused. An injected client must have both flags off.
* Error messages are fixed strings (no URL, headers or body) and chained `from None`.
* The model is **never pulled**. A missing model gives a configuration error that tells you to run
  `ollama pull <tag>` yourself.

### Context protection (nothing is silently shortened)

Ollama truncates an over-long prompt to `num_ctx` *without an error*. A truncated judge prompt would
silently lose the end of the answer or the rubric, so the client guards both sides:

* **Before sending:** a deliberately pessimistic estimate `ceil(utf8_bytes / 2.0) + 64` must be
  <= `num_ctx - max_output_tokens`, otherwise `LLMContextLimitError` is raised and **no request is
  made**. The estimate is a heuristic upper bound, not a proof.
* **After the response:** it is rejected if `done` is false, `done_reason` or either token count is
  missing, `done_reason == "length"`, `prompt_eval_count` >= the input budget, `eval_count` >=
  `num_predict`, or prompt + completion >= `num_ctx`.
* A context failure is recorded as a **failed item** (never as a low score) and excluded from valid
  counts.

Remaining limitations: Ollama's prefix cache can lower `prompt_eval_count`, so the count is only a
lower bound and does not by itself prove nothing was truncated; the client fails closed when counts
are missing, which could reject a real response on a full prompt-cache hit (repeatability runs use
`keep_alive 0` for this reason). A model swapped mid-process is only noticed by the next process.

### Runtime fingerprint

`RuntimeIdentity` = provider, model tag, **model digest**, quantization, parameter size, Ollama
version, `num_ctx`, max output tokens, temperature, seed, structured-output mode. Its fingerprint is
the SHA-256 of the canonical JSON. `keep_alive` is excluded (it changes speed, not output). The
fingerprint is stored in each judgement record and file and is part of the judge cache key *only when
it exists*, so **Gemini cache keys are unchanged** (golden-hash test). It identifies the runtime; it
is not a claim of determinism.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `COPILOT_OLLAMA_BASE_URL` | `http://localhost:11434` | Loopback endpoint |
| `COPILOT_OLLAMA_NUM_CTX` | `8192` | Context window requested per call |
| `COPILOT_OLLAMA_SEED` | `20260` | Sampling seed |
| `COPILOT_OLLAMA_KEEP_ALIVE` | `30m` | How long the model stays loaded |
| `COPILOT_OLLAMA_ALLOW_REMOTE` | `false` | Allow a non-loopback `https` endpoint |

Typed errors: 401/403 `LLMAuthenticationError`, 404 `LLMConfigurationError`, 429 `LLMRateLimitError`, 408/504
`LLMTimeoutError`, 3xx `LLMProviderError` ("redirect refused"), context failures
`LLMContextLimitError`, everything else `LLMProviderError`.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `COPILOT_LLM_PROVIDER` | `gemini` | Only `gemini` is implemented (`ollama` is accepted by the settings but the factory refuses it). |
| `COPILOT_LLM_MODEL` | **unset** | The model id. **Required for any live use.** |
| `GEMINI_API_KEY` | unset | API key (a `SecretStr`; never printed, logged, put in `repr` or exceptions). |
| `COPILOT_LLM_TEMPERATURE` | `0.0` | Conservative: keeps runs as repeatable as the provider allows (not a guarantee). |
| `COPILOT_LLM_MAX_OUTPUT_TOKENS` | `1024` | Cap on generated tokens (16..65536). |
| `COPILOT_LLM_TIMEOUT_SECONDS` | `60` | Request timeout. |
| `COPILOT_RAG_CONTEXT_MAX_TOKENS` | `6000` | Estimated-token budget for the evidence in one RAG prompt. |

**No model id is shipped as a default.** Model names and availability change and depend on your key,
region and quota, and this project will not invent one. Pick an id from Google's current model list
for the Gemini API (<https://ai.google.dev/gemini-api/docs/models>) that your key can use, set
`COPILOT_LLM_MODEL` in `.env`, and check it works with the cheap smoke command in
[`rag.md`](rag.md). An unknown model yields a clear `LLMConfigurationError` (HTTP 404 mapping).

### Thinking models and `MAX_TOKENS` (confirmed on a live Gemini call, see below)

Some Gemini models "think" before answering, and the hidden thought tokens are billed against the
*same* `max_output_tokens` budget as the visible answer. Per the installed `google-genai` 2.24.0
SDK's own field documentation, `usage_metadata.total_token_count` is the sum of
`prompt_token_count` + `candidates_token_count` + `tool_use_prompt_token_count` +
`thoughts_token_count`; `max_output_tokens` in `GenerateContentConfig` caps generation overall, and
a `thinking_config` (`thinking_budget` / `thinking_level`) exists per-model to control how much of
that budget the model may spend thinking, but this project does not set it and none is exposed as
a setting yet. Consequences for this project:

* A question that makes the model think a lot can leave few or zero tokens for the visible answer,
  finishing with `MAX_TOKENS` even though `completion_tokens` (`candidates_token_count`, the
  *visible* tokens) is small — the "missing" tokens went to thinking, not to a parsing error.
* `LLMResponse.reasoning_tokens` (from `usage_metadata.thoughts_token_count`) makes this visible:
  `prompt_tokens + completion_tokens + reasoning_tokens` approaches `total_tokens` when it is the
  explanation. It is provider-neutral by name (other reasoning-capable providers report an
  analogous count) and `None` when a provider/model does not report it; it changes no behaviour by
  itself. `python -m copilot.rag ask --show-context` and `GenerationInfo` surface it.
* This is a **generation/truncation** condition, not a client bug: the (possibly very short, but
  non-empty) visible text is still returned rather than raised as an error; only a *fully* empty
  or blank reply raises `LLMResponseError`. If truncation is a problem in practice, raise
  `COPILOT_LLM_MAX_OUTPUT_TOKENS`, since today the budget is shared with thinking and this project
  does not configure `thinking_budget`.
* A real Milestone 6 live run reproduced exactly this shape; see the Q3 entry in
  [`rag.md`](rag.md#live-smoke-test-native-windows-2026-09) for the actual numbers.

## Errors

All failures are typed subclasses of `LLMError` with **fixed messages** (provider name, HTTP status,
what to check). Provider exceptions are re-raised `from None`, because the SDK's own error text
includes the raw response body, which can echo request content.

| Situation | Error |
|---|---|
| model/key unset, provider unsupported, HTTP 404 (unknown model) | `LLMConfigurationError` |
| HTTP 401/403 | `LLMAuthenticationError` (note: Gemini returns HTTP 400 for an invalid key; that maps to `LLMProviderError` whose message says so) |
| HTTP 429 | `LLMRateLimitError` |
| HTTP 408/504 or client timeout | `LLMTimeoutError` |
| network/DNS/TLS failure | `LLMConnectionError` |
| other HTTP status or unexpected SDK failure | `LLMProviderError` (`.status_code`) |
| blocked prompt, empty or non-text reply | `LLMResponseError` |
| a secret in the outbound text | `RepositorySecretRiskError` (from `copilot.security`, raised before any provider call) |

There are no automatic retries: a rate-limit failure is reported once, so cost and behaviour stay
visible. The comparison harness offers `--delay-seconds` to space requests instead.

## Live tests (opt-in)

Normal `pytest` never contacts a hosted API and CI must not depend on Gemini. The live tests
(`tests/integration/test_llm_live.py`, marker `llm_live`) run only when
`COPILOT_RUN_LLM_LIVE=1` and both `COPILOT_LLM_MODEL` and `GEMINI_API_KEY` are available
(environment or `.env`); otherwise they skip with a reason. They use synthetic, non-sensitive
prompts and check mechanics only (non-empty reply, metadata, that the gate still refuses).

```powershell
$env:COPILOT_RUN_LLM_LIVE = "1"
uv run pytest tests/integration/test_llm_live.py -v -s
```

## Limitations

* Two providers (Gemini, Ollama). Adding another means one new class implementing `LLMClient`; nothing else changes. The Ollama provider was exercised against a real server for one model only (`llama3.1:8b`, one validation run); other models and Ollama versions are untested.
* No streaming, no response cache, no retry/backoff, no token-cost accounting beyond what the
  provider reports.
* Latency and token counts come from the provider; they vary between runs.
