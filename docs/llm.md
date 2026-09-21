# The LLM layer (Milestone 6)

Package `copilot.llm`. It is the only code that can send text to a hosted model, and every request
passes the content-secret gate first. It contains no retrieval or RAG logic.

## Components

| Piece | Role |
|---|---|
| `LLMClient` (Protocol) | `provider`, `model`, `generate(request) -> LLMResponse`. Small on purpose; only Gemini is implemented, so there is no registry or plug-in machinery. |
| `LLMRequest` | Frozen model of *exactly* what leaves the machine: `system_instruction`, `user_prompt`, `temperature`, `max_output_tokens`, `prompt_version`. `outbound_text()` returns the sent text; `scan_targets()` returns the same two strings for the gate. The texts are excluded from `repr`. |
| `LLMResponse` | `text` (never empty), `provider`, `model`, `finish_reason`, token usage, `latency_seconds`. No request text, no credentials. |
| `FakeLLMClient` | Deterministic, offline; records every request in `.requests`. Used by all normal tests. |
| `GeminiLLMClient` | The hosted client (official `google-genai` SDK). |
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

Note for "thinking" models: hidden reasoning tokens can consume `max_output_tokens` and yield an
empty reply with finish reason `MAX_TOKENS`. The client reports that explicitly; raise
`COPILOT_LLM_MAX_OUTPUT_TOKENS` if you see it.

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

* One provider. Adding another means one new class implementing `LLMClient`; nothing else changes.
* No streaming, no response cache, no retry/backoff, no token-cost accounting beyond what the
  provider reports.
* Latency and token counts come from the provider; they vary between runs.
