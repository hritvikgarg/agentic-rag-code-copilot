# Local judge validation: llama3.1:8b (Q4_K_M) on Ollama - result: FAIL

Milestone 11B, specification `judge-validation-v1`. This is a negative result and it is kept as a
finding. Nothing was changed to obtain a pass: not the controls, the expected verdicts, the
thresholds, the judge prompt (`gen-judge-2`), the rubric (`gen-rubric-1`) or the parser. Work on
local judge models was stopped by decision after this run.

## What was run

| | |
|---|---|
| Model | `llama3.1:8b`, quantization Q4_K_M, 8.0B parameters, Ollama 0.40.18, native Windows |
| Digest | `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e` |
| Runtime | context 8192, max output 1024, temperature 0, seed 20260, JSON mode; 100% on GPU (RTX 4060 Laptop, 8188 MiB) |
| Runtime fingerprint | `50a5ba7e73bf1ca7a01c19bca0c08fc0040bedbadd9a57f2a156c9b37ca626a8` |
| Spec / controls | `judge-validation-v1.json` sha256 `aa8af417...a81a`; `controls-v1.json` sha256 `c6e18e96...6476` |
| Executed | preflight, then the first pass of the 18 model controls plus the 10 parser controls and the security control |

## Results (observed)

| Criterion | Result |
|---|---|
| Overall | **FAIL** (`judge-validate` exit 3) |
| T2 structured-output parsing | **FAIL**: 15/18 parsed; the spec needs all 18 |
| Expected control outcomes (informational, not a spec criterion) | 8/18 met |
| S3 fabricated, contradicted and invented controls scoring 0 | **FAIL**: 4/6 |
| S4 abstention discrimination | **FAIL**: 3/4 |
| Parser regression controls | 10/10 passed |
| T3 context/truncation, T4 security, T5 runtime fingerprint | PASS |
| T1, T6, S1, S2, S5 (pilot judgements, repeatability, human agreement) | **not evaluated** (not run); P1-P4 not applicable yet |

"Not evaluated" is not a pass. Nothing about the judge's agreement with people, or its
repeatability, was measured.

### The three parsing failures

`ctl-io-07`, `ctl-cp-06` and `ctl-cp-07` were recorded as failed items. In each, the model returned
a well-formed JSON object whose `fact_verdicts` was `{}` together with `"abstained": true`,
`"response_type": "insufficient_evidence"`, `"groundedness": 0`. The three questions have key facts
(4, 2 and 2), so the parser raised `EvaluationError: ... verdicts cover [], key facts are [...]`
and refused the judgement, exactly as designed (a missing verdict is never defaulted).

The prompt says: "If the question has no key facts (the repository cannot answer it),
`fact_verdicts` must be `{}`". That empty-set rule is conditional on the question having no key
facts. The model applied it to refusals of answerable questions. **Observed:** the output, the
error, and that the same model produced complete verdicts for refusals in `ctl-io-06`.
**Hypothesis, not tested:** the model treated "the answer contains no repository facts" as "there are
no facts to grade". No prompt or parser variant was tried, so it is unknown whether wording would
change this. The strict parser was deliberately left unchanged.

A further observation from the raw output: for `ctl-io-07` and `ctl-cp-07` (expected
`generic_refusal`) the model answered `insufficient_evidence`, so those two would also have failed
the response-type expectation had they parsed.

### The semantic failures (observed)

| Control | Kind | Expected | Judge returned |
|---|---|---|---|
| `ctl-io-02`, `ctl-cp-02` | partially correct | score 1, one fact supported and two missing | score 2, all three facts "supported" (both) |
| `ctl-io-04` | contradicted fact | score 0 | score 1; one fact contradicted, two "missing", one "supported" |
| `ctl-io-09` | invented symbol | answered, score 0 | `insufficient_evidence`, abstained, score 1 |
| `ctl-cp-09` | invented symbol | answered, score 0 | score 0 but typed `generic_refusal` |
| `ctl-io-08` | mixed | `mixed`, abstained, score 1 | `answered`, not abstained, score 1 |
| `ctl-cp-08` | mixed | `mixed`, abstained, score 1 | `answered`, all three facts "supported", score 2 |

What passed: both correct-substantive answers (score 2), both correct abstentions (score 2,
abstained), `ctl-io-06` (incorrect refusal, score 0, abstained), the confident-fabrication controls
(`ctl-io-03`, `ctl-cp-03`: score 0) and `ctl-cp-04` (contradicted, score 0). The security control
stopped at the secret gate before any request, and the planted token appears in no artifact.

Across the 15 judged controls the model gave credit that the expected verdicts withheld: in three
partial or mixed answers it marked facts supported that were absent, and it misjudged one
contradiction and one invented symbol.

### What the technical passes mean

No judged prompt exceeded 1303 tokens (input budget 7168) and no completion exceeded 191 tokens
(cap 1024), the server returned token counts on every call, and no context-limit failure occurred.
So the infrastructure (provider, context guard, fingerprint, secret gate, isolation) behaved as
designed against a real Ollama server. The failure is in the judge's output, not in the plumbing.

## What this does and does not show

**Shown:** this specific model, quantization, prompt and rubric combination, at temperature 0 with
seed 20260, did not meet the predeclared thresholds on these 18 controls in one pass.

**Not shown, and not claimed:**

* that any other local model, any other 8B model, or the same model with a different prompt or
  quantization would fail or pass;
* that the model lacks the capability for judging (the explanations above are hypotheses; the
  controls cannot separate "capability limit" from "prompt/format mismatch");
* that the result is repeatable (one pass; the repeat was not run);
* anything about agreement with human raters.

With only 18 controls, the 0.95 parse threshold meant one parse failure fails T2; the three seen here
fail it by any reading.

## Consequences for the project

* No judgement from this model is used in any analysis. No pilot or primary judgements were made
  with it.
* The Gemini judge is now the *candidate* hosted judge, not a validated one. It must pass its own
  validation: see [`docs/hosted-judge-validation.md`](hosted-judge-validation.md).
* The predeclared thresholds were not relaxed because this run failed.

## Preserved artifacts

All under `generation-eval/judge-validation/results/ollama-llama3.1-8b-q4km-v1/`:

| File | What |
|---|---|
| `judge-controls-pass1.json` | raw machine-written per-control results, raw model outputs, runtime description and fingerprint, parser and security results |
| `judge-validation-report.json` | the `judge-validate` report (criterion by criterion) |
| `execution-metadata.json` | model digest, quantization, server version, hardware, warm-up, fingerprint, hashes of the two files above; marks which fields were transcribed from console output |

The specification (`generation-eval/judge-validation/judge-validation-v1.json`) and control
definitions with expected verdicts (`generation-eval/judge-controls/controls-v1.json`) stay at their
tracked paths, unchanged since commits `35fa1af` and `cc9f0fd`. A test
(`tests/unit/test_judge_validation_closeout.py`) pins their hashes to the ones recorded in the result
and re-derives the preserved report from the preserved results. The 15 Ollama response-cache files
were not kept. Nothing contained a credential or a machine-specific path (checked before tracking).
