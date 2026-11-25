# Hosted judge validation plan (Gemini as a candidate judge)

Milestone 11B closeout. **Nothing in this document has been executed.** No hosted call has been
made, no model has been checked for availability, and no hosted judge result exists. Gemini is a
*candidate* judge; it is validated only if it passes the same predeclared criteria the local judge
failed ([`judge-validation-llama3.1-8b.md`](judge-validation-llama3.1-8b.md)). The criteria,
thresholds, controls, judge prompt (`gen-judge-2`) and rubric (`gen-rubric-1`) are unchanged.

## 1. Hosted judge selection

The judge is chosen on the command line and is independent of the generation model:
`--judge-provider gemini --judge-model <ID>`. The generation model stays whatever the frozen
protocol confirms (`gemini/gemini-3.6-flash`). There is **no default judge model**: the commands
refuse to run without `--judge-model`.

| Setting | Value (code as it exists) |
|---|---|
| Provider | `gemini` through `GeminiLLMClient`, wrapped by the secret gate |
| Judge model id | **you name it**; this project does not and cannot (offline) say which ids your key may use |
| Temperature | 0.0 (protocol `judge.temperature`) |
| Max output tokens | 4096 (protocol `judge.max_output_tokens`); reasoning ("thinking") tokens count against it |
| Seed / JSON mode | none: the client sets neither, so hosted output is **not** claimed deterministic |
| Prompt / rubric | `gen-judge-2` / `gen-rubric-1`, unchanged |
| Key | `GEMINI_API_KEY` from the environment or `.env`; never printed |
| Cache | `<judgements-dir>/cache-judge` unless `--no-cache`; checked only after the secret gate |
| Retries | up to 5 attempts per request (`--retry-max-attempts`), only for HTTP 429/503/504 and timeouts; delays 5 s doubling to 60 s |

**Which id to pick.** Prefer a model that differs from the generator: an LLM judge can favour text
from its own model (self-preference). A different Gemini model reduces that risk but does not remove
it, because both come from one vendor family; the report states which case applies
(`same_model_as_generator`, `self_preference`). If you use `gemini-3.6-flash` itself, results that
depend on the judge must say the judge graded its own model's answers. `gemini-3.6-flash` is the
only id this project has used live; other names in `docs/ARCHITECTURE_PLAN.md` are planning-stage
candidates and **none is verified available to your key**. Check the model list and quota in Google
AI Studio yourself, then pick.

**Declare it before the first call.** The hosted profile deliberately does not name a model. Record
your choice in the repository history before any hosted call, so the timeline is verifiable:

```powershell
git commit --allow-empty -m "M11B: declared hosted judge model <the id you chose> before validation"
```

## 2. Offline preflight is not an availability check

`judge-preflight` for a hosted judge is **offline and makes zero provider calls** (its output says
`"hosted_calls_made": 0`). It checks:

* a judge model id was given and the provider is `gemini`;
* a `GEMINI_API_KEY` is configured (presence only, never shown);
* the judge/generation relation (same model, or same vendor family);
* the hosted profile is pinned to the exact `judge-validation-v1.json` (sha256);
* the controls file loads (its sha256 is shown) and the judgements directory is outside the primary one;
* the call budget of every stage (section 7).

It does **not** show that the key is valid, that the model id exists or is permitted for your key,
that you have quota left, or that the model accepts the request. Those become known only at the first
real call. A wrong id or key fails on that call with a non-retryable error (section 8 explains how to
stop early).

```powershell
$JUDGE_MODEL = "<the Gemini model id you declared>"      # no default; not named by this project
$GEN = "gemini/gemini-3.6-flash"
$V = "data/judge-validation/gemini-judge-1"                # validation artifacts; no model id in the path

uv run python -m copilot.evaluation.generation judge-preflight --judge-provider gemini --judge-model $JUDGE_MODEL --confirm-model $GEN --judgements-dir "$V/controls-1"
uv run python -m copilot.evaluation.generation plan --pilot --confirm-model $GEN
```

`plan` is also offline. Exit code 3 from the preflight means a check failed (for example no key).

## 3. Validation criteria for a hosted judge

All thresholds are read from `generation-eval/judge-validation/judge-validation-v1.json`; the
profile `judge-validation-v1-hosted.json` pins that file's hash and changes no number. Meeting the
T and S criteria means no pilot-level red flag; it does not validate the judge. Only P1-P4 on the
primary human audit do.

| Id | Hosted treatment |
|---|---|
| T1, T2, T4 | **Unchanged.** T2 still needs all 18 first-pass controls to parse; T4 (secret gate, planted token never in an artifact) matters more for a hosted provider |
| T3 | **Adapted to the output cap only**: no accepted judgement with completion + reasoning tokens at or above the protocol cap (4096), and token counts must be present. The input-window half has no hosted counterpart (no client-side window is configured) |
| T5 | **Replaced by `T5H`**: every artifact says provider `gemini` and exactly the id you pass to `--expect-judge-model`, consistently. A hosted API exposes no digest, quantization, seed or fingerprint |
| T6 | Independence checks unchanged (no cache, separate directory, nothing reused) plus: both passes used the same hosted model |
| S1, S2, S3, S4, S5 | **Unchanged** (agreement >= 0.75 on the 8 pilot answers with fewer than 4 rated being inconclusive; zero two-point disagreements; all fabricated/contradicted/invented controls score 0; abstention discrimination; >= 0.95 repeat consistency) |
| P1-P4 | **Unchanged**, evaluated once on the primary audit; undefined kappa is "not met"; failure labels conclusions EXPLORATORY |

Genuinely inapplicable for a hosted API (listed in the profile): model digest, quantization, input
context size, seed, a server-side JSON mode and the runtime fingerprint. A provider may re-point a
model id over time and this client records only the id you requested, so do not run validation
across a provider-announced model change, and record the date of each run.

Procedural requirement (also in the profile): **do not resume a run to re-attempt items that failed
with "invalid judge output"**. Re-calling a nondeterministic hosted model until the output parses
would hide the format unreliability T1/T2 exist to measure. Resume only after transport
interruptions (429/503/504, timeouts, a breaker stop).

## 4. The blinded human worksheet (independent of any judge)

`audit-worksheet --pilot` reads only the generated pilot answers and the benchmark. It reads **no
judgement file**, needs no model and no network, and can be run before, after or without any judge.
It includes **all 8** pilot answers (4 questions x 2 arms).

```powershell
uv run python -m copilot.evaluation.generation audit-worksheet --pilot
```

It writes, beside the pilot generation files (new files only; the generated answers are not touched):

| File | Content |
|---|---|
| `data/generation-eval/pilot/audit-worksheet-PILOT.csv` | the 8 items to rate; human fields empty |
| `data/generation-eval/pilot/audit-key-PILOT.json` | item id to repository, question and arm. **Do not open it while rating** |
| `data/generation-eval/pilot/audit-population-PILOT.json` | what was eligible (8 of 8) |

Verified offline on a scratch copy: 8 rows; two independent runs produced byte-identical worksheet
and key (the order is a seeded shuffle of the answers, so it is deterministic); the CSV has no arm,
model, judge-rating or expected-verdict column; the human columns are empty; `[Source n]` labels are
rewritten to `[cite]`; the key is a separate file. Nothing populates a rating automatically.

**Unavoidable arm-revealing characteristics. The worksheet is not truly blinded.** The arm is not
written anywhere, but the answers themselves differ:

* answers from the repository-aware arm carry `[cite]` markers (4 of the 8 answers do);
* answers from the plain arm state that they have no access to the repository (4 of the 8 do);
* the two sets above did not overlap in the scratch check, so an attentive rater can very likely
  tell the arms apart from the text alone (I did not open the key to confirm the mapping);
* each question appears twice, once per arm.

The `[cite]` rewrite is deliberate: the judge sees the same text, so a human-vs-judge comparison
uses identical input. Removing the markers or the access statements would change what is being
evaluated. Treat the ratings as unblinded-in-practice: rate each answer strictly against the
reference key facts and source evidence in its own row, and do not let the answer's style or its
citations influence a verdict. Do not describe this audit as blinded in a report without this
caveat.

### Rating format

Rate every row. Do not open the key or any judge output first. Save as CSV **UTF-8** (in Excel:
"CSV UTF-8 (Comma delimited)"); keep the column names.

| Column | What to enter |
|---|---|
| `human_fact_verdicts` | for answerable questions, one verdict per key fact listed in `reference_key_facts`: `f1=supported;f2=missing;f3=contradicted`. `supported`: the answer conveys the fact consistently with the reference. `missing`: it does not convey it. `contradicted`: it says something that conflicts with it. For rows that say "(none - the repository cannot answer this)" leave blank. **This column carries completeness**: there is no separate completeness field; a `missing` fact is an incomplete answer |
| `human_groundedness` | `0` most repository-specific claims are unsupported by the reference evidence; `1` mixed; `2` every repository-specific claim is supported (also `2` when the answer makes no repository-specific claim) |
| `human_unsupported_claims` | repository-specific claims a reader could act on that the reference evidence does not support or contradicts (an invented file, function, behaviour or number), separated by ` \| `; clearly labelled general programming knowledge is not counted. Blank if none |
| `human_abstained` | `yes` if the answer itself says it cannot answer, or that the repository evidence is insufficient, for the question or its main part (also when it says part is not shown but asserts repository content for the main part); `no` if it answers, or refuses without saying the repository lacks the information. Exactly `yes` or `no` |
| `human_correctness` | optional `0`, `1` or `2` cross-check only. The correctness used for agreement is **computed from your verdicts by the rubric**, not read from here |
| `human_notes` | optional |

Correctness rubric `gen-rubric-1` (applied by the tool): answerable: 0 if no fact is supported or
more are contradicted than supported; 2 if all are supported, none contradicted and no unsupported
claim; otherwise 1. Unanswerable: 2 if abstained with no unsupported claim; 1 if abstained with
unsupported claims; 0 if it answered. A half-filled row is an error, not a guess.

## 5. Staged execution plan (do not run anything until you decide to)

Use a **fresh directory** for each pass; never reuse one. Stop conditions are in the right column of
section 8. All commands: native Windows PowerShell, from the project root, with the variables from
section 2.

**Stage 0 - offline (0 calls).** Tests green; section 2 preflight; worksheet (section 4); the empty
commit declaring the judge id; check quota and the model list in AI Studio.

**Stage 1 - controls, first pass (<= 18 calls).**

```powershell
uv run python -m copilot.evaluation.generation judge-controls --judge-provider gemini --judge-model $JUDGE_MODEL --judgements-dir "$V/controls-1"
uv run python -m copilot.evaluation.generation judge-validate --judge-profile hosted --expect-judge-model $JUDGE_MODEL --controls-dir "$V/controls-1"
```

The second command is offline. Gate: if T2, S3 or S4 already **FAIL**, the overall result is FAIL
whatever the pilot shows. Stop, record it as a finding like the Llama result, and spend nothing more.

**Stage 2 - judge the four-question pilot (<= 8 calls).** The generated pilot answers are only read.

```powershell
uv run python -m copilot.evaluation.generation judge --pilot --confirm-model $GEN --judge-provider gemini --judge-model $JUDGE_MODEL --judgements-dir "$V/pilot"
```

Gate: the printed counts must show `failed: 0` and `skipped: 0` (T1). RAG system abstentions are
rule-scored (no call) and are not failures.

**Stage 3 - human ratings and agreement (offline, 0 calls).** Fill the worksheet (section 4), then:

```powershell
uv run python -m copilot.evaluation.generation audit-agree --pilot --worksheet "<path to your filled CSV>" --key data/generation-eval/pilot/audit-key-PILOT.json --judgements-dir "$V/pilot" --rater rater-1
```

Exit code 3 means invalid, duplicate, unexpected or missing rows; fix the CSV and re-run. It writes
`$V/pilot/audit-agreement-PILOT.json`. With about 8 answers a kappa is unstable; S1/S2 use exact
agreement and the two-point count, and `judge-compare` refuses a kappa below n = 30.

**Stage 4 - uncached repeatability (<= 18 + 8 calls).** Both passes bypass the cache and use new
directories.

```powershell
uv run python -m copilot.evaluation.generation judge-controls --judge-provider gemini --judge-model $JUDGE_MODEL --judgements-dir "$V/controls-2" --repeat
uv run python -m copilot.evaluation.generation judge --pilot --confirm-model $GEN --judge-provider gemini --judge-model $JUDGE_MODEL --judgements-dir "$V/pilot-repeat" --no-cache
```

**Stage 5 - pilot report and the validation verdict (offline).**

```powershell
uv run python -m copilot.evaluation.generation report --pilot --judgements-dir "$V/pilot"
uv run python -m copilot.evaluation.generation judge-validate --judge-profile hosted --expect-judge-model $JUDGE_MODEL --controls-dir "$V/controls-1" --controls-repeat-dir "$V/controls-2" --pilot-dir "$V/pilot" --pilot-repeat-dir "$V/pilot-repeat"
```

`judge-validate` writes `judge-validation-report-hosted.json` into `$V/pilot`; exit 0 all T/S
criteria met, 3 a criterion failed, 4 evidence still missing. Optional, only if a Gemini-judged pilot
already exists in the pilot directory: `judge-compare --pilot --dir-b "$V/pilot"` (offline).

**Stage 6 - primary experiment. Only after you approve it explicitly, and only after Stage 5 exits 0.**

```powershell
uv run python -m copilot.evaluation.generation plan --confirm-model $GEN --historical data/evaluation/compare-8192-4aaeab9c650ab76c.json
uv run python -m copilot.evaluation.generation run --confirm-model $GEN --historical data/evaluation/compare-8192-4aaeab9c650ab76c.json
uv run python -m copilot.evaluation.generation judge --confirm-model $GEN --judge-provider gemini --judge-model $JUDGE_MODEL
uv run python -m copilot.evaluation.generation audit-worksheet
# rate data/generation-eval/primary/audit-worksheet.csv without opening audit-key.json, then:
uv run python -m copilot.evaluation.generation audit-agree --worksheet "<filled.csv>" --key data/generation-eval/primary/audit-key.json
uv run python -m copilot.evaluation.generation judge-validate --judge-profile hosted --expect-judge-model $JUDGE_MODEL --controls-dir "$V/controls-1" --controls-repeat-dir "$V/controls-2" --pilot-dir "$V/pilot" --pilot-repeat-dir "$V/pilot-repeat" --primary-agreement data/generation-eval/primary/audit-agreement.json
uv run python -m copilot.evaluation.generation report
```

P1-P4 are evaluated here, once. If they fail, conclusions that depend on the judge are labelled
EXPLORATORY.

## 6. Resuming an interrupted run

Re-run the **identical** command with the same `--judgements-dir`, model and flags. Judged items are
kept; failed and missing items are re-attempted; a changed judge, spec hash or run is refused rather
than mixed. Before resuming, look at what failed:

```powershell
# controls
(Get-Content "$V/controls-1/judge-controls-pass1.json" | ConvertFrom-Json).results | Where-Object state -eq failed | Select-Object control_id, error
```

Resume only if every failure is a transport interruption (rate limit, 503/504, timeout). If any says
"invalid judge output", do **not** resume that directory (section 3): evaluate it as it stands.

## 7. Call budgets (upper bounds)

`max calls` = one attempt per item. `with retries` multiplies by the 5-attempt bound; it is reached
only if every request keeps failing transiently. Numbers come from `judge-preflight` and `plan`.

| Stage | Max calls | With retries (x5) |
|---|---|---|
| Controls, first pass | 18 | 90 |
| Controls, repeat | 18 | 90 |
| Pilot judging | 8 | 40 |
| Pilot repeat | 8 | 40 |
| **Validation and pilot total** | **52** | **260** |
| Primary generation | 136 | 680 |
| Primary judging | 144 | 720 |

These are bounds **before** cache reuse and rule-scored items: judge calls fall below them when an
answer is already in the judge cache, is a rule-scored RAG abstention (no model text) or was judged
earlier; primary generation reuses 8 historical RAG answers and any side already generated. The
parser controls (10) and the security control cost no call (the secret gate blocks the security
control before any provider request).

## 8. Quota, retry and stop-condition limitations

* **Retries add provider attempts** (up to 5 per item) and each attempt may consume quota, including
  rejected ones; the 260-attempt figure is the ceiling for one complete Stage 1-5 execution.
* **Resumes add cumulative attempts** beyond the per-run bound: every resume re-attempts failed
  items. Total spend across resumes is not capped by these tables.
* **Circuit breakers differ.** `judge` stops after 3 consecutive rate-limit failures and marks the
  rest `skipped` (T1 then fails until resumed). `run` has `--max-consecutive-rate-limit-failures`
  (default 3). **`judge-controls` has no breaker**: with a wrong model id or key every one of its 18
  items would try and fail (authentication/not-found/permission errors are not retried, so about 18
  calls), and sustained 429s would retry each item up to 5 times.
* **Quota is unknown here.** Per-minute and per-day limits depend on the model and your account and
  are not known to this project; if the daily limit is below 52 calls, split the stages across days
  (completed items are kept).
* **Thinking models** spend reasoning tokens from the same output budget; T3 counts them.
* **Offline preflight proves configuration only**, never availability (section 2).

| Situation | Stop condition |
|---|---|
| First 1-3 control items fail with authentication, permission or "model not found" | Press Ctrl+C. Do not resume. Fix the key or id, **use a new directory** |
| Repeated 429 retry messages during controls | Press Ctrl+C; resume later (the run is checkpointed per item) |
| Stage 1 `judge-validate` shows T2, S3 or S4 FAIL | Stop; record the failed validation; no pilot, repeat or primary spend |
| Stage 2 counts show `failed` > 0 with "invalid judge output" | Do not resume; T1 fails; stop |
| Stage 2 counts show `skipped` > 0 (breaker) | Wait for quota; resume the identical command |
| Stage 5 exit code 3 | Stop; the hosted judge failed; record it |
| Stage 5 exit code 4 | Evidence missing; complete the missing stage; never read it as a pass |
| Stage 5 exit code 0 | Pilot criteria met (not validated); ask for explicit approval before Stage 6 |

## 9. Artifact locations

| Artifact | Location (all under git-ignored `data/` unless noted) |
|---|---|
| Generated pilot answers (never modified) | `data/generation-eval/pilot/generation-*.json` |
| Human worksheet, key, population | `data/generation-eval/pilot/audit-*-PILOT.*` |
| Hosted controls | `$V/controls-1`, `$V/controls-2` |
| Hosted pilot judgements, repeat | `$V/pilot`, `$V/pilot-repeat` |
| Agreement, pilot report, validation report | `$V/pilot/audit-agreement-PILOT.json`, `report-PILOT.*`, `judge-validation-report-hosted.json` |
| Primary generation, judgements, audit | `data/generation-eval/primary` |
| Specification and hosted profile (tracked) | `generation-eval/judge-validation/judge-validation-v1.json`, `judge-validation-v1-hosted.json` |
| Control definitions (tracked) | `generation-eval/judge-controls/controls-v1.json` |
| Preserved Llama result (tracked) | `generation-eval/judge-validation/results/ollama-llama3.1-8b-q4km-v1/` |

After a hosted validation, copy the raw controls result, the validation report and a metadata file to
`generation-eval/judge-validation/results/<name>/` the way the Llama result was preserved, after
checking for credentials and private paths. Never track the judge cache directories.
