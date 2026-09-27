# Fast, noise-aware iteration for little-coder's self-improvement loop, v2

**Status:** v2. It revises v1 after three critiques: implementability against the code, statistical validity, and a search for missing pieces. Changes from v1 are listed in §15.

**Scope:** PR #16 (`benchmarks/self_improve/`, GEPA 0.1.4, live Aider Polyglot scoring). Inference is oMLX 0.6.3 serving Qwen3.6 hybrid models on one Apple Silicon Mac.

**Evidence base:**
- the PR #16 review against NVIDIA SoL-Pi (arXiv 2609.20519)
- a deep-research pass on replay and simulation, with adversarial verification
- the local oMLX source and settings
- code-level verification in Pi 0.83 `dist/`, GEPA 0.1.4 and the PR branch

File paths are relative to the little-coder repo unless marked.

---

## 1. Goal and budget reality

**Goal.** More *honest* accept/reject decisions per hour of Mac time. "Honest" means the false-accept rate is known, measured and bounded. "More" comes from parallelism, skipping work that provably can't change the outcome, and spending runs only on exercises that carry information.

**Budget reality.** Assume 1–2k exercise runs per experiment and about 180 s per run.
- A child evaluated properly costs about 64 runs.
- That buys **20–30 decisions at the expensive stage (T4) per experiment**.
- So the cheap stages must cut candidates 3–5× before T4, or T4 consumes the whole budget.
- With about 25% of exercises uncertain, the **minimum detectable effect is about +0.2 pass probability on uncertain exercises**. That is roughly +3–5 pp overall. Smaller gains are undetectable at this budget, and the manifest says so.

**Non-goals.**
- Replacing live scoring for accept decisions.
- Optimizing Pi internals.
- Simulated environments or LLM world models as scorers. The research found a sim-to-real gap of about 8 pts and silent state errors, so they are unsuitable.

## 2. Facts the design rests on (all verified)

| # | Fact | Evidence |
|---|---|---|
| F1 | Each Polyglot attempt is a **fresh `pi` process with a fresh session and exactly one user prompt**. Attempt 2's prompt is the original prompt plus the last 4000 characters of the test output. | `aider_polyglot.py:872-905`, `:1018-1040` |
| F2 | The harness already saves `workdir_<n>/` (pre-scoring copy) and `final_output_<n>.txt` for each attempt. | `aider_polyglot.py:430-445`, `:992` |
| F3 | skill-inject and knowledge-inject select **once per user prompt**, in `before_agent_start`. In Polyglot, skill selection is intent-only, because `lastFailedTool`/`recentToolCalls` start empty in each process. The error-recovery path never fires. | `skill-inject/index.ts:566-579`, `:170-205`; `knowledge-inject/index.ts:94` |
| F4 | Selected component ids are already reported through `ctx.ui.notify`, captured by `rpc_client`, and consumed by the adapter (`merge_component_usage`). Content hashes are not. | `skill-inject:644-658`, `knowledge-inject:163`, `polyglot_adapter.py:90-96` |
| F5 | Injected blocks are tail `custom` → `user` messages (default `message` mode, #73). The system prompt (AGENTS.md via `--system-prompt`) is not stored in sessions. | `_shared/inject.ts`, `agent-session.js:900-910` |
| F6 | pi-ai already parses oMLX `cached_tokens` into `usage.cacheRead`, and `rpc_client` sums it. It is **dropped** before the results JSON, `LiveRunResult` and `spend_log`. | `openai-completions.js:1076-1092`, `rpc_client.py:885-897`, `aider_polyglot.py:~1067` |
| F7 | GEPA 0.1.4 provides four hooks. (1) A custom `acceptance_criterion` sees `eval_before`/`eval_after`/`objective_scores`. (2) `evaluate` may run repeats and report `num_metric_calls`. (3) `objective_scores` plus `frontier_type="objective"\|"hybrid"` affect parent selection. (4) `propose_new_texts` returning `{}` skips the child evaluation, but the **parent minibatch has already been evaluated by then**. | `api.py:97`, `core/adapter.py:34-35,206-227`, `core/state.py:22`, `reflective_mutation.py:130-150,306` |
| F8 | `live_eval.run_batch` is **sequential, in a single worktree**. oMLX is configured for 8 concurrent requests. | `live_eval.py:390-432`; `~/.omlx/settings.json` |
| F9 | oMLX has paged, prefix-shared KV (256-token blocks), a 4 GB RAM tier, and a 50 GB SSD tier that persists across restarts (currently 51 GB used). It batches continuously (8). For hybrid GDN models it stores boundary snapshots, so reuse is block-aligned. | oMLX source and settings |
| F10 | Session resume via `--session` works only after rewriting the header `cwd`; Polyglot temp directories are deleted. The SDK can `agent.continue()` from a tool result but bypasses `before_agent_start`, so there is no turn cap or thinking budget. RPC has no continue command. `appendEntry(customType, data)` is positional and not in the model's context. | `main.js:498-521`, `agent.js:229-251`, `loader.js:247`, `rpc-mode.js` |
| F11 | Bugs: the cache key uses raw text while the sanitized text runs; `skip_perfect_score` is off; a wall-clock overrun loses output; `harness_error` scores 0; `split_train_val` is unstratified. | review §Bugs, `live_cache.py:36-60`, `exercises.py:65-76` |

## 3. Principles

- **P1 Pre-registered manifest.** Before search, the manifest fixes: metrics, tolerances, the minimum detectable effect, the α values, the exercise classes and splits, searchable components and knobs, the budget, and the environment fingerprint. The optimizer cannot edit it.
- **P2 Three disjoint splits:** search, acceptance and test. The acceptance and test splits are each touched only by frozen candidates, at most a capped number of times.
- **P3 Cheap stages only reject; only live runs accept.** Every filter logs a propensity score (the probability it let the candidate through), so its effect can be measured.
- **P4 Never score from replayed or simulated steps past the point where the candidate first differs.** Reuse is allowed only where the candidate *provably* could not have changed anything (§6).
- **P5 Lineage-level non-inferiority.** Correctness is guarded against the **frozen root**, not only the parent, so tolerances can't compound (SoL-Pi lost about 6% this way).
- **P6 Reproducibility.** Every sample records its full condition: candidate hash, environment fingerprint, concurrency, and cache condition.

## 4. The evaluation stages

```
GEPA iteration
 ├─ parent minibatch eval ── served from the sample store when the condition matches (§8)
 ├─ GEPA's default reflection → proposal      (propose_new_texts stays None; see below)
 ├─ child eval: adapter.evaluate(child) ──────────────────────────────────────────────
 │    T0  static gates:   sanitize · dedupe · leak lint · token budget
 │    T0r relevance:      deterministic re-selection on recorded prompts → which (exercise, attempt) can this candidate touch?
 │    T0v reviewer:       independent LLM review (coherence, leakage, addresses the stated failure cluster)
 │    T1  oracle screen:  LLM judge over recorded failing trajectories + diff (optional; enabled only after M5)
 │    → rejected: return with outputs.flag = "rejected:<stage>", zero rollouts, num_metric_calls = 0
 │      (25% of rejects pass through at random, propensity logged)
 │    T4  sequential paired test on the uncertain set; exact reuse via §6
 └─ acceptance_criterion ── custom: reads the flag + the T4 verdict + guards + root non-inferiority
Frozen candidates (≤3 per run, chosen by the rule in §7.5) ── T5 on the acceptance split, outside GEPA
Test split ── reported once, outside GEPA
```

**Why the filters run inside `evaluate()` and not in `propose_new_texts`.**
- Setting `propose_new_texts` **replaces** GEPA's reflective proposer (`reflective_mutation.py:146`). Using it would mean reimplementing reflection.
- The adapter pins it to `None` deliberately (`polyglot_adapter.py:136-148`, which documents the AttributeError bug this avoids).
- Filtering inside `evaluate()` costs no extra hook. It saves the same rollouts, because the parent minibatch is already evaluated either way.
- The custom acceptance criterion reads the reject flag, so a rejected child can never be accepted.
- T0v's one allowed revision is therefore dropped. A failed review simply rejects, and GEPA proposes again.

The v1 **T2 mid-attempt probe** and general **session forking** are deferred to §13 (Phase 3). They need an SDK runner that enforces turn caps from outside (F10). T3 has been folded into T4 as exact reuse (§6).

## 5. Exercise classes and splits

**Classes (from M1).** Classify each exercise by its baseline pass probability p̂_i:
- **uncertain**: 0.15 < p̂ < 0.85
- **solid**: p̂ ≥ 0.85
- **zero**: p̂ ≤ 0.15

**Splits.** Split the 34 Python exercises three ways, **stratified by class**, so that the acceptance and test splits each get about 4–5 uncertain exercises:
- search ≈ 14
- acceptance ≈ 10
- test ≈ 10

**Enforcement.** A new `split_stratified()` in `exercises.py` builds the splits, `select_exercises(exclude=)` keeps them disjoint, and the seed is recorded in the manifest.

**Second language.** Add one (e.g. JavaScript, 49 exercises) to the acceptance and test pools **for AGENTS.md candidates only**, because skill cards are Python-specific.

**Where power comes from.** Power is limited by the uncertain count, not the total, so it is bought with repeats (k), not more exercises.

## 6. Exact reuse ("forking" as Polyglot actually runs)

Polyglot attempts are separate processes (F1), and the harness already saves each attempt's files and output (F2). The only exact fork point is **between attempts**, and it needs no session machinery.

### 6.1 Deterministic relevance check (T0r)

For each recorded parent sample, re-run the injectors' **selection functions** offline on that sample's recorded prompts, under the candidate text.
- The functions are `selectSkills` / `scoreEntry`, exported for a Node CLI: `node .pi/extensions/_shared/select-cli.ts --components <dir> --prompt <file>`.
- Selection must be re-computed because card token cost and knowledge keywords change with the text.

Output per (sample, attempt): `untouched` | `touched`.
- A component is **touched** if the candidate's version of it would be injected in that attempt, or if the candidate changes which components are injected.
- An **AGENTS.md** candidate touches every attempt.

### 6.2 Reuse rules

| Case (per recorded parent sample) | Action | Validity |
|---|---|---|
| Both attempts untouched | **Reuse the parent sample as a child sample** | Exact: every prompt the model sees is identical, so the distributions match |
| Attempt 1 untouched, attempt 2 touched (the card is only selected from the test-output tail) | **Rerun attempt 2 only:** restore `workdir_1/`, prompt = original + `final_output_1[-4000:]`, run one attempt | Exact conditional on attempt 1 (§6.3 estimator) |
| Attempt 1 touched | Full live run | — |

Implementation: an `aider_polyglot.py --resume-attempt 2 --from-log <dir>` flag. Size M.

### 6.3 Estimator for mixed full runs and attempt-2 reruns

For binary pass (the primary metric):

ΔP = P(fail@1) · (q_child − q_parent), where q = P(pass@2 | fail@1)

- **P(fail@1)** is estimated from all full runs, parent and child pooled, because attempt 1 is identical when it is untouched.
- **q** is estimated from attempt-2 reruns on fail@1 records **drawn uniformly at random**, paired with parent vs child on the same record. Variance is clustered by record.
- **Never pool** rerun outcomes with full-run outcomes as if they measured the same quantity.
- **Score delta:** ΔScore = 0.7 · ΔP(fail@1 → pass@2).

### 6.4 What doesn't fork

- **AGENTS.md** candidates: always full runs. The prefix cache still helps (§9).
- **Mid-attempt forks** need the SDK runner (§13).

Expected savings are decided by M2: the share of (exercise, component) pairs untouched in attempt 1, or in both attempts.

### 6.5 Consequence: skill cards have a larger minimum detectable effect

Row 1 reuse makes untouched exercises free. It also means they carry **no information** about the candidate.
- **Effective sample.** A skill card's effective sample is `touched ∩ uncertain`, which may be only 2–3 exercises. Its MDE is therefore larger than an AGENTS.md candidate's, which touches everything.
- **Gate.** T0r **rejects** any card with fewer than 4 touched-uncertain search exercises. Such a card can't be evaluated at this budget.
- **Relevance scope.** §10.4 (searchable components) is derived from the same M2 counts.

## 7. Scoring and acceptance protocol

### 7.1 Metrics (pre-registered)

**Primary: binary pass.** Secondary is the attempt-weighted score. With `--max-attempts 2` the attempt-weighted score takes only the values 1.0 / 0.7 / 0; the 0.4 level is unreachable.

**Efficiency: decode tokens + total prompt tokens.**
- It is independent of cache state, and paired per exercise over **all** runs.
- It is not "tokens per pass", which conditions on the outcome.
- The cached ratio and wall-clock time are **operational metrics**, reported but never used to decide acceptance.

**GEPA frontier.** Pass `objective_scores = {pass, -tokens}` with `frontier_type="hybrid"`, so parent selection can explore efficient lineages (F7). Acceptance stays custom.

**Timeouts and errors.**
- `harness_error` (the harness failed; no agent outcome exists) is **excluded**. It is retried in place (`HARNESS_ERROR_RETRIES`), and if it persists the run stops with `LiveEvalHarnessError` (exit 4, partial results written). It is never scored or cached. A config `error` (missing exercise, unknown agent, missing JS deps) stops the run the same way, without retrying.
- Runtime `error` / `empty_response` are retried in place, then **scored 0.0** and kept out of `live_cache`. The environment or the candidate (for example, context overflow) can cause them, and the harness does not guess which. GEPA's own EvaluationCache still records the 0.0 for the rest of that `optimize()` call.
- Exposure to a dead model server is bounded by the live budget (wall clock and run cap), not by a breaker. It shows up in `spend_log.jsonl` as each exercise's `status`, plus `error` when one was recorded. Pi crashes and empty responses usually carry no reason, so the status is the signal. `summarize()` gives `by_status`.
- `fail_timeout` is **scored** (a looping agent is a candidate outcome) and not retried. Today's per-attempt deadline is wall clock, so it is kept out of `live_cache`. Once timeouts are measured in turns (planned), they would be cached like any other outcome.

### 7.2 T4: the child evaluation, inside `adapter.evaluate`

GEPA calls `adapter.evaluate(parent_minibatch)` and then `adapter.evaluate(child_minibatch)` in the same iteration. The adapter keeps the most recent parent-candidate id in its state, so the child call can pair against it. Valset re-evaluations go through `adapter.batch_evaluate` (`engine.py:183`). The adapter overrides `batch_evaluate` to run a plain fixed-k evaluation and **never** the sequential procedure (§7.5). Stages A–C are ordered cheapest first.

**A. Uncertain set** (about 8 search-split exercises):
1. Apply the §6 reuse rules.
2. Run the child in rounds: k=4, then k=8, at most k=12.
3. **The sequential decision uses the betting e-value alone.** The e-value is built on the paired per-exercise differences (child minus condition-matched parent samples) and is anytime-valid, so looking after every round does not inflate α.
   - **Success** when E ≥ 1/α_T4 = 100.
   - **Futility** when E < 0.1 at any look.
   - **Undecided at k=12** means reject.
4. The stratified pooled z-statistic, z = Σ(b_i − a_i) / sqrt(Σ 2k·p̂_i(1 − p̂_i)), is **reported only**. It is valid as a test only as a single pre-registered look at k=12, and it is never OR-ed with the e-value. Mixing a fixed-k z threshold into repeated looks is optional stopping, and v2-draft had that bug.

**B. Solid guard** (search-split solid exercises): k=1 each. **Reject on 2 or more failures** (about a 5% chance false alarm at p=0.97).

**C. Zero watch** (search-split zero exercises): k=1 each. A pass is a **separate signal**, not part of A. Including it in A would make which exercises count depend on the data.
- A pass triggers 4 more repeats on that exercise.
- Two or more passes out of 5 set `zero_flip = true`, which is reported and used to prioritise the candidate for T5 but does not by itself accept.

**Efficiency-only accept.** Correctness must be non-inferior: the lower bound of a one-sided 90% CI must exceed −2 pp against **both the parent and the root**. Tokens must show a paired 20%-trimmed-mean reduction with bootstrap p < 0.01.

**Parent samples.**
- **Condition match.** Parent samples are drawn from the sample store only when their condition matches (environment fingerprint, concurrency band, cache condition). Otherwise the parent is **re-sampled live in the same batch** as the child. This fixes v1's impossible "same cache path" rule.
- **Re-baseline.** After an accept, draw k **fresh** samples for the new parent on the uncertain set and never reuse the selection samples. This counters the winner's curse.

**Scores returned to GEPA.** The mean of k in `scores`; raw samples in `outputs`; the true `num_metric_calls`.

### 7.3 `acceptance_criterion` (custom)

Accept only if **all** of the following hold:
- **Correctness or efficiency:** T4 reached success, or the efficiency-only criteria are met.
- **Solid guard:** not tripped.
- **Root budget:** lineage non-inferiority against the root holds, with a total budget of 2 pp.
- **Ledger:** below the false-accept budget (§7.6).

### 7.4 T5 and test

- **Cap.** At most **3 frozen candidates** per run.
- **Acceptance split:**
  - uncertain exercises at k=12, one-sided stratified z-test at α = 0.05/3;
  - plus non-inferiority against the root;
  - plus the solid guard;
  - plus the **cross-benchmark canary**: a small, fixed gaia/tb subset. This is required for AGENTS.md and any shared protocol skill.
- **Test split:** evaluated once, with Wilson CIs, for reporting only.

### 7.5 GEPA wiring and split discipline

`gepa.optimize()` re-evaluates **every accepted candidate on its valset** to compute `best_idx`. The valset must therefore not be the acceptance split, or P2 breaks.

- **GEPA sets.** `trainset` and `valset` are both the **search split**. The valset is the search split's uncertain exercises at a plain fixed k=4, run through `batch_evaluate`. GEPA's `best_idx` is used only for parent selection inside the run.
- **T5 runs outside GEPA**, after `optimize()` returns or when the budget is hit. At most 3 frozen candidates are chosen by a pre-registered rule: the top of the search-split (pass, −tokens) Pareto frontier by pass, ties broken by tokens, with `zero_flip` candidates prioritised.
- **The test split** is evaluated only by a separate `report_test.py`, and only once per experiment.
- **Sampling temperature** is pre-registered in the manifest at the production default. It must be > 0, because temperature 0 removes the variance we are estimating and makes M1 meaningless.

### 7.6 False-accept ledger

- Track the expected false accepts: Σ α_T4 over every T4 test in the run.
- If it passes 0.5, stop the run or tighten α.
- M1b (§12) calibrates the whole protocol empirically.

## 8. Sample store and cache (replaces `live_cache`'s single-score semantics)

**Key.** (sanitized candidate hash, exercise, attempt-mode {full | a2-rerun:<record>}, env-fingerprint, sample index).

**Value.** A full `LiveRunResult` including `usage` (F6), turns, status, the injected component ids and content hashes, concurrency, and the cache condition (cached/prompt ratio).

**Env-fingerprint:**
- model id, quant and file hash
- oMLX build (e.g. `HEAD-a20ec09`)
- output-changing model settings: `turboquant_kv_enabled`, `specprefill_enabled`, `dflash_enabled`, `mtp_enabled`
- sampling params and `n_ctx`
- Pi version, and the little-coder commit of every non-candidate file (already hashed)

These are read from `~/.omlx/settings.json`, `~/.omlx/model_settings.json` and `/v1/models`.

**Rules.**
- `fail_timeout` is cached like any other outcome once it is turn-measured; today's wall-clock `fail_timeout` is kept out of `live_cache`.
- `harness_error`, `error` and `empty_response` are never cached.

## 9. Inference layer (oMLX)

1. **Parallelism.** A worktree pool of N scratch worktrees (N = 4–6, fixed per experiment) runs exercises concurrently. This fixes F8, the biggest cheap win.
   - Concurrency is part of the sample condition.
   - Parent and child that are compared must share a concurrency band.
2. **Prefix warmup.** Before each batch, send one `max_tokens=1` request with the candidate's system prompt and tools. Otherwise N concurrent cold starts all miss the shared prefix.
3. **Prompt order.** Stable header → tool definitions → optimizable AGENTS.md section **last**. Verify the rendered order with the chat template.
   - Never evaluate with `LITTLE_CODER_INJECT_MODE=system`.
   - Attempt 2 shares the system prompt, tools and original prompt with attempt 1, so it is prefix-cached across processes.
4. **Exercise-major scheduling.** Run every candidate, repeat and rerun for one exercise before moving to the next, so the trunk stays in the 4 GB RAM tier.
5. **One cache directory per experiment**, set via `cache.ssd_cache_dir` and sized from M4.
   - Not one per problem: blocks look content-addressed, the setting is server-wide (changing it needs a restart), and the start of the prompt is shared by every exercise.
   - Clear it only on purpose, never in the middle of a comparison.
6. **Speculative decoding.** Try DFlash or MTP (up to about 2–3.7× decode on Qwen3.6-27B has been reported). Measure *aggregate* tokens/s at the chosen N, since the gain shrinks under concurrency, and check that the outcome distribution matches M1 (M4b). The flags are part of the env-fingerprint.
7. **Mid-conversation system messages.** oMLX keeps them if the template supports them. Otherwise it folds them into user notes (`preserve_mid_system_cache: true`), or merges them into the leading system message and busts the cache. little-coder should send none. Assert this once by logging the outgoing roles in a `before_provider_request` hook.
8. **Hybrid models.** Reuse is block-aligned (256 tokens), so check the actual cached/prompt ratio (M4) rather than assuming it.
9. **mlx-serve:** no persistent paged store is evident. Keep oMLX for this work.

## 10. Candidate generation

1. **Propose from failure clusters, not single exercises.** Following SoL-Pi's Oracle Analysis, one analyzer per failing trajectory → a reducer → a clustered failure taxonomy (edit misuse, wrong pytest invocation, turn-cap hit, rereading files, …). Each reflection targets one cluster and cites the overhead or failure source. This lowers leak risk and improves transfer.
2. **Leak guard (T0).** Reject a rewrite that contains any of:
   - an exercise name;
   - an identifier from test or reference files;
   - an expected literal;
   - n-gram overlap of 8 or more tokens with tests or reference solutions.

   Run the same lint on the seed skills (`tree_zipper.md`).
3. **Reflection redaction.** Strip expected values from the pytest tail passed to the reflection model. Keep the assertion shape and the failing test name.
4. **Relevance scope.** Only components that the search-split injection data shows are injected are searchable. Add `PRINCIPLES.md` to `components.yaml`, or drop the claim.
5. **Dedupe.** Check the normalized-text hash, plus near-duplicate embedding similarity, before T0r.
6. **Adherence metric.** For each candidate, record whether its guidance was actually followed. Example: the edit-skill advice says X, and the edit calls show X. Use it as a secondary signal and for explaining results, not for acceptance.

## 11. Parameter track (beyond text)

- **Knobs:** turn cap, thinking budget, `COMPACT_AT_PERCENT`, skill-inject top-k, and sampling temperature.
- **Method:** these are low-dimensional, so use successive halving or TPE, not GEPA. The manifest lists which knobs are searchable.
- **Shared machinery:** the same T4, T5 and ledger. Each knob setting is part of the candidate hash.
- Tool-schema descriptions are text and can become GEPA components, grouped with AGENTS.md for caching.

## 12. Measurement plan

| Step | What | Runs | Go/no-go it decides |
|---|---|---|---|
| **M0** | S-size fixes (§14 phase 0) + worktree pool + usage plumbing | — | — |
| **M1** | Adaptive flip-rate: 4 repeats on all 34 exercises; 6 more on exercises with 1–3 passes; 4 more on exercises with 0 passes. Include 20 runs at concurrency 1 vs N and cold vs warm cache | ~250 | Classes and splits, whether concurrency or cache changes outcomes, the MDE statement |
| **M1b** | A/A: the full §7 protocol, parent vs itself, about 20 times, by resampling M1 data | ~0 | **Empirical false-accept rate.** Go if ≤ 2% |
| **M2** | Relevance: for each component, the share of (exercise, attempt) pairs it touches, via T0r on the M1 records | 0 | How much §6 reuse saves; which components are in scope |
| **M3** | a2-rerun fidelity: rerun attempt 2 from recorded `workdir_1` with the **same** text vs live attempt 2 on the same records. TOST with ±7 pp margin, 30 records × 5 | ~300 | Whether §6.2 row 2 is used. Skip if M2 shows under 15% of pairs qualify |
| **M4** | Cache and throughput: cached/prompt ratio, `du` per run, aggregate throughput at N ∈ {1, 4, 6, 8} | from M1 | N, cache-directory size |
| **M4b** | Speculative decoding on vs off × N; outcome equivalence against M1 | ~60 | Whether spec decoding is enabled |
| **M5** | Filter validation for T1/T0v, per (candidate, record) unit, calibrated with synthetic known-bad (scrambled or ablated card) and known-neutral (paraphrase) candidates | ~100 | Enable T1 only if known-bad rejection is ≥ 0.8 and known-neutral pass is ≥ 0.8 |
| **M6** | First real, budget-capped `gepa.optimize()` run, with the ledger | budget | — |

Order: M0 → M1 → (M1b, M2, M4, all free) → M4b → M3 (conditional) → M5 → M6.

## 13. Deferred (Phase 3)

- **SDK runner** (Node, `AgentSessionRuntime`, in-process) for mid-attempt continuation (`agent.continue()`). It must enforce the turn cap and thinking budget from outside, because `before_agent_start` does not fire (F10). It enables:
  - the **T2 counterfactual probe**: K concurrent continuations from recorded failure points; off-policy, so a filter only;
  - forking for non-Polyglot benchmarks (gaia/tb) via session truncation with a `cwd` rewrite, an fs-snapshot extension (`appendEntry("lc-fs-snapshot", {sha})`), and persisted extension state (inject dedupe, turn-cap, thinking-budget).
- **Early stop for clearly failing runs.**
  - Start rule-based: identical tool calls repeated, no file change in K turns, or a projected turn-cap hit. Apply it identically to parent and child, and audit 10% of stopped runs by running them to the end.
  - A learned predictor (FailFast- or EarlyEval-style) comes later. Calibration doesn't transfer across harness variants, so recalibrate per lineage.
- **Proxy-model screening.** Use a faster sibling model, only if Spearman ρ ≥ 0.5 against target T4 outcomes on M5 candidates.

## 14. Implementation plan

**Phase 0: correctness fixes (S each)**
1. Key the cache on the sanitized text; add the sample index; set `skip_perfect_score=True`.
2. Exclude `harness_error` from scoring and caching (retry in place, then stop the run); retry runtime `error`/`empty_response` in place, then score them 0.0 without caching them.
3. Write partial results on a wall-clock overrun (catch `LiveEvalBudgetExceeded`, then write `optimized_components.yaml` from the best-so-far state).
4. Carry `usage` (prompt, cacheRead, output) through the results JSON → `LiveRunResult` → `spend_log`.
5. Add content hashes to the skill-inject and knowledge-inject notify payloads.
6. Add the manifest schema (`self_improve/manifest.py`) and three disjoint splits using `exclude=`.

**Phase 1: throughput and statistics (M)**
7. Worktree pool with fixed N plus prefix warmup; exercise-major scheduler.
8. Sample store (§8) with the env-fingerprint reader for oMLX settings.
9. Custom T4 procedure in `adapter.evaluate` (e-value sequential test, guards, parent pairing through adapter state), a `batch_evaluate` override for plain fixed-k valset evaluations, and the custom `acceptance_criterion` (reads the reject flag, the T4 verdict, the ledger and root non-inferiority).
10. `split_stratified()` after M1; `run_t5.py` and `report_test.py` outside GEPA (§7.5).
11. T0 gates **inside `evaluate()`**: leak lint, dedupe, token budget. Plus reflection redaction, via the reflective dataset built in `make_reflective_dataset`. `propose_new_texts` stays `None`.
12. T0r selection CLI (export `selectSkills`/`scoreEntry`), the §6.2 reuse logic, and the touched-uncertain ≥ 4 gate; `--resume-attempt 2` in `aider_polyglot.py`.

**Phase 2: quality (M)**
13. Failure-cluster oracle analysis → cluster-targeted reflection prompts.
14. T0v independent reviewer; T1 judge behind the M5 gate.
15. Candidate cards: flipped exercises, token and turn deltas, first divergent injection, an LLM "why it won" drawn from flipped pairs only; a static HTML report from `runs/`.
16. Parameter track (successive halving over the §11 knobs).
17. Governance: a staged PR carrying the manifest, component hashes, lineage and evidence links; a leak-lint report in the review view; a git tag per accepted component version; the T5 canary.

**Phase 3:** §13.

**Testing.**
- Every phase 0–1 item gets `fake_pi.py` coverage. Extend `fake_pi` to emit `usage` and notify payloads with hashes, and to simulate per-exercise pass probabilities, so the §7 protocol can be unit-tested against known p_i. M1b in simulation doubles as a regression test.
- The SDK runner (phase 3) needs its own fake provider (Pi's scripted fake).

## 15. Changes from v1

- **§6 rewritten.** Polyglot attempts are separate processes, so session forking and fs-snapshots aren't needed there. Exact reuse is (a) skipping untouched samples entirely and (b) rerunning only attempt 2 from saved files. "Error-recovery cards on attempt 2" doesn't happen in Polyglot (F3).
- **The acceptance rule was replaced.** v1's P ≥ 0.9 bootstrap over all exercises (about 10% false accepts, 15–46% power) is now a sequential, stratified test on the uncertain set only, with guards, re-baselining, a root budget and a ledger. M1b measures the false-accept rate.
- **Funnel stages moved to the start of `adapter.evaluate(child)`.** They save the child's cost (F7). `propose_new_texts` stays `None`, because setting it would replace GEPA's reflection. Pass-through is 25% with propensity logging. The T2 probe is deferred.
- **Sequential test fixed.** The sequential test is e-value only. The earlier draft OR-ed it with a fixed-k z threshold, which is optional stopping.
- **GEPA's valset = the search split.** T5 and test run outside GEPA.
- **Zero-set passes are a separate signal.**
- **Skill-card MDE caveat** with a touched-uncertain ≥ 4 gate.
- **Temperature pre-registered** at > 0.
- **Cache and pairing.** v1's "same cache path for pairs" was impossible given GEPA's ordering. It is replaced by condition-matched sample reuse or co-batched parent re-sampling.
- **Recording layer shrank.** Most of it already exists (F4, F6); only hashes and usage plumbing are needed.
- **Added:** worktree parallelism, prefix warmup, speculative decoding, failure-cluster proposals, reviewer, adherence metric, parameter track, candidate cards, governance, and a second language for AGENTS.md candidates.
- **Metric fixes.** The 0.4 score level is unreachable with 2 attempts; binary pass is primary. The efficiency metric is decode plus total prompt tokens, not uncached prefill.

## 16. Open questions

1. How much of the prompt does the chat template actually render before AGENTS.md (tools-first or system-first)? This decides how much AGENTS.md candidates can share.
2. KV and GDN-snapshot bytes per token for q36-27b/35b on oMLX. M4 answers this.
3. Does DFlash or MTP in oMLX 0.6.3 give **identical output distributions** under batching? M4b answers this.
4. ~~e-value vs SPRT~~ Resolved: e-value only, for sequential decisions (§7.2). Still open is the betting strategy, meaning how the betting fraction is set. Options are a fixed λ from the MDE, or an adaptive (aGRAPA-style) λ. Choose it by simulation against M1's p_i before M1b.
5. ~~Canary subset~~ Resolved: see §18.2 (6 short always-pass TB 2.1 tasks, about 20–25 min per candidate). A gaia canary is still open.

## 17. Risks

- **The uncertain set is smaller than assumed** (e.g. under 6 exercises). Then even T4 is underpowered. Mitigation: add a second language to search, or accept a larger MDE.
- **The relevance check misses an indirect effect.** For example, a card changes token budgets, which changes which other cards are selected. Mitigation: T0r compares the entire selected set, not just the candidate component, and M3-style spot checks cover the untouched-reuse rows too.
- **Concurrency or speculative decoding change outcome distributions.** M1 and M4b measure this before use.
- **Overfitting to Python Polyglot.** The T5 canary and the second-language pool mitigate it.
- **Budget exhaustion at T4.** Mitigated by the ledger, the futility stop and the funnel reject rate. Track runs per decision from day one.

---

## 18. Terminal-Bench 2.1 via Harbor

**Source.** Dev supports TB 2.1 through Harbor 0.22.0 (`benchmarks/harbor_adapter/little_coder_agent.py`, `harbor_pilot.sh`, `tb21_split.py` → `tb21_splits.json`: 89 tasks, 16 train / 12 validate / 61 test). The PR #16 branch is 144 commits behind dev.

### 18.1 Facts

| # | Fact | Evidence |
|---|---|---|
| H1 | **pi runs on the host, not in the task container.** The container is reached only through Harbor's `environment.exec`, via the `__LC_TB_SHELL__` UI-proxy channel (`_shared/tb-proxy.ts`). `setup()` is a no-op. pi calls oMLX directly at `127.0.0.1:8000`, with no proxy. | dev `little_coder_agent.py:1370, 1693-1696`; `models.json:68-70` |
| H2 | Everything the agent reads comes from the checkout that Harbor imports: `rpc_client.REPO_ROOT` (bound at import), AGENTS.md (`--system-prompt`), `.pi/extensions` (`-e`), and skills (resolved relative to each extension's file). **A scratch worktree works** if Harbor is launched with `cwd=worktree` and `PYTHONPATH=worktree`, the same way `harbor_pilot.sh:101,170` does with the repo root. | dev `rpc_client.py:30-71, 322-324` |
| H3 | Command: `harbor run --dataset terminal-bench/terminal-bench-2-1 --include-task-name … --agent benchmarks.harbor_adapter.little_coder_agent:LittleCoderAgent --model omlx/<id> --jobs-dir … --n-concurrent 1 --timeout-multiplier 15 --override-cpus 4 -y --force-build`. The per-task deadline is `timeout_sec × 15 × 0.9`, so often 203 min. | `harbor_pilot.sh:135-156`; `lca:121, 977-978` |
| H4 | The per-trial `result.json` has `verifier_result.rewards.reward` ∈ {0, 1}, `exception_info`, `agent_execution` timing, `agent_result.n_input/n_cache/n_output_tokens`, and metadata (`stop_reason`, `n_turns`, `n_compactions`, …). `task_name` carries a `terminal-bench/` prefix. | `lca:2199-2237` |
| H5 | **Time.** 98 historical trials (`little-coder-dev/benchmarks/harbor_runs/`): median agent time per task ranges from 2.5 to 321 min. About 9% of trials are harness crashes. The 12-task validate set takes about 1–1.5 days per candidate at concurrency 1. | trial scan |
| H6 | **Allowed tools on TB** are the ShellSession family only. The GEPA components that can affect TB are therefore `agents_md`, `skills_tools_shell_session`, and knowledge cards without unmet `requires_tools`. **The strongest TB lever, `prompt_prefix`, is hardcoded** in the adapter (dev `little_coder_agent.py:~1720`), outside `components.yaml`. | `lca:111`; skill-inject `:176` |
| H7 | **Merging dev into pr16** conflicts in `benchmarks/rpc_client.py` and `.gitignore` (`git merge-tree`). The `rpc_client` conflict lies in the region that holds pr16's `LITTLE_CODER_PI_BIN_OVERRIDE`, which the worktree approach needs, and must also keep pr16's `PRINCIPLES.md` concatenation. The adapter itself merges cleanly. | merge-tree |
| H8 | PR16's `ingest/harbor_tb_ingest.py` has three problems against dev's format: it keeps the `terminal-bench/` prefix; it ignores `exception_info`, so a crash is scored as reward 0; and it treats the job-level `result.json` as a trial when pointed at `harbor_runs/`. | ingest `:90, 98-139` |

### 18.2 Role 1 (now): TB 2.1 is the T5 cross-benchmark canary

This resolves open question 5. TB runs are far too slow for T4 selection: the v2 budget assumed 180-s runs, while TB tasks take minutes to hours. TB's first role is therefore a **regression gate** on the at most 3 frozen candidates per run.

- **Tasks.** Six that passed in every historical trial and are short (median agent minutes):

  | Task | Median min |
  |---|---|
  | prove-plus-comm | 2.9 |
  | fix-git | 3.1 |
  | nginx-request-logging | 4.2 |
  | vulnerable-secret | 5.0 |
  | constraints-scheduling | 6.8 |
  | break-filter-js-from-html | 12.0 |

  These tasks are in the "contaminated" train split. That is fine for a regression gate: the harness is known to handle them, so breaking one is a real regression. It is **not** fine for measuring improvement, and the canary is never used for that.
- **Rule.** k=1 per task. Reject the frozen candidate on **2 or more failures**, plus the parent baseline check below. Harness crashes (`exception_info`) are re-run, never counted.
- **Cost.** About 35 agent-minutes serial. At `--n-concurrent 2` that is about 20–25 min per candidate, or at most about 75 min per run.
- **When it runs.** Required for AGENTS.md and shared protocol or knowledge candidates. Skipped for Polyglot-only tool cards whose T0r relevance on TB prompts is empty (the TB prompt goes through the same selection CLI).
- **Baseline.** n=3 per task is thin evidence for "always passes". Before first use, run the canary 3× on the root (about 1 hour at concurrency 2). Any task that fails on the root is dropped.
- **Pinned conditions.** `--timeout-multiplier` (15), `--override-cpus` (4), the dataset `ref` and the model are **manifest fields**. **Never lower the multiplier to save time**: it changes task difficulty. Any change re-measures the baseline.

### 18.3 `HarborLiveRunner` (Role 1 needs a minimal version)

It implements the same interface as `PolyglotLiveRunner`:
1. **`materialize()`**: unchanged (reset the worktree, write components, dirty-check). With TB, the adapter in the candidate's worktree is the one that runs, and it is part of the non-candidate files hash.
2. **Launch.** `harbor run …` (H3) as a subprocess with `cwd=worktree.path`, `PYTHONPATH=worktree.path`, `LITTLE_CODER_PI_BIN_OVERRIDE`, `start_new_session=True`, a deterministic `--job-name` (candidate hash + batch), and **`--jobs-dir` outside the worktree**, because `reset()` runs `git clean -fdx`. Drop `--force-build` after the first build of each image.
3. **Parse** `<jobs>/<job>/<trial>/result.json` (H4):
   - strip the `terminal-bench/` prefix;
   - `exception_info` → `harness_error` (uncached, re-queued, never 0);
   - reward, `stop_reason`, tokens (input/cache/output feed §7.1 efficiency and the cache-ratio ops metric), agent duration;
   - feedback paths: `verifier/test-stdout.txt`, `agent/little_coder.log`.
4. **Timeouts.** Kill the process group **and** remove the trial's Docker containers (Harbor's `environment.delete` runs only on a clean exit). Label containers via the job name, then `docker ps --filter` → `docker rm -f`.
5. **`run_config` / env-fingerprint.** Add the dataset `ref`, multiplier, cpus/memory overrides, Harbor version, and a hash of the worktree's `little_coder_agent.py` + `rpc_client.py`.
6. **Budget.** Wall-clock only ($0 per token). `--estimate-only` uses the per-task median from the historical scan.
7. **Fix the ingest** (H8), so historical TB trials can be used for M1-style baselines and T1 analysis.

### 18.4 Role 2 (deferred): a TB-native optimization track

This gets its own manifest and budget. **Do not start it** until the following hold:
- **B4.** `prompt_prefix` moves out of `little_coder_agent.py` into a component file listed in `components.yaml`. Without this, GEPA can't reach the strongest TB lever.
- **A TB flip-rate study (TB-M1).** The uncertain tasks (0.15 < p < 0.85, e.g. configure-git-webserver, reshard-c4-data, cobol-modernization, mteb-leaderboard, overfull-hbox, train-fasttext, chess-best-move) are mostly in the contaminated train split, and the validate tasks have n=1. So `tb21_splits.json` can't be class-stratified as §5 requires until TB-M1 exists. **TB-M1 is multi-day by itself.** 4 repeats × ~10 candidate tasks at a median of about 60 min is about 40 agent-hours, or about 1 day at concurrency 2.
- **A new split.** Build a TB-specific three-way split from TB-M1 classes, with its own contamination note. Drop always-0 tasks such as gpt2-codegolf (0/12) and write-compressor (1/13). They burn about 200 min each for no signal.
- **Expected throughput.** A handful of T4 decisions per week. That fits a few targeted AGENTS.md or `prompt_prefix` hypotheses, not open-ended GEPA search.

### 18.5 Prerequisites and order

1. **B1: first real-pi launch from a scratch worktree. DONE (2026-09-26, on local merge `9262fae`).**
   - **Polyglot:** `run_gepa --baseline-only` from an auto-created scratch worktree (no `node_modules`, `LITTLE_CODER_PI_BIN_OVERRIDE` → little-coder-dev's pi 0.83.0), model `omlx/tiel-coder-oq6e-fp16`. affine-cipher passed on attempt 1 (100 s); grade-school passed on attempt 1 (179 s). The worktree was cleaned up afterwards.
   - **Harbor:** `harbor_pilot.sh` from a scratch worktree. `terminal-bench/prove-plus-comm` scored reward 1.0 (16 turns, 185k input / 127k cached / 12k output tokens, 7.9 min of agent time, 8m14s total). No containers were left behind. All 40 extensions loaded with empty stderr.
   - **New finding:** Harbor 0.22 requires **dataset-prefixed task names** (`terminal-bench/<task>`). Bare names fail with "No tasks matched" (and `hello-world` is not in the TB 2.1 dataset). `HarborLiveRunner` must add the prefix, and `harbor_pilot.sh` usage/docs need updating.
   - Original note:  Real `pi` has never run from a worktree without `node_modules`, and extension imports (`typebox`, `pi-coding-agent`) may not resolve. **This gates Polyglot too.** Test it with one `--baseline-only` Polyglot exercise (about 3 min) or the Harbor `hello-world` task. **It needs explicit go-ahead**, since it uses real compute.
2. **B2: merge dev into the PR branch.** Resolve `rpc_client.py`, keeping `LITTLE_CODER_PI_BIN_OVERRIDE` and the `PRINCIPLES.md` concatenation alongside dev's +942 lines, and resolve `.gitignore`. This is the owner's call.
3. H8 ingest fixes plus a minimal `HarborLiveRunner` (§18.3), with `fake_pi`-style tests. Stub `harbor` with a fake CLI that writes `result.json` trees.
4. Measure the canary baseline 3× on the root, then wire it into `run_t5.py`.
5. Later, Role 2 prerequisites: B4, then TB-M1.

### 18.6 Changes to earlier sections
- §7.4: the canary is §18.2.
- §16 open question 5: resolved by §18.2.
- §14 phase 1 gains item 18: H8 ingest fixes, `HarborLiveRunner` (minimal), and canary wiring.
- §17 risks gain three items:
  - oMLX contention when the canary runs concurrently with Polyglot work. Never run them concurrently in a comparison; the canary runs after GEPA stops.
  - About 9% harness crashes on TB (re-queue with a cap of 2).
  - Docker VM memory: 8.3 GB, and some tasks request 8 GB. Keep `--n-concurrent` ≤ 2 for canary tasks.
