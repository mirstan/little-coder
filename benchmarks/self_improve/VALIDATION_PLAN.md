# Validation Plan — GEPA Self-Improvement Loop

Companion to `TDD_SPEC.md`. That document specs unit-level, test-first development
of the ingestion/metric/component code. This document covers everything unit tests
cannot reach: real trajectory data, real GEPA cost and behavior, and real mutation
of files a live agent depends on. Six layers, each a hard gate on the next — do not
proceed to layer N+1 until layer N passes.

**Design note (supersedes this doc's original Layer 3/4 framing)**: `run_gepa.py`
no longer scores GEPA candidates from frozen historical trajectories — that design
was confirmed structurally incapable of ever accepting a candidate (every
candidate's score was independent of its actual text; see `polyglot_adapter.py`'s
module docstring). `run_gepa.py` now implements `gepa.core.adapter.GEPAAdapter`
directly and calls `gepa.optimize()`, scoring every candidate by actually running
it against real `aider_polyglot.py` exercises in a disposable git worktree
(`live_eval.py`, `scratch_worktree.py`). This makes Layer 4 genuinely live by
construction — it is no longer possible to "pass" Layer 4 without a real rollout.
The old ingest-and-score dry-run pipeline still exists, unchanged in spirit, as
`report_trajectories.py` — a free, GEPA-independent reporting tool over historical
logs (Layers 2-3 below), decoupled from the live-eval loop it used to feed.

---

## Layer 1 — Unit tests (prerequisite, specified elsewhere)

```
python -m pytest benchmarks/self_improve/tests/ benchmarks/test_rpc_system_prompt.py \
  benchmarks/test_polyglot_env_overrides.py benchmarks/test_fake_pi_modes.py -q
```

Fast, deterministic, no API calls, no live model calls, no real git worktrees
(the session-scoped `_no_stray_real_worktrees` fixture in `tests/conftest.py`
fails loudly if any test touches the real repo's worktree list). Fully specified
in `TDD_SPEC.md`; the live-eval CLI's own gate/refusal logic is specified in
`tests/test_run_gepa_live_gate.py`. This is the gate before Layer 2 runs at all —
do not debug ingestion/live-eval behavior against real data while a unit test is
red; fix the unit test first, since it isolates the failure far more cheaply.

**Pass criterion**: zero failures, zero unexpected skips (once `dspy-ai` is
installed as a dev dependency, only `tests/conftest.py`'s own `importorskip`
guard — for hosts without the optional deps at all — should ever skip).

---

## Layer 2 — Ingestion against real (not synthetic) logs

**Why**: `TDD_SPEC.md`'s fixtures are hand-built to match documented shapes. Real
log directories accumulate quirks fixtures don't — partial/interrupted runs,
encoding issues, fields that are `null` in practice but never `null` in a
hand-typed fixture, etc. This layer is where the two flagged gaps (aider_polyglot
missing notification capture, gaia's per-task `result.json` missing `stop_reason`)
get confirmed against reality rather than assumed from reading source.

**Procedure**:
1. Locate at least one real, already-existing log tree per benchmark:
   - aider_polyglot: an existing `benchmarks/results_full_polyglot.json` +
     `benchmarks/full_polyglot_logs/` tree from a prior run (check
     `docs/benchmark-*.md` for which runs are referenced, use the freshest).
   - gaia: an existing `--run-name` output directory under wherever `gaia.py`
     writes runs.
   - harbor/tb: existing adapter log output, if any exists on disk; if none exists,
     this is itself a finding — note it and treat harbor/tb as untested until a run
     is available (do not fabricate a real-looking log to substitute).
2. Add a `if __name__ == "__main__":` CLI mode to each `ingest/*.py` module (or a
   shared `benchmarks/self_improve/ingest_report.py` script) that takes a log root,
   runs `load()`, and prints: total trajectory count, count by `success`, count with
   non-empty `components_used`, count with non-empty `failure_signals`, and one full
   sample record as JSON.
3. Run each ingestion module against its real log tree. Manually cross-check the
   printed sample record's fields against the actual files on disk (open the
   trajectory JSON / result.json / log file directly and compare).
4. Confirm explicitly:
   - aider_polyglot trajectories show `components_used == []` for all records (the
     documented gap) — if any record unexpectedly has non-empty `components_used`,
     that means the assumption in `TDD_SPEC.md` §0 is stale and needs re-checking
     against the current `_dump_trajectory` implementation before proceeding.
   - gaia trajectories' `stop_reason` fallback behavior matches what's actually in
     the manifest vs. defaulted.
   - No ingestion module raises on any real file in the sample set.

**Pass criterion**: for each benchmark with available real data, ingestion runs
clean over the full log tree with no exceptions, and the two documented gaps are
confirmed as still-accurate (or explicitly updated if reality has changed).
**harbor/tb absence of real data is a valid, documented outcome of this layer** —
proceed to Layer 3 only with the benchmarks that do have confirmed-clean real-data
ingestion; the others stay flagged as unvalidated until a run exists.

---

## Layer 3 — `report_trajectories.py` smoke test (no API cost, no live rollouts)

**Why**: this is the cheapest point at which ingestion, schema, and
component-loading are exercised together end-to-end, entirely independent of the
live-eval loop. It no longer rehearses what GEPA will do (GEPA now always scores
live — there is no offline stand-in for that), but it remains the fast, free check
that historical log ingestion and component loading both still work before
spending anything on Layer 4.

**Procedure**:
1. `python -m benchmarks.self_improve.report_trajectories --log-roots aider=<real log root from Layer 2>,<results.json> gaia=<real log root>` (harbor/tb included only if Layer 2 validated them).
2. This runs full ingestion, loads the actual current `AGENTS.md` + real skill
   files (via `components.yaml`), computes each trajectory's own recorded
   pass/partial score, the weighted aggregate across benchmarks, and per-component
   usage counts (from real `components_used` data) — then stops. No `gepa`, no LM
   of any kind, no live rollout.
3. Inspect the printed summary for:
   - No unhandled exceptions per-trajectory (a single bad trajectory should be
     logged and skipped, not crash the run — verify this by deliberately
     corrupting one fixture and re-running).
   - The weighted aggregate is a plausible number in `[0, 1]`.
   - Per-component usage counts look sane — e.g. `skills_tools_bash` should have a
     nonzero count of attributed trajectories for gaia (since gaia does capture
     notifications) but zero for aider_polyglot (per the confirmed gap) — an
     unexpected mismatch here means Layer 2's confirmation was wrong or
     `merge_component_usage`'s attribution logic has a bug.

**Pass criterion**: report completes without exceptions, produces a plausible
aggregate score, and per-component attribution counts match the benchmark-specific
capabilities confirmed in Layer 2.

---

## Layer 4 — Single-component, capped-cost real live GEPA run

**Why**: first time real money, a real model under test, AND real live rollouts
are involved. `run_gepa.py`'s `gepa.optimize()` call always scores candidates by
actually running them — there is no cheaper offline substitute for this layer
anymore, so scope tightly (one component, a small exercise set, a hard
`--max-metric-calls` cap) to bound cost and make the output easy to judge by eye.

**Procedure**:
1. Use the existing scoped config, `config/components_bash_only.yaml` (or create
   one with exactly one entry: `skills_tools_bash -> skills/tools/bash.md`).
2. **Always start with `--estimate-only`** (spends nothing, no worktree, no
   adapter) to see the projected live-run count and wall-clock ceiling before
   authorizing anything: `python -m benchmarks.self_improve.run_gepa --estimate-only --components-config <scoped yaml> --model <model under test> --confirm-live-rollouts --max-metric-calls <small N> --exercise-count <small> --val-count <small>`.
3. **Then run `--baseline-only`** (real live rollouts, no reflection LM) to confirm
   the whole pipeline works end to end and the seed isn't already saturated at a
   perfect score on the chosen exercises: same flags, swap `--estimate-only` for
   `--baseline-only --yes`. Inspect `<out-dir>/seed_baseline.json` and
   `spend_log.jsonl`.
4. Only then run for real: add `--reflection-model <configured model>` +
   `$REFLECTION_LM_API_KEY` + `--confirm-real-run` and drop `--baseline-only`.
5. Record wall-clock time and, if the reflection model API reports usage/cost,
   record that too. `spend_log.jsonl` (append-only, flushed per line) is the
   authoritative record of what actually ran, regardless of how the process exited.
6. Read the resulting `apply_results.py`-produced diff for `skills/tools/bash.md`:
   - Frontmatter block is byte-identical to before (this is the one thing Layer 1's
     roundtrip unit test already gives strong confidence on, but confirming it
     against a real GEPA-produced body — not a hand-typed test string — is the
     actual validation of the full pipeline, not just the string-manipulation
     function in isolation).
   - The rewritten body text is coherent, on-topic guidance for the `bash` tool,
     not degenerate output (e.g. not truncated mid-sentence, not repeating the
     original verbatim if a change was expected, not introducing formatting the
     skill-inject extension can't handle — check against
     `.pi/extensions/skill-inject/index.ts`'s expected `buildBlock`/injection shape).
   - The reported score delta is independently recomputable from `spend_log.jsonl`'s
     per-exercise records and the run's own `run_dir`/GEPA state — since scores now
     come from real live runs, this check is about confirming the reported numbers
     match what `spend_log.jsonl` actually recorded, not re-deriving them from a
     frozen dataset.

**Pass criterion**: run completes within a documented cost/time budget, PR diff is
human-legible and correct (frontmatter untouched, body coherent), reported score
delta matches `spend_log.jsonl`.

**If this layer fails** (e.g. degenerate output, cost wildly over budget, score
delta doesn't reproduce): do not proceed to Layer 5 or 6's full-scope work. Fix the
adapter/feedback-text construction or reflection_lm configuration and re-run Layer 4
before expanding scope.

---

## Layer 5 — Live re-run regression check (closes the loop)

**Why**: this is the layer most likely to be skipped by accident, and it remains
distinct from Layer 4 even though Layer 4's scoring is now live: Layer 4's whole
loop (train + val exercises, cache, reflection feedback) draws from one small,
fixed exercise pool selected up front (`exercises.py::select_exercises`) — a
rewrite can still overfit to the quirks of *that* pool and not generalize, or
subtly change the model's behavior in a way a pass/fail score on the same handful
of exercises doesn't surface (tone shifts, instructions that parse fine as text but
confuse the model in-context, etc). Layer 5 re-runs against exercises/tasks
independent of Layer 4's own pool.

**Procedure**:
1. Check out the PR branch from Layer 4 into a **separate scratch worktree** (not
   `dev`, not `main`, not the `self-improve/gepa-loop` worktree itself — a fourth,
   disposable one), so the live re-run doesn't collide with ongoing implementation
   work.
2. Identify the specific exercises/tasks that were in Layer 4's **validation
   split** (not the training split) — these are held-out examples GEPA did not use
   to shape the rewrite.
3. Re-run those specific exercises for real, through the actual harness
   (`aider_polyglot.py --exercise <name>` per exercise, or the equivalent gaia
   task-id filtering), once with the OLD `bash.md` (checked out from `dev`) and
   once with the NEW `bash.md` (from the PR branch) — same model, same seed/params
   where controllable, to isolate the skill-file change as the only variable.
4. Compare live pass/fail outcomes before vs. after on this held-out set. This is
   the number that actually matters — not GEPA's internally reported score delta
   from Layer 4, which only proves the optimizer's *own* objective went up.
5. Also spot-check: does the live agent's behavior when `bash.md` is injected still
   look sane in a manual transcript read (2-3 sample transcripts), independent of
   pass/fail — catches "technically still passes tests but now behaves oddly"
   regressions that a binary pass/fail metric would miss.

**Pass criterion**: held-out live pass rate is non-regressive (ideally improved)
compared to the old skill text. A regression here means the PR should not be
merged even if Layer 4's offline score delta was positive — treat Layer 5 as
overriding Layer 4, not merely confirming it.

---

## Layer 6 — Backward-compatibility guard for the runtime change

**Why**: `_build_system_prompt()` (from `TDD_SPEC.md` §8) touches the live
benchmark-running code path (`rpc_client.py`) directly, independent of whether any
GEPA optimization has happened yet. This must be validated on its own, since it's a
runtime behavior change, not an offline text-optimization artifact.

**Procedure**:
1. Confirm Layer 1's unit tests for `_build_system_prompt()` are green (backward
   compat when `PRINCIPLES.md` is absent, correct concatenation when present,
   refresh-on-each-call).
2. Beyond unit tests: with no `PRINCIPLES.md` present in the repo, run one real
   `PiRpc` session (e.g. a single aider_polyglot exercise) from before this change
   and one from after, and diff:
   - The exact subprocess `cmd` list passed to `Popen` (should be identical modulo
     any unrelated changes).
   - The actual resolved system prompt content pi receives (can be captured by
     temporarily logging `_build_system_prompt()`'s return value's file content, or
     by inspecting pi's own debug output if it echoes the resolved system prompt).
3. With `PRINCIPLES.md` present (a placeholder file for this test), run one live
   session and manually read the transcript to confirm pi's behavior reflects
   awareness of the added `# Principles` section content (e.g. add an
   easily-observable placeholder instruction like "always start responses with
   'ACK:'" temporarily, confirm it takes effect, then revert).

**Pass criterion**: byte-identical subprocess invocation and system prompt content
when `PRINCIPLES.md` is absent; observably-effective prompt content when present.

---

## Summary table

| Layer | Validates | Cost | Blocking on |
|---|---|---|---|
| 1 | Unit-level correctness | Free | Nothing — first gate |
| 2 | Ingestion vs. real log shapes | Free | Layer 1 green |
| 3 | Ingestion + component loading, no API spend | Free | Layer 2 confirmed (per-benchmark) |
| 4 | Real live GEPA output quality, single component | $ + compute (bounded by `--max-metric-calls`) | Layer 3 clean |
| 5 | Live regression check, held-out set | Compute (benchmark re-runs) | Layer 4 pass |
| 6 | Runtime backward-compatibility | Free/cheap | Independent — can run parallel to 2-5 |

Only after Layers 1-6 all pass for the single-component (`skills/tools/bash.md`)
case should scope expand to the full component set (all skills + AGENTS.md +
PRINCIPLES.md), per `TDD_SPEC.md`'s build-order step 7. Re-run Layers 4-5 (not
1-3, which are component-count-agnostic) at full scope before merging that PR.
