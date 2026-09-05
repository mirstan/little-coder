# Validation Plan — GEPA Self-Improvement Loop

Companion to `TDD_SPEC.md`. That document specs unit-level, test-first development
of the ingestion/metric/component code. This document covers everything unit tests
cannot reach: real trajectory data, real GEPA cost and behavior, and real mutation
of files a live agent depends on. Six layers, each a hard gate on the next — do not
proceed to layer N+1 until layer N passes.

---

## Layer 1 — Unit tests (prerequisite, specified elsewhere)

```
python -m pytest benchmarks/self_improve/tests/ benchmarks/test_rpc_system_prompt.py -q
```

Fast, deterministic, no API calls, no live `dspy` execution beyond import. Fully
specified in `TDD_SPEC.md`. This is the gate before Layer 2 runs at all — do not
debug ingestion/metric behavior against real data while a unit test is red; fix the
unit test first, since it isolates the failure far more cheaply.

**Pass criterion**: zero failures, zero skips (once `dspy-ai` is installed as a dev
dependency, no test should skip via `importorskip`).

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

## Layer 3 — `--dry-run` pipeline smoke test (no API cost)

**Why**: this is the cheapest point at which ingestion, schema, metric, and
component-loading are exercised together end-to-end, before any money is spent on
reflection_lm calls.

**Procedure**:
1. `run_gepa.py --dry-run --log-roots aider=<real log root from Layer 2> gaia=<real log root>` (harbor/tb included only if Layer 2 validated them).
2. `--dry-run` must: run full ingestion, build `HarnessProgram` from the actual
   current `AGENTS.md` + real skill files (via `components.yaml`), run `metric()`
   over every ingested trajectory for every `pred_name` that trajectory's
   `components_used` references plus the episode-level `pred_name=None` call, and
   compute the weighted aggregate score — then stop, printing a summary, without
   constructing a `dspy.LM` or calling GEPA's `.compile()`.
3. Inspect the printed summary for:
   - No `None` scores, no unhandled exceptions per-trajectory (a single bad
     trajectory should be logged and skipped, not crash the run — verify this by
     deliberately corrupting one fixture and re-running).
   - The weighted aggregate is a plausible number in `[0, 1]`.
   - Per-component score/feedback counts look sane — e.g. `skills_tools_bash`
     should have a nonzero count of attributed trajectories for gaia (since gaia
     does capture notifications) but zero for aider_polyglot (per the confirmed
     gap) — an unexpected mismatch here means Layer 2's confirmation was wrong or
     the metric's attribution logic has a bug.

**Pass criterion**: dry run completes without exceptions, produces a plausible
aggregate score, and per-component attribution counts match the benchmark-specific
capabilities confirmed in Layer 2.

---

## Layer 4 — Single-component, capped-cost real GEPA run

**Why**: first time real money and a real reflection model are involved. Scope
tightly to bound cost and make the output easy to judge by eye.

**Procedure**:
1. Create a scoped `components.yaml` containing exactly one entry:
   `skills_tools_bash -> skills/tools/bash.md`.
2. `--train-frac` small enough that the val set still has a handful of both
   pass and fail examples for `bash`-using trajectories specifically — pick this
   from Layer 2/3's printed counts, not a fixed default, since availability varies
   by which real logs exist.
3. Run for real: `run_gepa.py --components-config <scoped yaml> --reflection-model <configured model> ...` (no `--dry-run`).
4. Record wall-clock time and, if the reflection model API reports usage/cost,
   record that too.
5. Read the resulting `apply_results.py`-produced diff for `skills/tools/bash.md`:
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
   - The PR description's reported score delta is independently recomputable: take
     the val-set trajectories, manually run the OLD `bash.md` body and NEW body
     through `metric()`'s `_score_for_benchmark` logic and confirm the reported
     delta isn't fabricated or miscalculated.

**Pass criterion**: run completes within a documented cost/time budget, PR diff is
human-legible and correct (frontmatter untouched, body coherent), reported score
delta is independently verified.

**If this layer fails** (e.g. degenerate output, cost wildly over budget, score
delta doesn't reproduce): do not proceed to Layer 5 or 6's full-scope work. Fix the
metric/feedback-text construction or reflection_lm configuration and re-run Layer 4
before expanding scope.

---

## Layer 5 — Live re-run regression check (closes the loop)

**Why**: this is the layer most likely to be skipped by accident, and it's the only
one that validates against reality rather than against GEPA's own offline scoring
of the training data it just learned from. GEPA's internal score during
optimization is computed from the *same* trajectories it used to propose the
rewrite — a rewrite can look like a clear improvement on that data and still be
overfit, or subtly break the actual model's behavior in a way the offline metric
can't see (tone shifts, instructions that parse fine as text but confuse the model
in-context, etc).

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
| 3 | Full pipeline wiring, no API spend | Free | Layer 2 confirmed (per-benchmark) |
| 4 | Real GEPA output quality, single component | $ (bounded) | Layer 3 clean |
| 5 | Live regression check, held-out set | Compute (benchmark re-runs) | Layer 4 pass |
| 6 | Runtime backward-compatibility | Free/cheap | Independent — can run parallel to 2-5 |

Only after Layers 1-6 all pass for the single-component (`skills/tools/bash.md`)
case should scope expand to the full component set (all skills + AGENTS.md +
PRINCIPLES.md), per `TDD_SPEC.md`'s build-order step 7. Re-run Layers 4-5 (not
1-3, which are component-count-agnostic) at full scope before merging that PR.
