# self-improve: GEPA-based self-improvement loop for little-coder

Live-execution, human-reviewed optimization loop that proposes rewrites to
`AGENTS.md` and skill files (`skills/tools/*.md`, `skills/knowledge/*.md`,
`skills/protocols/*.md`) using [GEPA](https://github.com/gepa-ai/gepa)'s
reflective-mutation optimizer, called directly via `gepa.optimize()` against a
hand-built `PolyglotGEPAAdapter`. Never modifies the runtime agent loop, never
auto-commits, never auto-merges — results are always proposed as a PR for
human review.

Every GEPA candidate is scored by **actually running it**: a candidate's
component text is materialized into a disposable `git worktree`, and a real
`aider_polyglot.py --exercise <name>` subprocess is invoked against it. This
replaces an earlier design that scored candidates from frozen historical
trajectories — that design was confirmed structurally incapable of ever
accepting a candidate, since every candidate scored identically regardless of
its actual text (see `polyglot_adapter.py`'s module docstring for the full
story). There is no free/offline mode for this loop's actual optimization step
— `--estimate-only` and `--baseline-only` (below) are the free/cheap ways to
inspect cost and validate the pipeline before spending on real optimization.

Note: `PRINCIPLES.md`, when present, is concatenated into the runtime system
prompt by `rpc_client.py::_build_system_prompt()` (see Layer 6 below), but is
NOT currently an optimizable component -- `config/components.yaml` (the
single source of truth `load_components()` reads from) has no `principles_md`
entry. Add one there to make it a GEPA target; until then, editing it is a
manual, non-GEPA-driven action (confirmed by review: the opening paragraph
here previously implied otherwise).

See `TDD_SPEC.md` for the test-first implementation spec and `VALIDATION_PLAN.md`
for the 6-layer end-to-end validation gate this loop must clear before real
use. This README covers day-to-day usage.

## Setup

```bash
python3.11 -m venv .venv   # or any Python 3.11/3.12
source .venv/bin/activate
pip install -e .[dev]      # gepa, dspy-ai, pydantic, pyyaml, python-dotenv, pytest
```

Run the test suite (fast, deterministic, no external API calls; some E2E tests
create disposable local git worktrees, not against the real checkout):

```bash
python -m pytest benchmarks/self_improve/tests/ benchmarks/test_rpc_system_prompt.py \
  benchmarks/test_polyglot_env_overrides.py benchmarks/test_fake_pi_modes.py -q
```

## Usage

### Free: `--estimate-only`

Prints a pre-flight cost/wall-clock estimate for the run you're about to
authorize and exits — constructs no git worktree, no adapter, spends nothing.
This is deliberately **not** gated by `--confirm-live-rollouts`/
`--confirm-real-run`/API keys, since its whole purpose is to inform whether to
grant those.

```bash
python -m benchmarks.self_improve.run_gepa --estimate-only \
  --components-config benchmarks/self_improve/config/components_bash_only.yaml \
  --repo-root <path to the little-coder repo root> \
  --benchmark-root <path to a local polyglot-benchmark checkout> \
  --model <model under test> --confirm-live-rollouts --max-metric-calls <N>
```

### Free-ish (real compute, no reflection LM): `--baseline-only`

Runs the seed candidate once over the selected exercises for real — no
reflection LM involved, but still behind the live-rollout gate (real agent
compute, real wall-clock time). Confirms the whole pipeline works and the seed
isn't already saturated at a perfect score before spending on optimization.
Writes `<out-dir>/seed_baseline.json` and `<out-dir>/spend_log.jsonl`.

```bash
python -m benchmarks.self_improve.run_gepa --baseline-only --yes \
  --components-config benchmarks/self_improve/config/components_bash_only.yaml \
  --repo-root <path> --benchmark-root <path> \
  --model <model under test> --confirm-live-rollouts --max-metric-calls <N>
```

### Real (costs money AND real compute): two independent gates

```bash
export REFLECTION_LM_API_KEY=<your key>   # separate from any key used by the model under test
python -m benchmarks.self_improve.run_gepa \
  --components-config benchmarks/self_improve/config/components_bash_only.yaml \
  --repo-root <path> --benchmark-root <path> \
  --model <model under test> --confirm-live-rollouts --max-metric-calls <N> \
  --reflection-model <model, e.g. anthropic/claude-opus-4-6> --confirm-real-run
```

Two **independent** resources each need their own gate satisfied, or the run
refuses with a clear message and exits without spending anything — deliberate,
redundant safety, not a bug to work around:
1. **Reflection LM spend**: `--reflection-model` + `$REFLECTION_LM_API_KEY` +
   `--confirm-real-run` (tokens spent proposing rewrites).
2. **Live rollout spend**: `--model` + `--confirm-live-rollouts` +
   `--max-metric-calls` (real coding-agent runs against real exercises — real
   compute AND real wall-clock time, not just API dollars; never use GEPA's own
   `auto=` presets here — confirmed to wildly over-provision).

`$SELF_IMPROVE_NO_LIVE_ROLLOUTS=1` refuses regardless of flags — a hard,
machine-level deny for a shared host. A graceful stop is available mid-run via
`touch <out-dir>/gepa.stop` (printed in the startup banner) or Ctrl-C (a second
Ctrl-C aborts immediately instead of waiting for the current iteration).

**What actually costs money vs. compute**: both `reflection_lm` calls (the
model that reads a live run's diff/pytest output/transcript and proposes a
rewrite) AND every live rollout can cost real API dollars — a live rollout is
a real coding-agent run against a real exercise using whatever `--model` you
configure as the model under test, and if that's a hosted provider, each
rollout consumes real provider tokens on top of the real compute and
wall-clock time it takes. Rollout spend is capped hard by `--max-metric-calls`
(never GEPA's `auto=` presets — confirmed to over-provision by orders of
magnitude relative to a hand-picked minibatch size); reflection spend has no
equivalent per-call cap beyond `--reflection-minibatch-size` and how many
iterations the rollout budget allows.

**Cost/runtime expectation**: depends entirely on `--max-metric-calls`,
`--reflection-minibatch-size`, and how many exercises/components are in scope
— `--estimate-only` prints the exact projected live-run count and wall-clock
ceiling for your specific flags before you authorize anything; there is no
fixed rule of thumb since live rollouts (not reflection_lm calls) now dominate
both cost and time.

### Free, GEPA-independent: `report_trajectories.py`

The old ingest-and-score reporting pipeline (historical trajectories → weighted
aggregate → per-component usage counts) still exists, decoupled from the
live-eval loop it used to feed — useful for auditing what real benchmark data
is available, independent of whether you're about to run a live GEPA loop.

```bash
python -m benchmarks.self_improve.report_trajectories \
  --log-roots aider=<log_root>,<results.json> gaia=<gaia_run_dir> tb=<tb_run_dir> \
  --components-config benchmarks/self_improve/config/components.yaml \
  --repo-root <path to the little-coder repo root>
```

### Applying results

A real `run_gepa.py` run writes `<out-dir>/optimized_components.yaml`
(pred_name → optimized instruction text) and never touches the actual repo
files directly (the scratch worktree it ran in is destroyed on exit unless
`--keep-scratch` was passed). Use `apply_results.py`'s `apply_and_open_pr()` (or its
lower-level `create_branch_and_commit()`) to write the optimized text back
into the real files, preserving each skill file's YAML frontmatter untouched.
`push_and_open_pr=True` on `apply_and_open_pr()` is the one call in this
codebase that pushes a branch and opens a real GitHub PR — it defaults to
`False` and is never invoked by any test; run it deliberately, once you've
reviewed the local commit.

## Current validation status (see VALIDATION_PLAN.md for the full picture)

- **Layer 2** (real-data ingestion): passing for **aider_polyglot**, **tb**,
  and **harbor** — all three validated against real, freshly-generated data
  (not fixtures). harbor turned out to have a genuinely different real
  structure from tb (not a naming variant), which `harbor_tb_ingest.py` now
  handles via two separate internal loaders rather than one shared parser.
  **gaia** remains blocked on `gaia-benchmark/GAIA`'s gated HuggingFace access
  (request access at the dataset page, then re-attempt ingestion) — the only
  benchmark not yet validated against real data.
- **Layer 3** (`report_trajectories.py` smoke test): passing.
- **Layer 4** (real live GEPA run): the loop was rewritten from a frozen-
  historical-data design (which was confirmed structurally incapable of ever
  accepting a candidate — every candidate scored identically regardless of its
  actual text, so GEPA's strict-improvement acceptance criterion could never
  fire on any dataset size; the one real paid run under that design correctly
  reported "no improvement" for exactly this reason) to a live-execution
  design: every candidate is scored by actually running `aider_polyglot.py`
  against it in a disposable git worktree (`live_eval.py`,
  `polyglot_adapter.py`). The end-to-end pipeline (real git worktree, real
  subprocess, real pytest scoring) is proven with `fake_pi.py` at zero cost
  (`tests/test_live_eval_e2e_fake_pi.py`, including a regression test that two
  candidates differing only in text now score differently). A real,
  money-spending `--baseline-only` run (gated on the live-rollout gate alone)
  or a full `gepa.optimize()` run (gated on both the live-rollout gate and the
  reflection-LM gate) against a real model has not yet been executed under
  this design.
- **Layer 5** (held-out live regression check): comparison utility
  (`compare_runs.py::compare_pass_rates`) built and tested; execution is
  downstream of a real Layer 4 run producing a PR to check out and re-test.
- **Layer 6** (runtime backward-compatibility): passing, including a real
  subprocess-cmd diff proving `_build_system_prompt()` is byte-identical to
  prior behavior when `PRINCIPLES.md` is absent.

## Known gaps / next steps

1. ~~`aider_polyglot.py`'s `_dump_trajectory()` doesn't capture
   `rpc.notifications()`~~ **Closed**: `_dump_trajectory()` now takes a
   `notifications=` kwarg (delta-sliced per attempt, since `PiRpc.notifications()`
   accumulates for the whole session, not per-prompt), and
   `aider_polyglot_ingest.py` extracts `components_used` from it the same way
   `gaia_ingest.py` already did. Older `trajectory_*.json` files written
   before this change have no `"notifications"` key and degrade gracefully
   to `components_used=[]`.
2. gaia dataset access needs to be requested on HuggingFace before gaia can
   feed the training signal.
3. ~~harbor's real output format should be captured~~ **Closed**: captured
   against a real `harbor run` (hello-world, same `fake_pi.py`/
   `LITTLE_CODER_PI_BIN_OVERRIDE` technique as tb). It's genuinely different
   from tb's structure — single-level trial dirs, singular `result.json`,
   reward-float ground truth, and richer structured `agent_result.metadata`
   (no log regex needed for stop_reason/turn_count, unlike tb) —
   `harbor_tb_ingest.py` now has two separate internal loaders, not one
   shared parser.
4. Full-scope expansion (all ~32 components at once) should only happen after
   the single-component case (e.g. just `skills/tools/bash.md`) clears
   Layers 4 and 5, per `VALIDATION_PLAN.md`'s closing summary table.
5. `ComponentUsage.was_error_context` (`ingest/common.py`'s `follows_error`
   param to `merge_component_usage()`) is never set `True` by any real
   caller today -- `gaia_ingest.py`, `aider_polyglot_ingest.py`, and
   `polyglot_adapter.py`'s live-run call all pass the default `False`.
   `polyglot_adapter.py`'s `_component_feedback()` is written to cite
   "(including right after a tool error)" when this flag is set, but
   since no caller currently correlates a notification line's
   position with an adjacent tool-call error in `tool_calls.jsonl`, that
   refinement never actually fires against real data -- only in unit tests
   that set it directly. Confirmed by review; not fixed here because doing
   it correctly needs a real ordering/timestamp correlation across two
   separate log files per benchmark, which is more than a "safe, well-defined"
   fix -- it needs its own design pass on what "right after" should mean
   (same turn? N lines apart? within a time window?).
