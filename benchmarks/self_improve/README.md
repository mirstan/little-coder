# self-improve: GEPA-based self-improvement loop for little-coder

Offline, human-reviewed optimization loop that reads little-coder's collected
benchmark trajectories and proposes rewrites to `AGENTS.md` and skill files
(`skills/tools/*.md`, `skills/knowledge/*.md`, `skills/protocols/*.md`) using
[DSPy](https://dspy.ai)'s GEPA optimizer. Never modifies the runtime agent
loop, never auto-commits, never auto-merges — results are always proposed as
a PR for human review.

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
pip install -e .[dev]      # dspy-ai, pydantic, pyyaml, python-dotenv, pytest
```

Run the test suite (fast, deterministic, no API calls):

```bash
python -m pytest benchmarks/self_improve/tests/ benchmarks/test_rpc_system_prompt.py -q
```

## Usage

### Free: `--dry-run`

Validates the full pipeline (ingest real trajectories → build the component
map → score every trajectory through the metric → weighted aggregate) without
spending anything — no `dspy.LM` is constructed, `GEPA.compile()` is never
called.

```bash
python -m benchmarks.self_improve.run_gepa --dry-run \
  --log-roots aider=<log_root>,<results.json> gaia=<gaia_run_dir> tb=<tb_run_dir> \
  --components-config benchmarks/self_improve/config/components.yaml \
  --repo-root <path to the little-coder repo root>
```

`--log-roots` accepts any subset of `aider=<log_root>,<results.json>`,
`gaia=<run_dir>`, `harbor=<run_dir>`, `tb=<run_dir>` — omit whichever
benchmarks you don't have data for.

### Real (costs money): requires three independent confirmations

```bash
export REFLECTION_LM_API_KEY=<your key>   # separate from any key used by the model under test
python -m benchmarks.self_improve.run_gepa \
  --log-roots aider=<log_root>,<results.json> \
  --reflection-model <model, e.g. anthropic/claude-opus-4-6> \
  --confirm-real-run \
  --components-config benchmarks/self_improve/config/components.yaml \
  --repo-root <path to little-coder repo root>
```

All three of `--reflection-model`, `$REFLECTION_LM_API_KEY`, and
`--confirm-real-run` must be present or the run refuses with a clear message
and exits without spending anything — this is deliberate, redundant safety,
not a bug to work around.

**What actually costs money**: only `reflection_lm` calls (the model that
reads failing trajectories + feedback and proposes rewrites). The forward-pass
rollouts GEPA's own architecture requires are backed by `dspy.utils.dummies.DummyLM`
— a real, valid, zero-cost, network-free predictor call chain (confirmed
necessary by reading the installed `dspy-ai` GEPA source: an empty trace for
any predictor crashes GEPA's reflective-dataset construction, so `forward()`
must call every predictor for real, just not against a paid model).

**Cost/runtime expectation**: tens to low-hundreds of `reflection_lm` calls
across ~32 components and however many trajectories you provide — budget
roughly $20–100 and 30–90 minutes for a full run against a frontier reflection
model, dominated by `reflection_lm` latency/cost.

### Applying results

`run_gepa.py`'s real path writes `<out-dir>/optimized_components.yaml`
(pred_name → optimized instruction text) and never touches the actual repo
files directly. Use `apply_results.py`'s `apply_and_open_pr()` (or its
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
- **Layer 3** (dry-run smoke test): passing.
- **Layer 4** (real GEPA run): executed successfully once, with a real
  reflection_lm API key, against a tiny real fixture dataset scoped to
  `skills_tools_bash` only — completed without error and reported no
  improvement over the seed instructions on that dataset (a real, structural
  finding: with the current `metric()` design, GEPA's score is identical for
  every candidate on the same example regardless of what that candidate's
  instructions actually say, so its default strict-improvement acceptance
  criterion can never accept a proposed rewrite on ANY dataset size -- see
  `metric.py`'s module docstring). Confirmed by review; not yet resolved.
  Still safety-gated behind `--reflection-model` + `$REFLECTION_LM_API_KEY` +
  `--confirm-real-run` for any further real run.
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
   caller today -- both `gaia_ingest.py` and `aider_polyglot_ingest.py` pass
   the default `False`. The metric's `_component_feedback()` is written to
   cite "(including right after a tool error)" when this flag is set, but
   since no ingest module currently correlates a notification line's
   position with an adjacent tool-call error in `tool_calls.jsonl`, that
   refinement never actually fires against real data -- only in unit tests
   that set it directly. Confirmed by review; not fixed here because doing
   it correctly needs a real ordering/timestamp correlation across two
   separate log files per benchmark, which is more than a "safe, well-defined"
   fix -- it needs its own design pass on what "right after" should mean
   (same turn? N lines apart? within a time window?).
