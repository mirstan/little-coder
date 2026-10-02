# self-improve: GEPA-based self-improvement loop for little-coder

Live-execution, human-reviewed optimization loop that proposes rewrites to
`AGENTS.md` and skill files (`skills/tools/*.md`, `skills/knowledge/*.md`,
`skills/protocols/*.md`) using [GEPA](https://github.com/gepa-ai/gepa)'s
reflective-mutation optimizer, called directly via `gepa.optimize()` against a
hand-built `PolyglotGEPAAdapter`. Never modifies the runtime agent loop, never
auto-commits, never auto-merges — results are always proposed as a PR for
human review.

Every GEPA candidate is scored by **actually running it**: a candidate's
component text is materialized into a disposable private git repo (its own
object store, git dir kept beside the tree), and a real
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
create disposable private git repos from throwaway fixture repos, never from the
real checkout):

```bash
python -m pytest benchmarks/self_improve/tests/ benchmarks/test_rpc_system_prompt.py \
  benchmarks/test_polyglot_env_overrides.py benchmarks/test_fake_pi_modes.py -q
```

## Usage

### Free: `--estimate-only`

Prints a pre-flight cost/wall-clock estimate for the run you're about to
authorize and exits — constructs no scratch repo, no adapter, spends nothing.
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

One `--out-dir` holds one pre-registered run. A live run (baseline or real)
refuses with exit 1, before the confirmation prompt, when its `--out-dir`
already holds any of `manifest.yaml`, `gepa/gepa_state.bin`,
`optimized_components.yaml`, `seed_baseline.json` or `spend_log.jsonl`. A dir
holding only `live_cache/` is accepted. So after a baseline, run the real
optimization in a fresh `--out-dir` and point it at the baseline's warm cache:

```bash
python -m benchmarks.self_improve.run_gepa ... \
  --out-dir benchmarks/self_improve/runs/<new-run> \
  --live-cache-dir <baseline-out-dir>/live_cache
```

### Real (costs money AND real compute): two independent gates

```bash
# Use a distinct, spend-capped key. Either put REFLECTION_LM_API_KEY=<your key>
# in benchmarks/self_improve/.env, or set it for this one command as below
# rather than exporting it into the shell.
REFLECTION_LM_API_KEY=<your key> python -m benchmarks.self_improve.run_gepa \
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

`REFLECTION_LM_API_KEY`, however it was set, and every variable defined in
`benchmarks/self_improve/.env` are orchestrator-only: `run_gepa` removes them
from the environment of the agent-under-test, its bash tool, and the
exercise test runs. The startup banner lists the withheld names (never their
values). That environment also loses git's repository-location and
config-override variables: `GIT_DIR`, `GIT_WORK_TREE`, `GIT_INDEX_FILE`,
`GIT_COMMON_DIR`, `GIT_OBJECT_DIRECTORY`, `GIT_CONFIG_PARAMETERS`,
`GIT_CONFIG_COUNT`, `GIT_CONFIG_KEY_*`/`GIT_CONFIG_VALUE_*` and the rest of
`git rev-parse --local-env-vars`, plus `GIT_ATTR_SOURCE`, `GIT_NAMESPACE` and
`GIT_QUARANTINE_PATH`. Without that, starting `run_gepa` from a git hook
(a pre-commit hook in a linked worktree exports an absolute `GIT_DIR` and
`GIT_INDEX_FILE`) or from a shell that exports them would make the agent's
own git commands act on your checkout. `GIT_CONFIG_GLOBAL`,
`GIT_CONFIG_SYSTEM`, `GIT_CONFIG_NOSYSTEM` and other git settings such as
`GIT_EXEC_PATH` or `GIT_AUTHOR_*` reach the agent unchanged, since they
choose your git config or installation, not a repository. Put the model-under-test's provider key in your shell or pi's
config, not in this `.env`. A run knob the agent reads itself
(`ATTEMPT_TIMEOUT_S`, `CODEX_TIMEOUT_S`, or any `LITTLE_CODER_*` / `PI_*`
name) is refused if it appears in the `.env`, because the orchestrator and
the agent would otherwise run with different values: export it in your shell
instead.

This scrub stops *accidental* exposure: an `env` or `printenv`, or an error
dump, ending up in transcripts, run logs, cached results or the reflection
dataset. It does not contain a hostile agent. The agent runs as your user
with unrestricted bash and no filesystem sandbox. Its bash starts in a
per-exercise temp dir, but the scratch tree's path is in pi's argv, so treat
the tree and everything next to it as reachable. It can still:

- read the `.env` file on disk;
- read the orchestrator's launch environment (`ps eww <pid>` on macOS); the
  marker beside the scratch tree (`<tree>.marker.json`) records that pid;
- learn where your checkout is: `LITTLE_CODER_PI_BIN_OVERRIDE` in its own
  environment is the resolved pi path, which by default lies inside
  `<your checkout>/node_modules`;
- write to your real checkout and its `.git` by absolute path, or to the
  scratch repo's git dir (`<tree>.git`). The checks below cover that git
  dir's top level, config and `refs/` subtree, and what `reset()` checks
  out. Nothing stops the agent changing `objects/` or `refs/` (though a
  symlink or special file under `refs/` fails the check), but a change
  there that alters the checkout makes the next `reset()` fail.

What the orchestrator does about it is limited to its own git calls and
file operations. The scratch repo is a private repo (its own object store,
no hooks directory, no remote) created by a depth-1 fetch of the base
commit, so it shares no git admin with your checkout, and your checkout's
`.git` is only read. Its git dir sits beside the tree, not inside it. Every
git call on it:

- ignores user and system git config (`GIT_CONFIG_GLOBAL=/dev/null`,
  `GIT_CONFIG_NOSYSTEM=1`) and drops git's location/config environment
  variables;
- pins `GIT_COMMON_DIR` to `<tree>.git`, so a planted `commondir` file
  cannot redirect config, refs or objects elsewhere;
- pins `core.hooksPath=/dev/null`, `core.fsmonitor=false`,
  `core.attributesFile=/dev/null`, `core.excludesFile=/dev/null` and
  `core.useReplaceRefs=false`, and reads attributes only from the base
  commit (`--attr-source`, git 2.40 or later; on older git, `reset()`
  removes untracked files before checkout instead);
- also pins `core.commitGraph=false`, `core.multiPackIndex=false`,
  `core.splitIndex=false`, `core.untrackedCache=false`,
  `core.sparseCheckout=false` and `index.sparse=false`. These are defense
  in depth: a planted commit-graph did not change what checkout produced
  on git 2.54;
- pins `core.logAllRefUpdates=false`, so git never creates
  `<tree>.git/logs/`. git still appends to a reflog that already exists,
  through a symlink or hard link, so a `logs/` directory in `<tree>.git`
  fails the check below instead of being trusted. Without that, a link at
  `<tree>.git/logs/HEAD` would make `reset()` append a line to any file you
  can write, your checkout's `.git/config` included;
- runs `status` with `--no-optional-locks` and `--ignore-submodules=all`;
- runs without the orchestrator-only variables above in its environment;
- first checks, raising `ScratchWorktreeCorrupted` instead of running git
  if any check fails:
  - the tree's and `<tree>.git`'s paths still name the directories this
    run created. Each path is `lstat`ed and compared with a directory
    descriptor held since creation, so a symlink or another directory at
    either path fails the check;
  - `<tree>.git` contains `HEAD`, `config`, `objects/` and `refs/`, and
    nothing at its top level other than those plus `index`, `shallow` and
    `ORIG_HEAD`, each of the expected type (no symlinks). `commondir`,
    `gitdir`, `config.worktree`, `logs/`, `worktrees/`, `hooks/`, `info/`,
    `sharedindex.*` and anything else fail;
  - nothing under `<tree>.git/refs/` is a symlink or special file, and
    every directory there can be listed. A symlinked directory would let a
    ref update that follows a symbolic ref write into another directory,
    and git only needs search permission to pass through one the check
    cannot list;
  - `<tree>.git/config` has exactly the bytes creation wrote.

  The same checks run once at the end of creation.

`reset()` deletes `<tree>.git/index` before checking out, so a crafted index
(skip-worktree or assume-unchanged bits, split or sparse index, untracked
cache) is rebuilt from the base commit. It also deletes
`<tree>.git/ORIG_HEAD`: `reset --hard` updates it, and would follow a
symbolic one (`ref: refs/heads/<name>`) and write the base commit's sha into
the ref it names. `HEAD` needs no such step, because `checkout --detach`
rewrites it without following it before `reset --hard` runs. It deletes
every `.git` entry (any case, any depth) left in the tree, since git itself
skips them. Last, it
compares every path of the base commit with the `git ls-tree -r` listing
recorded at creation, before the agent ran: file type, executable bit, and
the blob hash of the file's content or the symlink's target. Any difference
raises `ScratchWorktreeCorrupted`. The check looks at the result, not at how
`<tree>.git/objects` stores it, so it catches a rewritten object, an added
pack or an `objects/info/alternates` file alike. Without it, such a change
would persist into every later candidate's harness, which the exercise
subprocess runs: on git 2.54, checkout writes a rewritten loose object's
content without complaint. `assert_only_expected_dirty()` also fails if any
index entry carries a skip-worktree or assume-unchanged bit, which `git
status` would not report.

Apart from the identity check itself, the orchestrator's own file
operations on the tree and `<tree>.git` (reading the git dir, deleting the
index and `ORIG_HEAD`, removing `.git` entries, the content check,
teardown) go through the held directory descriptors, never through the
paths, so swapping a path cannot redirect them. Teardown empties each directory through its
descriptor and then `rmdir`s the path, which cannot remove a non-empty
directory or a symlink. If the path no longer names the directory this run
created, teardown leaves whatever is there alone and warns; the directory
this run created is then wherever the agent moved it, with no marker, and
the GC will not find it.

git itself takes paths, and the identity checks run at points in time
(before each git call), not continuously. A process the agent leaves
running that survives the post-run cleanup (see "Processes a live run
leaves behind" under Known gaps / next steps) could swap a path just after
a check and swap it back before the next one. Nothing detects that swap, and for the git call in
between, git could be pointed at another directory, such as your checkout.

Use a separate, spend-limited key for reflection, and don't export it in
shells where you run other harnesses (harbor, tb, or `aider_polyglot.py`
directly), which do not remove it.

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
wall-clock time it takes. Rollout spend is bounded by `--max-metric-calls`
(never GEPA's `auto=` presets — confirmed to over-provision by orders of
magnitude relative to a hand-picked minibatch size) — but for a real
`gepa.optimize()` run (not `--baseline-only`), `--max-metric-calls` is not
itself the hard ceiling: live rollouts can run up to `--max-metric-calls`
**plus** a `2*reflection_minibatch_size + valset_size` overshoot allowance
(`--estimate-only` prints the real number as "LIVE exercise executions" —
budget for that, not the raw flag value). `--baseline-only` has no such
overshoot: it never touches `gepa.optimize()`'s own iteration loop, so its
live executions are capped at exactly `--max-metric-calls`. Reflection spend
has no equivalent per-call cap beyond `--reflection-minibatch-size` and how
many iterations the rollout budget allows.

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

### Cleaning up scratch repos

A run that is SIGKILLed leaves its scratch tree, `<tree>.git`,
`<tree>.marker.json` and `<tree>.lock` in the scratch parent dir. A run
passed `--keep-scratch` leaves the tree, `<tree>.git` and the marker, but
releases and deletes its lock on exit, so a preserved tree has no
`<tree>.lock` and the GC judges it by the marker alone. List or remove
orphans with:

```bash
python -m benchmarks.self_improve.gepa_scratch_gc --list  [--scratch-root DIR]
python -m benchmarks.self_improve.gepa_scratch_gc --clean [--scratch-root DIR] [--older-than-hours 6] [--yes]
```

`--older-than-hours` defaults to 0, which means no grace period: every
removable entry is removed, however recent. Pass it explicitly (6 above) to
remove only entries whose marker records a creation time at least that old.

`--scratch-root` defaults to the system temp dir, which is also where
`run_gepa` puts scratch repos by default; if the run used `--scratch-dir X`,
pass `--scratch-root X`. In the scratch root, nothing without a marker is
touched and a symlinked tree or git dir is never followed; an entry whose
owner still holds its lock is never removed. Worktrees left by older
runs (which used `git worktree add`) are handled by a legacy pass over the
repo given by `--repo-root` (default: the current directory).

### Applying results

A real `run_gepa.py` run writes `<out-dir>/optimized_components.yaml`
(pred_name → optimized instruction text) and never touches the actual repo
files directly (the scratch repo it ran in is destroyed on exit unless
`--keep-scratch` was passed). If the live budget backstop or a persistent
harness error stops `gepa.optimize()` early, the file still holds the best
candidate GEPA had scored on the valset so far; `spend_log.jsonl`'s `run_end`
then carries `partial: true` and the exit code is 3 (budget) or 4 (harness).
A run whose reflection attempts never produced an evaluated child anywhere
in the run (GEPA swallows reflection errors), so its best candidate is still
the seed, still writes the file, but it is not optimized: `run_end` carries
`reason: no_proposals` and the exit code is 5, whichever stop condition
ended the run, provided `gepa.optimize()` returned normally. A budget
backstop or harness error raised out of it exits 3 or 4 as above before this
check is reached. A run that evaluated children but found none better than the
seed exits 0 with the seed written.
Use `apply_results.py`'s `apply_and_open_pr()` (or its
lower-level `create_branch_and_commit()`) to write the optimized text back
into the real files, preserving each skill file's YAML frontmatter untouched
except for a possible `token_cost:` update when the rewritten body's estimated
cost differs from what's currently recorded (see `TDD_SPEC.md` §7.2).
`push_and_open_pr=True` on `apply_and_open_pr()` is the one call in this
codebase that pushes a branch and opens a real GitHub PR — it defaults to
`False` and is never invoked by any test; run it deliberately, once you've
reviewed the local commit.

Each `<out-dir>` is written by exactly one run: `manifest.yaml` is created
once and never rewritten, and a second run into the same dir is refused (see
the `--baseline-only` section). Apply results from the out-dir of the run you
mean; start any further run in a new `--out-dir`, reusing the warm cache with
`--live-cache-dir <earlier-out-dir>/live_cache`.

### How self-improve runs are scored

`live_eval.py` runs `aider_polyglot.py` with `POLYGLOT_RESTORE_TESTS=1` and
`POLYGLOT_CRASH_IS_ERROR=1`. A plain benchmark run sets neither and scores
exactly as before. With `POLYGLOT_RESTORE_TESTS=1`, each attempt is scored
in a tree the harness builds itself (`_score_gated` in `aider_polyglot.py`):

- the solution files hold the agent's bytes, and every other prepared file
  holds its prepared bytes. Files the agent added are left out.
- python runs `sys.executable -I -m pytest --noconftest -c /dev/null`, and
  JavaScript runs jest directly rather than through `npm test`. Each writes
  its report outside the scored tree. A pass needs that report to show
  every expected test passing; exit code 0 alone is not enough.
- an attempt that changed a test or runner-config file, or whose solution
  file is not a regular file or trips the python import tripwire, is scored
  as a failure by policy. The record sets `tests_tampered`, and
  `tamper_reasons` says why. Added hook files (`conftest.py`,
  `package-lock.json` and so on) and edited docs are only reported, because
  they cannot reach the scored tree.

What this does not close:

- **In-process forging.** Code in the solution runs inside the test process
  and can still write a passing report and exit. The python tripwire
  (`_python_tripwire`) rejects the imports and names such code usually
  needs, which raises the bar but does not rule it out. JavaScript has no
  tripwire.
- **A writable interpreter environment.** The agent runs as the same user
  as the scorer, so it can write to the scoring interpreter's
  `site-packages` (a `.pth` file runs at startup even under `-I`), to the
  shared JS `node_modules`, and to `$HOME`. Closing that needs a separate
  user or a sandbox.

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
  against it in a disposable private git repo (`live_eval.py`,
  `polyglot_adapter.py`). The end-to-end pipeline (real scratch repo, real
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
   `notifications=` kwarg, populated with that attempt's own
   `rpc.notifications()`. Each attempt opens a fresh `PiRpc` session, so
   `notifications()` is already scoped to just that one attempt — no
   delta-slicing across attempts is needed or possible. `aider_polyglot_ingest.py`
   extracts `components_used` from it the same way `gaia_ingest.py` already
   did. Older `trajectory_*.json` files written before this change have no
   `"notifications"` key and degrade gracefully to `components_used=[]`.
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

6. **Processes a live run leaves behind.** After every live run, before its
   result is parsed (and before the next run starts), `live_eval.py` kills
   the run's process group, then every process whose environment carries
   that run's token (`LITTLE_CODER_SELF_IMPROVE_RUN_TOKEN`, a fresh value
   per run), together with each such process's own process group. The
   group kill alone is not enough: pi starts every bash tool call in a
   session of its own, so an agent's `cmd &` is outside the run's group.
   What this still misses:
   - **macOS: Apple's own binaries.** `ps -E` does not show the
     environment of Apple platform binaries: `/bin/sh`, `/bin/bash`,
     `/bin/zsh`, `sleep`, `tail`, `perl` and the rest. A leftover made
     only of them is not found unless it shares a process group with a
     process that is found (python, node). pi's bash tool runs
     `/bin/bash`, so a plain shell loop the agent backgrounds is exactly
     this case. `test_a_setsid_shell_only_background_writer_is_killed` is
     marked as an expected failure on macOS for this reason.
   - **Linux** reads `/proc/<pid>/environ`, which shows every process of
     the same user, so for those the sweep is complete.
   - **Anywhere:** a process that clears or rewrites its environment
     (`env -i`, an exec with an explicit environment), or that runs as
     another user.
   - **Writes made before the cleanup.** The results file and the log root
     are inside the tree the agent can write to, so a leftover can still
     change them between the child exiting and being killed. Moving them
     into a directory only the orchestrator writes is the real fix.
   - **A daemon the agent started** during the run (watchman, a gradle
     daemon) carries the token and is killed with the rest.
   - **A container whose PID 1 does not reap orphans.** Killed leftovers
     stay zombies there. A run that left something in its own process
     group then waits out the 2 s grace plus the 5 s SIGKILL confirmation
     and logs a warning; each sweep round that found something waits up
     to its 1 s grace twice.
   - If the process list cannot be read at all, the sweep logs one warning
     and the run is scored without it.

   A leftover that survives can also make the next run's `reset()` raise
   `ScratchWorktreeCorrupted`, which stops the batch rather than scoring
   a tree it changed.
7. **A kept scratch tree holds only the last live run's logs.** Every live
   run, retries included, starts from a reset tree, and the reset deletes
   untracked files, so `benchmarks/full_polyglot_logs/` and
   `benchmarks/results_full_polyglot.json` inside a tree kept with
   `--keep-scratch` show only the last run. A cached result in `live_cache` keeps its
   own reflection material (diff, test output tail, transcript excerpts);
   for an unscoreable try, `spend_log.jsonl` keeps only its status and
   error.
