"""CLI entrypoint for the live-execution GEPA self-improvement run.

Two independent, real-cost resources need two independent gates:
1. Reflection LM (--reflection-model + $REFLECTION_LM_API_KEY +
   --confirm-real-run): tokens spent on the reflection step's rewrite
   proposals.
2. Live rollouts (--model + --confirm-live-rollouts + --max-metric-calls):
   real coding-agent runs against real exercises -- real compute AND real
   wall-clock time, not just API dollars. `auto=` is NEVER used (confirmed
   to over-provision wildly, e.g. ~11,852 metric calls for a 32-component
   set, since its internal budget heuristic uses a hardcoded minibatch size
   disconnected from the real --reflection-minibatch-size); max_metric_calls
   is always set explicitly.

--estimate-only prints the pre-flight cost estimate and exits 0 WITHOUT
constructing a worktree or an adapter -- provably free.
--baseline-only runs the seed candidate once over the selected exercises (no
reflection LM needed, but still behind the live-rollout gate) to validate
the whole pipeline and pre-warm the cache before any reflection spend. The
real run then goes in a fresh --out-dir with --live-cache-dir
<baseline-out-dir>/live_cache: a live run refuses an --out-dir that already
holds a previous run's files (_PRIOR_RUN_MARKERS).

See VALIDATION_PLAN.md for the full design.
Historical-log ingestion + frozen-data reporting (VALIDATION_PLAN Layers
2-3) lives in report_trajectories.py -- a separate, free pipeline this file
no longer touches.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import signal
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# litellm (imported lazily by gepa on the first reflection call) runs its own
# load_dotenv() on import while LITELLM_MODE is DEV, its default. That would
# load a .env this module never parsed, whose names _DOTENV_KEYS could not
# withhold from the agent. Set before any import can pull litellm in.
os.environ.setdefault("LITELLM_MODE", "PRODUCTION")

import yaml  # noqa: E402
from dotenv import dotenv_values, load_dotenv  # noqa: E402

from benchmarks.self_improve.components import load_component_token_costs, load_components
from benchmarks.self_improve.exercises import discover_exercises, practice_dir, split_three_way, split_train_val
from benchmarks.self_improve.live_budget import (
    LiveBudget,
    LiveEvalBudgetExceeded,
    UserStopStopper,
    estimate_cost,
    median_exercise_seconds_from_results,
    render_estimate,
)
from benchmarks.self_improve.live_cache import LiveResultCache
from benchmarks.self_improve.live_eval import (
    LiveEvalHarnessError,
    PolyglotLiveRunner,
    _attempt_timeout_s,
    _sanitize_candidate,
)
from benchmarks.self_improve.manifest import Manifest
from benchmarks.self_improve.polyglot_adapter import PolyglotGEPAAdapter
from benchmarks.self_improve.scratch_worktree import (  # noqa: E402
    ALWAYS_ORCHESTRATOR_ONLY_ENV,
    REFLECTION_LM_API_KEY_ENV,  # also re-exported from here
    scratch_worktree,
)
from benchmarks.self_improve.spend_log import SpendLog  # noqa: E402
#: Hard machine-level deny, checked before every other gate -- lets a shared
#: box refuse live rollouts no matter what command gets pasted into it.
NO_LIVE_ROLLOUTS_ENV = "SELF_IMPROVE_NO_LIVE_ROLLOUTS"
DEFAULT_BENCHMARK_ROOT = Path.home() / "Documents" / "polyglot-benchmark"
_ZERO_USAGE = {"input_tokens": 0, "cache_read_tokens": 0, "output_tokens": 0}

# SELF_IMPROVE_DOTENV exists so a test never touches the real .env
# (test_run_gepa_dotenv.py).
_DOTENV_PATH = Path(os.environ.get("SELF_IMPROVE_DOTENV", str(Path(__file__).parent / ".env")))
# Read once, so the names withheld from the agent are exactly the names that
# were loaded. load_dotenv() on the text keeps its own semantics (override=
# False, interpolation, PYTHON_DOTENV_DISABLED); the readability test mirrors
# python-dotenv's own, so a missing file loads nothing and a FIFO still loads.
_readable = os.path.isfile(_DOTENV_PATH) or (
    os.path.exists(_DOTENV_PATH) and stat.S_ISFIFO(os.stat(_DOTENV_PATH).st_mode)
)
_DOTENV_TEXT = _DOTENV_PATH.read_text(encoding="utf-8") if _readable else ""
load_dotenv(stream=io.StringIO(_DOTENV_TEXT))
#: Every name defined in the .env: the agent-under-test never sees any of them.
_DOTENV_KEYS = frozenset(dotenv_values(stream=io.StringIO(_DOTENV_TEXT), interpolate=False))
del _readable, _DOTENV_TEXT

#: Names the agent-under-test's process tree reads itself. One of these in
#: the .env would reach the orchestrator but be withheld from the agent, so
#: the two would run with different values; _dotenv_refusals() refuses it.
#: POLYGLOT_* is not listed: live_eval sets the ones the child reads.
_CHILD_ENV_NAMES = frozenset({"ATTEMPT_TIMEOUT_S", "CODEX_TIMEOUT_S"})
_CHILD_ENV_PREFIXES = ("LITTLE_CODER_", "PI_")


def _orchestrator_only_env_names() -> frozenset[str]:
    """Names ScratchWorktree.env() withholds from the agent-under-test.
    Reads _DOTENV_KEYS at call time so a test can replace it."""
    return _DOTENV_KEYS | ALWAYS_ORCHESTRATOR_ONLY_ENV


def _dotenv_refusals() -> list[str]:
    """One refusal per .env name that the agent-under-test also reads."""
    return [
        f"{name} is set in benchmarks/self_improve/.env, which is orchestrator-only; "
        "export it in your shell instead."
        for name in sorted(_DOTENV_KEYS)
        if name in _CHILD_ENV_NAMES or name.startswith(_CHILD_ENV_PREFIXES)
    ]


def _check_gates(args: argparse.Namespace) -> list[str]:
    """Returns human-readable refusal messages; empty means every gate the
    requested action needs is satisfied. Never constructs anything real."""
    if os.environ.get(NO_LIVE_ROLLOUTS_ENV):
        return [
            f"Refusing to run: ${NO_LIVE_ROLLOUTS_ENV} is set on this machine. This host is "
            "configured to never run live agent-under-test rollouts. Unset it deliberately "
            "if that is wrong."
        ]

    messages: list[str] = []

    missing_rollout = []
    if not args.model:
        missing_rollout.append("--model")
    if not args.confirm_live_rollouts:
        missing_rollout.append("--confirm-live-rollouts")
    if not args.max_metric_calls:
        missing_rollout.append("--max-metric-calls")

    missing_reflection = []
    if not args.baseline_only:
        if not args.reflection_model:
            missing_reflection.append("--reflection-model")
        if not os.environ.get(REFLECTION_LM_API_KEY_ENV):
            missing_reflection.append(f"${REFLECTION_LM_API_KEY_ENV}")
        if not args.confirm_real_run:
            missing_reflection.append("--confirm-real-run")

    if missing_reflection or missing_rollout:
        lines = ["Refusing to run: this spends TWO independent real budgets."]
        if missing_reflection:
            lines.append(f"  1. Reflection LM tokens. Missing: {', '.join(missing_reflection)}")
        if missing_rollout:
            lines.append(
                "  2. LIVE agent-under-test rollouts (real compute + real wall clock, not "
                f"just API dollars). Missing: {', '.join(missing_rollout)}"
            )
        lines.append("Nothing was spent. Use --estimate-only to see the cost for free.")
        messages.append("\n".join(lines))

    if args.max_metric_calls is not None and args.max_metric_calls <= 0:
        messages.append(
            f"--max-metric-calls must be > 0, got {args.max_metric_calls}. There is no "
            "'unlimited' mode here -- every metric call is a real live exercise run."
        )

    # Unlike the train-pool-compatibility check below (genuinely irrelevant
    # to baseline mode, which never touches minibatch sampling), this basic
    # sanity bound applies regardless of --baseline-only: estimate_cost() is
    # called unconditionally in BOTH modes to print the pre-authorization
    # banner, and a negative --reflection-minibatch-size there produces a
    # negative (nonsensical) cost/wall-clock estimate before the human even
    # gets to decide whether to authorize anything.
    if args.reflection_minibatch_size <= 0:
        messages.append(f"--reflection-minibatch-size must be > 0, got {args.reflection_minibatch_size}.")

    # select_exercises() ignores --exercise-count entirely when --exercises
    # is given (exercises.py: `chosen = list(explicit)`) -- computing these
    # two gates from the raw --exercise-count flag in that case checks a
    # number that isn't the real pool size, both wrongly passing (a small
    # --exercises list padded by GEPA's sampler, paying twice for the same
    # exercise) and wrongly refusing (a large --exercises list against a
    # small default --exercise-count).
    exercise_count = (
        len([e for e in args.exercises.split(",") if e.strip()]) if args.exercises else args.exercise_count
    )

    train_pool_size = max(0, exercise_count - args.val_count)
    if not args.baseline_only and args.reflection_minibatch_size > max(1, train_pool_size):
        messages.append(
            f"--reflection-minibatch-size ({args.reflection_minibatch_size}) must be <= the "
            f"train pool size ({train_pool_size}), or GEPA's batch sampler pads by repeating "
            "exercises -- you'd pay for the same exercise twice in one minibatch for no extra signal."
        )

    if args.val_count >= exercise_count:
        messages.append(
            f"--val-count ({args.val_count}) must be less than the number of exercises "
            f"({exercise_count}) -- there must be at least one training exercise."
        )

    return messages


def _load_component_paths(components_yaml_path: Path) -> dict[str, str]:
    return yaml.safe_load(Path(components_yaml_path).read_text()) or {}


def _check_components_clean(repo_root: Path, components_yaml_path: Path) -> list[str]:
    """Refuse if components.yaml itself, or any component file it maps to,
    has uncommitted changes -- the seed candidate is read from the working
    tree, but the scratch worktree is checked out at a pinned commit. A
    mismatch there means the run isn't evaluating what a human reviewing the
    resulting PR would think it is.

    components.yaml itself must be included: if it was just edited to point
    a pred_name at a different (already-committed) file, that file's own git
    status can be clean even though the MAPPING the scratch worktree will use
    (read from the pinned commit) differs from the mapping just used to build
    the seed candidate -- candidates would then be evaluated against the
    wrong file, or silently dropped by write_components_back()'s own
    mismatched-scope warning."""
    paths = [str(components_yaml_path)]
    mapping = _load_component_paths(components_yaml_path)
    paths.extend(str(repo_root / rel) for rel in mapping.values())
    result = subprocess.run(["git", "status", "--porcelain", "--", *paths],
                             cwd=repo_root, capture_output=True, text=True)
    if result.returncode != 0:
        # A nonzero exit (a components.yaml entry escaping repo_root and
        # producing a "outside repository" pathspec error, --repo-root not
        # being a git checkout at all, etc.) is NOT "nothing is dirty" -- it
        # means this check could not verify the invariant it exists to
        # enforce at all, so it must refuse rather than fail open.
        return [
            f"Could not verify components are clean (git status exited {result.returncode}): "
            f"{result.stderr.strip()}. Refusing rather than assuming a dirty check that failed "
            "to run means the tree is clean."
        ]
    if not result.stdout.strip():
        return []
    dirty = [line[3:].strip() for line in result.stdout.splitlines() if line.strip()]
    return [
        f"Component file(s) have uncommitted changes: {dirty}. The scratch worktree is "
        "checked out at a pinned commit, so a dirty working tree means the seed candidate "
        "differs from what a resulting PR would diff against. Commit or stash first, or "
        "pass --allow-dirty-components."
    ]


def _resolve_exercises(args: argparse.Namespace):
    # Must be absolute: this same path is later handed to a subprocess whose
    # cwd is the SCRATCH WORKTREE, not the caller's cwd -- a relative path
    # would resolve to a different (missing) directory there, turning every
    # exercise into a harness_error even though discovery just succeeded here.
    if args.benchmark_root:
        benchmark_root = Path(args.benchmark_root).expanduser().resolve()
    else:
        benchmark_root = DEFAULT_BENCHMARK_ROOT
    pdir = practice_dir(benchmark_root, args.language)
    available = discover_exercises(pdir)
    explicit = [e.strip() for e in args.exercises.split(",")] if args.exercises else None
    # specs is the search split: GEPA's trainset and valset both come from it,
    # exactly as before the three-way split existed. acceptance/test are only
    # recorded in the manifest for now (T5 and report_test run outside GEPA).
    specs, acceptance, test = split_three_way(
        available, search_count=args.exercise_count, seed=args.seed, language=args.language,
        explicit_search=explicit, acceptance_count=args.acceptance_count, test_count=args.test_count,
    )
    trainset, valset = split_train_val(specs, val_count=args.val_count, seed=args.seed)
    return benchmark_root, pdir, specs, trainset, valset, acceptance, test


def _git_head(repo_root: Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _build_manifest(args: argparse.Namespace, *, repo_root: Path, components_rel: Path, specs,
                    acceptance, test, seed_candidate: dict[str, str]) -> Manifest:
    return Manifest(
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        seed=args.seed,
        benchmark="aider_polyglot",
        language=args.language,
        splits={
            "search": [s.exercise for s in specs],
            "acceptance": [s.exercise for s in acceptance],
            "test": [s.exercise for s in test],
        },
        searchable_components=sorted(seed_candidate),
        sampling={"temperature": args.temperature},
        budget={"max_metric_calls": args.max_metric_calls, "max_wall_clock_s": args.max_wall_clock_s},
        env_fingerprint={
            "model": args.model,
            "thinking": args.thinking,
            "max_attempts": args.max_attempts,
            "retry": not args.no_retry,
            "components_config": str(components_rel),
            "repo_head": _git_head(repo_root),
        },
    )


#: Files a previous run leaves in its --out-dir. gepa/gepa_state.bin matters
#: most: gepa.optimize() silently resumes from it when run_dir= points at a
#: dir that holds one. live_cache/ and gepa.stop are deliberately absent --
#: a warm cache is safe to share, and a stop file has its own refusal.
_PRIOR_RUN_MARKERS = (
    "manifest.yaml",
    "gepa/gepa_state.bin",
    "optimized_components.yaml",
    "seed_baseline.json",
    "spend_log.jsonl",
)


def _prior_run_artifacts(out_dir: Path) -> list[str]:
    return [m for m in _PRIOR_RUN_MARKERS if (out_dir / m).exists()]


def _resolve_components_yaml(repo_root: Path, components_config: str) -> tuple[Path, Path]:
    """Returns (absolute path under repo_root, path relative to repo_root).
    components.yaml paths are conventionally repo-relative -- resolving
    against `cwd` instead would silently break whenever this CLI is
    invoked with --repo-root pointing elsewhere than the working directory."""
    rel = Path(components_config)
    if rel.is_absolute():
        try:
            rel = rel.relative_to(repo_root)
        except ValueError:
            raise ValueError(
                f"--components-config {components_config!r} is outside --repo-root {repo_root} -- "
                "the scratch worktree only ever contains repo_root's own files, so a path outside "
                "it could never be read there anyway."
            ) from None
    return repo_root / rel, rel


def _run_live(args: argparse.Namespace) -> int:
    # --estimate-only spends nothing (no worktree, no adapter, no LM call) --
    # it exists precisely to let a human decide whether to authorize the
    # spend gates below, so it must not itself be blocked by them.
    if not args.estimate_only:
        gate_errors = _check_gates(args) + [f"Refusing to run: {m}" for m in _dotenv_refusals()]
        if gate_errors:
            for msg in gate_errors:
                print(msg, file=sys.stderr)
            return 1

    repo_root = Path(args.repo_root).resolve()
    try:
        components_yaml, components_rel = _resolve_components_yaml(repo_root, args.components_config)
    except ValueError as e:
        print(f"Refusing to run: {e}", file=sys.stderr)
        return 1

    if not args.allow_dirty_components:
        dirty_errors = _check_components_clean(repo_root, components_yaml)
        if dirty_errors:
            for msg in dirty_errors:
                print(msg, file=sys.stderr)
            return 1

    try:
        benchmark_root, pdir, specs, trainset, valset, acceptance, test = _resolve_exercises(args)
    except ValueError as e:
        print(f"Refusing to run: {e}", file=sys.stderr)
        return 1

    component_paths = _load_component_paths(components_yaml)
    seed_candidate = load_components(components_yaml, repo_root=repo_root)
    # The stable rescale baseline PolyglotGEPAAdapter needs to report an
    # accurate live token_cost estimate to reflection (see
    # components.py::_estimate_token_cost's own docstring for why an
    # absolute chars/token estimate doesn't work for this corpus). Computed
    # once, up front, from the real pristine files -- --only-components
    # filtering below narrows seed_candidate but this can stay unfiltered,
    # since it's looked up by pred_name, not iterated wholesale.
    seed_token_costs = load_component_token_costs(components_yaml, repo_root=repo_root)
    if args.only_components:
        only = set(c.strip() for c in args.only_components.split(","))
        unknown = only - set(seed_candidate)
        if unknown:
            print(f"Refusing to run: --only-components names unknown component(s) {sorted(unknown)}",
                  file=sys.stderr)
            return 1
        seed_candidate = {k: v for k, v in seed_candidate.items() if k in only}
        component_paths = {k: v for k, v in component_paths.items() if k in only}

    if not seed_candidate:
        print("Refusing to run: zero components selected.", file=sys.stderr)
        return 1

    per_exercise_timeout_s = args.per_exercise_timeout_s or (args.max_attempts * (_attempt_timeout_s() + 90) + 180)
    assumed_seconds = args.assumed_exercise_seconds
    if assumed_seconds is None:
        assumed_seconds = median_exercise_seconds_from_results(
            repo_root / "benchmarks" / "results_full_polyglot.json"
        ) or 180.0

    est = estimate_cost(
        max_metric_calls=args.max_metric_calls or 0,
        valset_size=len(valset), trainset_size=len(trainset),
        minibatch_size=args.reflection_minibatch_size,
        component_count=len(seed_candidate),
        assumed_exercise_seconds=assumed_seconds,
        exercise_timeout_seconds=per_exercise_timeout_s,
        module_selector=args.module_selector,
        max_wall_clock_seconds=args.max_wall_clock_s,
    )

    print(render_estimate(est))
    print()
    print(f"  Exercises selected : {[s.exercise for s in specs]}")
    print(f"  Train / Val split  : {[s.exercise for s in trainset]} / {[s.exercise for s in valset]}")
    print(f"  Acceptance / Test  : {[s.exercise for s in acceptance]} / {[s.exercise for s in test]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stop_file = out_dir / "gepa.stop"
    print(f"\n  Run dir       : {out_dir}")
    print(f"  Graceful stop : touch {stop_file}")
    print(f"  Withheld from the agent's env : {', '.join(sorted(_orchestrator_only_env_names()))}")

    prior = _prior_run_artifacts(out_dir)
    if args.estimate_only:
        if prior:
            print(f"\n  Note: {out_dir} already holds a previous run ({', '.join(prior)}); "
                  "a real run there would be refused. Use a fresh --out-dir.")
        return 0

    # Built (and validated) before the confirmation prompt so a bad
    # pre-registration refuses before a human authorizes anything; written
    # only once the run is authorized, still before any spend.
    try:
        manifest = _build_manifest(args, repo_root=repo_root, components_rel=components_rel, specs=specs,
                                   acceptance=acceptance, test=test, seed_candidate=seed_candidate)
    except ValueError as e:
        print(f"Refusing to run: invalid run manifest: {e}", file=sys.stderr)
        return 1

    if stop_file.exists():
        # A leftover gepa.stop from a PREVIOUS run at this --out-dir (the
        # default is a fixed path, benchmarks/self_improve/runs/latest) would
        # otherwise stop the run at the very first check -- for a full
        # gepa.optimize() call that's AFTER paying for the entire seed
        # valset evaluation, silently writing back the untouched seed as
        # optimized_components.yaml and logging "completed". Refuse instead
        # of guessing whether this file is a leftover or a deliberate
        # pre-stop.
        print(
            f"Refusing to run: a stop file already exists at {stop_file}. If this is a leftover "
            f"from a previous run, remove it first: rm {stop_file}", file=sys.stderr,
        )
        return 1
    nested_stop_file = out_dir / "gepa" / "gepa.stop"
    if nested_stop_file.exists():
        # GEPA's own FileStopper (added whenever run_dir= is passed) checks
        # <run_dir>/gepa.stop, so a leftover there stops the run exactly like
        # the top-level one above.
        print(
            f"Refusing to run: a stop file already exists at {nested_stop_file}. If this is a "
            f"leftover from a previous run, remove it first: rm {nested_stop_file}", file=sys.stderr,
        )
        return 1

    if prior:
        # One out-dir is one pre-registered run. Reusing it would let
        # gepa.optimize() silently resume from gepa/gepa_state.bin, and the
        # new manifest would replace the one the earlier run was judged by.
        print(
            f"Refusing to run: {out_dir} already holds a previous run ({', '.join(prior)}). "
            f"Each run needs its own pre-registered out-dir: pass a fresh --out-dir. To reuse the "
            f"previous run's warm cache, add --live-cache-dir {out_dir / 'live_cache'} (or the "
            f"--live-cache-dir that run used)",
            file=sys.stderr,
        )
        return 1

    if not args.yes:
        if not sys.stdin.isatty():
            print(
                "Refusing to run non-interactively without --yes: the cost estimate above "
                "could not be confirmed by a human. Re-run with --yes if you have read it "
                "(e.g. under nohup).", file=sys.stderr,
            )
            return 1
        answer = input("Type 'run' to authorize the budget above, anything else to abort: ").strip()
        if answer != "run":
            print("Aborted at confirmation. Nothing was spent.")
            return 2

    stopper = UserStopStopper()

    def _handle_signal(signum, _frame):
        if stopper.requested:
            raise KeyboardInterrupt()  # second signal: abort immediately
        stopper.request_stop()
        print(f"\nReceived signal {signum} -- stopping gracefully after the current iteration. "
              "Press Ctrl-C again to abort immediately.", file=sys.stderr)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    manifest.save(out_dir / "manifest.yaml")

    with (
        SpendLog(out_dir / "spend_log.jsonl") as spend_log,
        scratch_worktree(
            repo_root,
            parent_dir=Path(args.scratch_dir) if args.scratch_dir else None,
            pi_bin=Path(args.pi_bin) if args.pi_bin else None,
            keep=args.keep_scratch,
            orchestrator_only_env=_orchestrator_only_env_names(),
        ) as wt,
    ):
        spend_log.run_start(argv=sys.argv, exercises=[s.task_id for s in specs])
        spend_log.estimate(**{k: v for k, v in vars(est).items()})

        cache = None if args.no_live_cache else LiveResultCache(
            Path(args.live_cache_dir) if args.live_cache_dir else out_dir / "live_cache"
        )
        # A baseline-only run never touches gepa.optimize()'s own iteration
        # loop, so est.max_live_runs' "+2B+V" overshoot allowance (meant for
        # GEPA's parent/child minibatch + valset re-evaluation pattern) does
        # not apply here -- with a small --max-metric-calls the padded
        # allowance would let the runner execute MORE exercises than the cap
        # actually authorized. Baseline mode's cap is exactly what was asked for.
        budget = LiveBudget(
            hard_deadline_monotonic=time.monotonic() + args.max_wall_clock_s,
            max_live_runs=args.max_metric_calls if args.baseline_only else est.max_live_runs,
        )
        runner = PolyglotLiveRunner(
            worktree=wt,
            components_yaml=wt.path / components_rel,
            model=args.model, language=args.language,
            max_attempts=args.max_attempts, retry=not args.no_retry,
            thinking=args.thinking, benchmark_root=benchmark_root,
            cache=cache, per_exercise_timeout_s=per_exercise_timeout_s,
            budget=budget,
            # Called per-result INSIDE run_batch() (cache hits included), not
            # after a whole batch returns -- a later exercise in the same
            # batch raising (e.g. the budget backstop) must not erase the
            # audit trail for exercises that already genuinely ran. A cache
            # hit logs duration_s=0.0 (not the original run's elapsed_s):
            # SpendLog.summarize()'s total_wall_s would otherwise double-count
            # the same real wall-clock time every time that result is reused.
            # Tokens are spend too, so a hit zeroes usage the same way (the
            # memoized LiveRunResult still carries the original numbers).
            # `error` is LiveRunResult.error capped at 500 characters (a
            # harness_error's carries a stderr tail, and every try is
            # logged). With `status`, it is what shows a run of runtime
            # errors from a dead server here; it is often None for those.
            on_result=lambda r: spend_log.exercise(
                exercise_id=r.task_id, status=r.status, score=r.score,
                memo_hit=r.from_cache, duration_s=0.0 if r.from_cache else r.elapsed_s,
                usage=_ZERO_USAGE if r.from_cache else r.usage,
                error=(r.error[:500] if r.error else r.error),
            ),
        )

        if args.baseline_only:
            # No gepa.optimize() call here, so no GEPA-managed FileStopper --
            # honor the same printed stop_file/signal-driven graceful stop by
            # checking it between exercises (one runner.run_batch() call per
            # exercise; each call still consults the on-disk memo first).
            # spend_log.exercise() is logged by the runner's on_result above,
            # not duplicated here.
            results = []
            try:
                for spec in specs:
                    if stopper.requested or stop_file.exists():
                        print(f"\nGraceful stop requested -- stopping baseline after "
                              f"{len(results)}/{len(specs)} exercises.", file=sys.stderr)
                        break
                    (result_one,) = runner.run_batch(seed_candidate, [spec])
                    results.append(result_one)
                    print(f"  {result_one.task_id}: status={result_one.status} score={result_one.score:.2f} "
                          f"{'(cached)' if result_one.from_cache else ''}")
            except LiveEvalBudgetExceeded as e:
                print(f"\nBudget backstop fired: {e}", file=sys.stderr)
                # Persist whatever DID run before the cap hit -- spend_log's
                # whole purpose is to answer "what actually happened", and a
                # backstop firing after partial progress is not a reason to
                # discard that progress.
                (out_dir / "seed_baseline.json").write_text(
                    json.dumps({r.task_id: r.to_dict() for r in results}, indent=2, default=str)
                )
                spend_log.run_end(reason="budget_backstop", error=str(e))
                return 3
            except LiveEvalHarnessError as e:
                # run_batch() raises instead of scoring on the paths listed
                # in LiveEvalHarnessError's docstring -- same partial
                # persistence as the budget backstop above, distinct reason
                # and exit code.
                print(f"\nHarness error, stopping baseline: {e}", file=sys.stderr)
                (out_dir / "seed_baseline.json").write_text(
                    json.dumps({r.task_id: r.to_dict() for r in results}, indent=2, default=str)
                )
                spend_log.run_end(reason="harness_error", error=str(e))
                return 4
            print("\n=== Baseline results ===")
            (out_dir / "seed_baseline.json").write_text(
                json.dumps({r.task_id: r.to_dict() for r in results}, indent=2, default=str)
            )
            spend_log.run_end(reason="completed" if len(results) == len(specs) else "stopped_early")
            return 0

        from benchmarks.self_improve.ingest.common import build_knowledge_topic_index
        knowledge_topic_index = build_knowledge_topic_index(repo_root)
        adapter = PolyglotGEPAAdapter(
            runner, component_paths=component_paths, practice_dir_path=pdir,
            knowledge_topic_index=knowledge_topic_index,
            seed_bodies=seed_candidate, seed_token_costs=seed_token_costs,
        )

        import gepa
        from gepa.utils.stop_condition import TimeoutStopCondition

        reflection_lm_kwargs = {"api_key": os.environ.get(REFLECTION_LM_API_KEY_ENV)}
        if args.reflection_reasoning_effort:
            reflection_lm_kwargs["reasoning_effort"] = args.reflection_reasoning_effort

        class _StopFileStopper:
            """GEPA also auto-adds its own FileStopper at
            <run_dir>/gepa.stop when run_dir= is passed, but that's a
            different, nested path from the single one advertised in the
            startup banner -- this checks the SAME path the banner prints,
            so 'touch <that path>' is literally true regardless of run_dir."""
            def __call__(self, gepa_state) -> bool:
                return stop_file.exists()

        class _SpendLogCallback:
            """Also remembers the best candidate GEPA has reported so far, in
            memory, so a budget/harness overrun escaping gepa.optimize() can
            still write it out -- without parsing GEPA's run_dir state format.
            is_best_program is GEPA's own verdict (its val_evaluation_policy's
            best program), the same rule result.best_candidate uses."""
            def __init__(self) -> None:
                self.best_candidate: dict[str, str] | None = None
                self.best_candidate_idx: int | None = None
                self.best_val_score: float | None = None

            def on_iteration_end(self, event) -> None:
                state = event.get("state")
                spend_log.iteration_end(
                    iteration=event.get("iteration"),
                    proposal_accepted=event.get("proposal_accepted"),
                    total_num_evals=getattr(state, "total_num_evals", None),
                )

            def on_valset_evaluated(self, event) -> None:
                if event.get("is_best_program"):
                    self.best_candidate = dict(event["candidate"])
                    self.best_candidate_idx = event.get("candidate_idx")
                    self.best_val_score = event.get("average_score")

        spend_log_callback = _SpendLogCallback()
        stop_callbacks = [stopper, _StopFileStopper()]
        if args.max_wall_clock_s:
            stop_callbacks.append(TimeoutStopCondition(args.max_wall_clock_s * 0.8))

        print(f"\nStarting real gepa.optimize(): {len(trainset)} train, {len(valset)} val exercises, "
              f"{len(seed_candidate)} components, reflection_model={args.reflection_model}, "
              f"task_model={args.model}")

        try:
            result = gepa.optimize(
                seed_candidate=seed_candidate,
                trainset=trainset, valset=valset,
                adapter=adapter,
                reflection_lm=args.reflection_model,
                reflection_lm_kwargs=reflection_lm_kwargs,
                max_metric_calls=args.max_metric_calls,
                reflection_minibatch_size=args.reflection_minibatch_size,
                module_selector=args.module_selector,
                use_merge=False,
                # Skip reflection when the parent already scored perfect on
                # every sampled exercise -- nothing to improve, and the child
                # minibatch would be pure spend. GEPA compares per exercise
                # (all(s >= perfect_score)); 1.0 is pass on attempt 1.
                skip_perfect_score=args.skip_perfect_score,
                perfect_score=1.0,
                cache_evaluation=True,
                track_best_outputs=True,
                display_progress_bar=False,
                run_dir=str(out_dir / "gepa"),
                callbacks=[spend_log_callback],
                seed=args.seed,
                raise_on_exception=True,
                stop_callbacks=stop_callbacks,
            )
        except (LiveEvalBudgetExceeded, LiveEvalHarnessError) as e:
            if isinstance(e, LiveEvalBudgetExceeded):
                reason, code = "budget_backstop", 3
                print(f"\nBudget backstop fired: {e}", file=sys.stderr)
            else:
                reason, code = "harness_error", 4
                print(f"\nHarness error, stopping optimization: {e}", file=sys.stderr)
            # Every valset evaluation up to here was paid for -- keep the best
            # one rather than only GEPA's run_dir state. Exit codes match
            # --baseline-only's (3 budget, 4 harness).
            best = spend_log_callback
            if best.best_candidate is None:
                print("No candidate finished a valset evaluation before the stop -- "
                      "nothing written.", file=sys.stderr)
            else:
                print(f"PARTIAL RESULT: the run did not finish; writing the best candidate so far, "
                      f"#{best.best_candidate_idx} (valset score {best.best_val_score}).",
                      file=sys.stderr)
                _write_optimized_components(out_dir, best.best_candidate)
            spend_log.run_end(
                reason=reason, error=str(e), partial=True,
                best_candidate_idx=best.best_candidate_idx, best_val_score=best.best_val_score,
            )
            return code

        _write_optimized_components(out_dir, result.best_candidate)
        spend_log.run_end(reason="completed", total_metric_calls=getattr(result, "total_metric_calls", None))

    return 0


def _write_optimized_components(out_dir: Path, candidate) -> None:
    # Sanitize before writing: GEPA's own candidate tracking (and hence
    # best_candidate) carries whatever text the reflection LM proposed,
    # including a re-emitted YAML frontmatter block if it "helpfully"
    # produced one -- live_eval.py's materialize() strips that before
    # every SCORING run, but that stripping never reaches back into
    # GEPA's own retained result. Without this, the exact frontmatter-
    # duplication bug the live-eval guard exists to prevent could still
    # land in the file this writes and get applied for real.
    optimized = _sanitize_candidate(dict(candidate))
    (out_dir / "optimized_components.yaml").write_text(yaml.dump(optimized, sort_keys=True))
    print(f"\nWrote optimized component text to {out_dir / 'optimized_components.yaml'}")
    print("Review the diff, then use apply_results.py to open a PR -- nothing was "
          "committed or pushed automatically.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--components-config", default="benchmarks/self_improve/config/components_bash_only.yaml")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--only-components", default=None,
                     help="Comma-separated pred_names to scope down to, without editing components.yaml.")

    ap.add_argument("--language", default="python")
    ap.add_argument("--benchmark-root", default=None)
    ap.add_argument("--exercises", default=None, help="Comma-separated exercise names, bypassing selection.")
    ap.add_argument("--exercise-count", type=int, default=6)
    ap.add_argument("--val-count", type=int, default=3)
    ap.add_argument("--acceptance-count", type=int, default=None,
                     help="Size of the held-out acceptance split (default: half of the exercises "
                          "outside the search split). Recorded in manifest.yaml; not used by GEPA.")
    ap.add_argument("--test-count", type=int, default=None,
                     help="Size of the held-out test split (default: the rest of the pool). "
                          "Recorded in manifest.yaml; not used by GEPA.")

    ap.add_argument("--model", default=None, help="The model UNDER TEST for live rollouts. Required for a real run.")
    ap.add_argument("--confirm-live-rollouts", action="store_true",
                     help="Required (with --model and --max-metric-calls) to run real live agent rollouts.")
    ap.add_argument("--max-metric-calls", type=int, default=None,
                     help="Hard cap on real live exercise executions. Required. Never use an 'auto' preset.")
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--no-retry", action="store_true")
    ap.add_argument("--thinking", default=None)
    ap.add_argument("--temperature", type=float, default=0.3,
                     help="Sampling temperature to pre-register in manifest.yaml (must be > 0). "
                          "Recorded only -- NOT passed to the agent, whose temperature comes from "
                          ".pi/settings.json's model profile (default_model_profile is 0.3) or the "
                          "model server. Set it to what the model under test actually uses.")
    ap.add_argument("--per-exercise-timeout-s", type=int, default=None)
    ap.add_argument("--max-wall-clock-s", type=float, default=14400.0)
    ap.add_argument("--assumed-exercise-seconds", type=float, default=None)

    ap.add_argument("--reflection-model", default=None)
    ap.add_argument("--reflection-reasoning-effort", default=None)
    ap.add_argument("--confirm-real-run", action="store_true",
                     help=f"Required (with --reflection-model and ${REFLECTION_LM_API_KEY_ENV}) "
                          "to spend reflection LM budget.")
    ap.add_argument("--reflection-minibatch-size", type=int, default=2)
    ap.add_argument("--module-selector", choices=["round_robin", "all"], default="round_robin")
    ap.add_argument("--skip-perfect-score", action=argparse.BooleanOptionalAction, default=True,
                     help="Skip reflection on a minibatch the parent already solved perfectly "
                          "(default on; --no-skip-perfect-score to disable).")

    ap.add_argument("--live-cache-dir", default=None)
    ap.add_argument("--no-live-cache", action="store_true")
    ap.add_argument("--scratch-dir", default=None)
    ap.add_argument("--keep-scratch", action="store_true")
    ap.add_argument("--pi-bin", default=None)
    ap.add_argument("--allow-dirty-components", action="store_true")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="benchmarks/self_improve/runs/latest")
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--yes", "-y", action="store_true")

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return _run_live(args)


if __name__ == "__main__":
    sys.exit(main())
