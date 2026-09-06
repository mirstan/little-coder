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
the whole pipeline and pre-warm the cache before any reflection spend.

See the live-eval plan doc and VALIDATION_PLAN.md for the full design.
Historical-log ingestion + frozen-data reporting (VALIDATION_PLAN Layers
2-3) lives in report_trajectories.py -- a separate, free pipeline this file
no longer touches.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

from benchmarks.self_improve.components import load_components
from benchmarks.self_improve.exercises import discover_exercises, practice_dir, select_exercises, split_train_val
from benchmarks.self_improve.live_budget import (
    LiveBudget,
    LiveEvalBudgetExceeded,
    UserStopStopper,
    estimate_cost,
    median_exercise_seconds_from_results,
    render_estimate,
)
from benchmarks.self_improve.live_cache import LiveResultCache
from benchmarks.self_improve.live_eval import PolyglotLiveRunner
from benchmarks.self_improve.polyglot_adapter import PolyglotGEPAAdapter
from benchmarks.self_improve.scratch_worktree import scratch_worktree
from benchmarks.self_improve.spend_log import SpendLog

REFLECTION_LM_API_KEY_ENV = "REFLECTION_LM_API_KEY"
#: Hard machine-level deny, checked before every other gate -- lets a shared
#: box refuse live rollouts no matter what command gets pasted into it.
NO_LIVE_ROLLOUTS_ENV = "SELF_IMPROVE_NO_LIVE_ROLLOUTS"
DEFAULT_BENCHMARK_ROOT = Path.home() / "Documents" / "polyglot-benchmark"

logger = logging.getLogger(__name__)

# See test_run_gepa_dotenv.py for why SELF_IMPROVE_DOTENV exists (never
# touch the real .env from a test) -- unchanged from the old design.
load_dotenv(Path(os.environ.get("SELF_IMPROVE_DOTENV", str(Path(__file__).parent / ".env"))))


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

    train_pool_size = max(0, args.exercise_count - args.val_count)
    if not args.baseline_only and args.reflection_minibatch_size > max(1, train_pool_size):
        messages.append(
            f"--reflection-minibatch-size ({args.reflection_minibatch_size}) must be <= the "
            f"train pool size ({train_pool_size}), or GEPA's batch sampler pads by repeating "
            "exercises -- you'd pay for the same exercise twice in one minibatch for no extra signal."
        )

    if args.val_count >= args.exercise_count:
        messages.append(
            f"--val-count ({args.val_count}) must be less than --exercise-count "
            f"({args.exercise_count}) -- there must be at least one training exercise."
        )

    return messages


def _load_component_paths(components_yaml_path: Path) -> dict[str, str]:
    return yaml.safe_load(Path(components_yaml_path).read_text()) or {}


def _check_components_clean(repo_root: Path, components_yaml_path: Path) -> list[str]:
    """Refuse if any component file has uncommitted changes -- the seed
    candidate is read from the working tree, but the scratch worktree is
    checked out at a pinned commit. A mismatch there means the run isn't
    evaluating what a human reviewing the resulting PR would think it is."""
    mapping = _load_component_paths(components_yaml_path)
    paths = [str(repo_root / rel) for rel in mapping.values()]
    if not paths:
        return []
    result = subprocess.run(["git", "status", "--porcelain", "--", *paths],
                             cwd=repo_root, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return []
    dirty = [line[3:].strip() for line in result.stdout.splitlines() if line.strip()]
    return [
        f"Component file(s) have uncommitted changes: {dirty}. The scratch worktree is "
        "checked out at a pinned commit, so a dirty working tree means the seed candidate "
        "differs from what a resulting PR would diff against. Commit or stash first, or "
        "pass --allow-dirty-components."
    ]


def _resolve_exercises(args: argparse.Namespace):
    benchmark_root = Path(args.benchmark_root) if args.benchmark_root else DEFAULT_BENCHMARK_ROOT
    pdir = practice_dir(benchmark_root, args.language)
    available = discover_exercises(pdir)
    explicit = [e.strip() for e in args.exercises.split(",")] if args.exercises else None
    specs = select_exercises(available, count=args.exercise_count, seed=args.seed,
                              language=args.language, explicit=explicit)
    trainset, valset = split_train_val(specs, val_count=args.val_count, seed=args.seed)
    return benchmark_root, pdir, specs, trainset, valset


def _resolve_components_yaml(repo_root: Path, components_config: str) -> tuple[Path, Path]:
    """Returns (absolute path under repo_root, path relative to repo_root).
    components.yaml paths are conventionally repo-relative -- resolving
    against `cwd` instead would silently break whenever this CLI is
    invoked with --repo-root pointing elsewhere than the working directory."""
    rel = Path(components_config)
    if rel.is_absolute():
        rel = rel.relative_to(repo_root)
    return repo_root / rel, rel


def _run_live(args: argparse.Namespace) -> int:
    # --estimate-only spends nothing (no worktree, no adapter, no LM call) --
    # it exists precisely to let a human decide whether to authorize the
    # spend gates below, so it must not itself be blocked by them.
    if not args.estimate_only:
        gate_errors = _check_gates(args)
        if gate_errors:
            for msg in gate_errors:
                print(msg, file=sys.stderr)
            return 1

    repo_root = Path(args.repo_root).resolve()
    components_yaml, components_rel = _resolve_components_yaml(repo_root, args.components_config)

    if not args.allow_dirty_components:
        dirty_errors = _check_components_clean(repo_root, components_yaml)
        if dirty_errors:
            for msg in dirty_errors:
                print(msg, file=sys.stderr)
            return 1

    try:
        benchmark_root, pdir, specs, trainset, valset = _resolve_exercises(args)
    except ValueError as e:
        print(f"Refusing to run: {e}", file=sys.stderr)
        return 1

    component_paths = _load_component_paths(components_yaml)
    seed_candidate = load_components(components_yaml, repo_root=repo_root)
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

    per_exercise_timeout_s = args.per_exercise_timeout_s or (args.max_attempts * (900 + 90) + 180)
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Run dir       : {out_dir}")
    print(f"  Graceful stop : touch {out_dir / 'gepa.stop'}")

    if args.estimate_only:
        return 0

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

    with (
        SpendLog(out_dir / "spend_log.jsonl") as spend_log,
        scratch_worktree(
            repo_root,
            parent_dir=Path(args.scratch_dir) if args.scratch_dir else None,
            pi_bin=Path(args.pi_bin) if args.pi_bin else None,
            keep=args.keep_scratch,
        ) as wt,
    ):
        spend_log.run_start(argv=sys.argv, exercises=[s.task_id for s in specs])
        spend_log.estimate(**{k: v for k, v in vars(est).items()})

        cache = None if args.no_live_cache else LiveResultCache(
            Path(args.live_cache_dir) if args.live_cache_dir else out_dir / "live_cache"
        )
        budget = LiveBudget(
            hard_deadline_monotonic=time.monotonic() + args.max_wall_clock_s,
            max_live_runs=est.max_live_runs,
        )
        runner = PolyglotLiveRunner(
            worktree=wt,
            components_yaml=wt.path / components_rel,
            model=args.model, language=args.language,
            max_attempts=args.max_attempts, retry=not args.no_retry,
            thinking=args.thinking, benchmark_root=benchmark_root,
            cache=cache, per_exercise_timeout_s=per_exercise_timeout_s,
            budget=budget,
        )

        if args.baseline_only:
            try:
                results = runner.run_batch(seed_candidate, specs)
            except LiveEvalBudgetExceeded as e:
                print(f"\nBudget backstop fired: {e}", file=sys.stderr)
                spend_log.run_end(reason="budget_backstop", error=str(e))
                return 3
            print("\n=== Baseline results ===")
            for r in results:
                print(f"  {r.task_id}: status={r.status} score={r.score:.2f} "
                      f"{'(cached)' if r.from_cache else ''}")
                spend_log.exercise(exercise_id=r.task_id, status=r.status, score=r.score,
                                    memo_hit=r.from_cache, duration_s=r.elapsed_s)
            (out_dir / "seed_baseline.json").write_text(
                yaml.dump({r.task_id: r.to_dict() for r in results})
            )
            spend_log.run_end(reason="completed")
            return 0

        adapter = PolyglotGEPAAdapter(runner, component_paths=component_paths, practice_dir_path=pdir)

        import gepa
        from gepa.utils.stop_condition import TimeoutStopCondition

        reflection_lm_kwargs = {"api_key": os.environ.get(REFLECTION_LM_API_KEY_ENV)}
        if args.reflection_reasoning_effort:
            reflection_lm_kwargs["reasoning_effort"] = args.reflection_reasoning_effort

        stop_callbacks = [stopper]
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
                skip_perfect_score=args.skip_perfect_score,
                cache_evaluation=True,
                track_best_outputs=True,
                display_progress_bar=False,
                run_dir=str(out_dir / "gepa"),
                seed=args.seed,
                raise_on_exception=True,
                stop_callbacks=stop_callbacks,
            )
        except LiveEvalBudgetExceeded as e:
            print(f"\nBudget backstop fired: {e}", file=sys.stderr)
            spend_log.run_end(reason="budget_backstop", error=str(e))
            return 3

    optimized = dict(result.best_candidate)
    (out_dir / "optimized_components.yaml").write_text(yaml.dump(optimized, sort_keys=True))
    print(f"\nWrote optimized component text to {out_dir / 'optimized_components.yaml'}")
    print("Review the diff, then use apply_results.py to open a PR -- nothing was "
          "committed or pushed automatically.")
    spend_log.run_end(reason="completed", total_metric_calls=getattr(result, "total_metric_calls", None))
    return 0


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

    ap.add_argument("--model", default=None, help="The model UNDER TEST for live rollouts. Required for a real run.")
    ap.add_argument("--confirm-live-rollouts", action="store_true",
                     help="Required (with --model and --max-metric-calls) to run real live agent rollouts.")
    ap.add_argument("--max-metric-calls", type=int, default=None,
                     help="Hard cap on real live exercise executions. Required. Never use an 'auto' preset.")
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--no-retry", action="store_true")
    ap.add_argument("--thinking", default=None)
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
    ap.add_argument("--skip-perfect-score", action="store_true")

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
