"""CLI entrypoint for the GEPA self-improvement run.

--dry-run (VALIDATION_PLAN.md Layer 3): ingest real trajectories, build
HarnessProgram from the real components.yaml-mapped files, run metric() over
every trajectory for every pred_name its components_used references plus the
episode-level pred_name=None call, compute the weighted aggregate score, and
print a summary -- WITHOUT constructing a dspy.LM or calling GEPA.compile().
Real (non-dry-run) GEPA runs are not implemented yet; that's Layer 4 and
requires explicit reflection_lm/API-key configuration.
"""
import argparse
import logging
import os
import random
import sys
from pathlib import Path

import dspy
import yaml
from dspy.utils.dummies import DummyLM

from benchmarks.self_improve.components import build_harness_program, load_components
from benchmarks.self_improve.ingest import aider_polyglot_ingest, gaia_ingest, harbor_tb_ingest
from benchmarks.self_improve.metric import metric, weighted_aggregate
from benchmarks.self_improve.schema import NormalizedTrajectory

# Separate from any env var used by the model UNDER TEST -- reflection_lm is
# always a distinct, typically stronger model (architecture §6).
REFLECTION_LM_API_KEY_ENV = "REFLECTION_LM_API_KEY"

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {"aider_polyglot": 0.4, "gaia": 0.3, "harbor": 0.15, "tb": 0.15}


class _Example:
    def __init__(self, trajectory: NormalizedTrajectory):
        self.trajectory = trajectory


def _ingest_all(log_roots: dict[str, str]) -> list[NormalizedTrajectory]:
    """log_roots keys: 'aider' (value: 'log_root,results_json'), 'gaia', 'harbor', 'tb'."""
    trajectories: list[NormalizedTrajectory] = []

    if "aider" in log_roots:
        raw = log_roots["aider"]
        log_root_str, _, results_json_str = raw.partition(",")
        try:
            trajs = aider_polyglot_ingest.load(Path(log_root_str), Path(results_json_str))
            trajectories.extend(trajs)
            logger.info("ingested %d aider_polyglot trajectories", len(trajs))
        except Exception as e:
            logger.warning("aider_polyglot ingest failed: %s", e)

    if "gaia" in log_roots:
        try:
            trajs = gaia_ingest.load(Path(log_roots["gaia"]))
            trajectories.extend(trajs)
            logger.info("ingested %d gaia trajectories", len(trajs))
        except Exception as e:
            logger.warning("gaia ingest failed: %s", e)

    for key in ("harbor", "tb"):
        if key in log_roots:
            try:
                trajs = harbor_tb_ingest.load(Path(log_roots[key]), benchmark=key)
                trajectories.extend(trajs)
                logger.info("ingested %d %s trajectories", len(trajs), key)
            except Exception as e:
                logger.warning("%s ingest failed: %s", key, e)

    return trajectories


def _dry_run(trajectories: list[NormalizedTrajectory], components: dict[str, str], weights: dict[str, float]):
    print(f"\n=== Dry run: {len(trajectories)} trajectories, {len(components)} components ===\n")

    if not trajectories:
        print("WARNING: no trajectories ingested -- nothing to score.")
        return

    per_benchmark_scores: dict[str, list[float]] = {}
    per_component_counts: dict[str, int] = {}
    errors = 0

    for traj in trajectories:
        example = _Example(traj)
        try:
            episode_result = metric(example, pred=None, trace=None, pred_name=None, pred_trace=None)
        except Exception as e:
            logger.warning("metric() failed for %s/%s (episode-level): %s", traj.benchmark, traj.task_id, e)
            errors += 1
            continue

        if episode_result.score is None:
            logger.warning("metric() returned None score for %s/%s", traj.benchmark, traj.task_id)
            errors += 1
            continue

        per_benchmark_scores.setdefault(traj.benchmark, []).append(episode_result.score)

        for usage in traj.components_used:
            if usage.pred_name not in components:
                continue
            try:
                comp_result = metric(example, pred=None, trace=None,
                                      pred_name=usage.pred_name, pred_trace=None)
            except Exception as e:
                logger.warning("metric() failed for %s/%s pred_name=%s: %s",
                                traj.benchmark, traj.task_id, usage.pred_name, e)
                errors += 1
                continue
            if comp_result.feedback is not None:
                per_component_counts[usage.pred_name] = per_component_counts.get(usage.pred_name, 0) + 1

    per_benchmark_avg = {
        b: sum(scores) / len(scores) for b, scores in per_benchmark_scores.items()
    }
    aggregate = weighted_aggregate(per_benchmark_avg, weights)

    print("Per-benchmark trajectory counts:")
    for b, scores in sorted(per_benchmark_scores.items()):
        print(f"  {b}: {len(scores)} trajectories, avg score {per_benchmark_avg[b]:.3f}")

    print(f"\nWeighted aggregate score: {aggregate:.3f}")
    assert 0.0 <= aggregate <= 1.0, f"aggregate out of [0,1] range: {aggregate}"

    print(f"\nPer-component attributed-feedback counts (nonzero only):")
    if per_component_counts:
        for name, count in sorted(per_component_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {name}: {count}")
    else:
        print("  (none -- no trajectory attributed feedback to any component)")

    print(f"\nErrors encountered: {errors}")
    if errors:
        print("WARNING: some trajectories/components failed to score -- see log above.")

    print("\n=== Dry run complete (no dspy.LM constructed, no GEPA.compile() called) ===")


def build_examples(trajectories: list[NormalizedTrajectory]) -> list:
    """One dspy.Example per trajectory. task_context is the only forward()
    input; .trajectory is read directly by metric() and never used as a
    model input (avoids leaking the outcome into the forward pass)."""
    examples = []
    for traj in trajectories:
        ex = dspy.Example(
            trajectory=traj,
            task_context=traj.summarized_transcript or traj.task_id,
        ).with_inputs("task_context")
        examples.append(ex)
    return examples


def split_train_val(trajectories: list[NormalizedTrajectory], train_frac: float, seed: int):
    """Stratify by (benchmark, success) per architecture §6."""
    from collections import defaultdict

    groups = defaultdict(list)
    for t in trajectories:
        groups[(t.benchmark, t.success)].append(t)

    rng = random.Random(seed)
    train, val = [], []
    for group in groups.values():
        shuffled = group[:]
        rng.shuffle(shuffled)
        cut = max(1, round(len(shuffled) * train_frac)) if len(shuffled) > 1 else len(shuffled)
        train.extend(shuffled[:cut])
        val.extend(shuffled[cut:] or shuffled[:cut])  # tiny groups: reuse for val rather than empty
    return train, val


def _real_run(
    trajectories: list[NormalizedTrajectory],
    components: dict[str, str],
    args,
) -> int:
    """Wires the real (non-dry-run) GEPA path. Requires explicit, redundant
    confirmation before GEPA.compile() -- which spends real reflection_lm
    API calls -- ever executes: --reflection-model, REFLECTION_LM_API_KEY
    env var, AND --confirm-real-run must ALL be present. Missing any one
    prints a clear message and exits without calling compile()."""
    missing = []
    if not args.reflection_model:
        missing.append("--reflection-model")
    api_key = os.environ.get(REFLECTION_LM_API_KEY_ENV)
    if not api_key:
        missing.append(f"${REFLECTION_LM_API_KEY_ENV}")
    if not args.confirm_real_run:
        missing.append("--confirm-real-run")

    if missing:
        print(
            "Refusing to run GEPA for real: this spends real API budget on "
            f"reflection_lm calls. Missing: {', '.join(missing)}.\n"
            "Use --dry-run to validate the pipeline for free first.",
            file=sys.stderr,
        )
        return 1

    if not trajectories:
        print("No trajectories ingested -- nothing to optimize against.", file=sys.stderr)
        return 1

    train_trajs, val_trajs = split_train_val(trajectories, args.train_frac, args.seed)
    trainset = build_examples(train_trajs)
    valset = build_examples(val_trajs)

    program = build_harness_program(components)

    # forward-pass rollouts are backed by DummyLM: free, no network, no API
    # key -- see components.py::HarnessProgram for why this is required
    # (GEPA needs a real trace entry per predictor, but never needs the
    # actual generated text, since metric() scores from gold.trajectory).
    dspy.settings.configure(lm=DummyLM([{"guidance": "n/a"}] * 10_000))

    reflection_lm = dspy.LM(model=args.reflection_model, api_key=api_key)

    gepa = dspy.GEPA(metric=metric, reflection_lm=reflection_lm, auto="light")
    print(f"Starting real GEPA.compile(): {len(trainset)} train, {len(valset)} val examples, "
          f"{len(components)} components, reflection_model={args.reflection_model}")
    optimized_program = gepa.compile(program, trainset=trainset, valset=valset)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    optimized_texts = {
        name: predictor.signature.instructions
        for name, predictor in optimized_program.predictors.items()
    }
    (out_dir / "optimized_components.yaml").write_text(yaml.dump(optimized_texts))
    print(f"Wrote optimized component text to {out_dir / 'optimized_components.yaml'}")
    print("Review the diff, then use apply_results.py to open a PR -- nothing was "
          "committed or pushed automatically.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-roots", nargs="+", default=[],
                     help="key=value pairs, e.g. aider=<log_root>,<results.json> gaia=<log_root>")
    ap.add_argument("--components-config",
                     default="benchmarks/self_improve/config/components.yaml")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reflection-model", default=None)
    ap.add_argument("--confirm-real-run", action="store_true",
                     help="Required in addition to --reflection-model and "
                          f"${REFLECTION_LM_API_KEY_ENV} to actually spend API budget.")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="benchmarks/self_improve/runs/latest")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    log_roots = {}
    for kv in args.log_roots:
        key, _, value = kv.partition("=")
        log_roots[key] = value

    trajectories = _ingest_all(log_roots)
    components = load_components(Path(args.components_config), repo_root=Path(args.repo_root))

    weights_path = Path(args.components_config).parent / "benchmark_weights.yaml"
    weights = DEFAULT_WEIGHTS
    if weights_path.exists():
        weights = yaml.safe_load(weights_path.read_text()) or DEFAULT_WEIGHTS

    if args.dry_run:
        _dry_run(trajectories, components, weights)
        return 0

    return _real_run(trajectories, components, args)


if __name__ == "__main__":
    sys.exit(main())
