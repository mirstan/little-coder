"""GEPA metric: score + grounded feedback text, keyed by (benchmark, pred_name).

Per TDD_SPEC.md §6.2: define a local ScoreWithFeedback fallback so this
module (and its tests) import cleanly without dspy installed; swap to the
real dspy.teleprompt.gepa type once real GEPA wiring begins.

CONFIRMED REAL BUG (third one found via an actual GEPA run): the package
`dspy.teleprompt.gepa` does NOT re-export ScoreWithFeedback at its
top level -- `from dspy.teleprompt.gepa import ScoreWithFeedback` silently
raises ImportError and falls through to the local dataclass fallback below,
even with dspy fully installed. The local fallback isn't dict-subscriptable,
but dspy's OWN internal feedback_fn (dspy/teleprompt/gepa/gepa.py) does
`o["feedback"]` on whatever metric() returns -- crashing with "'ScoreWithFeedback'
object is not subscriptable" the moment GEPA's reflective step tried to use
it. The real, subscriptable class lives one level deeper, at
dspy.teleprompt.gepa.gepa (re-exported from gepa_utils).

CONFIRMED ARCHITECTURAL FINDING, NOT YET RESOLVED (found via PR review,
verified against installed gepa 3.3.1 source,
gepa/strategies/acceptance.py::StrictImprovementAcceptance -- the DEFAULT
acceptance criterion dspy.GEPA uses whenever the caller doesn't override it,
which run_gepa.py does not): `base_score` below is computed ENTIRELY from
`gold.trajectory` (the historical, already-collected ground truth) and never
reads `pred` (the candidate program's actual rollout output for this
example) -- by design, per HarnessProgram's own docstring in components.py.
This means every candidate gets the IDENTICAL score for the same example,
regardless of what that candidate's instructions say. StrictImprovementAcceptance
accepts a proposed rewrite only if `new_sum > old_sum` over a minibatch --
since old_sum and new_sum are always equal here, that condition can never be
true, on any dataset of any size. GEPA can therefore never accept a single
proposed rewrite as currently designed; this fully explains why the real
Layer 4 run (see README.md) reported no improvement. Fixing this needs a
design decision (e.g. switching acceptance criteria and accepting a
zero-quality-gate "always take the newest LLM rewrite" loop, making scoring
genuinely candidate-sensitive by replaying `pred` through a real evaluator
per candidate, or dropping GEPA's accept/reject loop for a pure
single-proposal-per-review workflow) -- not a metric.py code fix on its own.
"""
from dataclasses import dataclass

from benchmarks.self_improve.schema import ComponentUsage, NormalizedTrajectory

try:
    from dspy.teleprompt.gepa.gepa import ScoreWithFeedback  # type: ignore
except ImportError:
    @dataclass
    class ScoreWithFeedback:  # type: ignore[no-redef]
        score: float
        feedback: str | None = None


def _score_for_benchmark(traj: NormalizedTrajectory) -> float:
    if traj.partial_score is not None:
        return traj.partial_score
    return 1.0 if traj.success else 0.0


def _find_usage(traj: NormalizedTrajectory, pred_name: str) -> ComponentUsage | None:
    for usage in traj.components_used:
        if usage.pred_name == pred_name:
            return usage
    return None


def _component_feedback(traj: NormalizedTrajectory, usage: ComponentUsage, pred_name: str) -> str:
    excerpt = traj.summarized_transcript[-500:] if traj.summarized_transcript else ""
    outcome = "passed" if traj.success else "failed"
    parts = [
        f"Task {traj.task_id} ({traj.benchmark}) {outcome}.",
        f"stop_reason={traj.stop_reason}.",
        f"{pred_name} was injected {usage.invocation_count} time(s)"
        + (" (including right after a tool error)" if usage.was_error_context else "")
        + ".",
    ]
    if traj.failure_signals:
        parts.append(f"failure_signals={traj.failure_signals}.")
    if excerpt:
        parts.append(f"Excerpt: {excerpt}")
    return " ".join(parts)


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None):
    """Return type depends on the caller, confirmed via two real GEPA runs:

    1. trace/pred_name/pred_trace/program_trace all default to None: dspy's
       plain Evaluate utility (used by GEPA for full valset/trainset Pareto
       tracking, NOT just its own reflective credit-assignment step) calls
       metric(gold, pred, trace) with only 3 positional args. Without these
       defaults, every such call crashed with "missing 2 required positional
       arguments" -- silently swallowed by dspy's parallelizer into a 0.0,
       producing a fully broken optimization pass with no real signal.

    2. When pred_name is None (i.e. that same plain-Evaluate path), the
       return value must be a bare float/bool, not a ScoreWithFeedback --
       confirmed via TypeError: unsupported operand type(s) for +: 'int' and
       'ScoreWithFeedback' inside dspy's progress-reporting sum(vals). The
       real dspy.teleprompt.gepa.gepa_utils.ScoreWithFeedback extends
       dspy.Prediction (dict-like), not float -- genuinely incompatible with
       sum(). ScoreWithFeedback is only valid on the OTHER calling
       convention: GEPA's reflective credit-assignment step, which always
       supplies a real pred_name.
    """
    traj: NormalizedTrajectory = gold.trajectory
    base_score = _score_for_benchmark(traj)

    if pred_name is None:
        return base_score

    usage = _find_usage(traj, pred_name)
    if usage is None:
        # Real bug, confirmed by review: HarnessProgram.forward() calls
        # EVERY predictor for every example (required so GEPA has a real
        # trace entry per predictor -- see components.py), so "this
        # component wasn't used in this trajectory" is the COMMON case, not
        # rare. feedback=None was passed straight through into dspy's
        # reflection prompt renderer, which stringifies it -- the paid
        # reflection LM was reading a literal "# Feedback\nNone" for most
        # rows. A real, informative string costs nothing extra and gives
        # the reflection step an actual signal instead of noise.
        outcome = "passed" if traj.success else "failed"
        return ScoreWithFeedback(
            score=base_score,
            feedback=f"{pred_name} was not injected in this trajectory "
                     f"(task {traj.task_id}, {traj.benchmark}, {outcome}). "
                     f"Its text had no effect on this outcome.",
        )

    return ScoreWithFeedback(
        score=base_score,
        feedback=_component_feedback(traj, usage, pred_name),
    )


def weighted_aggregate(per_benchmark_avg: dict[str, float], weights: dict[str, float]) -> float:
    """Combine per-benchmark average scores using weights, renormalized over
    only the benchmarks actually present in per_benchmark_avg."""
    present = {k: weights.get(k, 0.0) for k in per_benchmark_avg}
    total_weight = sum(present.values())
    if total_weight == 0:
        return sum(per_benchmark_avg.values()) / len(per_benchmark_avg) if per_benchmark_avg else 0.0
    return sum(per_benchmark_avg[k] * (present[k] / total_weight) for k in per_benchmark_avg)
