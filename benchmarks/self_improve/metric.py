"""GEPA metric: score + grounded feedback text, keyed by (benchmark, pred_name).

Per TDD_SPEC.md §6.2: define a local ScoreWithFeedback fallback so this
module (and its tests) import cleanly without dspy installed; swap to the
real dspy.teleprompt.gepa type once real GEPA wiring begins.
"""
from dataclasses import dataclass

from benchmarks.self_improve.schema import ComponentUsage, NormalizedTrajectory

try:
    from dspy.teleprompt.gepa import ScoreWithFeedback  # type: ignore
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


def metric(gold, pred, trace, pred_name, pred_trace, program_trace=None) -> ScoreWithFeedback:
    traj: NormalizedTrajectory = gold.trajectory
    base_score = _score_for_benchmark(traj)

    if pred_name is None:
        return ScoreWithFeedback(score=base_score, feedback=None)

    usage = _find_usage(traj, pred_name)
    if usage is None:
        return ScoreWithFeedback(score=base_score, feedback=None)

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
