"""Pre-flight cost/wall-clock estimate and hard-backstop budget tracking for
a live-eval GEPA run.

`auto=` (dspy.GEPA's light/medium/heavy presets) is never used for a live
run -- confirmed to wildly over-provision (e.g. ~11,852 metric calls for a
32-component set, since the preset's internal budget heuristic uses a
hardcoded minibatch size disconnected from the real
reflection_minibatch_size). max_metric_calls is set explicitly by hand, and
this module turns that (plus the other knobs) into a human-readable
estimate BEFORE anything is spent.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CostEstimate:
    max_metric_calls: int
    valset_size: int
    trainset_size: int
    minibatch_size: int
    component_count: int
    module_selector: str
    overshoot_allowance: int
    max_live_runs: int
    iterations_worst_case: int
    iterations_best_case: int
    components_covered_worst: int
    max_reflection_calls: int
    assumed_exercise_seconds: float
    exercise_timeout_seconds: float
    expected_wall_clock_seconds: float
    ceiling_wall_clock_seconds: float
    warnings: list[str] = field(default_factory=list)


def _ceil_div(numerator: int, denominator: int) -> int:
    """max(0, ceil(numerator / denominator)); 0 whenever numerator <= 0 or
    denominator is 0 (a non-positive remaining budget affords zero
    iterations, not a nonsensical negative or division-by-zero result)."""
    if numerator <= 0 or not denominator:
        return 0
    return -(-numerator // denominator)


def estimate_cost(
    *, max_metric_calls: int, valset_size: int, trainset_size: int, minibatch_size: int,
    component_count: int, assumed_exercise_seconds: float, exercise_timeout_seconds: float,
    module_selector: str = "round_robin", max_wall_clock_seconds: float | None = None,
) -> CostEstimate:
    """Formula: total live runs ~= V (seed valset eval) + iterations*2*B
    (parent+child evaluated on the same minibatch each iteration) +
    accepted*V (a full valset re-eval on every accepted proposal).
    overshoot_allowance (2B+V) exists because GEPA's `_should_stop` is only
    checked at the TOP of its optimization loop, so a run can complete one
    more full iteration after the nominal budget is exhausted."""
    overshoot = 2 * minibatch_size + valset_size
    max_live_runs = max_metric_calls + overshoot
    denom_worst = 2 * minibatch_size + valset_size
    denom_best = 2 * minibatch_size
    # Ceiling, not floor: GEPA's _should_stop is only checked at the TOP of
    # the loop (the same reason overshoot_allowance exists above), so a
    # partial remainder still buys one more iteration attempt, not zero.
    remaining = max_metric_calls - valset_size
    iterations_worst_case = _ceil_div(remaining, denom_worst)
    iterations_best_case = _ceil_div(remaining, denom_best)
    max_reflection_calls = iterations_best_case

    if module_selector == "round_robin":
        components_covered_worst = min(component_count, iterations_worst_case)
    else:
        # "all" touches every component every iteration -- but if the budget
        # can't afford even one iteration, no component is actually reached.
        components_covered_worst = component_count if iterations_worst_case > 0 else 0

    expected_wall = max_live_runs * assumed_exercise_seconds
    ceiling_wall = max_live_runs * exercise_timeout_seconds

    warnings: list[str] = []
    if module_selector == "round_robin" and components_covered_worst < component_count:
        needed = valset_size + component_count * denom_worst
        warnings.append(
            f"round_robin touches ONE component per iteration. At this budget only "
            f"{components_covered_worst} of {component_count} components can be reached "
            f"in the worst case. Raise --max-metric-calls to >= {needed}, or scope down "
            f"with --only-components."
        )
    if max_wall_clock_seconds is not None and expected_wall > max_wall_clock_seconds:
        warnings.append(
            f"expected wall clock ({expected_wall / 3600:.1f}h) exceeds --max-wall-clock-s "
            f"({max_wall_clock_seconds / 3600:.1f}h). The run will be cut off before "
            f"exhausting --max-metric-calls; results will be a partial search."
        )
    if iterations_worst_case == 0:
        warnings.append(
            f"--max-metric-calls {max_metric_calls} barely covers the {valset_size}-example "
            f"startup evaluation; zero optimization iterations are affordable."
        )

    return CostEstimate(
        max_metric_calls=max_metric_calls, valset_size=valset_size, trainset_size=trainset_size,
        minibatch_size=minibatch_size, component_count=component_count, module_selector=module_selector,
        overshoot_allowance=overshoot, max_live_runs=max_live_runs,
        iterations_worst_case=iterations_worst_case, iterations_best_case=iterations_best_case,
        components_covered_worst=components_covered_worst, max_reflection_calls=max_reflection_calls,
        assumed_exercise_seconds=assumed_exercise_seconds, exercise_timeout_seconds=exercise_timeout_seconds,
        expected_wall_clock_seconds=expected_wall, ceiling_wall_clock_seconds=ceiling_wall,
        warnings=warnings,
    )


def render_estimate(est: CostEstimate) -> str:
    lines = [
        "=== Pre-flight cost estimate (nothing spent yet) ===",
        "",
        f"  Components under optimization : {est.component_count}",
        f"  Component selector            : {est.module_selector}",
        f"  Held-out val exercises  (V)   : {est.valset_size}",
        f"  Train exercise pool           : {est.trainset_size}",
        f"  Reflection minibatch    (B)   : {est.minibatch_size}",
        f"  max_metric_calls        (M)   : {est.max_metric_calls}",
        "",
        f"  Iterations affordable          : {est.iterations_worst_case} (worst case, all accepted)",
        f"                                   {est.iterations_best_case} (best case, none accepted)",
        f"  Components reachable          : {est.components_covered_worst} of {est.component_count}",
        f"  Reflection LM calls           : up to {est.max_reflection_calls}",
        "",
        f"  LIVE exercise executions      : up to {est.max_live_runs}  (M + 2B + V, since GEPA's "
        f"stop check only runs at iteration boundaries)",
        f"  Per exercise                  : up to {est.exercise_timeout_seconds:.0f}s",
        "",
        f"  Expected wall clock           : {est.expected_wall_clock_seconds / 3600:.1f}h "
        f"(at {est.assumed_exercise_seconds:.0f}s/exercise)",
        f"  Worst-case wall clock         : {est.ceiling_wall_clock_seconds / 3600:.1f}h",
    ]
    for w in est.warnings:
        lines.append("")
        lines.append(f"  WARNING: {w}")
    return "\n".join(lines)


def median_exercise_seconds_from_results(results_json: Path) -> float | None:
    """Median of exercises[*].elapsed_s from a real results_full_polyglot.json
    -shaped file, for a grounded (rather than default-guessed)
    assumed_exercise_seconds. Never raises -- a missing/malformed file just
    means the caller falls back to its own default."""
    path = Path(results_json)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    exercises = data.get("exercises", {})
    if not isinstance(exercises, dict):
        return None
    values = sorted(
        v["elapsed_s"] for v in exercises.values()
        if isinstance(v, dict) and isinstance(v.get("elapsed_s"), (int, float))
    )
    if not values:
        return None
    n = len(values)
    mid = n // 2
    return values[mid] if n % 2 else (values[mid - 1] + values[mid]) / 2


class LiveEvalBudgetExceeded(RuntimeError):
    """Raised by LiveBudget.check_before_exercise() rather than the caller
    silently returning a fabricated 0.0 score for a run it refuses to
    start -- a fake score would poison the search with a reason that has
    nothing to do with the candidate."""


@dataclass
class LiveBudget:
    hard_deadline_monotonic: float
    max_live_runs: int
    live_runs: int = 0

    def check_before_exercise(self, exercise_id: str) -> None:
        if time.monotonic() >= self.hard_deadline_monotonic:
            raise LiveEvalBudgetExceeded(f"wall-clock hard deadline reached before running {exercise_id!r}")
        if self.live_runs >= self.max_live_runs:
            raise LiveEvalBudgetExceeded(f"max_live_runs ({self.max_live_runs}) reached before running {exercise_id!r}")

    def remaining_seconds(self) -> float:
        """How long until the hard wall-clock deadline. check_before_exercise()
        only gates whether an exercise may START -- once started, a
        subprocess's own (much larger) per-exercise timeout would otherwise
        let --max-wall-clock-s be exceeded by up to one full exercise
        timeout. A caller should bound its own per-exercise timeout by this
        value so the deadline is enforced during an exercise too, not just
        between them."""
        return self.hard_deadline_monotonic - time.monotonic()

    def record_live_run(self) -> None:
        self.live_runs += 1


class UserStopStopper:
    """StopperProtocol-compatible (callable(gepa_state) -> bool) for
    gepa.optimize()'s stop_callbacks. Do NOT also pass gepa's own
    SignalStopper -- both install process-wide SIGINT/SIGTERM handlers, and
    whichever is installed last silently wins, disabling the other. This
    class does not install any signal handler itself; the caller wires
    request_stop() to its own signal.signal(...) registration."""

    def __init__(self):
        self.requested = False

    def request_stop(self, *_args) -> None:
        self.requested = True

    def __call__(self, gepa_state) -> bool:
        return self.requested
