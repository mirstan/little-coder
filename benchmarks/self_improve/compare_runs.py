"""Before/after pass-rate comparison for VALIDATION_PLAN.md Layer 5's held-out
live regression check. Pure comparison over two NormalizedTrajectory lists --
does not execute any benchmark itself; the caller re-runs the harness with the
old and new component text and passes the resulting trajectories here.
"""
from benchmarks.self_improve.schema import NormalizedTrajectory


def compare_pass_rates(
    before: list[NormalizedTrajectory], after: list[NormalizedTrajectory]
) -> dict:
    # Keyed by (benchmark, task_id), not task_id alone: real bug, confirmed
    # by review -- task_id is NOT namespaced per benchmark (aider uses the
    # exercise key, gaia the task-dir name, harbor/tb their own id), so a
    # before/after list spanning more than one benchmark could silently
    # collide two different trajectories onto one dict entry, under-counting
    # n and corrupting the exact regression signal this layer exists for.
    before_by_id = {(t.benchmark, t.task_id): t for t in before}
    after_by_id = {(t.benchmark, t.task_id): t for t in after}

    if set(before_by_id) != set(after_by_id):
        raise ValueError(
            "compare_pass_rates requires the same (benchmark, task_id) set in both runs; "
            f"before-only: {set(before_by_id) - set(after_by_id)}, "
            f"after-only: {set(after_by_id) - set(before_by_id)}"
        )

    regressions = []
    improvements = []
    for key in sorted(before_by_id):
        was = before_by_id[key].success
        now = after_by_id[key].success
        if was and not now:
            regressions.append(key)
        elif not was and now:
            improvements.append(key)

    n = len(before_by_id)
    before_pass_rate = sum(t.success for t in before_by_id.values()) / n if n else 0.0
    after_pass_rate = sum(t.success for t in after_by_id.values()) / n if n else 0.0

    return {
        "before_pass_rate": before_pass_rate,
        "after_pass_rate": after_pass_rate,
        "regressions": regressions,
        "improvements": improvements,
        "is_regression": len(regressions) > 0,
    }
