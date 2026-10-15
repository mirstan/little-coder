"""Before/after pass-rate comparison for VALIDATION_PLAN.md Layer 5's held-out
live regression check. Pure comparison over two NormalizedTrajectory lists --
does not execute any benchmark itself; the caller re-runs the harness with the
old and new component text and passes the resulting trajectories here.
"""
from benchmarks.self_improve.schema import NormalizedTrajectory


def compare_pass_rates(
    before: list[NormalizedTrajectory], after: list[NormalizedTrajectory]
) -> dict:
    # Keyed by (benchmark, task_id), not task_id alone: task_id is NOT
    # namespaced per benchmark (aider uses the exercise key, gaia the
    # task-dir name, harbor/tb their own id), so a list spanning more than
    # one benchmark could collide two different trajectories onto one entry,
    # under-counting n and corrupting the regression signal.
    before_by_id = {(t.benchmark, t.task_id): t for t in before}
    after_by_id = {(t.benchmark, t.task_id): t for t in after}

    # The comprehensions above keep only the LAST trajectory for a duplicate
    # key (e.g. a harbor/tb run with several trials of one task), which could
    # hide the regression this layer exists to catch. Reject rather than
    # guess which trial should win.
    if len(before_by_id) != len(before):
        raise ValueError("compare_pass_rates: duplicate (benchmark, task_id) trials in `before`")
    if len(after_by_id) != len(after):
        raise ValueError("compare_pass_rates: duplicate (benchmark, task_id) trials in `after`")

    # With both sides empty the set-equality check below passes vacuously,
    # and a Layer 5 comparison over nothing at all (e.g. an upstream ingest
    # failure) would report 0.0/0.0 and is_regression=False -- i.e. "safe".
    # Both, not either: a one-sided input gets the accurate before-only /
    # after-only diagnostic from the set-mismatch check instead.
    if not before_by_id and not after_by_id:
        raise ValueError("compare_pass_rates: no trajectories to compare (both `before` and `after` are empty)")

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

    n = len(before_by_id)  # always > 0 here -- the empty-input case raised above
    before_pass_rate = sum(t.success for t in before_by_id.values()) / n
    after_pass_rate = sum(t.success for t in after_by_id.values()) / n

    return {
        "before_pass_rate": before_pass_rate,
        "after_pass_rate": after_pass_rate,
        "regressions": regressions,
        "improvements": improvements,
        "is_regression": len(regressions) > 0,
    }
