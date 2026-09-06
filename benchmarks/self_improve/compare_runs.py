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

    # Real bug, confirmed by review: a dict comprehension silently keeps only
    # the LAST trajectory for a duplicate (benchmark, task_id) key (e.g. a
    # harbor/tb run with multiple trials of the same task) -- discarding an
    # earlier trial's outcome could hide exactly the regression this layer
    # exists to catch. Reject duplicates outright rather than guess which
    # trial should win.
    if len(before_by_id) != len(before):
        raise ValueError("compare_pass_rates: duplicate (benchmark, task_id) trials in `before`")
    if len(after_by_id) != len(after):
        raise ValueError("compare_pass_rates: duplicate (benchmark, task_id) trials in `after`")

    # Real bug, confirmed by review: with both `before` and `after` empty,
    # the set-equality check below passes vacuously (set() == set()) and n=0
    # made both pass rates default to 0.0 with is_regression=False --
    # reporting a Layer 5 comparison as safe when nothing was actually
    # compared (e.g. an upstream ingest failure silently produced no
    # trajectories at all). Real follow-up bug, confirmed by review: the
    # first version of this guard fired on `before` alone being empty even
    # when `after` was NOT -- misreporting a genuinely one-sided input as
    # "both empty" instead of letting the set-mismatch check below report
    # the accurate before-only/after-only diagnostic.
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
