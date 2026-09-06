"""compare_runs.py: before/after pass-rate comparison for VALIDATION_PLAN.md
Layer 5's held-out live regression check. Pure comparison logic against
NormalizedTrajectory lists -- no execution of any benchmark happens here."""
import pytest

from benchmarks.self_improve.compare_runs import compare_pass_rates
from benchmarks.self_improve.schema import NormalizedTrajectory


def _traj(task_id, success, benchmark="aider_polyglot"):
    return NormalizedTrajectory(
        benchmark=benchmark, task_id=task_id, success=success,
        stop_reason="agent_end" if success else "fail", turn_count=1,
        partial_score=1.0 if success else 0.0,
    )


def test_compare_pass_rates_matches_by_task_id():
    before = [_traj("a", True), _traj("b", False)]
    after = [_traj("a", True), _traj("b", True)]
    result = compare_pass_rates(before, after)
    assert result["before_pass_rate"] == 0.5
    assert result["after_pass_rate"] == 1.0


def test_compare_pass_rates_flags_regressions():
    """A task that passed before and fails after is a REGRESSION -- must be
    surfaced explicitly, since it's the exact failure mode Layer 5 exists to
    catch (an offline-looking improvement that breaks a real case)."""
    before = [_traj("a", True), _traj("b", True)]
    after = [_traj("a", True), _traj("b", False)]
    result = compare_pass_rates(before, after)
    assert result["regressions"] == [("aider_polyglot", "b")]
    assert result["improvements"] == []
    assert result["is_regression"] is True


def test_compare_pass_rates_flags_improvements():
    before = [_traj("a", False)]
    after = [_traj("a", True)]
    result = compare_pass_rates(before, after)
    assert result["improvements"] == [("aider_polyglot", "a")]
    assert result["regressions"] == []
    assert result["is_regression"] is False


def test_compare_pass_rates_does_not_collide_same_task_id_across_benchmarks():
    """Real bug, confirmed by review: task_id is NOT namespaced per
    benchmark (aider uses the exercise key, gaia the task-dir name, harbor/
    tb their own id) -- keying by task_id alone would silently collide two
    DIFFERENT trajectories that happen to share a task_id string in
    different benchmarks, under-counting n and corrupting the regression
    signal this whole layer exists to compute correctly."""
    before = [
        _traj("task-1", True, benchmark="gaia"),
        _traj("task-1", True, benchmark="harbor"),  # same task_id, different benchmark
    ]
    after = [
        _traj("task-1", True, benchmark="gaia"),
        _traj("task-1", False, benchmark="harbor"),  # only harbor's regressed
    ]
    result = compare_pass_rates(before, after)
    assert result["before_pass_rate"] == 1.0
    assert result["after_pass_rate"] == 0.5  # both counted, not collided into one
    assert result["regressions"] == [("harbor", "task-1")]


def test_compare_pass_rates_raises_on_mismatched_task_sets():
    """Comparing runs over different task sets produces a meaningless
    pass-rate delta -- this must be caught loudly, not silently averaged
    over whatever partial overlap happens to exist."""
    before = [_traj("a", True)]
    after = [_traj("b", True)]
    with pytest.raises(ValueError):
        compare_pass_rates(before, after)


def test_compare_pass_rates_raises_on_duplicate_trials_in_before():
    """Real bug, confirmed by review: a dict comprehension keyed by
    (benchmark, task_id) silently keeps only the LAST of two duplicate
    trials (e.g. a harbor/tb run with multiple trials of the same task) --
    discarding an earlier trial's outcome could hide exactly the regression
    Layer 5 exists to catch. Must raise, not silently pick a winner."""
    before = [_traj("a", True), _traj("a", False)]  # two trials, same (benchmark, task_id)
    after = [_traj("a", True)]
    with pytest.raises(ValueError, match="duplicate"):
        compare_pass_rates(before, after)


def test_compare_pass_rates_raises_on_duplicate_trials_in_after():
    before = [_traj("a", True)]
    after = [_traj("a", True), _traj("a", False)]
    with pytest.raises(ValueError, match="duplicate"):
        compare_pass_rates(before, after)


def test_compare_pass_rates_raises_mismatch_not_empty_error_for_one_sided_input():
    """Real follow-up bug, confirmed by review: the empty-input guard fired
    on `before` alone being empty even when `after` was NOT, misreporting a
    genuinely one-sided input as "both empty" instead of the accurate
    before-only/after-only mismatch diagnostic."""
    with pytest.raises(ValueError, match="before-only|after-only"):
        compare_pass_rates([], [_traj("a", True)])


def test_compare_pass_rates_raises_on_empty_input_instead_of_reporting_safe():
    """Real bug, confirmed by review: with both `before` and `after` empty,
    the set-equality check passed vacuously and n=0 made both pass rates
    default to 0.0 with is_regression=False -- reporting a Layer 5
    comparison as SAFE when nothing was actually compared (e.g. an upstream
    ingest failure silently produced zero trajectories)."""
    with pytest.raises(ValueError, match="no trajectories"):
        compare_pass_rates([], [])


def test_compare_pass_rates_no_change_is_not_a_regression():
    before = [_traj("a", True), _traj("b", False)]
    after = [_traj("a", True), _traj("b", False)]
    result = compare_pass_rates(before, after)
    assert result["is_regression"] is False
    assert result["regressions"] == []
    assert result["improvements"] == []
