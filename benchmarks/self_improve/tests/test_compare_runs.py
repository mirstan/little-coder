"""compare_runs.py: before/after pass-rate comparison for VALIDATION_PLAN.md
Layer 5's held-out live regression check. Pure comparison logic against
NormalizedTrajectory lists -- no execution of any benchmark happens here."""
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
    import pytest
    with pytest.raises(ValueError):
        compare_pass_rates(before, after)


def test_compare_pass_rates_no_change_is_not_a_regression():
    before = [_traj("a", True), _traj("b", False)]
    after = [_traj("a", True), _traj("b", False)]
    result = compare_pass_rates(before, after)
    assert result["is_regression"] is False
    assert result["regressions"] == []
    assert result["improvements"] == []
