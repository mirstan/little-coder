import time

import pytest

from benchmarks.self_improve.live_budget import (
    LiveBudget,
    LiveEvalBudgetExceeded,
    UserStopStopper,
    estimate_cost,
    median_exercise_seconds_from_results,
    render_estimate,
)


def test_max_live_runs_includes_the_iteration_boundary_overshoot():
    """GEPA's _should_stop is checked only at the TOP of its optimization
    loop (confirmed against engine.py), so a run can complete one more full
    iteration (2*minibatch_size parent+child, plus a valset re-eval) after
    the nominal budget is exhausted."""
    est = estimate_cost(
        max_metric_calls=100, valset_size=4, trainset_size=8, minibatch_size=2,
        component_count=1, assumed_exercise_seconds=180, exercise_timeout_seconds=1800,
    )
    assert est.overshoot_allowance == 2 * 2 + 4
    assert est.max_live_runs == 100 + 2 * 2 + 4


def test_iterations_worst_case_accounts_for_full_valset_reeval_on_acceptance():
    est = estimate_cost(
        max_metric_calls=100, valset_size=4, trainset_size=8, minibatch_size=2,
        component_count=1, assumed_exercise_seconds=180, exercise_timeout_seconds=1800,
    )
    # worst case: every proposal accepted -> denominator includes valset re-eval
    assert est.iterations_worst_case == (100 - 4) // (2 * 2 + 4)
    # best case: nothing accepted -> denominator is just parent+child
    assert est.iterations_best_case == (100 - 4) // (2 * 2)
    assert est.iterations_best_case >= est.iterations_worst_case


def test_estimate_warns_when_round_robin_cannot_reach_every_component():
    est = estimate_cost(
        max_metric_calls=200, valset_size=4, trainset_size=8, minibatch_size=2,
        component_count=33, assumed_exercise_seconds=180, exercise_timeout_seconds=1800,
        module_selector="round_robin",
    )
    assert est.components_covered_worst < 33
    assert any("round_robin" in w for w in est.warnings)


def test_estimate_does_not_warn_about_coverage_under_module_selector_all():
    est = estimate_cost(
        max_metric_calls=200, valset_size=4, trainset_size=8, minibatch_size=2,
        component_count=33, assumed_exercise_seconds=180, exercise_timeout_seconds=1800,
        module_selector="all",
    )
    assert est.components_covered_worst == 33
    assert not any("round_robin" in w for w in est.warnings)


def test_estimate_warns_when_expected_wall_clock_exceeds_the_hard_cap():
    est = estimate_cost(
        max_metric_calls=1000, valset_size=4, trainset_size=8, minibatch_size=2,
        component_count=1, assumed_exercise_seconds=180, exercise_timeout_seconds=1800,
        max_wall_clock_seconds=3600,
    )
    assert any("wall clock" in w for w in est.warnings)


def test_estimate_warns_when_budget_barely_covers_startup():
    est = estimate_cost(
        max_metric_calls=4, valset_size=4, trainset_size=8, minibatch_size=2,
        component_count=1, assumed_exercise_seconds=180, exercise_timeout_seconds=1800,
    )
    assert est.iterations_worst_case == 0
    assert any("zero optimization iterations" in w for w in est.warnings)


def test_estimate_is_monotonic_in_max_metric_calls():
    small = estimate_cost(max_metric_calls=50, valset_size=4, trainset_size=8, minibatch_size=2,
                           component_count=1, assumed_exercise_seconds=180, exercise_timeout_seconds=1800)
    large = estimate_cost(max_metric_calls=500, valset_size=4, trainset_size=8, minibatch_size=2,
                           component_count=1, assumed_exercise_seconds=180, exercise_timeout_seconds=1800)
    assert large.max_live_runs > small.max_live_runs
    assert large.iterations_worst_case >= small.iterations_worst_case


def test_render_estimate_names_every_number_the_confirmation_prompt_promises():
    est = estimate_cost(
        max_metric_calls=100, valset_size=4, trainset_size=8, minibatch_size=2,
        component_count=5, assumed_exercise_seconds=180, exercise_timeout_seconds=1800,
    )
    rendered = render_estimate(est)
    for expected in (str(est.max_metric_calls), str(est.max_live_runs), str(est.valset_size)):
        assert expected in rendered


def test_median_exercise_seconds_reads_a_real_shaped_results_json(tmp_path):
    import json
    results = tmp_path / "results_full_polyglot.json"
    results.write_text(json.dumps({"exercises": {
        "pi/python/a": {"elapsed_s": 100.0},
        "pi/python/b": {"elapsed_s": 200.0},
        "pi/python/c": {"elapsed_s": 300.0},
    }}))
    assert median_exercise_seconds_from_results(results) == 200.0


def test_median_exercise_seconds_returns_none_when_missing(tmp_path):
    assert median_exercise_seconds_from_results(tmp_path / "does-not-exist.json") is None


def test_median_exercise_seconds_returns_none_on_malformed_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not valid json{{{")
    assert median_exercise_seconds_from_results(bad) is None


def test_live_budget_raises_when_max_live_runs_reached():
    budget = LiveBudget(hard_deadline_monotonic=time.monotonic() + 3600, max_live_runs=2)
    budget.check_before_exercise("python/a")
    budget.record_live_run()
    budget.check_before_exercise("python/b")
    budget.record_live_run()
    with pytest.raises(LiveEvalBudgetExceeded, match="max_live_runs"):
        budget.check_before_exercise("python/c")


def test_live_budget_raises_when_wall_clock_deadline_passed():
    budget = LiveBudget(hard_deadline_monotonic=time.monotonic() - 1, max_live_runs=1000)
    with pytest.raises(LiveEvalBudgetExceeded, match="deadline"):
        budget.check_before_exercise("python/a")


def test_user_stop_stopper_starts_false_and_flips_on_request():
    stopper = UserStopStopper()
    assert stopper(gepa_state=None) is False
    stopper.request_stop()
    assert stopper(gepa_state=None) is True
