import pytest

from benchmarks.self_improve.metric import (
    _score_for_benchmark,
    metric,
    weighted_aggregate,
)
from benchmarks.self_improve.schema import ComponentUsage, NormalizedTrajectory


class FakeExample:
    def __init__(self, trajectory):
        self.trajectory = trajectory


def test_metric_episode_level_returns_base_score_for_pass():
    traj = NormalizedTrajectory(benchmark="aider_polyglot", task_id="t", success=True,
                                 stop_reason="agent_end", turn_count=3, partial_score=1.0)
    result = metric(FakeExample(traj), pred=None, trace=None, pred_name=None, pred_trace=None)
    assert result.score == 1.0


def test_metric_episode_level_returns_base_score_for_fail():
    traj = NormalizedTrajectory(benchmark="gaia", task_id="t", success=False,
                                 stop_reason="deadline", turn_count=3, partial_score=0.0)
    result = metric(FakeExample(traj), pred=None, trace=None, pred_name=None, pred_trace=None)
    assert result.score == 0.0


def test_metric_pass_2_scores_partial_credit():
    traj = NormalizedTrajectory(benchmark="aider_polyglot", task_id="t", success=True,
                                 stop_reason="agent_end", turn_count=5, partial_score=0.7)
    result = metric(FakeExample(traj), pred=None, trace=None, pred_name=None, pred_trace=None)
    assert result.score == 0.7


def test_metric_component_not_used_returns_score_no_feedback():
    """When pred_name is a specific component but it wasn't active in this
    trajectory, there's no attributable signal -- score still reflects the
    episode outcome, but feedback must be None (not fabricated text)."""
    traj = NormalizedTrajectory(benchmark="gaia", task_id="t", success=False,
                                 stop_reason="deadline", turn_count=2, partial_score=0.0,
                                 components_used=[])
    result = metric(FakeExample(traj), pred=None, trace=None,
                     pred_name="skills_tools_bash", pred_trace=None)
    assert result.score == 0.0
    assert result.feedback is None


def test_metric_component_used_in_failure_returns_grounded_feedback():
    traj = NormalizedTrajectory(
        benchmark="gaia", task_id="task-42", success=False, stop_reason="deadline",
        turn_count=10, partial_score=0.0, failure_signals=["repeated_tool_call"],
        components_used=[ComponentUsage(pred_name="skills_tools_bash", invocation_count=3,
                                         was_error_context=True)],
    )
    result = metric(FakeExample(traj), pred=None, trace=None,
                     pred_name="skills_tools_bash", pred_trace=None)
    assert result.score == 0.0
    assert result.feedback is not None
    assert "task-42" in result.feedback
    assert "deadline" in result.feedback
    assert "repeated_tool_call" in result.feedback
    assert "3" in result.feedback  # invocation count cited


def test_metric_component_used_in_success_returns_positive_feedback():
    """A component active in a PASSING trajectory should also get feedback
    -- GEPA needs positive signal too, not only failure critique, to avoid
    only ever pushing components toward defensiveness."""
    traj = NormalizedTrajectory(
        benchmark="aider_polyglot", task_id="ex-1", success=True, stop_reason="agent_end",
        turn_count=4, partial_score=1.0,
        components_used=[ComponentUsage(pred_name="skills_tools_bash", invocation_count=1)],
    )
    result = metric(FakeExample(traj), pred=None, trace=None,
                     pred_name="skills_tools_bash", pred_trace=None)
    assert result.score == 1.0
    assert result.feedback is not None


def test_score_for_benchmark_dispatches_by_benchmark_literal():
    """Unit-test the internal dispatcher directly -- each benchmark's
    partial_score->score mapping must be independently verifiable, not only
    exercised transitively through metric()."""
    assert _score_for_benchmark(NormalizedTrajectory(
        benchmark="gaia", task_id="t", success=True, stop_reason="agent_end",
        turn_count=1)) == 1.0


def test_aggregate_score_applies_benchmark_weights():
    """benchmark_weights.yaml lets aggregate valset scoring combine the three
    benchmarks without the largest sample count dominating -- test the
    weighting function directly against a hand-built per-benchmark score dict."""
    weights = {"aider_polyglot": 0.5, "gaia": 0.3, "harbor": 0.1, "tb": 0.1}
    per_benchmark_avg = {"aider_polyglot": 0.8, "gaia": 0.4, "harbor": 1.0, "tb": 0.0}
    agg = weighted_aggregate(per_benchmark_avg, weights)
    assert agg == pytest.approx(0.8 * 0.5 + 0.4 * 0.3 + 1.0 * 0.1 + 0.0 * 0.1)


def test_weighted_aggregate_missing_benchmark_treated_as_zero_weight():
    """If a benchmark has no examples in a given val split, it must not
    contribute NaN or crash the aggregate -- treat as absent, renormalize
    remaining weights."""
    agg = weighted_aggregate({"gaia": 1.0}, {"aider_polyglot": 0.5, "gaia": 0.5})
    assert agg == 1.0  # renormalized: gaia is the only present benchmark
