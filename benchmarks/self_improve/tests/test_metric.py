import pytest

from benchmarks.self_improve.metric import (
    ScoreWithFeedback,
    _score_for_benchmark,
    metric,
    weighted_aggregate,
)
from benchmarks.self_improve.schema import ComponentUsage, NormalizedTrajectory


def test_score_with_feedback_resolves_to_the_real_dspy_class_not_the_fallback():
    """Confirmed real bug (third one found via an actual GEPA run):
    `from dspy.teleprompt.gepa import ScoreWithFeedback` silently
    ImportErrors (the package doesn't re-export it at that top level) and
    falls through to the local dataclass fallback, even with dspy fully
    installed. The fallback isn't dict-subscriptable, but dspy's own
    internal feedback_fn does o["feedback"] on whatever metric() returns --
    crashing GEPA's reflective step the moment it tried to use a real
    component-targeted result. This test pins the module the class actually
    resolves from, so a future import-path regression fails immediately in
    CI rather than only inside a real, paid GEPA run."""
    assert ScoreWithFeedback.__module__ != "benchmarks.self_improve.metric", (
        "ScoreWithFeedback silently fell back to the local dataclass -- "
        "the real dspy.teleprompt.gepa.gepa import is broken again"
    )


def test_score_with_feedback_instances_are_dict_subscriptable():
    """The real dspy class supports o["feedback"]/o["score"] -- dspy's own
    GEPA internals rely on this. The local fallback dataclass does not."""
    inst = ScoreWithFeedback(score=1.0, feedback="hi")
    assert inst["feedback"] == "hi"
    assert inst["score"] == 1.0


class FakeExample:
    def __init__(self, trajectory):
        self.trajectory = trajectory


def test_metric_episode_level_returns_bare_float_not_score_with_feedback():
    """pred_name=None (dspy's plain Evaluate path, used by GEPA for
    valset/trainset Pareto tracking) must get a bare float back -- NOT
    ScoreWithFeedback, confirmed via a real GEPA run: TypeError:
    unsupported operand type(s) for +: 'int' and 'ScoreWithFeedback' inside
    dspy's own sum(vals) progress-reporting code. The real
    dspy.teleprompt.gepa.gepa_utils.ScoreWithFeedback extends
    dspy.Prediction (dict-like), genuinely incompatible with sum()."""
    traj = NormalizedTrajectory(benchmark="aider_polyglot", task_id="t", success=True,
                                 stop_reason="agent_end", turn_count=3, partial_score=1.0)
    result = metric(FakeExample(traj), pred=None, trace=None, pred_name=None, pred_trace=None)
    assert result == 1.0
    assert isinstance(result, float)


def test_metric_works_with_standard_3_arg_dspy_evaluate_call_signature():
    """Real bug, confirmed via an actual GEPA run: dspy's plain Evaluate
    (used for full valset/trainset scoring, not just GEPA's reflective
    credit-assignment step) calls metric(gold, pred, trace) with only 3
    positional args -- pred_name/pred_trace must default, or every such
    call crashes with 'missing 2 required positional arguments', which
    dspy's parallelizer silently swallows into a 0.0 score, producing a
    fully broken optimization run with no real signal."""
    traj = NormalizedTrajectory(benchmark="aider_polyglot", task_id="t", success=True,
                                 stop_reason="agent_end", turn_count=1, partial_score=1.0)
    result = metric(FakeExample(traj), None, None)  # exactly 3 positional args, no keywords
    assert result == 1.0


def test_metric_works_with_2_arg_call_signature():
    """Some dspy code paths call metric(gold, pred) with no trace at all."""
    traj = NormalizedTrajectory(benchmark="gaia", task_id="t", success=False,
                                 stop_reason="deadline", turn_count=1, partial_score=0.0)
    result = metric(FakeExample(traj), None)
    assert result == 0.0


def test_metric_episode_level_returns_base_score_for_fail():
    traj = NormalizedTrajectory(benchmark="gaia", task_id="t", success=False,
                                 stop_reason="deadline", turn_count=3, partial_score=0.0)
    result = metric(FakeExample(traj), pred=None, trace=None, pred_name=None, pred_trace=None)
    assert result == 0.0


def test_metric_pass_2_scores_partial_credit():
    traj = NormalizedTrajectory(benchmark="aider_polyglot", task_id="t", success=True,
                                 stop_reason="agent_end", turn_count=5, partial_score=0.7)
    result = metric(FakeExample(traj), pred=None, trace=None, pred_name=None, pred_trace=None)
    assert result == 0.7


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
