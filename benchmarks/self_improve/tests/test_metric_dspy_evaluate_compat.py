"""metric() must work when called through dspy's REAL Evaluate utility, not
just via hand-built direct calls or a stubbed GEPA. This is exactly the
boundary that broke TWICE during real GEPA runs:

1. metric() required 5 positional args, but dspy.Evaluate calls
   metric(gold, pred, trace) with only 3 -- crashed every call.
2. Even after fixing (1), metric()'s return value (ScoreWithFeedback, a
   dspy.Prediction subclass) broke dspy.Evaluate's own sum(vals) progress
   reporting -- TypeError: unsupported operand type(s) for +: 'int' and
   'ScoreWithFeedback'.

The prior "happy path" test (test_run_gepa_real_path_happy.py) stubbed out
dspy.GEPA entirely, so it never actually exercised dspy's real Evaluate
call path and could not have caught either bug. This test uses dspy.Evaluate
for real (backed by DummyLM, so it's still free/zero-network) specifically
to close that gap.
"""
import dspy
from dspy.utils.dummies import DummyLM

from benchmarks.self_improve.components import build_harness_program
from benchmarks.self_improve.metric import metric
from benchmarks.self_improve.schema import NormalizedTrajectory


def _traj(task_id, success=True):
    return NormalizedTrajectory(
        benchmark="aider_polyglot", task_id=task_id, success=success,
        stop_reason="agent_end", turn_count=1,
        partial_score=1.0 if success else 0.0,
    )


def test_metric_survives_a_real_dspy_evaluate_call():
    """The exact call path that broke twice: dspy.Evaluate(devset=...,
    metric=metric)(program). Must complete without exception and return a
    plausible aggregate score."""
    program = build_harness_program({"agents_md": "Body text."})
    devset = [
        dspy.Example(trajectory=_traj("t1", True), task_context="t1").with_inputs("task_context"),
        dspy.Example(trajectory=_traj("t2", False), task_context="t2").with_inputs("task_context"),
    ]

    with dspy.context(lm=DummyLM([{"guidance": "n/a"}] * 100)):
        evaluator = dspy.Evaluate(devset=devset, metric=metric, display_progress=False)
        result = evaluator(program)

    # EvaluationResult exposes .score as a percentage (0-100), averaging the
    # per-example bare-float returns from metric() -- 1 pass + 1 fail -> 50.
    assert result.score == 50.0


def test_metric_survives_a_real_dspy_evaluate_call_all_passing():
    program = build_harness_program({"agents_md": "Body text."})
    devset = [
        dspy.Example(trajectory=_traj("t1", True), task_context="t1").with_inputs("task_context"),
        dspy.Example(trajectory=_traj("t2", True), task_context="t2").with_inputs("task_context"),
    ]

    with dspy.context(lm=DummyLM([{"guidance": "n/a"}] * 100)):
        evaluator = dspy.Evaluate(devset=devset, metric=metric, display_progress=False)
        result = evaluator(program)

    assert result.score == 100.0
