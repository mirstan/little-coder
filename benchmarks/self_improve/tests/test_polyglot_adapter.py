"""PolyglotGEPAAdapter: pure unit tests against a stubbed FakeRunner (no
subprocess, no real git worktree) -- the real end-to-end pipeline is
covered by test_live_eval_e2e_fake_pi.py. Here we test only the adapter's
OWN logic: EvaluationBatch construction and reflective-dataset content."""
import json

import pytest

from benchmarks.self_improve.exercises import ExerciseSpec
from benchmarks.self_improve.live_eval import LiveRunResult
from benchmarks.self_improve.polyglot_adapter import PolyglotGEPAAdapter


class FakeRunner:
    """Records every (candidate, specs) call it receives, and returns
    canned results keyed by exercise task_id."""

    def __init__(self, results_by_exercise: dict[str, LiveRunResult]):
        self.results_by_exercise = results_by_exercise
        self.calls: list[tuple[dict, list[ExerciseSpec]]] = []

    def run_batch(self, candidate, specs):
        self.calls.append((dict(candidate), list(specs)))
        return [self.results_by_exercise[spec.task_id] for spec in specs]


def _result(task_id, exercise, *, status="pass_1", score=1.0, success=True, **kw) -> LiveRunResult:
    return LiveRunResult(task_id=task_id, exercise=exercise, language="python",
                          status=status, score=score, success=success, **kw)


def _adapter(runner, component_paths=None, practice_dir_path="."):
    return PolyglotGEPAAdapter(
        runner, component_paths=component_paths or {"skills_tools_bash": "skills/tools/bash.md"},
        practice_dir_path=practice_dir_path,
    )


def test_evaluate_returns_scores_aligned_with_batch_order():
    specs = [ExerciseSpec("a"), ExerciseSpec("b"), ExerciseSpec("c")]
    runner = FakeRunner({
        "python/a": _result("python/a", "a", score=1.0),
        "python/b": _result("python/b", "b", score=0.0, success=False, status="fail"),
        "python/c": _result("python/c", "c", score=0.7, status="pass_2"),
    })
    batch = _adapter(runner).evaluate(specs, {"skills_tools_bash": "text"})
    assert len(batch.outputs) == len(batch.scores) == len(specs)
    assert batch.scores == [1.0, 0.0, 0.7]
    assert [o.task_id for o in batch.outputs] == ["python/a", "python/b", "python/c"]


def test_evaluate_trajectories_none_when_capture_traces_false():
    specs = [ExerciseSpec("a")]
    runner = FakeRunner({"python/a": _result("python/a", "a")})
    batch = _adapter(runner).evaluate(specs, {}, capture_traces=False)
    assert batch.trajectories is None


def test_evaluate_trajectories_populated_when_capture_traces_true():
    specs = [ExerciseSpec("a")]
    runner = FakeRunner({"python/a": _result("python/a", "a")})
    batch = _adapter(runner).evaluate(specs, {}, capture_traces=True)
    assert len(batch.trajectories) == 1
    assert batch.trajectories[0].spec == specs[0]


def test_num_metric_calls_counts_only_non_cached_results():
    specs = [ExerciseSpec("a"), ExerciseSpec("b")]
    runner = FakeRunner({
        "python/a": _result("python/a", "a", from_cache=True),
        "python/b": _result("python/b", "b", from_cache=False),
    })
    batch = _adapter(runner).evaluate(specs, {})
    assert batch.num_metric_calls == 1


def test_num_metric_calls_zero_when_all_cached():
    specs = [ExerciseSpec("a")]
    runner = FakeRunner({"python/a": _result("python/a", "a", from_cache=True)})
    batch = _adapter(runner).evaluate(specs, {})
    assert batch.num_metric_calls == 0


def test_evaluate_passes_the_full_candidate_dict_to_the_runner():
    """The regression test this whole rewrite exists for: the runner must
    actually RECEIVE the candidate text -- under the old frozen-data design
    the equivalent metric() call never read `pred` at all."""
    specs = [ExerciseSpec("a")]
    candidate = {"agents_md": "distinctive text", "skills_tools_bash": "other text"}
    runner = FakeRunner({"python/a": _result("python/a", "a")})
    _adapter(runner).evaluate(specs, candidate)
    assert runner.calls[0][0] == candidate


def test_two_different_candidates_yield_different_scores_via_the_runner():
    """FakeRunner here stands in for "a real run whose outcome depends on
    candidate text" -- the adapter itself must not collapse that
    distinction; it just passes scores through."""
    specs = [ExerciseSpec("a")]
    runner_good = FakeRunner({"python/a": _result("python/a", "a", score=1.0)})
    runner_bad = FakeRunner({"python/a": _result("python/a", "a", score=0.0, success=False, status="fail")})
    good = _adapter(runner_good).evaluate(specs, {"agents_md": "good candidate"})
    bad = _adapter(runner_bad).evaluate(specs, {"agents_md": "bad candidate"})
    assert good.scores != bad.scores


def test_evaluate_does_not_mutate_the_candidate_dict():
    specs = [ExerciseSpec("a")]
    candidate = {"agents_md": "text"}
    original = dict(candidate)
    runner = FakeRunner({"python/a": _result("python/a", "a")})
    _adapter(runner).evaluate(specs, candidate)
    assert candidate == original


def test_reflective_dataset_has_one_key_per_requested_component():
    specs = [ExerciseSpec("a")]
    runner = FakeRunner({"python/a": _result("python/a", "a")})
    adapter = _adapter(runner)
    batch = adapter.evaluate(specs, {"skills_tools_bash": "text"}, capture_traces=True)
    dataset = adapter.make_reflective_dataset({"skills_tools_bash": "text"}, batch, ["skills_tools_bash"])
    assert set(dataset.keys()) == {"skills_tools_bash"}
    assert len(dataset["skills_tools_bash"]) == 1


def test_reflective_dataset_record_has_required_fields():
    specs = [ExerciseSpec("a")]
    runner = FakeRunner({"python/a": _result("python/a", "a", status="fail", score=0.0, success=False,
                                              test_output_tail="AssertionError: boom")})
    adapter = _adapter(runner)
    batch = adapter.evaluate(specs, {"skills_tools_bash": "text"}, capture_traces=True)
    record = adapter.make_reflective_dataset({"skills_tools_bash": "text"}, batch, ["skills_tools_bash"])["skills_tools_bash"][0]
    assert "Inputs" in record and "Generated Outputs" in record and "Feedback" in record
    assert record["Inputs"]["exercise"] == "python/a"
    assert "AssertionError: boom" in record["Feedback"]


def test_reflective_dataset_says_not_injected_when_component_absent_from_notifications():
    specs = [ExerciseSpec("a")]
    runner = FakeRunner({"python/a": _result("python/a", "a", notifications=[])})
    adapter = _adapter(runner)
    batch = adapter.evaluate(specs, {"skills_tools_bash": "text"}, capture_traces=True)
    feedback = adapter.make_reflective_dataset({"skills_tools_bash": "text"}, batch, ["skills_tools_bash"])["skills_tools_bash"][0]["Feedback"]
    assert "NOT injected" in feedback


def test_reflective_dataset_says_injected_when_component_present_in_notifications():
    specs = [ExerciseSpec("a")]
    runner = FakeRunner({"python/a": _result(
        "python/a", "a", notifications=["[info] skill-inject: +1 [bash]"],
    )})
    adapter = _adapter(runner)
    batch = adapter.evaluate(specs, {"skills_tools_bash": "text"}, capture_traces=True)
    feedback = adapter.make_reflective_dataset({"skills_tools_bash": "text"}, batch, ["skills_tools_bash"])["skills_tools_bash"][0]["Feedback"]
    assert "was injected 1 time" in feedback


def test_reflective_dataset_treats_agents_md_as_always_active():
    specs = [ExerciseSpec("a")]
    runner = FakeRunner({"python/a": _result("python/a", "a", notifications=[])})
    adapter = _adapter(runner, component_paths={"agents_md": "AGENTS.md"})
    batch = adapter.evaluate(specs, {"agents_md": "text"}, capture_traces=True)
    feedback = adapter.make_reflective_dataset({"agents_md": "text"}, batch, ["agents_md"])["agents_md"][0]["Feedback"]
    assert "always active" in feedback
    assert "NOT injected" not in feedback


def test_reflective_dataset_harness_error_disclaims_component_causality():
    specs = [ExerciseSpec("a")]
    runner = FakeRunner({"python/a": _result(
        "python/a", "a", status="harness_error", score=0.0, success=False, error="results file missing",
    )})
    adapter = _adapter(runner)
    batch = adapter.evaluate(specs, {"skills_tools_bash": "text"}, capture_traces=True)
    feedback = adapter.make_reflective_dataset({"skills_tools_bash": "text"}, batch, ["skills_tools_bash"])["skills_tools_bash"][0]["Feedback"]
    assert "NOTHING about the quality" in feedback
    assert "results file missing" in feedback


def test_reflective_dataset_is_fully_json_serializable():
    specs = [ExerciseSpec("a")]
    runner = FakeRunner({"python/a": _result(
        "python/a", "a", status="fail", score=0.0, success=False,
        test_output_tail="output", diff_summary="diff", notifications=["[info] x"],
    )})
    adapter = _adapter(runner)
    batch = adapter.evaluate(specs, {"skills_tools_bash": "text"}, capture_traces=True)
    dataset = adapter.make_reflective_dataset({"skills_tools_bash": "text"}, batch, ["skills_tools_bash"])
    json.dumps(dataset)  # must not raise


def test_reflective_dataset_caps_do_not_blow_up_on_a_huge_pytest_tail():
    specs = [ExerciseSpec("a")]
    huge_output = "x" * 1_000_000
    runner = FakeRunner({"python/a": _result(
        "python/a", "a", status="fail", score=0.0, success=False, test_output_tail=huge_output,
    )})
    adapter = _adapter(runner)
    batch = adapter.evaluate(specs, {"skills_tools_bash": "text"}, capture_traces=True)
    record = adapter.make_reflective_dataset({"skills_tools_bash": "text"}, batch, ["skills_tools_bash"])["skills_tools_bash"][0]
    # live_eval.py itself caps test_output_tail at _MAX_TAIL_CHARS before it
    # ever reaches here -- this just confirms the adapter doesn't ALSO
    # explode the record with duplicated content.
    assert len(record["Feedback"]) < 2_000_000
