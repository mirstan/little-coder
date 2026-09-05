"""Happy-path test for run_gepa.py's _real_run(): with all three safety
gates satisfied (a FAKE api key, never a real one -- this test makes no
network call), verify the code actually reaches and correctly calls
GEPA.compile() without crashing, and writes optimized_components.yaml
correctly from its result.

Why this matters: every other test of _real_run() only exercises the
REFUSAL paths (missing model/key/confirm-flag). This is the only test that
ever executes the trainset/valset construction, HarnessProgram building,
DummyLM configuration, and dspy.LM/dspy.GEPA construction lines -- a bug
anywhere in that path would otherwise only surface after a human spends
real reflection_lm budget finding it. dspy.GEPA itself is monkeypatched to
a stub whose .compile() returns the input program unchanged, so this test
makes zero network calls and costs nothing.
"""
import argparse

import dspy
import yaml

import benchmarks.self_improve.run_gepa as run_gepa_module
from benchmarks.self_improve.run_gepa import _real_run
from benchmarks.self_improve.schema import NormalizedTrajectory


class _StubGEPA:
    """Stands in for dspy.GEPA: .compile() returns the input program
    unchanged (no network call, no reflection). Records the args it was
    called with so the test can assert on them."""
    last_instance = None

    def __init__(self, metric, reflection_lm, auto):
        self.metric = metric
        self.reflection_lm = reflection_lm
        self.auto = auto
        self.compile_calls = []
        _StubGEPA.last_instance = self

    def compile(self, student, *, trainset, valset):
        self.compile_calls.append({"student": student, "trainset": trainset, "valset": valset})
        return student


def _traj(task_id, success=True):
    return NormalizedTrajectory(
        benchmark="aider_polyglot", task_id=task_id, success=success,
        stop_reason="agent_end", turn_count=1,
        partial_score=1.0 if success else 0.0,
    )


def _args(tmp_path, **overrides):
    defaults = dict(
        reflection_model="fake-provider/fake-model",
        confirm_real_run=True,
        train_frac=0.7, seed=42, out_dir=str(tmp_path / "out"),
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_real_run_happy_path_reaches_compile_and_writes_output(tmp_path, monkeypatch):
    monkeypatch.setenv("REFLECTION_LM_API_KEY", "sk-fake-not-real-never-used")
    monkeypatch.setattr(dspy, "GEPA", _StubGEPA)

    trajectories = [_traj("t1"), _traj("t2", success=False), _traj("t3")]
    components = {"agents_md": "Body text.", "skills_tools_bash": "Bash guidance."}

    code = _real_run(trajectories, components, _args(tmp_path))

    assert code == 0
    stub = _StubGEPA.last_instance
    assert stub is not None
    assert len(stub.compile_calls) == 1
    call = stub.compile_calls[0]
    # t2 is the ONLY (aider_polyglot, success=False) trajectory -- a
    # singleton stratification group can't be split into two disjoint
    # non-empty sets, so split_train_val() deliberately duplicates it into
    # both train and val (see its own "tiny groups" comment) rather than
    # leaving val without any failure example for that group. t1/t3 (the
    # size-2 success=True group) split cleanly with none dropped.
    train_ids = {ex.trajectory.task_id for ex in call["trainset"]}
    val_ids = {ex.trajectory.task_id for ex in call["valset"]}
    assert train_ids | val_ids == {"t1", "t2", "t3"}
    assert "t2" in train_ids and "t2" in val_ids  # the documented singleton-duplication case

    out_file = tmp_path / "out" / "optimized_components.yaml"
    assert out_file.exists()
    written = yaml.safe_load(out_file.read_text())
    assert set(written.keys()) == {"agents_md", "skills_tools_bash"}
    # stub compile() returned the program unchanged, so instructions match
    # exactly what build_harness_program() set from the original bodies
    assert written["agents_md"] == "Body text."
    assert written["skills_tools_bash"] == "Bash guidance."


def test_real_run_happy_path_configures_dummy_lm_not_a_real_one(tmp_path, monkeypatch):
    """Confirms the forward-pass LM is the free DummyLM, not something that
    would make a real network call -- the actual safety property this whole
    design depends on.

    Reads what _real_run() actually PASSED to dspy.settings.configure(),
    rather than reading dspy.settings.lm back afterward: the latter is
    global, un-reset state shared across the whole test session, so it could
    stay green from a previous test's configure() call even if this run's
    own configure() call were removed or broken -- a real gap, confirmed by
    review.

    Patches Settings.configure on the CLASS, not the dspy.settings instance:
    Settings overrides __setattr__ to route any instance-attribute set
    through self.configure(**{name: value}) itself, so
    monkeypatch.setattr(dspy.settings, "configure", ...) silently never
    takes effect -- normal attribute lookup still finds the real class
    method first. Patching the class is the only way to actually intercept
    the call."""
    monkeypatch.setenv("REFLECTION_LM_API_KEY", "sk-fake-not-real-never-used")
    monkeypatch.setattr(dspy, "GEPA", _StubGEPA)

    settings_cls = type(dspy.settings)
    configure_calls = []
    original_configure = settings_cls.configure

    def recording_configure(self, **kwargs):
        configure_calls.append(kwargs)
        return original_configure(self, **kwargs)

    monkeypatch.setattr(settings_cls, "configure", recording_configure)

    trajectories = [_traj("t1")]
    components = {"agents_md": "Body."}

    _real_run(trajectories, components, _args(tmp_path))

    from dspy.utils.dummies import DummyLM
    assert len(configure_calls) == 1
    assert isinstance(configure_calls[0]["lm"], DummyLM)


def test_real_run_happy_path_passes_reflection_model_through(tmp_path, monkeypatch):
    monkeypatch.setenv("REFLECTION_LM_API_KEY", "sk-fake-not-real-never-used")
    monkeypatch.setattr(dspy, "GEPA", _StubGEPA)

    _real_run([_traj("t1")], {"agents_md": "Body."}, _args(tmp_path, reflection_model="anthropic/claude-opus-4-6"))

    stub = _StubGEPA.last_instance
    assert stub.reflection_lm.model == "anthropic/claude-opus-4-6"


def test_real_run_happy_path_passes_reasoning_effort_through_when_set(tmp_path, monkeypatch):
    monkeypatch.setenv("REFLECTION_LM_API_KEY", "sk-fake-not-real-never-used")
    monkeypatch.setattr(dspy, "GEPA", _StubGEPA)

    _real_run(
        [_traj("t1")], {"agents_md": "Body."},
        _args(tmp_path, reflection_model="openai/gpt-5.6-luna", reflection_reasoning_effort="xhigh"),
    )

    stub = _StubGEPA.last_instance
    assert stub.reflection_lm.kwargs.get("reasoning_effort") == "xhigh"


def test_real_run_happy_path_omits_reasoning_effort_when_not_set(tmp_path, monkeypatch):
    """Models that don't support reasoning_effort must not get the kwarg at
    all -- absence, not a null/empty value, since dspy.LM forwards kwargs
    verbatim to the underlying provider call."""
    monkeypatch.setenv("REFLECTION_LM_API_KEY", "sk-fake-not-real-never-used")
    monkeypatch.setattr(dspy, "GEPA", _StubGEPA)

    _real_run([_traj("t1")], {"agents_md": "Body."}, _args(tmp_path, reflection_model="anthropic/claude-opus-4-6"))

    stub = _StubGEPA.last_instance
    assert "reasoning_effort" not in stub.reflection_lm.kwargs
