"""HarnessProgram construction: one dspy.Predict per component, docstring =
the component's body text (per architecture §3). No LLM/API calls happen
here -- this only exercises dspy.Signature/Module construction, which is
free and deterministic."""
import dspy
from dspy.utils.dummies import DummyLM

from benchmarks.self_improve.components import build_harness_program


def test_build_harness_program_creates_one_predictor_per_component():
    components = {
        "agents_md": "You are little-coder.",
        "skills_tools_bash": "## bash tool\nExecute a shell command.",
    }
    program = build_harness_program(components)
    assert set(program.predictors.keys()) == {"agents_md", "skills_tools_bash"}
    for pred_name, predictor in program.predictors.items():
        assert isinstance(predictor, dspy.Predict)


def test_build_harness_program_signature_docstring_is_component_body():
    components = {"agents_md": "You are little-coder.\n\nBe concise."}
    program = build_harness_program(components)
    sig = program.predictors["agents_md"].signature
    assert sig.instructions == "You are little-coder.\n\nBe concise."


def test_build_harness_program_predictors_are_independently_addressable():
    """Two components with different bodies must produce two distinct
    signatures -- GEPA targets each by pred_name, so they must not share
    instructions by accident (e.g. from a closure-capture bug)."""
    components = {
        "a": "Body A",
        "b": "Body B",
    }
    program = build_harness_program(components)
    assert program.predictors["a"].signature.instructions == "Body A"
    assert program.predictors["b"].signature.instructions == "Body B"


def test_build_harness_program_empty_components_is_valid():
    program = build_harness_program({})
    assert program.predictors == {}


def test_forward_calls_every_predictor_and_produces_a_full_trace():
    """GEPA requires a real (predictor, inputs, prediction) trace entry per
    predictor it might optimize -- an empty trace for any predictor crashes
    GEPA's reflective-dataset construction (confirmed against installed
    dspy-ai 3.3.1 source). forward() must call every predictor, not a
    caller-selected subset, so this never happens regardless of which
    predictor GEPA is currently reflecting on."""
    components = {"a": "Body A", "b": "Body B", "c": "Body C"}
    program = build_harness_program(components)

    with dspy.context(lm=DummyLM([{"guidance": "canned"}] * 10)):
        with dspy.settings.context(trace=[]):
            result = program(task_context="some task")
            trace = dspy.settings.trace

    instructions_seen = {predictor.signature.instructions for predictor, _inputs, _pred in trace}
    assert instructions_seen == {"Body A", "Body B", "Body C"}
    assert result.a == "canned"
    assert result.b == "canned"
    assert result.c == "canned"


def test_forward_is_free_no_real_lm_configured():
    """Confirms forward() never requires (or accidentally reaches for) a
    real, paid LM client -- DummyLM alone is sufficient for a full,
    successful forward pass with no network access and no API key."""
    program = build_harness_program({"only": "Body"})
    with dspy.context(lm=DummyLM([{"guidance": "ok"}])):
        result = program(task_context="x")
    assert result.only == "ok"
