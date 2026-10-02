"""Two scoring guards that only self-improve's live_eval turns on.

POLYGLOT_CRASH_IS_ERROR=1: a pi crash on any attempt of an exercise that did
not pass is an "error" (retried, never cached), not a "fail" GEPA learns from.
POLYGLOT_RESTORE_TESTS=1: tests are scored from their pristine copy, so an
agent that edits its test file cannot manufacture a pass.
Both change the benchmark's accounting, so a plain run leaves them off.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aider_polyglot as AP  # noqa: E402


# ── crash on a later attempt ─────────────────────────────────────────────
def test_a_later_crash_stays_a_fail_by_default():
    assert AP._classify_status(False, None, ["completed", "process_exit"]) == "fail"


def test_a_later_crash_is_an_error_when_opted_in():
    def status(passed, attempt, outcomes):
        return AP._classify_status(passed, attempt, outcomes, crash_is_error=True)
    assert status(False, None, ["completed", "process_exit"]) == "error"
    assert status(False, None, ["empty_response", "process_exit"]) == "error"
    # a pass still wins, and a crash-free failure is still a fail
    assert status(True, "pass_2", ["completed", "process_exit"]) == "pass_2"
    assert status(False, None, ["completed", "completed"]) == "fail"
    assert status(False, None, ["completed", "deadline"]) == "fail_timeout"


def test_run_exercise_reads_the_crash_gate(monkeypatch, tmp_path):
    """A pi crash on attempt 2 of a failing exercise, end to end."""
    monkeypatch.setenv("POLYGLOT_CRASH_IS_ERROR", "1")
    monkeypatch.delenv("POLYGLOT_RESTORE_TESTS", raising=False)
    _exercise(monkeypatch, tmp_path)
    reasons = iter(["agent_end", "process_exit"])

    class FakeRpc:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def notifications(self):
            return []

        def prompt_and_collect(self, message, timeout=900):
            return AP.PromptResult(turn_count=2, agent_ended=True, stop_reason=next(reasons),
                                   assistant_text="tried", tool_calls=[{"name": "edit"}])

    monkeypatch.setattr(AP, "PiRpc", FakeRpc)
    record = AP._run_exercise("faker", "ex", "fake/model", agent="pi", verbose=False,
                              retry=True, max_attempts=2)
    assert record["stop_reasons"] == ["agent_end", "process_exit"]
    assert record["status"] == "error"


# ── edited tests ─────────────────────────────────────────────────────────
def _exercise(monkeypatch, tmp_path, *, score_in_copy=True, config_tests=None):
    """A python-shaped exercise whose tests pass only if ex_test.py says so,
    and an agent that 'solves' it by rewriting ex_test.py."""
    src = tmp_path / "practice" / "ex"
    (src / ".meta").mkdir(parents=True)
    (src / "ex.py").write_text("stub")
    (src / "ex_test.py").write_text("real tests")
    (src / "test_utils.py").write_text("real helper")
    if config_tests is not None:
        (src / ".meta" / "config.json").write_text(json.dumps({"files": {"test": config_tests}}))

    def prepare(s, w):
        AP._copy_exercise(s, w)
        # like _prepare_python: test_utils.py is not *_test.py, so it is a "stub"
        return [w / "ex.py", w / "test_utils.py"], [w / "ex_test.py"]

    def run_tests(work, timeout):
        ok = (work / "ex_test.py").read_text() == "trivial" or \
             (work / "test_utils.py").read_text() == "trivial"
        return ok, "ok" if ok else "FAILED"

    monkeypatch.setitem(AP.LANG_DESCRIPTORS, "faker", {
        "score_in_copy": score_in_copy,
        "practice_dir": tmp_path / "practice",
        "prepare": prepare,
        "run_tests": run_tests,
        "syntax_hint": "",
        "timeout_s": 5,
    })
    monkeypatch.setattr(AP, "LOG_ROOT", tmp_path / "logs")


def _run(monkeypatch, tmp_path, tamper="ex_test.py"):
    seen = {}

    class FakeRpc:
        def __init__(self, *a, cwd=None, **kw):
            self.cwd = cwd

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def notifications(self):
            return []

        def prompt_and_collect(self, message, timeout=900):
            from pathlib import Path
            (Path(self.cwd) / tamper).write_text("trivial")
            seen["work"] = Path(self.cwd)
            seen["after"] = (Path(self.cwd) / tamper).read_text()
            return AP.PromptResult(turn_count=2, agent_ended=True, stop_reason="agent_end",
                                   assistant_text="edited the test")

    monkeypatch.setattr(AP, "PiRpc", FakeRpc)
    record = AP._run_exercise("faker", "ex", "fake/model", agent="pi", verbose=False,
                              retry=True, max_attempts=2)
    return record, seen


def test_edited_tests_still_pass_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("POLYGLOT_RESTORE_TESTS", raising=False)
    _exercise(monkeypatch, tmp_path)
    record, _ = _run(monkeypatch, tmp_path)
    assert record["status"] == "pass_1"
    assert "tests_tampered" not in record


@pytest.mark.parametrize("score_in_copy", [True, False])
def test_edited_tests_are_scored_from_the_pristine_copy_when_opted_in(
        monkeypatch, tmp_path, score_in_copy):
    monkeypatch.setenv("POLYGLOT_RESTORE_TESTS", "1")
    _exercise(monkeypatch, tmp_path, score_in_copy=score_in_copy)
    record, seen = _run(monkeypatch, tmp_path)
    assert record["status"] == "fail"
    assert record["tests_tampered"] is True
    if score_in_copy:
        # the agent's own tree is left as the agent wrote it
        assert seen["after"] == "trivial"


def test_config_listed_test_fixtures_are_restored_too(monkeypatch, tmp_path):
    """paasio ships test_utils.py as a test file in .meta/config.json, though
    _prepare_python's *_test.py glob hands it to the agent as a stub."""
    monkeypatch.setenv("POLYGLOT_RESTORE_TESTS", "1")
    _exercise(monkeypatch, tmp_path, config_tests=["ex_test.py", "test_utils.py"])
    record, _ = _run(monkeypatch, tmp_path, tamper="test_utils.py")
    assert record["status"] == "fail"
    assert record["tests_tampered"] is True


def test_untouched_tests_are_not_reported_as_tampered(monkeypatch, tmp_path):
    monkeypatch.setenv("POLYGLOT_RESTORE_TESTS", "1")
    _exercise(monkeypatch, tmp_path)
    record, _ = _run(monkeypatch, tmp_path, tamper="ex.py")
    assert record["status"] == "fail"
    assert record["tests_tampered"] is False


def test_scoring_params_record_the_gates_only_when_on(monkeypatch):
    desc = {"timeout_s": 5}
    monkeypatch.delenv("POLYGLOT_CRASH_IS_ERROR", raising=False)
    monkeypatch.delenv("POLYGLOT_RESTORE_TESTS", raising=False)
    off = AP._scoring_params("m", "python", True, desc)
    assert "crash_is_error" not in off and "restore_tests" not in off
    monkeypatch.setenv("POLYGLOT_CRASH_IS_ERROR", "1")
    monkeypatch.setenv("POLYGLOT_RESTORE_TESTS", "1")
    on = AP._scoring_params("m", "python", True, desc)
    assert on["crash_is_error"] is True and on["restore_tests"] is True
