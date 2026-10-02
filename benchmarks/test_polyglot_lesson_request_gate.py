"""The LESSON: request in the attempt-2 retry prompt is opt-in.

Asking every retry for a LESSON: line changes the benchmark protocol, so a
plain aider_polyglot.py run's pass@2 would stop being comparable with
published results. Only POLYGLOT_REQUEST_LESSONS=1 (set by self-improve's
live_eval) adds the request; without it both agents get dev's retry text.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aider_polyglot as AP  # noqa: E402


def _faker(monkeypatch, tmp_path):
    src = tmp_path / "practice" / "ex"
    src.mkdir(parents=True)
    (src / "ex.py").write_text("stub")
    (src / "ex_test.py").write_text("test")

    def prepare(s, w):
        AP._copy_exercise(s, w)
        return [w / "ex.py"], [w / "ex_test.py"]

    monkeypatch.setitem(AP.LANG_DESCRIPTORS, "faker", {
        "practice_dir": tmp_path / "practice",
        "prepare": prepare,
        "run_tests": lambda work, timeout: (False, "boom"),  # always fail -> retry
        "syntax_hint": "",
        "timeout_s": 5,
    })
    monkeypatch.setattr(AP, "LOG_ROOT", tmp_path / "logs")


def _pi_prompts(monkeypatch, tmp_path):
    prompts = []

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
            prompts.append(message)
            return AP.PromptResult(turn_count=1, agent_ended=True, stop_reason="agent_end",
                                   assistant_text="tried")

    _faker(monkeypatch, tmp_path)
    monkeypatch.setattr(AP, "PiRpc", FakeRpc)
    AP._run_exercise("faker", "ex", "fake/model", agent="pi", verbose=False, retry=True)
    return prompts


def _codex_prompts(monkeypatch, tmp_path):
    prompts = []

    def fake_turn(model, work, prompt, session_id, log_dir, attempt_name):
        prompts.append(prompt)
        return (AP.PromptResult(turn_count=1, agent_ended=True, stop_reason="agent_end",
                                assistant_text="tried"), "fake-session-id")

    _faker(monkeypatch, tmp_path)
    monkeypatch.setattr(AP, "_run_codex_turn", fake_turn)
    AP._run_exercise("faker", "ex", "fake/model", agent="codex", verbose=False, retry=True)
    return prompts


def test_pi_retry_prompt_is_devs_original_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("POLYGLOT_REQUEST_LESSONS", raising=False)
    prompts = _pi_prompts(monkeypatch, tmp_path)
    assert len(prompts) == 2
    assert prompts[1].endswith("boom\n```\n\nFix the implementation and try again.")
    assert "LESSON" not in prompts[1]


def test_codex_retry_prompt_is_devs_original_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("POLYGLOT_REQUEST_LESSONS", raising=False)
    prompts = _codex_prompts(monkeypatch, tmp_path)
    assert len(prompts) == 2
    assert prompts[1] == ("The tests failed. Output:\n\n```\nboom\n```\n\n"
                          "The test file(s) are for reference only -- "
                          "do not edit them. Fix the implementation and try again.")


def test_pi_retry_prompt_requests_a_lesson_when_opted_in(tmp_path, monkeypatch):
    monkeypatch.setenv("POLYGLOT_REQUEST_LESSONS", "1")
    prompts = _pi_prompts(monkeypatch, tmp_path)
    assert "starting with 'LESSON:'" in prompts[1]
    assert prompts[1].endswith("Then fix the implementation and try again.")


def test_codex_retry_prompt_requests_a_lesson_when_opted_in(tmp_path, monkeypatch):
    monkeypatch.setenv("POLYGLOT_REQUEST_LESSONS", "1")
    prompts = _codex_prompts(monkeypatch, tmp_path)
    assert "starting with 'LESSON:'" in prompts[1]
    assert "do not edit them." in prompts[1]


def test_scoring_params_record_the_lesson_request_only_when_on(monkeypatch):
    """Recorded so --resume cannot blend runs made under the two retry
    prompts; absent when off, so params of default runs (and every results
    file written before the gate existed) are unchanged."""
    desc = {"timeout_s": 5}
    monkeypatch.delenv("POLYGLOT_REQUEST_LESSONS", raising=False)
    off = AP._scoring_params("m", "python", True, desc)
    assert "request_lessons" not in off
    monkeypatch.setenv("POLYGLOT_REQUEST_LESSONS", "1")
    on = AP._scoring_params("m", "python", True, desc)
    assert on["request_lessons"] is True
    assert AP._param_mismatches(off, on) == ["request_lessons: None -> True"]
