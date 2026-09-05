"""Scoring parameters, resume compatibility, exit code, turn accounting."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aider_polyglot as AP  # noqa: E402

HARNESS = Path(__file__).parent / "aider_polyglot.py"


def test_params_capture_what_changes_a_score():
    d = AP.LANG_DESCRIPTORS["python"]
    p = AP._scoring_params("m/x", "python", True, d)
    for key in ("model", "language", "attempt_timeout_s", "test_timeout_s",
                "retry", "score_in_copy", "allowed_tools", "env"):
        assert key in p, key


def test_attempt_timeout_comes_from_the_env_not_argparse():
    """ATTEMPT_TIMEOUT_S is a module constant read at import; the guard must
    compare against that, not an args field that does not exist."""
    p = AP._scoring_params("m", "python", True, AP.LANG_DESCRIPTORS["python"])
    assert p["attempt_timeout_s"] == AP.ATTEMPT_TIMEOUT_S


@pytest.mark.parametrize("recorded,current,expected_count", [
    ({"a": 1}, {"a": 1}, 0),
    ({"a": 1}, {"a": 2}, 1),
    ({}, {"a": 1}, 1),                      # nothing recorded == incompatible
    ({"a": 1}, {"a": 1, "b": 2}, 1),        # a new parameter is a difference
])
def test_param_mismatches(recorded, current, expected_count):
    assert len(AP._param_mismatches(recorded, current)) == expected_count


@pytest.mark.parametrize("records,expected", [
    ({}, 0),
    ({"a": {"status": "pass_1"}}, 0),
    ({"a": {"status": "fail"}}, 0),
    ({"a": {"status": "fail_timeout"}}, 0),
    ({"a": {"status": "empty_response"}}, 0),
    ({"a": {"status": "error"}}, 1),
    ({"a": {"status": "pass_1"}, "b": {"status": "error"}}, 1),
])
def test_exit_code_only_for_harness_errors(records, expected):
    assert AP._exit_code(records) == expected


def test_exit_code_ignores_records_this_run_did_not_write():
    """A resumed run must not fail for an inherited error it never touched."""
    inherited = {"old": {"status": "error"}}
    written = {"new": {"status": "pass_1"}}
    assert AP._exit_code(written) == 0
    assert AP._exit_code({**inherited, **written}) == 1  # sanity: it does see errors


def test_missing_exercise_dir_makes_the_process_exit_nonzero(tmp_path):
    """End-to-end: a broken environment must not look like a finished 0% run."""
    env = dict(os.environ, PYTHONPATH=str(HARNESS.parent))
    r = subprocess.run(
        [sys.executable, str(HARNESS), "--language", "python",
         "--exercise", "definitely-not-an-exercise", "--model", "fake/model"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert r.returncode != 0, f"exited 0 despite a harness error\n{r.stdout}\n{r.stderr}"
    assert "error" in (r.stdout + r.stderr).lower()


class _CountingRpc:
    """Fake agent whose attempts report distinct turn counts."""
    turns = [3, 4]

    def __init__(self, *a, **kw):
        self.cwd = Path(kw["cwd"])
        # Session ids are now "poly-<lang>-<ex>-attempt<i>" -- one fresh PiRpc
        # per attempt -- rather than one id shared across the whole retry, so
        # derive which attempt this instance represents from that id instead
        # of a per-instance call counter (each attempt gets its own fresh
        # instance now, not a shared one whose call count tracked the
        # attempt number).
        self.n = int(kw["session_id"].rsplit("attempt", 1)[-1]) - 1

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def close(self):
        pass

    def prompt_and_collect(self, message, timeout=900):
        class R:
            stop_reason = "agent_end"
            agent_ended = True
            turn_count = _CountingRpc.turns[self.n]
            compaction_events = 0
            assistant_text = "work"
            tool_calls = [{"name": "read"}]
        return R()


def test_turn_count_sums_both_attempts_on_failure(tmp_path, monkeypatch):
    """The old expression dropped r2.turn_count whenever the exercise failed,
    so a two-attempt failure under-reported its own effort."""
    src = tmp_path / "practice" / "ex"
    src.mkdir(parents=True)
    (src / "ex.py").write_text("stub")

    monkeypatch.setitem(AP.LANG_DESCRIPTORS, "faker", {
        "practice_dir": tmp_path / "practice",
        "prepare": lambda s, w: (AP._copy_exercise(s, w), ([w / "ex.py"], []))[1],
        "run_tests": lambda where, timeout: (False, "nope"),   # always fail -> retry
        "syntax_hint": "",
        "timeout_s": 5,
    })
    monkeypatch.setattr(AP, "PiRpc", _CountingRpc)
    monkeypatch.setattr(AP, "LOG_ROOT", tmp_path / "logs")

    rec = AP._run_exercise("faker", "ex", "fake/model", agent="pi", verbose=False, retry=True)
    assert rec["status"] == "fail"
    assert rec["turn_count"] == 7, f"expected 3+4, got {rec['turn_count']}"


class _EmptyThenRealRpc:
    """Fake agent: attempt 1 returns a provider-side empty completion (no
    tool calls, no assistant text, work tree untouched); attempt 2 does
    real work and fixes the exercise."""

    def __init__(self, *a, **kw):
        self.cwd = Path(kw["cwd"])
        self.n = int(kw["session_id"].rsplit("attempt", 1)[-1])

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def prompt_and_collect(self, message, timeout=900):
        n = self.n

        class R:
            stop_reason = "agent_end"
            agent_ended = True
            turn_count = 0 if n == 1 else 2
            compaction_events = 0
            assistant_text = "" if n == 1 else "did work"
            tool_calls = [] if n == 1 else [{"name": "write"}]
        if n != 1:
            (self.cwd / "ex.py").write_text("fixed")
        return R()


def test_empty_response_does_not_abort_remaining_attempts(tmp_path, monkeypatch):
    """empty_response must fall through to a retry rather than stopping the
    loop, unlike deadline/process_exit. Per _is_empty_response, it means the
    provider returned an empty completion with the work tree untouched -- a
    transient fault, not an attempt with nothing left to build on. Before
    this fix it was lumped in with deadline/process_exit and aborted the
    rest of --max-attempts on the single most retryable outcome."""
    src = tmp_path / "practice" / "ex"
    src.mkdir(parents=True)
    (src / "ex.py").write_text("stub")

    calls = {"n": 0}

    def run_tests(where, timeout):
        calls["n"] += 1
        passed = (Path(where) / "ex.py").read_text() == "fixed"
        return passed, ("ok" if passed else "still stub")

    monkeypatch.setitem(AP.LANG_DESCRIPTORS, "faker", {
        "practice_dir": tmp_path / "practice",
        "prepare": lambda s, w: (AP._copy_exercise(s, w), ([w / "ex.py"], []))[1],
        "run_tests": run_tests,
        "syntax_hint": "",
        "timeout_s": 5,
    })
    monkeypatch.setattr(AP, "PiRpc", _EmptyThenRealRpc)
    monkeypatch.setattr(AP, "LOG_ROOT", tmp_path / "logs")

    rec = AP._run_exercise("faker", "ex", "fake/model", agent="pi", verbose=False, retry=True, max_attempts=2)
    assert calls["n"] == 2, "attempt 2 never ran -- empty_response aborted the loop"
    assert rec["status"] == "pass_2"
    assert rec["stop_reasons"] == ["agent_end", "agent_end"]


def test_config_label_recorded_in_meta_not_scoring_params(tmp_path, monkeypatch):
    """--config-label is result-file bookkeeping for distinguishing runs of the
    same --agent/--model under different tuning configs (e.g. server-side
    sampling params, invisible to this script) -- compare_agents.py groups on
    meta.config_label. It must land there, but stay out of _scoring_params:
    unlike --thinking/--max-attempts it doesn't change how an attempt runs,
    so a --resume under a different --config-label must not be flagged as a
    scoring-parameter mismatch."""
    monkeypatch.setattr(AP, "RESULTS_FILE", tmp_path / "results.json")
    monkeypatch.setattr(sys, "argv", [
        "aider_polyglot.py", "--language", "python",
        "--exercise", "definitely-not-an-exercise",
        "--model", "fake/model", "--config-label", "tuned",
    ])
    AP.main()

    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["meta"]["config_label"] == "tuned"
    assert "config_label" not in saved["meta"]["scoring_params"]


def test_config_label_persists_across_resume_without_the_flag(tmp_path, monkeypatch):
    """A --resume run that omits --config-label must not blank out the label
    recorded by the run that originally set it."""
    monkeypatch.setattr(AP, "RESULTS_FILE", tmp_path / "results.json")
    monkeypatch.setattr(sys, "argv", [
        "aider_polyglot.py", "--language", "python",
        "--exercise", "definitely-not-an-exercise",
        "--model", "fake/model", "--config-label", "tuned",
    ])
    AP.main()

    monkeypatch.setattr(sys, "argv", [
        "aider_polyglot.py", "--language", "python",
        "--exercise", "definitely-not-an-exercise",
        "--model", "fake/model", "--resume",
    ])
    AP.main()

    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["meta"]["config_label"] == "tuned"
