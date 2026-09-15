"""_run_codex_turn's subprocess construction, driven by a fake Popen.

The real behavior is verified end-to-end against the actual codex binary
during development (see the PR description), but that requires a live
ChatGPT-subscription auth and real API calls, so it can't run in CI and
wasn't run before every commit -- which is exactly how a broken `resume`
invocation (codex exec resume rejecting --approve-for-me and --cd outright)
shipped once already. These tests are hermetic and deterministic: they
never invoke a real subprocess, and they exist specifically to pin the argv
shape so a regression there fails loudly and immediately.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aider_polyglot as AP  # noqa: E402


class _FakeCompletedPopen:
    """Stands in for subprocess.Popen(codex exec ...), capturing the argv
    and cwd it was called with, and returning a scripted JSONL stdout."""

    #: Set by each test before constructing _run_codex_turn's call, so the
    #: class (not just an instance) can record what it was invoked with --
    #: _run_codex_turn only ever gets our patched subprocess.Popen callable.
    calls: list[dict] = []
    stdout_lines: list[str] = []
    returncode = 0

    def __init__(self, args, stdout=None, stderr=None, text=None, stdin=None,
                 cwd=None, start_new_session=None):
        _FakeCompletedPopen.calls.append({"args": args, "cwd": cwd})
        self.pid = 999
        self.returncode = _FakeCompletedPopen.returncode
        self._stdout = "\n".join(_FakeCompletedPopen.stdout_lines) + "\n"

    def communicate(self, timeout=None):
        return (self._stdout, "")


def _clean_turn(thread_id="01a0000-thread"):
    return [
        json.dumps({"type": "thread.started", "thread_id": thread_id}),
        json.dumps({"type": "item.completed",
                    "item": {"id": "item_0", "type": "file_change",
                              "changes": [{"path": "x", "kind": "add"}]}}),
        json.dumps({"type": "turn.completed", "usage": {}}),
    ]


def test_fresh_attempt_uses_exec_with_approve_and_cd(tmp_path, monkeypatch):
    """Attempt 1 (session_id=None) must use `codex exec` with
    --approve-for-me and --cd -- the flags codex actually needs to run
    unattended in a fresh sandboxed workspace."""
    monkeypatch.setattr(AP.subprocess, "Popen", _FakeCompletedPopen)
    _FakeCompletedPopen.calls = []
    _FakeCompletedPopen.stdout_lines = _clean_turn()
    _FakeCompletedPopen.returncode = 0

    work = tmp_path / "work"
    work.mkdir()
    r, session_id = AP._run_codex_turn("gpt-5.6-luna", work, "do it", None, tmp_path, "1")

    assert len(_FakeCompletedPopen.calls) == 1
    args = _FakeCompletedPopen.calls[0]["args"]
    assert args[:2] == ["codex", "exec"]
    assert "resume" not in args
    assert "--approve-for-me" in args
    assert "--cd" in args and str(work) in args
    # cwd= is passed explicitly to Popen too, not relied on via --cd alone --
    # `resume` has no --cd equivalent, so this must hold on both paths.
    assert _FakeCompletedPopen.calls[0]["cwd"] == str(work)
    assert session_id == "01a0000-thread"
    assert r.stop_reason == "agent_end"
    assert r.agent_ended is True


def test_retry_uses_resume_with_session_id_not_approve_or_cd(tmp_path, monkeypatch):
    """Regression guard for the shipped bug: `codex exec resume` is a
    DIFFERENT subcommand with a smaller flag set than `codex exec` --
    --approve-for-me and --cd are both rejected outright by the real binary
    (confirmed: 'unexpected argument --approve-for-me found'). A retry must
    never include them, must include `resume` + the captured session id
    (not `--last`, which is a global newest-session selector with no
    guarantee of picking THIS exercise's session), and must still pass
    cwd= explicitly to Popen since resume has no --cd of its own."""
    monkeypatch.setattr(AP.subprocess, "Popen", _FakeCompletedPopen)
    _FakeCompletedPopen.calls = []
    _FakeCompletedPopen.stdout_lines = _clean_turn(thread_id="01a0000-thread")
    _FakeCompletedPopen.returncode = 0

    work = tmp_path / "work"
    work.mkdir()
    AP._run_codex_turn("gpt-5.6-luna", work, "do it", None, tmp_path, "1")

    _FakeCompletedPopen.calls = []
    r2, session_id2 = AP._run_codex_turn(
        "gpt-5.6-luna", work, "fix it", "01a0000-thread", tmp_path, "2")

    assert len(_FakeCompletedPopen.calls) == 1
    args = _FakeCompletedPopen.calls[0]["args"]
    assert args[:3] == ["codex", "exec", "resume"]
    assert "01a0000-thread" in args
    assert "--last" not in args
    assert "--approve-for-me" not in args
    assert "--cd" not in args
    assert _FakeCompletedPopen.calls[0]["cwd"] == str(work)
    assert session_id2 == "01a0000-thread"
    assert r2.stop_reason == "agent_end"


def test_nonzero_exit_is_process_exit_not_agent_end(tmp_path, monkeypatch):
    """A rejected/failed codex invocation must not be recorded as a clean
    completion -- that is precisely how the broken resume path (a rejected
    argv, exit code 2) went unnoticed: returncode was never checked."""
    monkeypatch.setattr(AP.subprocess, "Popen", _FakeCompletedPopen)
    _FakeCompletedPopen.calls = []
    _FakeCompletedPopen.stdout_lines = []
    _FakeCompletedPopen.returncode = 2

    work = tmp_path / "work"
    work.mkdir()
    r, session_id = AP._run_codex_turn("gpt-5.6-luna", work, "do it", None, tmp_path, "1")

    assert r.stop_reason == "process_exit"
    assert r.agent_ended is False


def test_turn_failed_event_is_process_exit_even_with_exit_zero(tmp_path, monkeypatch):
    """codex can exit 0 while still reporting turn.failed (stream error,
    context overflow, moderation) -- that must not be scored as completed
    either, or a broken turn silently burns the retry budget as if it were
    a genuine attempt."""
    monkeypatch.setattr(AP.subprocess, "Popen", _FakeCompletedPopen)
    _FakeCompletedPopen.calls = []
    _FakeCompletedPopen.stdout_lines = [
        json.dumps({"type": "thread.started", "thread_id": "01a0000-thread"}),
        json.dumps({"type": "turn.failed", "error": {"message": "boom"}}),
    ]
    _FakeCompletedPopen.returncode = 0

    work = tmp_path / "work"
    work.mkdir()
    r, session_id = AP._run_codex_turn("gpt-5.6-luna", work, "do it", None, tmp_path, "1")

    assert r.stop_reason == "process_exit"
    assert r.agent_ended is False
