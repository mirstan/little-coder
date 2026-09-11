"""Terminal-reason behaviour of PiRpc, driven by a fake pi subprocess.

These cover the JSONL event loop, which the existing tests cannot: they spawn a
real pi and skip when node_modules/.bin/pi is absent. EOF/deadline/crash need a
process that exits on cue.
"""
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rpc_client  # noqa: E402
from rpc_client import PiRpc  # noqa: E402

FAKE = Path(__file__).parent / "fake_pi.py"


@pytest.fixture
def fake_pi(monkeypatch):
    """Point PiRpc at the fake instead of the real pi binary."""
    monkeypatch.setattr(rpc_client, "PI_BIN", FAKE)
    monkeypatch.setattr(rpc_client, "_extension_paths", lambda: [])
    def _spawn(mode, cwd, **kw):
        monkeypatch.setenv("FAKE_PI_MODE", mode)
        return PiRpc(model="fake/model", cwd=str(cwd), **kw)
    return _spawn


def test_thinking_flag_appears_in_argv(fake_pi, tmp_path):
    """Hermetic companion to test_rpc_thinking_flag_reaches_pi (which needs a
    real pi binary and is machine-dependent on ambient settings): this checks
    the literal subprocess argv against a fake pi, so it can never be fooled
    by a local ~/.pi/agent/settings.json default and runs in CI."""
    with fake_pi("clean", tmp_path, thinking="high") as rpc:
        args = rpc._proc.args
    assert "--thinking" in args
    assert args[args.index("--thinking") + 1] == "high"


def test_no_thinking_flag_when_unset(fake_pi, tmp_path):
    """The flag must be genuinely absent when unset, not passed as an empty
    or default value -- pi's own default should apply unmodified."""
    with fake_pi("clean", tmp_path) as rpc:
        args = rpc._proc.args
    assert "--thinking" not in args


def test_clean_run_reports_agent_end(fake_pi, tmp_path):
    with fake_pi("clean", tmp_path) as rpc:
        r = rpc.prompt_and_collect("go", timeout=30)
    assert r.agent_ended is True
    assert r.stop_reason == "agent_end"
    assert r.tool_calls and r.tool_calls[0]["name"] == "read"
    assert "hello" in r.assistant_text


def test_on_event_fires_for_every_event_in_order(fake_pi, tmp_path):
    seen = []
    with fake_pi("clean", tmp_path) as rpc:
        rpc.prompt_and_collect("go", timeout=30, on_event=lambda ev: seen.append(ev["type"]))
    assert seen == [
        "agent_start", "message_update", "tool_execution_start",
        "tool_execution_end", "turn_end", "agent_end", "agent_settled",
    ]


def test_on_event_exception_does_not_drop_later_events(fake_pi, tmp_path):
    """Regression test: an earlier batch-snapshot version of
    _drain_events_until cleared the whole queue up front and only requeued
    the unconsumed remainder on the predicate-match path, so a callback
    raising mid-batch silently dropped every event after it -- including a
    possible agent_end. Events are now popped one at a time, so a raising
    callback only ever discards this call's own `collected` (the events it
    already handed to on_event) -- everything still sitting in the queue,
    including agent_end, survives untouched for the next call to see."""
    def flaky_on_event(ev):
        if ev["type"] == "tool_execution_start":
            raise RuntimeError("simulated live-log write failure")

    with fake_pi("clean", tmp_path) as rpc:
        with pytest.raises(RuntimeError, match="simulated live-log write failure"):
            rpc.prompt_and_collect("go", timeout=30, on_event=flaky_on_event)

        # The events queued after the one that raised (tool_execution_end,
        # turn_end, agent_end) must still be sitting in the queue, not lost.
        remaining = rpc._drain_events_until(
            lambda ev: ev.get("type") == "agent_end", timeout=5,
        )
    remaining_types = [ev["type"] for ev in remaining]
    assert remaining_types == ["tool_execution_end", "turn_end", "agent_end"]


def test_crash_after_ack_is_process_exit_not_deadline(fake_pi, tmp_path):
    """A pi that dies mid-run must not burn the whole budget or look like a timeout."""
    with fake_pi("crash_after_ack", tmp_path) as rpc:
        t0 = time.time()
        r = rpc.prompt_and_collect("go", timeout=30)
        elapsed = time.time() - t0
    assert elapsed < 10, f"took {elapsed:.1f}s -- EOF did not wake the drain"
    assert r.agent_ended is False
    assert r.stop_reason == "process_exit"


def test_hang_is_deadline(fake_pi, tmp_path):
    with fake_pi("hang_after_ack", tmp_path) as rpc:
        t0 = time.time()
        r = rpc.prompt_and_collect("go", timeout=2)
        elapsed = time.time() - t0
    assert elapsed >= 2
    assert r.agent_ended is False
    assert r.stop_reason == "deadline"


def test_agent_end_and_eof_together_still_agent_end(fake_pi, tmp_path):
    """Ordering hazard: queue must be drained before the EOF flag is consulted."""
    with fake_pi("end_then_exit", tmp_path) as rpc:
        r = rpc.prompt_and_collect("go", timeout=30)
    assert r.agent_ended is True
    assert r.stop_reason == "agent_end"


def test_abort_then_followup_returns_recovered_answer(fake_pi, tmp_path):
    """Reproduces the thinking-budget-abort bug directly: an extension's
    ctx.abort() makes pi emit agent_end mid-thought, but a queued follow-up
    survives the abort and pi immediately runs a second, real turn on the
    same connection. Draining only to the first agent_end (the old
    behaviour) would return here with empty/partial assistant_text and miss
    the recovery turn entirely, because the caller would treat the abort's
    agent_end as the whole run finishing. This test fails on the old
    single-phase drain and passes once the drain continues through to
    agent_settled."""
    with fake_pi("abort_then_followup", tmp_path) as rpc:
        r = rpc.prompt_and_collect("go", timeout=30)
    assert "recovered answer" in r.assistant_text
    assert r.agent_ended is True
    assert r.stop_reason == "agent_end"
    assert r.turn_count == 1


def test_end_never_settles_falls_back_after_grace(fake_pi, tmp_path, capsys):
    """Defensive fallback: a pi that emits agent_end but never agent_settled
    (and never exits) must not hang the caller forever -- settle_grace
    bounds the wait, stop_reason still reports agent_end, and a warning is
    logged since this means the caller fell back to the weaker signal."""
    with fake_pi("end_never_settles", tmp_path) as rpc:
        t0 = time.time()
        r = rpc.prompt_and_collect("go", timeout=30, settle_grace=1)
        elapsed = time.time() - t0
    assert elapsed < 10, f"took {elapsed:.1f}s -- settle_grace did not bound the wait"
    assert r.agent_ended is True
    assert r.stop_reason == "agent_end"
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "agent_settled" in err


def test_exit_before_ack_raises_with_stderr(fake_pi, tmp_path):
    with fake_pi("exit_before_ack", tmp_path) as rpc:
        t0 = time.time()
        with pytest.raises(rpc_client.PiProcessExited) as exc:
            rpc.prompt_and_collect("go", timeout=30)
        elapsed = time.time() - t0
    assert elapsed < 10, f"took {elapsed:.1f}s -- did not notice the dead process"
    assert "dying before ack" in str(exc.value)


def test_stderr_is_complete_after_crash(fake_pi, tmp_path):
    """gaia/tb/harbor read stderr exactly on failure; returning fast must not race it."""
    with fake_pi("crash_after_ack", tmp_path) as rpc:
        rpc.prompt_and_collect("go", timeout=30)
        assert "boom" in rpc.stderr()


def test_prompt_after_close_raises(fake_pi, tmp_path):
    rpc = fake_pi("clean", tmp_path)
    rpc.close()
    with pytest.raises(RuntimeError):
        rpc.prompt_and_collect("go", timeout=5)


def test_close_is_idempotent(fake_pi, tmp_path):
    rpc = fake_pi("clean", tmp_path)
    rpc.close()
    rpc.close()


def test_busy_then_ready_recovers_without_stale_event(fake_pi, tmp_path):
    """Regression guard: a rejected send must not let a leftover event from
    the turn pi was still finishing be mistaken for the retry's own
    completion. Without discarding stale events, _drain_events_until would
    pop the leftover agent_end first and return empty -- indistinguishable
    from a genuinely empty response."""
    with fake_pi("busy_then_ready", tmp_path) as rpc:
        r = rpc.prompt_and_collect("go", timeout=30)
    assert r.agent_ended is True
    assert r.stop_reason == "agent_end"
    assert "real answer" in r.assistant_text


def test_busy_then_late_stale_event_still_discarded(fake_pi, tmp_path):
    """Harder ordering than the above: the leftover event is written AFTER
    the retry has already been sent, not before. A fix that discards stale
    events only once, right before resending, passes the case above but
    fails this one -- the leftover lands in the queue after that one-time
    clear and is never removed. Only a fix keyed to the retry's own
    acceptance (not to when the resend happened) handles both."""
    with fake_pi("busy_then_late_stale", tmp_path) as rpc:
        r = rpc.prompt_and_collect("go", timeout=30)
    assert r.agent_ended is True
    assert r.stop_reason == "agent_end"
    assert "real answer" in r.assistant_text


def test_stray_event_on_reused_session_does_not_corrupt_next_turn(fake_pi, tmp_path):
    """The stale-event bug doesn't require a rejection at all: PiRpc's own
    docstring advertises reuse "across prompts within a session" (line 72).
    If pi emits a stray duplicate agent_end right after a turn's real one,
    and the NEXT prompt is accepted outright with no readiness retry
    involved, a fix gated on readiness_attempt > 0 never runs -- the stray
    event sits ahead of the second turn's own completion and the drain
    returns on it immediately, same silent-empty-response symptom as the
    readiness-retry case.

    Timing pin: this fixture's turn 1 has nothing left to settle-wait for
    (fake pi is blocked on read_prompt() for turn 2, which the test can't
    send until turn 1 returns), so turn 1 can only end via the settle-grace
    fallback. settle_grace is passed explicitly (small) so that fallback --
    and therefore this whole test -- is fast; the bug this pins made turn 1
    instead re-arm to the FULL remaining `timeout` on the stray duplicate
    (measured 35.04s here vs 5.03s pre-regression), which passed every
    assertion below while being 7x slower -- exactly why the regression
    shipped unnoticed."""
    t0 = time.time()
    with fake_pi("stray_end_then_clean_reuse", tmp_path) as rpc:
        r1 = rpc.prompt_and_collect("go", timeout=30, settle_grace=2)
        r2 = rpc.prompt_and_collect("go again", timeout=30, settle_grace=2)
    elapsed = time.time() - t0
    assert elapsed < 10, f"took {elapsed:.1f}s -- stray duplicate re-armed the timeout again"
    assert "first answer" in r1.assistant_text
    assert r2.agent_ended is True
    assert r2.stop_reason == "agent_end"
    assert "second answer" in r2.assistant_text


def test_settled_after_extra_event_returns_fast(fake_pi, tmp_path):
    """Pins Bug A: the old Phase 1 predicate only matched agent_end, so an
    agent_settled arriving after some other event (not immediately after
    agent_end) was treated as an ordinary event and Phase 1 waited out the
    entire remaining timeout. Reproduced at ~12.2s; must now return in well
    under 2s."""
    with fake_pi("settled_after_extra_event", tmp_path) as rpc:
        t0 = time.time()
        r = rpc.prompt_and_collect("go", timeout=12)
        elapsed = time.time() - t0
    assert elapsed < 2, f"took {elapsed:.1f}s -- agent_settled after an extra event was missed"
    assert r.agent_ended is True
    assert r.settled is True
    assert r.stop_reason == "agent_end"


def test_retry_then_settled_does_not_rearm_timeout(fake_pi, tmp_path):
    """Pins the PR28 repro: auto_retry_end between agent_end and
    agent_settled must not be mistaken for a continuation (which would
    re-arm the full remaining timeout). Reproduced at 6.29s with
    timeout=6, settle_grace=2; must now return in well under 2s."""
    with fake_pi("retry_then_settled", tmp_path) as rpc:
        t0 = time.time()
        r = rpc.prompt_and_collect("go", timeout=6, settle_grace=2)
        elapsed = time.time() - t0
    assert elapsed < 2, f"took {elapsed:.1f}s -- auto_retry_end re-armed the timeout"
    assert r.agent_ended is True
    assert r.settled is True
    assert r.stop_reason == "agent_end"


def test_abort_then_followup_survives_slow_on_event_eof_race(fake_pi, tmp_path):
    """Pins Bug B directly: the old `if self._eof: break` guards raced the
    reader thread against a slow on_event (Harbor's live-log writer) --
    when the reader ran ahead and set _eof before on_event had drained an
    already-queued recovery turn, the whole queued turn was discarded.
    Reproduced failing 3/3 with a 50ms on_event; must now recover the
    followup every time."""
    def slow_on_event(ev):
        time.sleep(0.05)

    with fake_pi("abort_then_followup_then_exit", tmp_path) as rpc:
        r = rpc.prompt_and_collect("go", timeout=30, on_event=slow_on_event)
    assert "recovered answer" in r.assistant_text
    assert r.turn_count == 1


def test_stray_end_must_not_rearm_settle_window(fake_pi, tmp_path, capsys):
    """Pins Bug C's specific 'stray must not re-arm' case: a duplicate
    agent_end seen while SETTLING must stay in SETTLING (bounded by the
    ORIGINAL settle deadline), not restart the window and not be read as
    renewed work. pi never emits agent_settled here, so this also exercises
    the defensive fallback."""
    with fake_pi("end_then_stray_then_hang", tmp_path) as rpc:
        t0 = time.time()
        r = rpc.prompt_and_collect("go", timeout=30, settle_grace=1)
        elapsed = time.time() - t0
    assert elapsed < 5, f"took {elapsed:.1f}s -- stray agent_end re-armed the settle window"
    assert r.agent_ended is True
    assert r.settled is False
    assert r.stop_reason == "agent_end"
    err = capsys.readouterr().err
    assert "WARNING" in err


def test_continuation_that_never_finishes_is_deadline_not_agent_end(fake_pi, tmp_path):
    """Pins Bug D: stop_reason must reflect how the call actually ended, not
    a latched 'did we ever see an agent_end'. Here a genuine continuation
    starts (agent_start) but never finishes -- the call must run the full
    outer timeout and report stop_reason == 'deadline', even though an
    agent_end was seen earlier in this same call."""
    with fake_pi("continuation_never_finishes", tmp_path) as rpc:
        t0 = time.time()
        r = rpc.prompt_and_collect("go", timeout=3, settle_grace=1)
        elapsed = time.time() - t0
    assert elapsed >= 3
    assert r.agent_ended is True
    assert r.settled is False
    assert r.stop_reason == "deadline"


def test_chatty_stream_bounded_by_absolute_settle_window(fake_pi, tmp_path):
    """Pins the absolute (not per-event) settle window: a continuous stream
    of non-terminal events after agent_end must not extend SETTLING past
    the settle_deadline computed on entry. A per-event reset would let this
    hang until the outer `timeout` instead."""
    with fake_pi("chatty_then_hang", tmp_path) as rpc:
        t0 = time.time()
        r = rpc.prompt_and_collect("go", timeout=30, settle_grace=1)
        elapsed = time.time() - t0
    assert elapsed < 10, f"took {elapsed:.1f}s -- chatty post-turn stream extended settling"
    assert r.agent_ended is True
    assert r.settled is False
    assert r.stop_reason == "agent_end"


def test_promptresult_still_constructible_with_no_args():
    r = rpc_client.PromptResult()
    assert r.agent_ended is False
    assert r.tool_calls == []


class _NeverFinishingThread:
    """Stands in for a reader thread that's still alive after join()'s
    timeout elapses -- without actually waiting out a real timeout."""

    def __init__(self, alive: bool):
        self._alive = alive

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return self._alive


class _AlreadyExitedProc:
    class _Stdin:
        closed = False

        def close(self):
            pass

    def __init__(self):
        self.stdin = self._Stdin()

    def wait(self, timeout=None):
        return 0


def test_close_warns_if_reader_thread_never_finishes(capsys):
    """A reader thread still alive after close()'s join() timeout must not
    be silently swallowed -- it means notifications()/responses may be
    missing their tail, and that's surprising enough to warrant a loud
    warning rather than just proceeding as if nothing happened."""
    rpc = object.__new__(PiRpc)
    rpc._closed = False
    rpc._session_id = "wedged-session"
    rpc._proc = _AlreadyExitedProc()
    rpc._reader = _NeverFinishingThread(alive=True)
    rpc._stderr_reader = _NeverFinishingThread(alive=False)

    rpc.close()

    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "wedged-session" in err
    assert "reader thread" in err


def test_close_silent_when_reader_threads_finish_in_time(capsys):
    rpc = object.__new__(PiRpc)
    rpc._closed = False
    rpc._session_id = "clean-session"
    rpc._proc = _AlreadyExitedProc()
    rpc._reader = _NeverFinishingThread(alive=False)
    rpc._stderr_reader = _NeverFinishingThread(alive=False)

    rpc.close()

    err = capsys.readouterr().err
    assert "WARNING" not in err
