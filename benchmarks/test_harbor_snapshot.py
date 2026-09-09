"""Plan 4b (shell-proxy-snapshot) -- the deadline-snapshot mechanism added to
_HarborShellProxy / LittleCoderAgent.run() in little_coder_agent.py:

  - the bounded, atomically-staged snapshot shell command itself
  - _compute_snapshot_delay_sec()'s scheduling arithmetic (incl. the
    short-task skip edge case)
  - _HarborShellProxy.run_harness() staying on the caller's event loop
    (never routing through asyncio.run_coroutine_threadsafe(), which is
    exactly the sync thread-bridge path that would deadlock a same-loop
    asyncio task -- Codex finding 1)
  - _exec_lock actually serializing a model-issued command (via the sync
    run() thread-bridge, exercised from a real background thread since
    calling it from the loop's own thread would itself deadlock) against a
    harness-issued run_harness() call

Requires the `harbor` package (only installed in harbor's own uv-tool venv,
not the plain system Python these other benchmark tests run under) purely
because little_coder_agent.py imports harbor.agents.base at module load
time; none of this file's actual assertions touch Harbor itself. Same
scaffolding convention as test_harbor_adapter_timeout.py.
"""
import asyncio
import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "harbor_adapter"))
import little_coder_agent as lca  # noqa: E402


def _logger() -> logging.Logger:
    return logging.getLogger("test-harbor-snapshot")


class _RecordingEnv:
    """Fake harbor BaseEnvironment: records every command it was asked to
    exec and returns a canned success result."""

    def __init__(self):
        self.commands: list[str] = []

    async def exec(self, command: str, timeout_sec: int | None = None, **kwargs) -> SimpleNamespace:
        self.commands.append(command)
        return SimpleNamespace(stdout="", stderr="", return_code=0)


class _OverlapTrackingEnv:
    """Fake environment that records the max number of concurrently-active
    exec() calls, to detect whether two callers' commands ever overlapped
    inside the container."""

    def __init__(self, hold_sec: float = 0.1):
        self.hold_sec = hold_sec
        self.current = 0
        self.max_concurrent = 0
        self.commands: list[str] = []

    async def exec(self, command: str, timeout_sec: int | None = None, **kwargs) -> SimpleNamespace:
        self.commands.append(command)
        self.current += 1
        self.max_concurrent = max(self.max_concurrent, self.current)
        try:
            await asyncio.sleep(self.hold_sec)
        finally:
            self.current -= 1
        return SimpleNamespace(stdout="", stderr="", return_code=0)


# ── 1. Snapshot command construction ────────────────────────────────────────

def test_snapshot_command_has_bounded_caps_and_staged_publish():
    """Every Codex-review-motivated cap and the atomic-publish mechanism must
    actually be present in the composite command -- this is the command that
    literally runs in the container, so a typo here silently defeats one of
    the bounds."""
    cmd = lca._SNAPSHOT_COMMAND
    # per-file size cap
    assert "-size -10M" in cmd
    # freshness filter keyed to the start marker
    assert "-newer /tmp/.lc-start" in cmd
    assert lca.SNAPSHOT_START_MARKER in cmd
    # aggregate file-count cap
    assert "head -z -n 500" in cmd
    # aggregate byte cap: 200MB
    assert "209715200" in cmd
    # free-space reserve: 500MB
    assert "524288000" in cmd
    # staged-then-atomically-published, not written in place
    assert "STAGE=/tmp/.lc-snapshot.stage.$$" in cmd
    assert 'mv "$STAGE" /tmp/.lc-snapshot' in cmd
    assert lca.SNAPSHOT_PUBLISH_PATH in cmd
    # internal timeout + outer stage-glob cleanup for the kill-mid-copy case
    assert "timeout 20 sh -c" in cmd
    assert "rm -rf /tmp/.lc-snapshot.stage.*" in cmd
    # refuses rather than partially copies when oversize
    assert "set -e" in cmd


# ── 2. Scheduling arithmetic (_compute_snapshot_delay_sec) ─────────────────

def test_snapshot_delay_normal_budget_fires_lead_seconds_before_deadline():
    eff = 3600.0
    delay = lca._compute_snapshot_delay_sec(eff)
    assert delay == pytest.approx(eff - lca.SNAPSHOT_LEAD_SEC)


def test_snapshot_delay_clamped_to_zero_when_lead_exceeds_budget():
    """Budget is between the min-budget floor and the lead time -- the naive
    delay would be negative, so it's clamped to 0 (snapshot fires almost
    immediately) rather than skipped."""
    eff = 400.0
    assert eff >= lca.SNAPSHOT_MIN_BUDGET_SEC
    assert eff - lca.SNAPSHOT_LEAD_SEC < 0
    assert lca._compute_snapshot_delay_sec(eff) == 0.0


def test_snapshot_delay_skips_entirely_below_min_budget():
    """Plan 4b's short-task edge case: below SNAPSHOT_MIN_BUDGET_SEC the
    snapshot is skipped outright (None), not merely delay-clamped."""
    eff = lca.SNAPSHOT_MIN_BUDGET_SEC - 1.0
    assert lca._compute_snapshot_delay_sec(eff) is None


def test_snapshot_delay_at_exact_min_budget_boundary_is_not_skipped():
    eff = lca.SNAPSHOT_MIN_BUDGET_SEC
    assert lca._compute_snapshot_delay_sec(eff) is not None


# ── 3. _snapshot_at_deadline: fires via run_harness after its delay, and is
#      cleanly cancellable before it fires ──────────────────────────────────

def test_snapshot_at_deadline_fires_run_harness_after_delay():
    env = _RecordingEnv()

    async def scenario():
        loop = asyncio.get_running_loop()
        proxy = lca._HarborShellProxy(env, loop, _logger())
        task = asyncio.create_task(lca._snapshot_at_deadline(proxy, 0.01, _logger()))
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(scenario())
    assert len(env.commands) == 1
    assert "STAGE=/tmp/.lc-snapshot.stage.$$" in env.commands[0]


def test_snapshot_at_deadline_cancelled_before_firing_never_execs():
    """Mirrors run()'s own finally: task.cancel() + suppressed
    CancelledError. A snapshot scheduled far in the future and cancelled
    immediately (as happens on early trial completion) must never touch the
    container."""
    env = _RecordingEnv()

    async def scenario():
        loop = asyncio.get_running_loop()
        proxy = lca._HarborShellProxy(env, loop, _logger())
        task = asyncio.create_task(lca._snapshot_at_deadline(proxy, 10.0, _logger()))
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())
    assert env.commands == []


# ── 4. run_harness stays on the caller's loop -- never the thread bridge ───

def test_run_harness_never_routes_through_run_coroutine_threadsafe(monkeypatch):
    """Pins the deadlock fix: run_harness must await the async exec path
    directly. Under the original (rejected) design, the snapshot task called
    proxy.run() -- which submits via asyncio.run_coroutine_threadsafe(...)
    and then blocks on fut.result(). Called from an asyncio task on the same
    loop, fut.result() would block that very loop, deadlocking it. Patching
    run_coroutine_threadsafe to blow up turns any regression back to that
    call path into a hard test failure instead of a hang."""

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError(
            "run_harness routed through run_coroutine_threadsafe -- this is "
            "the thread-bridge path that deadlocks when called from an "
            "asyncio task on the same loop (Plan 4b / Codex finding 1)"
        )

    monkeypatch.setattr(lca.asyncio, "run_coroutine_threadsafe", _must_not_be_called)

    env = _RecordingEnv()

    async def scenario():
        loop = asyncio.get_running_loop()
        proxy = lca._HarborShellProxy(env, loop, _logger())
        return await proxy.run_harness("echo harness-only", 5)

    asyncio.run(scenario())
    assert len(env.commands) == 1


# ── 5. _exec_lock serializes model-issued (run()) and harness-issued
#      (run_harness()) commands so they never overlap inside the container ─

def test_exec_lock_serializes_run_and_run_harness():
    """Simulates PiRpc's reader thread (which calls the sync proxy.run()
    thread-bridge) issuing a model command concurrently with the snapshot
    task's run_harness() call on the loop itself. Before this lock existed
    there was nothing to serialize the two paths against each other; without
    it, two coroutines scheduled on the same loop that each `await
    asyncio.sleep(...)` mid-exec would interleave, and OverlapTrackingEnv's
    max_concurrent would show 2."""
    env = _OverlapTrackingEnv(hold_sec=0.15)
    thread_result: dict[str, str] = {}

    async def scenario():
        loop = asyncio.get_running_loop()
        proxy = lca._HarborShellProxy(env, loop, _logger())

        def thread_target():
            thread_result["out"] = proxy.run("echo model-issued", 5)

        t = threading.Thread(target=thread_target)
        t.start()
        # Let the reader-thread's command actually start executing on the
        # loop before we submit the harness-issued one, so the two windows
        # genuinely overlap in time rather than trivially running back-to-back.
        await asyncio.sleep(0.05)
        harness_out = await proxy.run_harness("echo harness-issued", 5)
        t.join(timeout=5)
        assert not t.is_alive()
        return harness_out

    harness_out = asyncio.run(scenario())

    assert env.max_concurrent == 1, (
        f"exec calls overlapped (max_concurrent={env.max_concurrent}) -- "
        "_exec_lock failed to serialize model- and harness-issued commands"
    )
    assert len(env.commands) == 2
    assert any("echo model-issued" in c for c in env.commands)
    assert any("echo harness-issued" in c for c in env.commands)
    assert thread_result.get("out")  # sync run() returned formatted output
    assert harness_out
