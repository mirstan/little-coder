"""Tests for the deadline-snapshot mechanism in _HarborShellProxy /
LittleCoderAgent.run() in little_coder_agent.py:

  - the bounded, atomically-staged snapshot shell command itself
  - _wrap_command() actually composing a parseable shell script -- both for
    a harness call's cwd=None form and a model call's cd/pwd-tracking form
    -- verified by actually feeding the composed string to `sh -n`/`bash -n`
    rather than merely substring-matching pieces of it. This is the
    regression test for an unterminated trailing `#` comment on
    _SNAPSHOT_COMMAND's last line silently swallowing everything _exec_async
    appended after it -- a bug that every existing substring-matching test
    missed, because none of them ever actually composed and parsed the real
    string.
  - _compute_snapshot_delay_sec()'s scheduling arithmetic (incl. the
    short-task skip edge case)
  - _HarborShellProxy.run_harness() staying on the caller's event loop
    (never routing through asyncio.run_coroutine_threadsafe(), which is
    exactly the sync thread-bridge path that would deadlock a same-loop
    asyncio task)
  - _exec_lock actually serializing a model-issued command (via the sync
    run() thread-bridge, exercised from a real background thread since
    calling it from the loop's own thread would itself deadlock) against a
    harness-issued run_harness() call, and run_harness's track_cwd=False
    never mutating proxy.cwd even when the underlying exec's stdout happens
    to look like it echoed back a pwd line

Requires the `harbor` package (only installed in harbor's own uv-tool venv,
not the plain system Python these other benchmark tests run under) purely
because little_coder_agent.py imports harbor.agents.base at module load
time; none of this file's actual assertions touch Harbor itself. Same
scaffolding convention as test_harbor_adapter_timeout.py.
"""
import asyncio
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
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
    inside the container.

    fake_pwd_line, if set, fabricates a `{sentinel}:0:{fake_pwd_line}` tail
    on the response to any command that does NOT itself end in ` ; pwd`
    (i.e. a track_cwd=False / harness-issued composition, which never asks
    for one) -- simulating an environment that echoes back something that
    *looks* like a pwd line even though none was requested, so a test can
    confirm run_harness's track_cwd=False guard (not merely the absence of a
    real trailing `pwd` in the command) is what keeps self.cwd untouched.
    """

    def __init__(self, hold_sec: float = 0.1, fake_pwd_line: str | None = None):
        self.hold_sec = hold_sec
        self.fake_pwd_line = fake_pwd_line
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
        stdout = ""
        if self.fake_pwd_line is not None and not command.rstrip().endswith("pwd"):
            m = re.search(r"(__LC_END_\w+__):", command)
            if m:
                stdout = f"\n{m.group(1)}:0:{self.fake_pwd_line}\n"
        return SimpleNamespace(stdout=stdout, stderr="", return_code=0)


# ── 1. Snapshot command construction ────────────────────────────────────────

def test_snapshot_command_has_bounded_caps_and_staged_publish():
    """Every safety cap and the atomic-publish mechanism must actually be
    present in the composite command -- this is the command that literally
    runs in the container, so a typo here silently defeats one of the
    bounds."""
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
    # 2.3: publish is gated on a non-empty candidate list, so a zero-match
    # find (or a deleted start marker) refuses rather than publishing an
    # empty snapshot directory
    assert '[ -s "$STAGE/.list" ]' in cmd


def test_snapshot_command_last_line_has_no_trailing_inline_comment():
    """Regression pin for 2.1: a trailing `#` comment (with no newline after
    it) on _SNAPSHOT_COMMAND's last line is exactly what silently swallowed
    everything _exec_async/_wrap_command appended after it, producing a
    command that never parsed. Guard against re-introducing one."""
    last_line = lca._SNAPSHOT_COMMAND.rsplit("\n", 1)[-1]
    assert "#" not in last_line, (
        f"_SNAPSHOT_COMMAND's last line contains a trailing inline comment "
        f"with no newline after it -- this swallows everything appended "
        f"after the constant when it's wrapped (2.1): {last_line!r}"
    )


# ── 1b. The composed command must actually shell-parse ─────────────────────
# (not just substring-match -- substring matching is exactly what let 2.1's
# unterminated-comment bug ship undetected)

_SHELLS_TO_TRY = ("sh", "bash")


@pytest.mark.parametrize("shell", _SHELLS_TO_TRY)
def test_composed_snapshot_command_parses_under_sh_n(shell):
    """The model-command (track_cwd=True) composition of _SNAPSHOT_COMMAND,
    fed to a real shell's syntax checker -- this is the actual regression
    test for 2.1: `sh -n`/`bash -n` reported "unexpected end of file", rc=2,
    every single time before the fix."""
    if shutil.which(shell) is None:
        pytest.skip(f"{shell} not found on PATH")
    composed = lca._wrap_command(lca._SNAPSHOT_COMMAND, "/app", "__LC_END_test__")
    result = subprocess.run([shell, "-n"], input=composed, text=True, capture_output=True)
    assert result.returncode == 0, (
        f"{shell} -n rejected the composed snapshot command "
        f"(rc={result.returncode}): {result.stderr!r}"
    )
    assert result.stderr == ""


@pytest.mark.parametrize("shell", _SHELLS_TO_TRY)
def test_composed_harness_command_parses_under_sh_n(shell):
    """Same, for the track_cwd=False composition (2.2) -- run_harness's
    actual call shape: no leading `cd`, no trailing `pwd`."""
    if shutil.which(shell) is None:
        pytest.skip(f"{shell} not found on PATH")
    composed = lca._wrap_command(lca._SNAPSHOT_COMMAND, None, "__LC_END_test__")
    result = subprocess.run([shell, "-n"], input=composed, text=True, capture_output=True)
    assert result.returncode == 0, (
        f"{shell} -n rejected the composed harness (track_cwd=False) command "
        f"(rc={result.returncode}): {result.stderr!r}"
    )
    assert result.stderr == ""


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
    """Short-task edge case: below SNAPSHOT_MIN_BUDGET_SEC the snapshot is
    skipped outright (None), not merely delay-clamped."""
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
            "asyncio task on the same loop"
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
    max_concurrent would show 2.

    Also (2.2) pins that run_harness's track_cwd=False never mutates
    proxy.cwd -- the fake env is set up to fabricate a bogus pwd-looking
    tail on any non-pwd-ending command (i.e. exactly the harness-issued
    composition), so a regression that let run_harness read/write self.cwd
    would show up here as proxy.cwd changing to that bogus value.
    """
    env = _OverlapTrackingEnv(hold_sec=0.15, fake_pwd_line="/bogus/should-not-be-used")
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
        return harness_out, proxy.cwd

    harness_out, cwd_after = asyncio.run(scenario())

    assert env.max_concurrent == 1, (
        f"exec calls overlapped (max_concurrent={env.max_concurrent}) -- "
        "_exec_lock failed to serialize model- and harness-issued commands"
    )
    assert len(env.commands) == 2
    assert any("echo model-issued" in c for c in env.commands)
    assert any("echo harness-issued" in c for c in env.commands)
    assert thread_result.get("out")  # sync run() returned formatted output
    assert harness_out
    assert cwd_after == "/app", (
        f"proxy.cwd changed to {cwd_after!r} -- run_harness's track_cwd=False "
        "must never read or write self.cwd, even when the underlying exec's "
        "stdout looks like it echoed back a pwd line"
    )


# ── 6. Overflow capture: pushing the untruncated output into the container ─
#
# Phase 1 capped ShellSession output at ~48KB and told the model nothing about
# where the discarded bytes went, because a host tmp path is unreachable from
# inside the container. Phase 2 uploads the full cleaned output through
# env.upload_file() (BaseEnvironment's portable host->container primitive) and
# points the model at the container-side path.

_OVERFLOW_STDOUT = "x" * 200_000  # one line, so only the byte cap can catch it


class _OverflowEnv:
    """Fake harbor BaseEnvironment for the overflow-capture tests.

    exec() returns `stdout` only for model-issued commands (the ones carrying
    an _exec_async sentinel) -- there is no longer any other exec the code
    issues (the mkdir bootstrap was removed; harbor's own tar-fallback mkdir
    -p's the target as root when the fast `cp` path needs it, so this fake
    never needs to simulate that side).

    upload_file() reads the host file's content and mode *during* the call:
    _capture_overflow_inner unlinks the tmp file in a finally before control
    returns, so a stub that only recorded the path would find nothing left to
    read by assertion time.
    """

    def __init__(self, stdout: str = "", stderr: str = "",
                 upload_error: Exception | None = None, upload_delay_sec: float = 0.0):
        self.stdout = stdout
        self.stderr = stderr
        self.upload_error = upload_error
        self.upload_delay_sec = upload_delay_sec
        self.commands: list[str] = []
        # (source_path, target_path, content_at_call_time, mode_at_call_time)
        self.uploads: list[tuple[str, str, str | None, int | None]] = []

    async def exec(self, command: str, timeout_sec: int | None = None, **kwargs) -> SimpleNamespace:
        self.commands.append(command)
        return SimpleNamespace(stdout=self.stdout, stderr=self.stderr, return_code=0)

    async def upload_file(self, source_path, target_path: str) -> None:
        content, mode = None, None
        try:
            mode = os.stat(source_path).st_mode & 0o777
            with open(source_path, encoding="utf-8") as f:
                content = f.read()
        except OSError:
            pass
        self.uploads.append((str(source_path), target_path, content, mode))
        if self.upload_delay_sec:
            await asyncio.sleep(self.upload_delay_sec)
        if self.upload_error is not None:
            raise self.upload_error


def _run_exec(env, command: str = "echo hi", timeout: int = 5) -> str:
    async def scenario():
        loop = asyncio.get_running_loop()
        proxy = lca._HarborShellProxy(env, loop, _logger())
        return await proxy._exec_async(command, timeout)

    return asyncio.run(scenario())


def _host_tmp_paths(env) -> list[str]:
    return [src for src, _target, _content, _mode in env.uploads]


def test_byte_capped_output_uploads_the_full_content_and_points_at_it():
    """The headline case. Asserts all of: upload happens exactly once, the
    target is under this proxy's randomized container overflow dir, the
    uploaded bytes are the full untruncated cleaned output (not the capped
    body), the host file was widened to 0644 (mkstemp's 0600 would be
    unreadable to a non-root task's shell after docker cp lands it as root),
    and the note is spliced between the body and the footer rather than
    after it."""
    env = _OverflowEnv(stdout=_OVERFLOW_STDOUT)
    out = _run_exec(env)

    assert len(env.uploads) == 1
    _src, target, content, mode = env.uploads[0]
    assert target.startswith(lca._CONTAINER_OVERFLOW_DIR_PREFIX)
    assert target.endswith(".out")
    assert content == lca._cleaned_output(_OVERFLOW_STDOUT, "")
    assert len(content) == len(_OVERFLOW_STDOUT)  # the full output, not the capped body
    assert mode == 0o644, f"host tmp file was mode {oct(mode or 0)}, not 0644"

    lines = out.split("\n")
    assert lines[-1].startswith("[exit=0 ")
    assert "output_truncated=true" in lines[-1]
    assert "byte_capped=true" in lines[-1]
    assert lines[-2] == f"Full output: {target}"


def test_normal_output_never_uploads_anything():
    """Regression guard: capture must fire on truncation only, not on every
    single ShellSession call."""
    env = _OverflowEnv(stdout="hello\nworld\n")
    out = _run_exec(env)

    assert env.uploads == []
    assert "Full output:" not in out
    assert "output_truncated=true" not in out


def test_body_containing_the_truncation_flag_does_not_trigger_capture():
    """The gate reads the footer line only. A command's own output can
    legitimately contain the literal `byte_capped=true` -- e.g. the model
    catting back an earlier transcript, or one of this feature's own overflow
    files -- and a whole-string scan would upload needlessly and splice a
    lying "Full output:" note onto an untruncated result."""
    env = _OverflowEnv(stdout="[exit=0 cwd=/app timed_out=false byte_capped=true]\nreal output")
    out = _run_exec(env)

    assert env.uploads == []
    assert "Full output:" not in out
    assert "byte_capped=true" not in out.rsplit("\n", 1)[-1]


def test_line_only_truncation_does_not_trigger_capture():
    """Routine output that trips only the 200-line cap (pip install, pytest
    -v, git log) sets output_truncated=true but not byte_capped=true, and
    must not cost a docker-cp round trip -- that would make the common case
    pay for a feature built for the rare, pathological one."""
    many_short_lines = "\n".join(f"line {i}" for i in range(400))  # far under any byte cap
    env = _OverflowEnv(stdout=many_short_lines)
    out = _run_exec(env)

    assert env.uploads == []
    footer = out.rsplit("\n", 1)[-1]
    assert "output_truncated=true" in footer
    assert "byte_capped=true" not in footer
    assert "Full output:" not in out


def test_no_mkdir_bootstrap_and_dir_stays_stable_across_calls():
    """The mkdir bootstrap was removed entirely (harbor's own tar-fallback
    mkdir -p's the target as root when the fast `cp` path needs it), so no
    exec beyond the model's own wrapped command is ever issued, and every
    capture on one proxy instance targets the same per-proxy directory."""
    env = _OverflowEnv(stdout=_OVERFLOW_STDOUT)

    async def scenario():
        loop = asyncio.get_running_loop()
        proxy = lca._HarborShellProxy(env, loop, _logger())
        first = await proxy._exec_async("echo one", 5)
        second = await proxy._exec_async("echo two", 5)
        return proxy, first, second

    proxy, _first, _second = asyncio.run(scenario())
    assert not any(c.strip().startswith("mkdir") for c in env.commands), (
        f"unexpected mkdir exec: {env.commands}"
    )
    assert len(env.uploads) == 2
    dirs = {target.rsplit("/", 1)[0] for _src, target, _content, _mode in env.uploads}
    assert dirs == {proxy._overflow_dir}


def test_host_stage_dir_is_removable_after_the_trial():
    """Per-file cleanup (see test_host_tmp_file_is_always_unlinked below)
    unlinks each staged file as it's uploaded, but the private 0700
    directory those files lived in (LittleCoderAgent.run()'s job to remove,
    via shutil.rmtree(proxy._host_stage_dir, ignore_errors=True) in its
    outer finally) otherwise outlives the trial -- one leaked empty
    directory per trial that ever byte-capped, forever, on the shared
    harness host. This pins the mechanism run() relies on: after a capture,
    the directory exists, is empty (no leaked files), and is safely
    removable by exactly the call run() makes."""
    env = _OverflowEnv(stdout=_OVERFLOW_STDOUT)

    async def scenario():
        loop = asyncio.get_running_loop()
        proxy = lca._HarborShellProxy(env, loop, _logger())
        await proxy._exec_async("echo one", 5)
        return proxy

    proxy = asyncio.run(scenario())
    assert proxy._host_stage_dir is not None
    assert os.path.isdir(proxy._host_stage_dir)
    assert os.listdir(proxy._host_stage_dir) == [], "a staged file leaked past its upload"

    shutil.rmtree(proxy._host_stage_dir, ignore_errors=True)
    assert not os.path.exists(proxy._host_stage_dir)


def test_budget_exhaustion_stops_uploading_and_says_so(monkeypatch):
    """The per-trial ceiling exists to stop a pathological loop from filling
    the container's /tmp. Once it's hit the model is told the file wasn't
    saved rather than being pointed at a path that doesn't exist -- and the
    message must not itself look like a "Full output: <path>" line, or a
    naive prefix-split would hand a model this sentence as a path."""
    monkeypatch.setattr(lca, "_OVERFLOW_BUDGET_BYTES", 300_000)
    env = _OverflowEnv(stdout=_OVERFLOW_STDOUT)  # 200_000 bytes cleaned

    async def scenario():
        loop = asyncio.get_running_loop()
        proxy = lca._HarborShellProxy(env, loop, _logger())
        first = await proxy._exec_async("echo one", 5)
        second = await proxy._exec_async("echo two", 5)
        return first, second

    first, second = asyncio.run(scenario())

    assert f"Full output: {lca._CONTAINER_OVERFLOW_DIR_PREFIX}" in first
    assert len(env.uploads) == 1, "second capture uploaded despite an exhausted budget"
    assert "Full output:" not in second
    assert "not saved" in second and "budget exhausted" in second
    assert "output_truncated=true" in second.rsplit("\n", 1)[-1]


def test_upload_failure_degrades_to_phase_1_disclosure():
    """Capture is a bonus on top of the truncation footer. If it fails the
    result must be exactly what it was before this feature existed -- never
    an exception, and never a "Full output:" line pointing at a file the
    upload didn't actually produce."""
    env = _OverflowEnv(stdout=_OVERFLOW_STDOUT, upload_error=RuntimeError("docker cp exploded"))
    out = _run_exec(env)

    assert "Full output:" not in out
    footer = out.rsplit("\n", 1)[-1]
    assert "output_truncated=true" in footer
    assert f"raw_bytes={len(_OVERFLOW_STDOUT)}" in footer
    (host_tmp,) = _host_tmp_paths(env)
    assert not os.path.exists(host_tmp)


def test_capture_timeout_degrades_the_same_way_and_returns_promptly(monkeypatch):
    """A docker cp that hangs must not eat run()'s fixed timeout+30 slack and
    turn a truncated-but-present result into a hard shell-proxy error. The
    real constant is pinned by the first assertion; the test itself then
    swaps in a short one, so a would-be 60s hang shows up as a fast,
    clean degrade instead of a 15s one."""
    assert lca._OVERFLOW_CAPTURE_TIMEOUT_SEC < 30, (
        "the capture timeout must fit inside run()'s timeout+30 slack"
    )
    monkeypatch.setattr(lca, "_OVERFLOW_CAPTURE_TIMEOUT_SEC", 0.05)
    env = _OverflowEnv(stdout=_OVERFLOW_STDOUT, upload_delay_sec=60)

    started = time.monotonic()
    out = _run_exec(env)
    elapsed = time.monotonic() - started

    assert elapsed < 5, f"a hung upload blocked the result for {elapsed:.1f}s"
    assert "Full output:" not in out
    footer = out.rsplit("\n", 1)[-1]
    assert "output_truncated=true" in footer
    assert f"raw_bytes={len(_OVERFLOW_STDOUT)}" in footer


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"upload_error": RuntimeError("boom")}],
    ids=["upload-succeeds", "upload-raises"],
)
def test_host_tmp_file_is_always_unlinked(kwargs):
    """The staging file lives on the *host* (the harness machine, shared
    across concurrent trials), so leaking one per capped command is a real
    disk leak. The unlink is in a finally; pin that it covers the failure
    path too."""
    env = _OverflowEnv(stdout=_OVERFLOW_STDOUT, **kwargs)
    _run_exec(env)

    paths = _host_tmp_paths(env)
    assert len(paths) == 1
    assert not os.path.exists(paths[0]), f"host tmp file {paths[0]} survived the call"
