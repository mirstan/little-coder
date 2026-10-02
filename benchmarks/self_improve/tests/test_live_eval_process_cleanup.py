"""Unit tests for the post-run cleanup in live_eval: the process-group kill
and the run-token sweep. Every process started here is killed by its
recorded pid or group in a `finally`."""
import logging
import os
import signal
import subprocess
import sys
import time

import pytest

import benchmarks.self_improve.live_eval as live_eval
from benchmarks.self_improve.live_eval import PolyglotLiveRunner

# The orphans these tests leave for the kernel to reparent are never reaped
# when pytest itself is PID 1 (a bare container); they stay zombies and the
# group kill waits out its full grace. See README "Known gaps / next steps".
pytestmark = pytest.mark.skipif(os.getpid() == 1, reason="orphans are never reaped under PID 1")


def _group_alive(pgid: int) -> bool:
    """macOS answers EPERM for a group whose only members are zombies."""
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _start_group(script: str, **env) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["/bin/sh", "-c", script], env={**os.environ, **env}, start_new_session=True,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    proc.wait()  # the leader exits at once; its background member lives on
    return proc


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def test_kill_process_group_gives_a_reaped_leaders_group_its_grace(tmp_path):
    """On the success path the leader is always reaped already, so waiting
    on it returned at once and the members were SIGKILLed with no grace.
    This member exits cleanly on SIGTERM and leaves a marker when it does."""
    marker = tmp_path / "terminated_cleanly"
    proc = _start_group(
        '( trap \'echo ok > "$M"; exit 0\' TERM; while :; do sleep 0.05; done ) & exit 0', M=str(marker),
    )
    try:
        time.sleep(0.2)  # let the member install its trap
        started = time.monotonic()
        PolyglotLiveRunner._kill_process_group(proc)
        assert marker.read_text() == "ok\n"
        assert time.monotonic() - started < 7
        assert not _group_alive(proc.pid)
    finally:
        _kill_group(proc)


def test_kill_process_group_kills_a_member_that_ignores_sigterm_after_the_grace():
    proc = _start_group('( trap "" TERM; while :; do sleep 0.05; done ) & exit 0')
    try:
        time.sleep(0.2)
        started = time.monotonic()
        assert PolyglotLiveRunner._kill_process_group(proc, grace_s=0.5) is True
        elapsed = time.monotonic() - started
        assert 0.5 <= elapsed < 0.5 + live_eval._KILL_CONFIRM_S
        assert not _group_alive(proc.pid)
    finally:
        _kill_group(proc)


def test_kill_process_group_is_a_no_op_once_the_group_is_gone():
    proc = subprocess.Popen(["/bin/sh", "-c", "exit 0"], start_new_session=True)
    proc.wait()
    assert PolyglotLiveRunner._kill_process_group(proc, grace_s=0.5) is False


def test_kill_process_group_refuses_the_orchestrators_own_group(monkeypatch):
    class _FakeProc:
        pid = os.getpgrp()

        def poll(self):
            return 0

    sent = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))
    assert PolyglotLiveRunner._kill_process_group(_FakeProc(), grace_s=0.1) is False
    assert sent == []


def test_ps_matching_needs_the_whole_entry_and_skips_this_process():
    token = "a" * 32
    me = os.getpid()
    listing = b"\n".join([
        f"{me} python x {live_eval._RUN_TOKEN_ENV}={token} HOME=/h".encode(),
        f"  101 bash {live_eval._RUN_TOKEN_ENV}={token}".encode(),
        f"102 node {live_eval._RUN_TOKEN_ENV}={token}b".encode(),
        f"103 node X{live_eval._RUN_TOKEN_ENV}={token}".encode(),
        b"104 perl \xff\xfe not utf-8",
        f"105 python {live_eval._RUN_TOKEN_ENV}={'b' * 32}".encode(),
        b"",
        b"garbage",
    ])
    assert live_eval._tagged_pids_from_ps(listing, token) == {me, 101}


def test_list_tagged_pids_never_returns_this_process(monkeypatch):
    token = "c" * 32
    me = os.getpid()
    monkeypatch.setattr(live_eval, "_USE_PROC", False)
    monkeypatch.setattr(live_eval, "_ps_environ_listing",
                        lambda: f"{me} python {live_eval._RUN_TOKEN_ENV}={token}".encode())
    assert live_eval._list_tagged_pids(token) == set()


def test_sweep_kills_a_tagged_process_and_its_whole_group(tmp_path):
    """The tagged member's group holds an untagged shell loop too, the shape
    of a macOS leftover whose /bin/sh the sweep cannot see."""
    token = os.urandom(16).hex()
    proc = subprocess.Popen(
        ["/bin/sh", "-c", '( while :; do sleep 0.05; done ) & "$PY" -c "import time; time.sleep(30)" & wait'],
        env={**os.environ, "PY": sys.executable},
        start_new_session=True,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        # Only the python member carries the token in this test; the shell
        # loop is found through its group.
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            out = subprocess.run(["pgrep", "-g", str(proc.pid), "-f", "time.sleep"],
                                 capture_output=True, text=True).stdout.split()
            if out:
                break
            time.sleep(0.05)
        assert out, "the python member never started"
        tagged = int(out[0])
        scans = []

        def _lister(tok):
            # Like the real listers, which never show a killed process again
            # (a zombie has no environment left to match).
            scans.append(tok)
            return {tagged} if tok == token and len(scans) == 1 else set()

        killed = live_eval.PolyglotLiveRunner._sweep_tagged_processes(token, _lister=_lister)
        assert killed == [tagged]
        proc.wait(timeout=5)
        deadline = time.monotonic() + 3
        while _group_alive(proc.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _group_alive(proc.pid)
    finally:
        _kill_group(proc)
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_sweep_warns_once_and_kills_nothing_when_listing_fails(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="benchmarks.self_improve.live_eval")

    def _broken(token):
        raise OSError("ps unavailable")

    monkeypatch.setattr(os, "kill", lambda *a: pytest.fail("must not signal anything"))
    assert live_eval.PolyglotLiveRunner._sweep_tagged_processes("d" * 32, _lister=_broken) == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "ps unavailable" in warnings[0].getMessage()


def test_sweep_stands_down_if_the_token_is_in_the_orchestrators_own_environment(monkeypatch, caplog):
    """Would make every other child of this orchestrator a target."""
    caplog.set_level(logging.WARNING, logger="benchmarks.self_improve.live_eval")
    token = "e" * 32
    monkeypatch.setenv(live_eval._RUN_TOKEN_ENV, token)

    def _must_not_list(tok):
        raise AssertionError("listed processes despite the tripwire")

    assert live_eval.PolyglotLiveRunner._sweep_tagged_processes(token, _lister=_must_not_list) == []
    assert any(live_eval._RUN_TOKEN_ENV in r.getMessage() for r in caplog.records)
