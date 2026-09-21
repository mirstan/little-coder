"""Light tests for rpc_client.PiRpc.

These exercise startup + RPC round-trips but don't require a live LLM —
they use pi's built-in get_state command, which responds without hitting
the provider.

Run with:
    python -m pytest benchmarks/test_rpc_client.py -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rpc_client as RC  # noqa: E402
from rpc_client import PiRpc, _extension_paths, REPO_ROOT  # noqa: E402

PI_BIN = REPO_ROOT / "node_modules" / ".bin" / "pi"


@pytest.fixture(scope="module", autouse=True)
def _skip_if_no_pi():
    if not PI_BIN.exists():
        pytest.skip(f"pi CLI not installed at {PI_BIN} — run `npm install`")


@pytest.fixture(autouse=True)
def _isolate_bench_agent_dir(tmp_path, monkeypatch):
    """Keep constructing a PiRpc out of the checkout's own scratch agent dir.

    _bench_agent_dir() mkdirs it and deletes its settings.json, so without
    this a test run mutates whatever a real benchmark run left at
    REPO_ROOT/.cache. Tests that pin the path override this with their own
    setattr, which monkeypatch applies after the fixture's.
    """
    monkeypatch.setattr(RC, "_BENCH_AGENT_DIR", tmp_path / "bench-agent")


def test_extension_enumeration_finds_scaffold():
    paths = _extension_paths()
    assert len(paths) > 0
    names = {Path(p).parent.name for p in paths}
    for required in ["write-guard", "skill-inject", "knowledge-inject", "evidence", "shell-session"]:
        assert required in names, f"missing required extension: {required}"


def test_rpc_get_state_from_arbitrary_cwd(tmp_path):
    """PiRpc should work from any cwd, not just the repo root."""
    rpc = PiRpc(model="llamacpp/qwen3.6-35b-a3b", cwd=str(tmp_path))
    try:
        rid = str(uuid.uuid4())
        rpc._send({"id": rid, "type": "get_state"})
        resp = rpc._await_response(rid, timeout=20)
        assert resp["success"] is True
        assert resp["data"]["model"]["id"] == "qwen3.6-35b-a3b"
        assert resp["data"]["model"]["provider"] == "llamacpp"
    finally:
        rpc.close(timeout=3)


def test_rpc_allowed_tools_env_propagates(tmp_path):
    """LITTLE_CODER_ALLOWED_TOOLS should reach the tool-gating extension."""
    rpc = PiRpc(
        model="llamacpp/qwen3.6-35b-a3b",
        cwd=str(tmp_path),
        allowed_tools=["Read", "Bash"],
    )
    try:
        rid = str(uuid.uuid4())
        rpc._send({"id": rid, "type": "get_state"})
        resp = rpc._await_response(rid, timeout=20)
        assert resp["success"] is True
    finally:
        rpc.close(timeout=3)


def test_rpc_thinking_flag_reaches_pi(tmp_path):
    """thinking= should reach pi's own --thinking CLI flag, not just be
    accepted and silently dropped. get_state reports the level pi actually
    resolved, so this checks the real effect, not just that no error was
    raised.

    Regression note: this test originally used thinking="high" and would
    have passed even with the parameter deleted entirely, because this
    machine's ~/.pi/agent/settings.json sets defaultThinkingLevel: "high" --
    the exact value pi resolves to with NO --thinking flag at all. "low" and
    "off" are both confirmed distinguishable from every ambient default
    pi has (its own compiled default is "medium"), so either exposes a
    reverted parameter as a real assertion failure instead of a silent pass.
    """
    rpc = PiRpc(
        model="llamacpp/qwen3.6-35b-a3b",
        cwd=str(tmp_path),
        thinking="low",
    )
    try:
        rid = str(uuid.uuid4())
        rpc._send({"id": rid, "type": "get_state"})
        resp = rpc._await_response(rid, timeout=20)
        assert resp["success"] is True
        assert resp["data"]["thinkingLevel"] == "low"
    finally:
        rpc.close(timeout=3)


def test_rpc_tb_mode_env_propagates(tmp_path):
    """tb_mode=True sets LITTLE_CODER_TB_MODE=1 for the subprocess."""
    rpc = PiRpc(
        model="llamacpp/qwen3.6-35b-a3b",
        cwd=str(tmp_path),
        tb_mode=True,
        session_id="test-tb",
    )
    try:
        rid = str(uuid.uuid4())
        rpc._send({"id": rid, "type": "get_state"})
        resp = rpc._await_response(rid, timeout=20)
        assert resp["success"] is True
    finally:
        rpc.close(timeout=3)


class _FakeStream:
    """Minimal stand-in for a subprocess pipe: EOF immediately, no I/O."""

    def __init__(self):
        self.closed = False

    def readline(self):
        return ""

    def write(self, *_a, **_k):
        pass

    def flush(self):
        pass

    def close(self):
        self.closed = True


class _FakeProc:
    """Records the `env` kwarg Popen was called with, without spawning
    anything real -- used to assert on PiRpc's full_env construction
    directly, rather than going through turn-cap.ts's own resolution, which
    is covered separately in TS."""

    captured_env: dict | None = None

    def __init__(self, *_args, **kwargs):
        _FakeProc.captured_env = kwargs.get("env")
        self.stdin = _FakeStream()
        self.stdout = _FakeStream()
        self.stderr = _FakeStream()

    def poll(self):
        return 0

    def terminate(self):
        pass

    def kill(self):
        pass

    def wait(self, timeout=None):
        return 0


def test_max_turns_explicit_zero_clobbers_ambient_env(tmp_path, monkeypatch):
    """max_turns=0 is a deliberate "no cap" choice and must still WRITE
    LITTLE_CODER_MAX_TURNS=0 into the subprocess env, clobbering any ambient
    value inherited from the calling process's own environment (full_env
    starts as a copy of os.environ). A truthiness check (`if max_turns:`)
    would treat 0 as falsy and skip the write entirely, silently leaving a
    leaked ambient cap (e.g. from a wrapper script) in place."""
    monkeypatch.setenv("LITTLE_CODER_MAX_TURNS", "40")
    pi_bin = tmp_path / "pi"
    pi_bin.write_text("")
    monkeypatch.setattr(RC, "PI_BIN", pi_bin)
    monkeypatch.setattr(RC.subprocess, "Popen", _FakeProc)

    rpc = PiRpc(model="llamacpp/qwen3.6-35b-a3b", cwd=str(tmp_path), max_turns=0)
    try:
        assert _FakeProc.captured_env["LITTLE_CODER_MAX_TURNS"] == "0"
    finally:
        rpc.close(timeout=1)


def test_agent_dir_is_isolated_from_the_users_real_pi_config(tmp_path, monkeypatch):
    """Every benchmark pi subprocess must get PI_CODING_AGENT_DIR pointed at
    the scratch dir, never the default ~/.pi/agent: thinking-budget's
    setThinkingLevel() latches persist through it, and a run was observed
    rewriting the user's own interactive defaultThinkingLevel."""
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
    scratch = tmp_path / "scratch-agent"
    pi_bin = tmp_path / "pi"
    pi_bin.write_text("")
    monkeypatch.setattr(RC, "PI_BIN", pi_bin)
    monkeypatch.setattr(RC, "_BENCH_AGENT_DIR", scratch)
    monkeypatch.setattr(RC.subprocess, "Popen", _FakeProc)

    rpc = PiRpc(model="llamacpp/qwen3.6-35b-a3b", cwd=str(tmp_path))
    try:
        assert _FakeProc.captured_env["PI_CODING_AGENT_DIR"] == str(scratch)
        assert scratch.is_dir()
    finally:
        rpc.close(timeout=1)


def test_each_session_clears_a_previous_runs_thinking_latch(tmp_path, monkeypatch):
    """thinking-budget calls setThinkingLevel() mid-trial and pi persists it
    to <agent dir>/settings.json. Because the scratch dir is a fixed path
    reused by every later run, a run that latched to "off" would otherwise
    become the startup default of unrelated runs. bin/ must survive the
    reset -- it is the ripgrep symlink that keeps pi's Grep tool from
    downloading mid-trial."""
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
    scratch = tmp_path / "scratch-agent"
    scratch.mkdir()
    (scratch / "settings.json").write_text('{"defaultThinkingLevel": "off"}')
    (scratch / "bin").mkdir()
    (scratch / "auth.json").write_text("{}")
    pi_bin = tmp_path / "pi"
    pi_bin.write_text("")
    monkeypatch.setattr(RC, "PI_BIN", pi_bin)
    monkeypatch.setattr(RC, "_BENCH_AGENT_DIR", scratch)
    monkeypatch.setattr(RC.subprocess, "Popen", _FakeProc)

    rpc = PiRpc(model="llamacpp/qwen3.6-35b-a3b", cwd=str(tmp_path))
    try:
        assert not (scratch / "settings.json").exists()
        assert (scratch / "bin").is_dir()
        assert (scratch / "auth.json").exists()
    finally:
        rpc.close(timeout=1)


def test_an_exported_agent_dir_still_wins(tmp_path, monkeypatch):
    """The isolation only fills the var in when absent, so an operator who
    deliberately exports PI_CODING_AGENT_DIR keeps control of where pi's
    state goes."""
    monkeypatch.setenv("PI_CODING_AGENT_DIR", "/somewhere/deliberate")
    pi_bin = tmp_path / "pi"
    pi_bin.write_text("")
    monkeypatch.setattr(RC, "PI_BIN", pi_bin)
    monkeypatch.setattr(RC, "_BENCH_AGENT_DIR", tmp_path / "scratch-agent")
    monkeypatch.setattr(RC.subprocess, "Popen", _FakeProc)

    rpc = PiRpc(model="llamacpp/qwen3.6-35b-a3b", cwd=str(tmp_path))
    try:
        assert _FakeProc.captured_env["PI_CODING_AGENT_DIR"] == "/somewhere/deliberate"
    finally:
        rpc.close(timeout=1)


def test_max_turns_unset_leaves_ambient_env_untouched(tmp_path, monkeypatch):
    """max_turns=None (the default, e.g. interactive use / callers that
    never pass the kwarg) must NOT touch LITTLE_CODER_MAX_TURNS at all --
    only an explicit value (including 0) is authoritative."""
    monkeypatch.setenv("LITTLE_CODER_MAX_TURNS", "40")
    pi_bin = tmp_path / "pi"
    pi_bin.write_text("")
    monkeypatch.setattr(RC, "PI_BIN", pi_bin)
    monkeypatch.setattr(RC.subprocess, "Popen", _FakeProc)

    rpc = PiRpc(model="llamacpp/qwen3.6-35b-a3b", cwd=str(tmp_path))
    try:
        assert _FakeProc.captured_env["LITTLE_CODER_MAX_TURNS"] == "40"
    finally:
        rpc.close(timeout=1)


def test_a_scratch_dir_redirected_into_the_real_pi_config_is_refused(tmp_path, monkeypatch):
    """mkdir(exist_ok=True) succeeds through a symlink, so a planted link
    back to ~/.pi/agent would have the settings-reset delete the user's own
    file and route every later latch through it. polyglot and gaia give the
    model under test a host shell, so the link is plantable."""
    fake_home = tmp_path / "home"
    (fake_home / ".pi" / "agent").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    victim = fake_home / ".pi" / "agent" / "settings.json"
    victim.write_text('{"defaultThinkingLevel": "xhigh"}')

    scratch = tmp_path / "repo" / ".cache" / "pi-bench-agent"
    scratch.parent.mkdir(parents=True)
    scratch.symlink_to(fake_home / ".pi" / "agent", target_is_directory=True)
    monkeypatch.setattr(RC, "_BENCH_AGENT_DIR", scratch)

    with pytest.raises(RuntimeError, match="refusing to touch it"):
        RC._bench_agent_dir()
    assert victim.read_text() == '{"defaultThinkingLevel": "xhigh"}'


def test_a_relocated_scratch_dir_outside_the_repo_is_still_allowed(tmp_path, monkeypatch):
    """The guard names ~/.pi specifically rather than "outside the repo" --
    symlinking .cache/ onto another disk is a legitimate thing to do."""
    fake_home = tmp_path / "home"
    (fake_home / ".pi" / "agent").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    elsewhere = tmp_path / "big-disk" / "pi-bench-agent"
    elsewhere.mkdir(parents=True)
    monkeypatch.setattr(RC, "_BENCH_AGENT_DIR", elsewhere)
    assert RC._bench_agent_dir() == str(elsewhere)


def test_a_dangling_bin_symlink_is_repaired(tmp_path, monkeypatch):
    """exists() follows symlinks, so a dead link reads as absent and
    symlink_to then raises FileExistsError into the OSError handler --
    leaving pi's Grep tool to re-download ripgrep every run, silently."""
    fake_home = tmp_path / "home"
    real_bin = fake_home / ".pi" / "agent" / "bin"
    real_bin.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "bin").symlink_to(tmp_path / "gone", target_is_directory=True)
    monkeypatch.setattr(RC, "_BENCH_AGENT_DIR", scratch)

    RC._bench_agent_dir()
    assert (scratch / "bin").readlink() == real_bin


def test_a_bin_symlink_pointing_somewhere_else_is_repointed(tmp_path, monkeypatch):
    """A link to an attacker-chosen dir would otherwise persist and pi would
    resolve rg from it."""
    fake_home = tmp_path / "home"
    real_bin = fake_home / ".pi" / "agent" / "bin"
    real_bin.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    other = tmp_path / "other-bin"
    other.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "bin").symlink_to(other, target_is_directory=True)
    monkeypatch.setattr(RC, "_BENCH_AGENT_DIR", scratch)

    RC._bench_agent_dir()
    assert (scratch / "bin").readlink() == real_bin


def test_a_correct_bin_symlink_is_left_alone(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    real_bin = fake_home / ".pi" / "agent" / "bin"
    real_bin.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "bin").symlink_to(real_bin, target_is_directory=True)
    monkeypatch.setattr(RC, "_BENCH_AGENT_DIR", scratch)

    RC._bench_agent_dir()
    assert (scratch / "bin").readlink() == real_bin
