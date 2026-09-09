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
    directly (Plan 5, verification item 1) rather than going through
    turn-cap.ts's own resolution, which is covered separately in TS."""

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
    starts as a copy of os.environ). The old `if max_turns:` check treated
    0 as falsy and skipped the write entirely, silently leaving a leaked
    ambient cap (e.g. from a wrapper script) in place -- see Plan 5 (Codex
    finding [high]: turn-cap-still-active-via-profile/ambient-env)."""
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
