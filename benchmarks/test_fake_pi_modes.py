"""Direct unit tests of fake_pi.py's new modes through a real PiRpc, so a
broken fake mode fails here in a 1-line test rather than as a confusing
failure deep inside the live-eval end-to-end test. Existing modes (clean,
crash_after_ack, hang_after_ack, etc., covered by test_rpc_terminal.py) are
untouched."""
import base64
import json
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


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def test_solve_from_env_writes_the_requested_files(fake_pi, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"solution.py": _b64("x = 1\n")}))
    with fake_pi("solve_from_env", tmp_path) as rpc:
        r = rpc.prompt_and_collect("go", timeout=30)
    assert r.stop_reason == "agent_end"
    assert (tmp_path / "solution.py").read_text() == "x = 1\n"
    assert r.tool_calls and r.tool_calls[0]["name"] == "write"


def test_noop_then_solve_does_not_write_on_first_attempt(fake_pi, tmp_path, monkeypatch):
    """Attempts are separate processes -- a fresh PiRpc per attempt -- so the
    attempt counter must live out-of-process (FAKE_PI_STATE_FILE), never an
    in-memory counter that a new process wouldn't see."""
    state_file = tmp_path / "state.txt"
    monkeypatch.setenv("FAKE_PI_STATE_FILE", str(state_file))
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"solution.py": _b64("x = 1\n")}))
    with fake_pi("noop_then_solve", tmp_path) as rpc:
        rpc.prompt_and_collect("go", timeout=30)
    assert not (tmp_path / "solution.py").exists()
    assert state_file.exists()


def test_noop_then_solve_writes_on_second_attempt(fake_pi, tmp_path, monkeypatch):
    state_file = tmp_path / "state.txt"
    monkeypatch.setenv("FAKE_PI_STATE_FILE", str(state_file))
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"solution.py": _b64("x = 1\n")}))
    with fake_pi("noop_then_solve", tmp_path) as rpc:
        rpc.prompt_and_collect("go", timeout=30)
    # Second attempt: a FRESH PiRpc (fresh fake_pi process), same state file.
    with fake_pi("noop_then_solve", tmp_path) as rpc:
        rpc.prompt_and_collect("go", timeout=30)
    assert (tmp_path / "solution.py").read_text() == "x = 1\n"


def test_read_system_prompt_echo_captures_the_real_system_prompt_content(fake_pi, tmp_path, monkeypatch):
    agents_md = tmp_path / "repo" / "AGENTS.md"
    agents_md.parent.mkdir(parents=True)
    agents_md.write_text("distinctive candidate text 12345\n")
    monkeypatch.setattr(rpc_client, "REPO_ROOT", agents_md.parent)
    echo_file = tmp_path / "echo.txt"
    monkeypatch.setenv("FAKE_PI_ECHO_FILE", str(echo_file))
    with fake_pi("read_system_prompt_echo", tmp_path) as rpc:
        rpc.prompt_and_collect("go", timeout=30)
    assert "distinctive candidate text 12345" in echo_file.read_text()


def test_solve_if_prompt_contains_writes_when_token_present(fake_pi, tmp_path, monkeypatch):
    agents_md = tmp_path / "repo" / "AGENTS.md"
    agents_md.parent.mkdir(parents=True)
    agents_md.write_text("some instructions with MAGIC-TOKEN-ABC embedded\n")
    monkeypatch.setattr(rpc_client, "REPO_ROOT", agents_md.parent)
    monkeypatch.setenv("FAKE_PI_MAGIC_TOKEN", "MAGIC-TOKEN-ABC")
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"solution.py": _b64("x = 1\n")}))
    with fake_pi("solve_if_prompt_contains", tmp_path) as rpc:
        rpc.prompt_and_collect("go", timeout=30)
    assert (tmp_path / "solution.py").read_text() == "x = 1\n"


def test_solve_if_prompt_contains_does_not_write_when_token_absent(fake_pi, tmp_path, monkeypatch):
    """The regression test for the bug this whole live-eval rewrite exists to
    fix: two candidates differing only in instruction text must be able to
    produce different outcomes."""
    agents_md = tmp_path / "repo" / "AGENTS.md"
    agents_md.parent.mkdir(parents=True)
    agents_md.write_text("some unrelated instructions\n")
    monkeypatch.setattr(rpc_client, "REPO_ROOT", agents_md.parent)
    monkeypatch.setenv("FAKE_PI_MAGIC_TOKEN", "MAGIC-TOKEN-ABC")
    monkeypatch.setenv("FAKE_PI_WRITE_FILES", json.dumps({"solution.py": _b64("x = 1\n")}))
    with fake_pi("solve_if_prompt_contains", tmp_path) as rpc:
        rpc.prompt_and_collect("go", timeout=30)
    assert not (tmp_path / "solution.py").exists()


def test_sleep_forever_honors_a_short_configurable_duration(fake_pi, tmp_path, monkeypatch):
    """Like hang_after_ack, but a deadline test shouldn't have to wait out a
    hardcoded 3600s -- a short FAKE_PI_SLEEP_S well above the request timeout
    still produces "deadline", proving the process was genuinely still
    sleeping (not exited early) when the timeout fired."""
    monkeypatch.setenv("FAKE_PI_SLEEP_S", "5")
    with fake_pi("sleep_forever", tmp_path) as rpc:
        t0 = time.time()
        r = rpc.prompt_and_collect("go", timeout=1)
        elapsed = time.time() - t0
    assert elapsed >= 1
    assert r.stop_reason == "deadline"
