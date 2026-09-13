"""LITTLE_CODER_PI_BIN_OVERRIDE: lets a host-side subprocess (e.g. `tb run`,
which runs pi on the host per benchmarks/tb_adapter's own comment) point
PI_BIN at fake_pi.py for integration testing, without modifying rpc_client.py
itself or any tracked file. Backward-compat: unset -> byte-identical to the
existing hardcoded default."""
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _restore_top_level_rpc_client_module():
    """_reload_rpc_client() below replaces
    sys.modules["rpc_client"] with a fresh top-level module object -- a
    DIFFERENT identity than benchmarks.rpc_client (this file's sys.path
    insertion above makes benchmarks/ importable as a top-level location
    too). Left in place for the rest of the pytest session, a later test
    anywhere else that happens to `import rpc_client` as a top-level name
    would silently receive this file's env-var-driven reload state (e.g.
    PI_BIN pointing at fake_pi.py) instead of the real module. Restore
    whatever (if anything) occupied that slot before this test ran."""
    before = sys.modules.get("rpc_client")
    yield
    if before is None:
        sys.modules.pop("rpc_client", None)
    else:
        sys.modules["rpc_client"] = before


def _reload_rpc_client():
    if "rpc_client" in sys.modules:
        del sys.modules["rpc_client"]
    import rpc_client
    return rpc_client


def test_pi_bin_defaults_to_node_modules_path_when_unset(monkeypatch):
    monkeypatch.delenv("LITTLE_CODER_PI_BIN_OVERRIDE", raising=False)
    rpc_client = _reload_rpc_client()
    assert rpc_client.PI_BIN == rpc_client.REPO_ROOT / "node_modules" / ".bin" / "pi"


def test_pi_bin_uses_override_when_set(monkeypatch, tmp_path):
    fake = tmp_path / "fake_pi.py"
    fake.write_text("#!/usr/bin/env python3\n")
    monkeypatch.setenv("LITTLE_CODER_PI_BIN_OVERRIDE", str(fake))
    rpc_client = _reload_rpc_client()
    assert rpc_client.PI_BIN == fake


def test_pi_bin_resolves_relative_override_to_absolute(monkeypatch, tmp_path):
    """On POSIX, subprocess.Popen with both a
    RELATIVE executable path and an explicit cwd= resolves that path against
    the CHILD's cwd, not the launcher's. A relative override would exist
    from the launcher's own cwd at import time, then silently fail to launch
    once PiRpc is constructed with a task-specific cwd (an aider_polyglot
    exercise dir, etc). Must be resolved to absolute once, up front."""
    fake = tmp_path / "fake_pi.py"
    fake.write_text("#!/usr/bin/env python3\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LITTLE_CODER_PI_BIN_OVERRIDE", "./fake_pi.py")
    rpc_client = _reload_rpc_client()
    assert rpc_client.PI_BIN.is_absolute()
    assert rpc_client.PI_BIN == fake.resolve()


def test_pi_bin_falls_back_to_default_when_override_is_empty_string(monkeypatch):
    """os.environ.get(name, default) returns
    "" (not the default) when the var is exported EMPTY, e.g. via
    `export LITTLE_CODER_PI_BIN_OVERRIDE="$SOME_UNSET_VAR"` in a harness
    script. Path("") == Path("."), whose .exists() is True, which silently
    defeated the "pi CLI not found" check and surfaced later as an opaque
    Popen error instead."""
    monkeypatch.setenv("LITTLE_CODER_PI_BIN_OVERRIDE", "")
    rpc_client = _reload_rpc_client()
    assert rpc_client.PI_BIN == rpc_client.REPO_ROOT / "node_modules" / ".bin" / "pi"
