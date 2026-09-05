"""LITTLE_CODER_PI_BIN_OVERRIDE: lets a host-side subprocess (e.g. `tb run`,
which runs pi on the host per benchmarks/tb_adapter's own comment) point
PI_BIN at fake_pi.py for integration testing, without modifying rpc_client.py
itself or any tracked file. Backward-compat: unset -> byte-identical to the
existing hardcoded default."""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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
