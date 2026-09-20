"""_TmuxShellProxy.run()'s sentinel-missing timeout path.

Loaded with `terminal_bench` stubbed (test_format_output.py's convention):
neither package is installed next to these tests, and the tmux-timeout logic
under test touches neither package's real behavior.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

BENCHMARKS = Path(__file__).resolve().parent

_STUBBED_ROOTS: set[str] = set()


def _install_stub(dotted: str, *names: str) -> None:
    root = dotted.split(".")[0]
    if root not in _STUBBED_ROOTS:
        try:
            __import__(root)
            return
        except ImportError:
            _STUBBED_ROOTS.add(root)
    parts = dotted.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod
            if i > 1:
                setattr(sys.modules[".".join(parts[: i - 1])], parts[i - 1], mod)
    leaf = sys.modules[dotted]
    for n in names:
        if not hasattr(leaf, n):
            setattr(leaf, n, type(n, (), {}))


_install_stub("terminal_bench.agents.base_agent", "AgentResult", "BaseAgent")
_install_stub("terminal_bench.agents.failure_mode", "FailureMode")
_install_stub("terminal_bench.terminal.tmux_session", "TmuxSession")

_spec = importlib.util.spec_from_file_location(
    "_lca_tb_adapter_timeout", BENCHMARKS / "tb_adapter" / "little_coder_agent.py"
)
tba = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tba
_spec.loader.exec_module(tba)


class _FakeExecResult:
    exit_code = 0


class _FakeContainer:
    def exec_run(self, args):
        return _FakeExecResult()


class _FakeTmux:
    """panes[0] backs _init_once's cursor-priming capture; panes[1] backs
    the capture taken after run()'s send_keys -- mirrors two real,
    successive capture_pane() calls on a session that never got the
    sentinel written to it."""

    def __init__(self, panes):
        self.container = _FakeContainer()
        self._panes = list(panes)
        self._calls = 0

    def send_keys(self, *args, **kwargs):
        pass

    def capture_pane(self, capture_entire=True):
        idx = min(self._calls, len(self._panes) - 1)
        self._calls += 1
        return self._panes[idx]


def _footer(result: str) -> str:
    return result.rsplit("\n", 1)[-1]


def test_sentinel_missing_reports_timed_out_true_with_tmux_warning():
    tmux = _FakeTmux(["", "partial output, no sentinel ever showed up"])
    proxy = tba._TmuxShellProxy(tmux, "sess-1")
    result = proxy.run("sleep 999", 30)
    assert "timed_out=true" in _footer(result)
    assert "exit=-1" in _footer(result)
    assert "partial output, no sentinel ever showed up" in result
    assert "STILL RUNNING" in result
    assert "its 30s timeout" in result


def test_sentinel_present_has_no_warning():
    sentinel = "__LC_END_deadbeef__"
    pane_with_sentinel = f"hello\n{sentinel}:0:/app\n"
    tmux = _FakeTmux(["", pane_with_sentinel])
    proxy = tba._TmuxShellProxy(tmux, "sess-1")
    # run() mints its own sentinel internally, so build a pane using a
    # sentinel this call will actually search for by patching uuid.
    import uuid

    real_uuid4 = uuid.uuid4

    class _FixedUUID:
        hex = "deadbeef" * 4

    uuid.uuid4 = lambda: _FixedUUID()
    try:
        result = proxy.run("echo hello", 30)
    finally:
        uuid.uuid4 = real_uuid4
    assert "timed_out=false" in _footer(result)
    assert "STILL RUNNING" not in result
    assert "WARNING" not in result
