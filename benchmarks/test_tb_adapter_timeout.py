"""_TmuxShellProxy.run()'s sentinel-missing timeout path.

Loaded with `terminal_bench` stubbed (test_format_output.py's convention): it
is not installed next to these tests, and the tmux-timeout logic under test
touches none of its real behavior.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

BENCHMARKS = Path(__file__).resolve().parent

_STUBBED_ROOTS: set[str] = set()
_STUBBED_MODULES: list[str] = []


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
            _STUBBED_MODULES.append(name)
            if i > 1:
                setattr(sys.modules[".".join(parts[: i - 1])], parts[i - 1], mod)
    leaf = sys.modules[dotted]
    for n in names:
        if not hasattr(leaf, n):
            setattr(leaf, n, type(n, (), {}))


_install_stub("terminal_bench.agents.base_agent", "AgentResult", "BaseAgent")
_install_stub("terminal_bench.agents.failure_mode", "FailureMode")
_install_stub("terminal_bench.terminal.tmux_session", "TmuxSession")

try:
    _spec = importlib.util.spec_from_file_location(
        "_lca_tb_adapter_timeout", BENCHMARKS / "tb_adapter" / "little_coder_agent.py"
    )
    tba = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = tba
    _spec.loader.exec_module(tba)
finally:
    # tba keeps its own references to the stubbed classes, so the stubs have
    # done their job. Leaving them in sys.modules would turn a sibling test
    # file's `pytest.importorskip("terminal_bench")` into a false positive
    # when this file is collected first (test_format_output.py's convention).
    for _name in reversed(_STUBBED_MODULES):
        sys.modules.pop(_name, None)


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


class _SendKeysRaisingTmux(_FakeTmux):
    """send_keys raises on the run()-issued call (but not _init_once's), so
    the command's `source <script>` was never actually typed into the pane --
    a non-timeout failure that still lands in the sentinel-missing branch
    because run() swallows the exception."""

    def send_keys(self, *args, **kwargs):
        if self._calls > 0:
            raise RuntimeError("tmux client gone")


class _CapturePaneRaisingTmux(_FakeTmux):
    """capture_pane raises on the post-send_keys call, so run() falls back to
    pane="" -- there is no output to show at all, regardless of whether the
    command ran, is running, or never started."""

    def capture_pane(self, capture_entire=True):
        idx = self._calls
        self._calls += 1
        if idx == 0:
            return self._panes[0]
        raise RuntimeError("capture failed")


def _footer(result: str) -> str:
    return result.rsplit("\n", 1)[-1]


def test_sentinel_missing_reports_timed_out_true_with_tmux_warning():
    tmux = _FakeTmux(["", "partial output, no sentinel ever showed up"])
    proxy = tba._TmuxShellProxy(tmux, "sess-1")
    result = proxy.run("sleep 999", 30)
    assert "timed_out=true" in _footer(result)
    assert "exit=-1" in _footer(result)
    assert "partial output, no sentinel ever showed up" in result
    # Hedged, not asserted as settled fact: this branch also fires for
    # non-timeout send_keys/capture_pane failures (see the two tests below),
    # so the warning may not claim a real timeout or a definitely-running
    # process.
    assert "no completion sentinel was seen" in result
    assert "may be STILL RUNNING" in result
    assert "its 30s timeout" in result


def test_send_keys_failure_reports_hedged_warning_not_a_timeout_claim():
    """send_keys can fail immediately for reasons that have nothing to do
    with the {N}s timeout elapsing (e.g. the tmux client itself is gone).
    run() swallows that exception, so this still lands in the
    sentinel-missing branch -- the warning must not assert the command
    'hit its timeout' as settled fact."""
    tmux = _SendKeysRaisingTmux(["", "whatever was on the pane before"])
    proxy = tba._TmuxShellProxy(tmux, "sess-1")
    result = proxy.run("echo hi", 30)
    assert "timed_out=true" in _footer(result)
    assert "no completion sentinel was seen" in result
    assert "failed to start" in result


def test_capture_pane_failure_reports_hedged_warning_with_no_output():
    """capture_pane raising leaves pane == "" -- there is no output to show
    at all, so the warning must not claim the process is definitely still
    running (it may have crashed, finished, or never started)."""
    tmux = _CapturePaneRaisingTmux(["", "unused"])
    proxy = tba._TmuxShellProxy(tmux, "sess-1")
    result = proxy.run("echo hi", 30)
    assert "timed_out=true" in _footer(result)
    assert "no completion sentinel was seen" in result
    assert "may not have been captured" in result


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
