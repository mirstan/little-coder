"""_HarborShellProxy timeout-truth regression tests.

Distinct from test_harbor_adapter_timeout.py, which covers
_resolve_trial_timeout_sec() (an unrelated function) and is gated on
pytest.importorskip("harbor"). These tests exercise _exec_async()/run()'s
exception handling and touch nothing that needs the real harbor package, so
they follow test_format_output.py's stub-and-load convention instead.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import importlib.util
import logging
import sys
import types
from pathlib import Path

BENCHMARKS = Path(__file__).resolve().parent

_STUBBED_ROOTS: set[str] = set()
_STUBBED_MODULES: list[str] = []


def _install_stub(dotted: str, *names: str) -> None:
    """Register a stub module (and its parent packages) under `dotted`, unless
    the real package is importable. Mirrors test_format_output.py's helper --
    _STUBBED_ROOTS must gate the real-import attempt, since a root stubbed by
    an earlier call would otherwise "import" successfully as itself."""
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
            mod.__path__ = []  # marks it a package, so submodule imports resolve
            sys.modules[name] = mod
            _STUBBED_MODULES.append(name)
            if i > 1:
                setattr(sys.modules[".".join(parts[: i - 1])], parts[i - 1], mod)
    leaf = sys.modules[dotted]
    for n in names:
        if not hasattr(leaf, n):
            setattr(leaf, n, type(n, (), {}))


_install_stub("harbor.agents.base", "BaseAgent")
_install_stub("harbor.environments.base", "BaseEnvironment")
_install_stub("harbor.models.agent.context", "AgentContext")

try:
    _spec = importlib.util.spec_from_file_location(
        "_lca_harbor_exec_timeout", BENCHMARKS / "harbor_adapter" / "little_coder_agent.py"
    )
    lca = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = lca
    _spec.loader.exec_module(lca)
finally:
    # lca keeps its own references to the stubbed classes, so the stubs have
    # done their job. Leaving them in sys.modules would turn a sibling test
    # file's `pytest.importorskip("harbor")` into a false positive when this
    # file is collected first (test_format_output.py's convention).
    for _name in reversed(_STUBBED_MODULES):
        sys.modules.pop(_name, None)


class _ExecResult:
    def __init__(self, stdout="", stderr="", return_code=0):
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code


class _RaisingEnv:
    """Fake BaseEnvironment whose exec() always raises a fixed error."""

    def __init__(self, error: BaseException):
        self._error = error

    async def exec(self, command, timeout_sec):
        raise self._error

    async def upload_file(self, *a, **kw):
        raise NotImplementedError


def _new_proxy(env) -> "lca._HarborShellProxy":
    return lca._HarborShellProxy(env, loop=None, logger=logging.getLogger("test-harbor-timeout"))


def _footer(result: str) -> str:
    return result.rsplit("\n", 1)[-1]


def test_docker_timeout_runtimeerror_reports_timed_out_true():
    proxy = _new_proxy(_RaisingEnv(RuntimeError("Command timed out after 30 seconds")))
    result = asyncio.run(proxy._exec_async("sleep 100", 30))
    assert "timed_out=true" in _footer(result)
    assert "exit=-1" in _footer(result)
    assert "this command hit its 30s timeout" in result


def test_unrelated_runtimeerror_does_not_over_match():
    """Regression guard: a docker-compose failure that happens to mention
    Stdout/Stderr must NOT be mistaken for a timeout -- the anchored
    fullmatch, not a substring scan, is what keeps this false."""
    msg = (
        "Docker compose command failed with exit code 1. "
        "Stdout: some output. Stderr: some error including the words "
        "Command timed out after 30 seconds somewhere in a log line"
    )
    proxy = _new_proxy(_RaisingEnv(RuntimeError(msg)))
    result = asyncio.run(proxy._exec_async("sleep 100", 30))
    assert "timed_out=false" in _footer(result)
    assert "WARNING" not in result


def test_asyncio_timeout_error_reports_timed_out_true():
    proxy = _new_proxy(_RaisingEnv(asyncio.TimeoutError()))
    result = asyncio.run(proxy._exec_async("sleep 100", 45))
    assert "timed_out=true" in _footer(result)
    assert "this command hit its 45s timeout" in result


def test_bridge_timeout_reports_timed_out_true(monkeypatch):
    class _FakeFuture:
        def result(self, timeout=None):
            raise concurrent.futures.TimeoutError("bridge timed out")

    def fake_run_coroutine_threadsafe(coro, loop):
        coro.close()  # never actually scheduled; avoid an unawaited-coroutine warning
        return _FakeFuture()

    monkeypatch.setattr(lca.asyncio, "run_coroutine_threadsafe", fake_run_coroutine_threadsafe)

    proxy = _new_proxy(_RaisingEnv(RuntimeError("unused")))
    result = proxy.run("sleep 100", 30)
    assert "timed_out=true" in _footer(result)
    assert "shell proxy error" in result
    # Unlike the confirmed-kill docker/asyncio timeout paths, the bridge
    # timeout only means _exec_async hasn't returned yet -- it must NOT claim
    # the connection was killed, and must hedge that the command may still be
    # running (and warn against a duplicate re-run).
    assert "its connection was killed" not in result
    assert "may still be running" in result
    assert "duplicate" in result


def test_mutation_regex_substring_would_over_match():
    """Pins the finding from this PR's by-hand mutation test: loosening
    _HARBOR_TIMEOUT_MSG_RE's anchors AND its call site's .fullmatch() to
    .search() together made test_unrelated_runtimeerror_does_not_over_match
    fail (loosening only one of the two left it passing -- .fullmatch()
    enforces a full-string match regardless of anchors). This asserts the
    same contrast directly at the regex level: an unanchored search over-
    matches the decoy message; the real anchored fullmatch does not."""
    loose = __import__("re").compile(r"Command timed out after \d+ seconds")
    over_matching_msg = (
        "Docker compose command failed. Stderr: ... Command timed out after 30 seconds ..."
    )
    assert loose.search(over_matching_msg) is not None
    assert lca._HARBOR_TIMEOUT_MSG_RE.fullmatch(over_matching_msg) is None
