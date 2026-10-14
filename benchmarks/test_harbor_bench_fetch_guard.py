"""_HarborShellProxy benchmark-repo fetch guard tests.

Uses test_harbor_exec_timeout.py's stub-and-load convention: nothing here
needs the real harbor package.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import importlib.util
import logging
import re
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
        "_lca_harbor_bench_fetch_guard", BENCHMARKS / "harbor_adapter" / "little_coder_agent.py"
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



import pytest

# Verbatim (trimmed) from the three trials that fetched Terminal-Bench's own
# tests or solution: filter-js-from-html 2026-10-11, regex-log and
# mteb-retrieve 2026-10-12.
OFFENDING = [
    'cd /tmp && for f in task.yaml tests/test_outputs.py solution.sh; do curl -sf --max-time 15 '
    '"https://raw.githubusercontent.com/laude-institute/terminal-bench/main/tasks/filter-js-from-html/$f"; done',
    'perl /tmp/fetch.pl api.github.com "/repos/laude-institute/terminal-bench/contents/tasks" > /tmp/tasks.json',
    'perl /tmp/fetch.pl api.github.com "/search/repositories?q=terminal-bench&per_page=5"',
    'python3 - <<EOF\nfor repo in ["LAUDE-INSTITUTE/terminal-bench"]:\n    pass\nEOF',
    "python3 -c \"import urllib.request; urllib.request.urlopen('https://huggingface.co/datasets/x/terminal-bench')\"",
]

ALLOWED = [
    "curl -sL https://raw.githubusercontent.com/idsoftware/Doom-Shareware/master/doom1.wad -o doom1.wad",
    'curl -s "https://api.github.com/search/repositories?q=povray"',
    "python3 -c 'query = \"terminal-bench\"; print(query)'",
    "grep -rn terminal-bench /app/data.txt",
    "ls /app",
]


@pytest.mark.parametrize("command", OFFENDING)
def test_benchmark_repo_fetch_is_detected(command):
    assert lca._is_benchmark_repo_fetch(command)


@pytest.mark.parametrize("command", ALLOWED)
def test_unrelated_github_and_bare_name_pass(command):
    assert not lca._is_benchmark_repo_fetch(command)


class _RecordingEnv:
    def __init__(self, stdout=""):
        self.commands = []
        self._stdout = stdout

    async def exec(self, command, timeout_sec):
        self.commands.append(command)
        raise AssertionError("a blocked command must never reach the container")

    async def upload_file(self, *a, **kw):
        raise NotImplementedError


def test_blocked_command_never_reaches_container_and_is_counted():
    env = _RecordingEnv()
    proxy = lca._HarborShellProxy(env, loop=None, logger=logging.getLogger("test-bench-guard"))
    out = proxy.run(OFFENDING[0], timeout=30)
    assert env.commands == []
    assert "Blocked:" in out
    assert "exit=1" in out
    assert proxy.n_blocked_benchmark_fetches == 1


def test_canary_in_output_is_counted_not_blocked():
    canary = "# terminal-bench-canary GUID 00000000-0000-0000-0000-000000000000"

    class _Env:
        async def exec(self, command, timeout_sec):
            sentinel = re.search(r"__LC_END_[0-9a-f]+__", command).group(0)
            return types.SimpleNamespace(stdout=f"{canary}\n{sentinel}:0:/app\n", stderr="", return_code=0)

        async def upload_file(self, *a, **kw):
            raise NotImplementedError

    loop = asyncio.new_event_loop()
    import threading
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    try:
        proxy = lca._HarborShellProxy(_Env(), loop=loop, logger=logging.getLogger("test-bench-guard"))
        out = proxy.run("cat /app/user.cpp", timeout=30)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(5)
    assert "terminal-bench-canary" in out
    assert proxy.n_canary_outputs == 1
    assert proxy.n_blocked_benchmark_fetches == 0
