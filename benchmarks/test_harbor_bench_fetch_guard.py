"""_HarborShellProxy benchmark-repo fetch guard and canary file scan tests.

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
    'FP=80 timeout 10 perl fetch.pl 151.101.0.223 "/api/datasets?search=terminal-bench" huggingface.co',
    # Harbor's registry and the harbor / tb CLIs serve a task with its tests
    # and solution, and none of these commands carries a GitHub URL.
    "harbor download terminal-bench/regex-log --export -o /tmp/t",
    "uvx harbor download terminal-bench/regex-log --export -o /tmp/t",
    "pip install harbor && harbor datasets list",
    "pip3 install -q harbor",
    "uv pip install terminal-bench",
    "pipx install terminal-bench",
    "tb datasets download --dataset terminal-bench-core",
    "tb run --task-id regex-log",
    "curl -s https://hlqxxzsirfrgeqasvaps.supabase.co/storage/v1/object/public/x/solve.sh",
    "curl -s https://registry.harborframework.com/api/tasks",
]

ALLOWED = [
    "curl -sL https://raw.githubusercontent.com/idsoftware/Doom-Shareware/master/doom1.wad -o doom1.wad",
    'curl -s "https://api.github.com/search/repositories?q=povray"',
    "python3 -c 'query = \"terminal-bench\"; print(query)'",
    "grep -rn terminal-bench /app/data.txt",
    "ls /app",
    # mteb-retrieve: its instruction query is "terminal-bench" and the task
    # legitimately downloads a model from huggingface.co.
    "cat > /app/solve.py <<'EOF'\nquery = \"terminal-bench\"\n"
    "model_url = \"https://huggingface.co/BAAI/bge-small-zh-v1.5\"\nEOF",
    "HF_ENDPOINT=https://huggingface.co python3 solve.py --query terminal-bench",
    "pip install sentence-transformers && python3 solve.py --query terminal-bench",
    "uv run python3 solve.py --query terminal-bench",
    "python3 -m pytest --tb short tests/",
    "echo 'the harbor is calm'",
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


def _harness_out(body: str, code: int = 0, timed_out: bool = False) -> str:
    return lca._format_output(body, "", code, "/app", timed_out)


def test_canary_scan_command_parses_and_prunes_harness_copies():
    import subprocess
    for newer in (None, lca.SNAPSHOT_START_MARKER):
        cmd = lca._build_canary_scan_command(newer)
        subprocess.run(["sh", "-n", "-c", lca._wrap_command(cmd, None, "__S__")], check=True)
        assert "-xdev" in cmd and "/tmp/.lc*" in cmd and lca.CANARY_STRING in cmd
        assert f"timeout {lca._CANARY_SCAN_TIMEOUT_SEC}" in cmd
        assert (f"-newer {newer}" in cmd) == (newer is not None)


def test_parse_ok_lists_paths():
    out = _harness_out(f"/app/user.cpp\n/tmp/solve.sh\n{lca._CANARY_SCAN_STATUS_PREFIX}ok")
    assert lca._parse_canary_scan(out) == ("ok", ["/app/user.cpp", "/tmp/solve.sh"])


def test_parse_missing_marker_is_skipped():
    out = _harness_out(f"{lca._CANARY_SCAN_STATUS_PREFIX}skipped")
    assert lca._parse_canary_scan(out) == ("skipped", [])


def test_parse_timeout_without_status_token():
    assert lca._parse_canary_scan(_harness_out("/app/a", code=124))[0] == "timeout"
    assert lca._parse_canary_scan(_harness_out("", code=-1, timed_out=True))[0] == "timeout"


def test_parse_error_never_reads_as_clean():
    assert lca._parse_canary_scan(_harness_out("", code=0)) == ("error", [])
    assert lca._parse_canary_scan(_harness_out("sh: find: not found", code=127))[0] == "error"
    assert lca._parse_canary_scan(None)[0] == "error"


def test_parse_over_cap_is_error():
    paths = "\n".join(f"/app/f{i}" for i in range(lca._CANARY_SCAN_MAX_PATHS + 1))
    out = _harness_out(f"{paths}\n{lca._CANARY_SCAN_STATUS_PREFIX}ok")
    assert lca._parse_canary_scan(out)[0] == "error"


def test_result_reports_only_paths_not_in_baseline():
    r = lca._canary_scan_result(("ok", ["/app/user.cpp"]), ("ok", ["/app/user.cpp", "/tmp/sol.sh"]))
    assert r == {
        "canary_scan_status": "ok",
        "canary_new_paths": ["/tmp/sol.sh"],
        "contamination_suspected": True,
    }
    clean = lca._canary_scan_result(("ok", ["/app/user.cpp"]), ("ok", ["/app/user.cpp"]))
    assert clean["contamination_suspected"] is False and clean["canary_scan_status"] == "ok"


def test_result_status_fails_closed_on_either_scan():
    r = lca._canary_scan_result(("timeout", []), ("ok", ["/app/user.cpp"]))
    assert r["canary_scan_status"] == "timeout"
    # No usable baseline: every hit is reported rather than dropped.
    assert r["canary_new_paths"] == ["/app/user.cpp"] and r["contamination_suspected"]
    assert lca._canary_scan_result(None, ("ok", []))["canary_scan_status"] == "error"
    assert lca._canary_scan_result(("ok", []), ("skipped", []))["canary_scan_status"] == "skipped"


def test_result_caps_reported_paths():
    end = ("ok", [f"/tmp/f{i}" for i in range(50)])
    r = lca._canary_scan_result(("ok", []), end)
    assert len(r["canary_new_paths"]) == lca._CANARY_REPORTED_PATHS_MAX


def _proxy_with_counts(blocked=2, canary=1):
    proxy = lca._HarborShellProxy(_RecordingEnv(), loop=None, logger=logging.getLogger("t"))
    proxy.n_blocked_benchmark_fetches = blocked
    proxy.n_canary_outputs = canary
    return proxy


GUARD_KEYS = {
    "n_blocked_benchmark_fetches", "n_canary_outputs", "canary_scan_status",
    "canary_new_paths", "contamination_suspected",
}


def test_record_guard_metadata_creates_dict_on_cancel_path():
    ctx = types.SimpleNamespace(metadata=None)
    scan = lca._canary_scan_result(None, ("error", []))
    lca._record_guard_metadata(ctx, _proxy_with_counts(), scan, logging.getLogger("t"))
    assert GUARD_KEYS <= set(ctx.metadata)
    assert ctx.metadata["n_blocked_benchmark_fetches"] == 2
    assert ctx.metadata["canary_scan_status"] == "error"


def test_record_guard_metadata_merges_into_success_dict():
    ctx = types.SimpleNamespace(metadata={"stop_reason": "end_turn"})
    scan = lca._canary_scan_result(("ok", []), ("ok", []))
    lca._record_guard_metadata(ctx, _proxy_with_counts(0, 0), scan, logging.getLogger("t"))
    assert ctx.metadata["stop_reason"] == "end_turn"
    assert ctx.metadata["canary_scan_status"] == "ok"
    assert ctx.metadata["contamination_suspected"] is False


def test_record_guard_metadata_never_raises():
    class _Frozen:
        metadata = None

        def __setattr__(self, k, v):
            raise RuntimeError("frozen")

    lca._record_guard_metadata(_Frozen(), _proxy_with_counts(), {}, logging.getLogger("t"))


def test_run_canary_scan_goes_through_run_harness_and_maps_failures():
    calls = []

    class _P:
        async def run_harness(self, command, timeout):
            calls.append(command)
            return _harness_out(f"/tmp/x\n{lca._CANARY_SCAN_STATUS_PREFIX}ok")

        def run(self, *a, **kw):
            raise AssertionError("the scan must never use proxy.run()")

    assert asyncio.run(lca._run_canary_scan(_P(), None, logging.getLogger("t"))) == ("ok", ["/tmp/x"])
    assert len(calls) == 1

    class _Boom:
        async def run_harness(self, command, timeout):
            raise RuntimeError("container gone")

    assert asyncio.run(lca._run_canary_scan(_Boom(), None, logging.getLogger("t"))) == ("error", [])


def test_run_canary_scan_does_not_swallow_cancellation():
    class _Slow:
        async def run_harness(self, command, timeout):
            await asyncio.sleep(10)

    async def main():
        task = asyncio.create_task(lca._run_canary_scan(_Slow(), None, logging.getLogger("t")))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
