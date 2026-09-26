"""The trial-budget window both TB adapters publish to pi's extensions.

tb-finalize-guard's progress checkpoints fire on the FRACTION of the budget
spent, which needs both ends of [trial start, deadline]. The deadline half
already existed; the start half is new, and the two adapters have to agree
about it -- an extension that behaved differently on TB1.0 than on TB2.x
because one adapter published half an interval would be diagnosable only
from a trial log.

Both adapters are loaded with `harbor` / `terminal_bench` stubbed rather than
importorskip'd (test_format_output.py's convention): neither package is
installed next to these tests, and skipping would leave this symmetry
untested on exactly the machines that run the suite.
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import textwrap
import types
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parent

_STUBBED_ROOTS: set[str] = set()
_STUBBED_MODULES: list[str] = []


def _install_stub(dotted: str, *names: str) -> None:
    """Register a stub module (and its parent packages) under `dotted`, unless
    the real package is importable."""
    root = dotted.split(".")[0]
    if root not in _STUBBED_ROOTS:
        try:
            importlib.import_module(root)
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


def _load(adapter: str):
    """Import an adapter's little_coder_agent.py under a unique module name --
    both files share a basename, so a plain import would alias them."""
    path = BENCHMARKS / adapter / "little_coder_agent.py"
    spec = importlib.util.spec_from_file_location(f"_lca_budget_{adapter}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_install_stub("harbor.agents.base", "BaseAgent")
_install_stub("harbor.environments.base", "BaseEnvironment")
_install_stub("harbor.models.agent.context", "AgentContext")
_install_stub("terminal_bench.agents.base_agent", "AgentResult", "BaseAgent")
_install_stub("terminal_bench.agents.failure_mode", "FailureMode")
_install_stub("terminal_bench.terminal.tmux_session", "TmuxSession")
try:
    _HARBOR = _load("harbor_adapter")
    _TB = _load("tb_adapter")
finally:
    # The adapters keep their own references, so the stubs have done their
    # job. Leaving them in sys.modules would turn the other benchmark tests'
    # `pytest.importorskip("harbor")` into a false positive.
    for _name in reversed(_STUBBED_MODULES):
        sys.modules.pop(_name, None)


START = 1_700_000_000_000
TOTAL_MS = 3_600_000


def _env(mod) -> dict[str, str]:
    """_pi_env over the two adapters' differing signatures -- harbor's also
    takes the initial-snapshot outcome, which TB1.0 has no equivalent of."""
    kwargs = {
        "budget_start_epoch_ms": START,
        "deadline_epoch_ms": START + TOTAL_MS,
    }
    if "initial_snapshot" in inspect.signature(mod._pi_env).parameters:
        kwargs["initial_snapshot"] = None
    return mod._pi_env(**kwargs)


ADAPTERS = [pytest.param(_HARBOR, id="harbor"), pytest.param(_TB, id="tb")]

# The two adapters' trial entrypoints are named differently (harbor's
# BaseAgent calls run(), terminal_bench's calls perform_task()), which the
# source-inspection pin below has to follow.
ENTRYPOINTS = [
    pytest.param(_HARBOR, "run", id="harbor"),
    pytest.param(_TB, "perform_task", id="tb"),
]


@pytest.mark.parametrize("mod", ADAPTERS)
def test_both_ends_of_the_budget_are_published(mod):
    env = _env(mod)
    assert env["LITTLE_CODER_BUDGET_START_EPOCH_MS"] == str(START)
    assert env["LITTLE_CODER_DEADLINE_EPOCH_MS"] == str(START + TOTAL_MS)


@pytest.mark.parametrize("mod", ADAPTERS)
def test_the_published_window_is_the_trial_budget(mod):
    """What the fraction is computed against. A start later than the deadline
    -- or equal to it -- makes the fraction undefined and silently disables
    every progress checkpoint."""
    env = _env(mod)
    start = int(env["LITTLE_CODER_BUDGET_START_EPOCH_MS"])
    deadline = int(env["LITTLE_CODER_DEADLINE_EPOCH_MS"])
    assert start < deadline
    assert deadline - start == TOTAL_MS


@pytest.mark.parametrize("mod,entrypoint", ENTRYPOINTS)
def test_the_trial_entrypoint_derives_both_ends_from_one_instant(mod, entrypoint):
    """Wiring pin: the assertions above are on a pure helper, so an
    entrypoint that read the clock twice -- or did not pass the start at all
    -- would pass them while publishing a window whose width is not the
    trial's budget."""
    src = textwrap.dedent(inspect.getsource(getattr(mod.LittleCoderAgent, entrypoint)))
    assert "budget_start_epoch_ms = int(time.time() * 1000)" in src
    assert "deadline_epoch_ms = budget_start_epoch_ms + int(" in src
    assert "budget_start_epoch_ms=budget_start_epoch_ms," in src


def test_only_the_harbor_adapter_can_publish_an_initial_snapshot():
    """TB1.0 stages no start-of-trial copy, so its env must never tell an
    extension one exists -- explicitly empty, not merely omitted (see the
    leak test below for why omission isn't enough)."""
    assert _env(_TB)["LITTLE_CODER_INITIAL_SNAPSHOT"] == ""
    assert "initial_snapshot" not in inspect.signature(_TB._pi_env).parameters


def test_tb_clobbers_an_initial_snapshot_var_leaked_from_the_calling_process():
    """rpc_client.PiRpc builds the child's env as a copy of this process's own
    os.environ, updated with _pi_env's dict -- a key _pi_env does not mention
    would pass through untouched. If LITTLE_CODER_INITIAL_SNAPSHOT were still
    set in this process from an earlier harbor trial (same worker, same
    shell), an extension in a TB1.0 trial would wrongly read a snapshot as
    real and point the model at a path that was never staged."""
    os.environ["LITTLE_CODER_INITIAL_SNAPSHOT"] = "succeeded"
    try:
        assert _env(_TB)["LITTLE_CODER_INITIAL_SNAPSHOT"] == ""
    finally:
        del os.environ["LITTLE_CODER_INITIAL_SNAPSHOT"]
