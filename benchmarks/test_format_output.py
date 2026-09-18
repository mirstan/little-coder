"""_format_output()'s byte cap, in both adapters.

The bug this pins: output was capped by line count only, so a `grep` hit on a
single-line 1MB file was one "line", passed through whole, and blew the
model's context window mid-trial.

Both adapters are loaded with `harbor` / `terminal_bench` stubbed rather than
importorskip'd (the convention in test_harbor_adapter_timeout.py and friends):
neither package is installed next to these tests, and skipping would leave the
formatter -- which touches neither package -- untested everywhere it runs.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARKS))

from rpc_client import preview_tool_result  # noqa: E402
from syntax_error_report import iter_results  # noqa: E402


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
    spec = importlib.util.spec_from_file_location(f"_lca_{adapter}", path)
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
    # The adapters keep their own references, so the stubs have done their job.
    # Leaving them in sys.modules would turn the other benchmark tests'
    # `pytest.importorskip("harbor")` into a false positive.
    for _name in reversed(_STUBBED_MODULES):
        sys.modules.pop(_name, None)


class _Adapter:
    """One calling convention over the two adapters' differing signatures."""

    def __init__(self, name, mod, fmt, backend_note):
        self.name = name
        self.mod = mod
        self.fmt = fmt
        self.backend_note = backend_note


ADAPTERS = [
    _Adapter(
        "harbor",
        _HARBOR,
        lambda raw, code=0, cwd="/tmp", timed_out=False: _HARBOR._format_output(
            raw, "", code, cwd, timed_out
        ),
        "backend=harbor-env",
    ),
    _Adapter(
        "tb",
        _TB,
        lambda raw, code=0, cwd="/tmp", timed_out=False: _TB._format_output(
            raw, code, cwd, timed_out, "backend=tmux-proxy"
        ),
        "backend=tmux-proxy",
    ),
]

pytestmark = pytest.mark.parametrize("ad", ADAPTERS, ids=lambda a: a.name)

GIANT_LEN = 1042304  # the real vocab.json grep result that crashed a trial


# ── capBytesHeadTail parity with the TypeScript implementation ────────────
# The same vectors are asserted in .pi/extensions/shell-session/helpers.test.ts.
# There is no shared code across the three implementations, only this spec.


def test_parity_two_byte_characters(ad):
    text, dropped = ad.mod._cap_bytes_head_tail("é" * 30, 11, 11)
    assert text == "é" * 5 + "\n  [... 40B truncated ...]\n" + "é" * 5
    assert dropped == 40


def test_parity_three_byte_characters(ad):
    text, dropped = ad.mod._cap_bytes_head_tail("日" * 20, 11, 11)
    assert text == "日" * 3 + "\n  [... 42B truncated ...]\n" + "日" * 3
    assert dropped == 42


def test_cuts_at_last_newline_in_head_window(ad):
    # A first-newline rule would cut after "a" and throw away the budget.
    text, _ = ad.mod._cap_bytes_head_tail("a\nbb\n" + "c" * 40, 16, 8)
    assert text.startswith("a\nbb\n  [... ")


def test_cuts_at_first_newline_in_tail_window(ad):
    # A last-newline rule would keep only "ee".
    text, _ = ad.mod._cap_bytes_head_tail("x" * 40 + "\ndd\nee", 8, 12)
    assert text.endswith(" truncated ...]\ndd\nee")


def test_passes_through_content_within_budget(ad):
    assert ad.mod._cap_bytes_head_tail("hello", 10, 10) == ("hello", 0)


# ── _format_output ────────────────────────────────────────────────────────


def test_caps_a_single_giant_line(ad):
    line = "x" * GIANT_LEN
    out = ad.fmt(line)

    assert len(out.encode()) < 64 * 1024
    assert "truncated ...]" in out
    assert "output_truncated=true" in out
    assert f"raw_bytes={GIANT_LEN}" in out

    # The guard against reaching for pi's truncateHead, which returns EMPTY
    # content for exactly this input.
    head = out.split("\n")[0]
    assert len(head) == 32 * 1024
    assert line.startswith(head)


def test_caps_many_short_lines_by_line_count(ad):
    big = "\n".join(str(i).zfill(39) for i in range(300000))
    out = ad.fmt(big)

    assert "lines truncated" in out
    assert "output_truncated=true" in out
    assert len(out.encode()) <= 48 * 1024
    assert len(out.split("\n")) <= ad.mod.MAX_LINES


def test_byte_capped_true_only_when_the_byte_cap_actually_fired(ad):
    """byte_capped=true is distinct from output_truncated=true, which also
    fires for the plain 200-line cap -- the harbor overflow-capture path
    gates on the narrower flag specifically so a long-but-small pip/pytest
    log (common) doesn't cost a docker-cp round trip that a genuinely
    byte-capped giant line (rare) should."""
    if ad.name != "harbor":
        pytest.skip("harbor-only: byte_capped=true only exists in the harbor formatter")

    giant = ad.fmt("x" * GIANT_LEN)
    assert "output_truncated=true" in giant
    assert "byte_capped=true" in giant

    # 300 short lines (a few KB total) trips only the 200-line cap -- nowhere
    # near either byte-cap stage (384KB raw pre-cap, 48KB body cap), unlike
    # test_caps_many_short_lines_by_line_count's 300,000-line/~12MB input,
    # which is large enough in aggregate to trip the raw pre-cap too.
    small_many_lines = ad.fmt("\n".join(f"line {i}" for i in range(300)))
    assert "output_truncated=true" in small_many_lines
    assert "byte_capped=true" not in small_many_lines


def test_dedup_still_rescues_identical_lines(ad):
    out = ad.fmt("same\n" * 100000)
    assert "duplicate line(s) collapsed" in out
    assert len(out.encode()) < 1024


def test_never_cuts_a_multibyte_line_mid_character(ad):
    line = "é" * 600000
    out = ad.fmt(line)
    assert "�" not in out
    assert line.startswith(out.split("\n")[0])


def test_normal_output_is_unchanged(ad):
    out = ad.fmt("hello\nworld\n", cwd="/tmp")
    assert out == f"hello\nworld\n\n[exit=0 cwd=/tmp timed_out=false {ad.backend_note}]"


def test_harbor_stderr_giant_line_is_capped(ad):
    """The [stderr] concatenation happens before cleaning, so a huge stderr
    has to be capped by the same path. Only harbor takes stderr separately;
    tb gets the same shape already concatenated."""
    if ad.name == "harbor":
        out = _HARBOR._format_output("", "e" * GIANT_LEN, 1, "/app", False)
    else:
        out = ad.fmt("\n[stderr]\n" + "e" * GIANT_LEN, code=1)
    assert len(out.encode()) < 64 * 1024
    assert "output_truncated=true" in out
    assert "[exit=1 " in out.split("\n")[-1]


def test_byte_capped_output_never_claims_an_overflow_file(ad):
    """_format_output itself never promises a file: it is pure and
    synchronous, and the harbor proxy splices the "Full output:" line in
    afterwards only once the upload has actually succeeded."""
    assert "Full output:" not in ad.fmt("x" * GIANT_LEN)


# ── _compose_raw, the seam overflow capture reconstructs content through ──


def test_compose_raw_matches_the_composition_it_was_extracted_from(ad):
    """Pins the behavior-preserving extraction: _compose_raw must reproduce
    the stdout/stderr concatenation _format_output used to inline, including
    the falsy-stderr and None cases."""
    if ad.name != "harbor":
        pytest.skip("harbor-only: tb_adapter's _format_output takes already-composed raw")
    cr = _HARBOR._compose_raw
    assert cr("out", "err") == "out\n[stderr]\nerr"
    assert cr("out", "") == "out"
    assert cr("", "err") == "\n[stderr]\nerr"
    assert cr(None, None) == ""
    assert cr("out", None) == "out"


def test_cleaned_output_matches_format_outputs_own_pipeline(ad, monkeypatch):
    """_format_output and the overflow-capture path both call the same
    _cleaned_output(stdout, stderr) helper now, specifically so the two
    computations cannot silently drift apart the way two independent
    recomputations of "_strip_ansi(_compose_raw(...)).replace(...)" could --
    this pins that _format_output actually routes through the shared helper
    rather than re-inlining the expression.

    _format_output's internal `cleaned` is observed where it is handed to
    _cap_bytes_head_tail -- the first thing downstream of the composition.
    """
    if ad.name != "harbor":
        pytest.skip("harbor-only: _cleaned_output is the harbor proxy's seam")
    stdout, stderr = "é" * 600000, "\x1b[31mé\r\né" * 10
    seen: list[str] = []
    real_cap = _HARBOR._cap_bytes_head_tail

    def spy(s, head_bytes, tail_bytes):
        seen.append(s)
        return real_cap(s, head_bytes, tail_bytes)

    monkeypatch.setattr(_HARBOR, "_cap_bytes_head_tail", spy)
    _HARBOR._format_output(stdout, stderr, 0, "/app", False)
    monkeypatch.undo()

    assert seen, "_format_output no longer routes cleaned through _cap_bytes_head_tail"
    assert seen[0] == _HARBOR._cleaned_output(stdout, stderr)


# ── downstream footer consumers ───────────────────────────────────────────


def test_extract_exit_code_still_parses_a_capped_footer(ad):
    # _extract_exit_code is harbor's alone; running it on tb-shaped output
    # would only re-test harbor's parser and imply tb has one too.
    if ad.name != "harbor":
        pytest.skip("harbor-only: tb has no _extract_exit_code")
    out = ad.fmt("x" * GIANT_LEN, code=7)
    assert _HARBOR._extract_exit_code(out) == 7


def test_extract_exit_code_binds_to_the_real_trailing_footer(ad):
    """A mid-line byte cut can create a line that merely looks like a footer."""
    if ad.name != "harbor":
        pytest.skip("harbor-only: tb has no _extract_exit_code")
    out = ad.fmt("[exit=99 cwd=/fake timed_out=false]\nreal output", code=0)
    assert _HARBOR._extract_exit_code(out) == 0


def test_preview_tool_result_keeps_the_capped_footer(ad):
    out = ad.fmt("x" * GIANT_LEN, code=3)
    preview = preview_tool_result(out)
    assert preview.split("\n")[-1] == out.split("\n")[-1]
    assert "raw_bytes=" in preview


def test_syntax_error_report_still_finds_the_result_body(ad):
    out = ad.fmt("SyntaxError: bad\n" + "x" * GIANT_LEN, code=1)
    log = f">> ShellSession({{'command': 'python3 x.py'}})\n<< {out}\n"
    bodies = list(iter_results(log))
    assert len(bodies) == 1
    assert "SyntaxError" in bodies[0]
