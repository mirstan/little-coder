"""The bounded retry both adapters wrap prompt_and_collect() in.

The policy itself lives in rpc_client.prompt_with_error_retry (shared, so the
Harbor and TB adapters cannot drift apart on it) and is driven here against a
stubbed rpc and an injected clock -- no harbor, no terminal_bench, no real
seconds. test_rpc_terminal.py covers the same loop against a real fake-pi
subprocess; this file covers the policy's edges, which a scripted subprocess
cannot express cheaply.

The adapters' own two-line wiring (call the helper, copy n_error_retries /
error_message into the metadata they already build) is guarded textually at
the bottom, because importing either adapter module needs a benchmark harness
package that isn't installed alongside these tests.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rpc_client  # noqa: E402
from rpc_client import ERROR_RETRY_PROMPT, PromptResult, prompt_with_error_retry  # noqa: E402


class _Clock:
    """Monotonic clock the test advances by hand."""

    def __init__(self, t=1000.0):
        self.t = t

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


class _StubRpc:
    """Replays a scripted list of PromptResults, recording how it was called.

    `consume_sec` makes each attempt cost wall-clock time on the injected
    clock, which is what lets the remaining-budget assertions below be real
    rather than vacuous.
    """

    def __init__(self, results, clock=None, consume_sec=0.0):
        self._results = list(results)
        self._clock = clock
        self._consume_sec = consume_sec
        self.calls = []  # [(message, timeout)]

    def prompt_and_collect(self, message, timeout=900, on_event=None):
        self.calls.append((message, timeout))
        if self._clock is not None:
            self._clock.t += self._consume_sec
        assert self._results, "prompt_and_collect called more often than scripted"
        return self._results.pop(0)


def _err(message="upstream 500"):
    return PromptResult(stop_reason="error", error_message=message)


def _ok():
    return PromptResult(stop_reason="agent_end", assistant_text="done")


def _run(rpc, clock, timeout=3600.0, **kw):
    return prompt_with_error_retry(
        rpc, "original prompt", timeout,
        sleep=clock.sleep, now=clock.now, **kw,
    )


# ── the happy path is untouched ──────────────────────────────────────────


def test_clean_first_attempt_is_not_retried():
    clock = _Clock()
    rpc = _StubRpc([_ok()], clock)
    outcome = _run(rpc, clock)
    assert len(rpc.calls) == 1
    assert outcome.n_error_retries == 0
    assert outcome.error_message == ""
    assert outcome.result.assistant_text == "done"


@pytest.mark.parametrize("stop_reason", ["deadline", "process_exit"])
def test_deadline_and_process_exit_are_not_retried(stop_reason):
    """Neither is retryable: the budget is already gone, or there is no
    session left to prompt."""
    clock = _Clock()
    rpc = _StubRpc([PromptResult(stop_reason=stop_reason)], clock)
    outcome = _run(rpc, clock)
    assert len(rpc.calls) == 1
    assert outcome.n_error_retries == 0


# ── retrying ─────────────────────────────────────────────────────────────


def test_error_is_retried_on_the_same_session_and_recovers():
    clock = _Clock()
    rpc = _StubRpc([_err("upstream 500"), _ok()], clock)
    outcome = _run(rpc, clock)
    assert len(rpc.calls) == 2
    assert rpc.calls[0][0] == "original prompt"
    assert rpc.calls[1][0] == ERROR_RETRY_PROMPT
    assert outcome.n_error_retries == 1
    assert outcome.result.stop_reason == "agent_end"


def test_the_recovered_error_is_still_reported():
    """A trial that recovered must not look like one that never failed --
    n_error_retries with no error text would be unexplainable in result.json."""
    clock = _Clock()
    rpc = _StubRpc([_err("upstream 500"), _ok()], clock)
    outcome = _run(rpc, clock)
    assert outcome.error_message == "upstream 500"


def test_retry_prompt_leaves_a_finished_agent_an_exit():
    """The nudge has to be able to lose: without the "if it's actually done,
    say so" clause it can talk an agent that already finished into undoing
    its own work."""
    assert "already complete" in ERROR_RETRY_PROMPT
    assert "stop" in ERROR_RETRY_PROMPT


def test_retry_cap_is_three_attempts_total():
    clock = _Clock()
    # Distinct messages, so the identical-error fast-fail can't fire first
    # and mask the cap.
    rpc = _StubRpc([_err("err one"), _err("err two"), _err("err three")], clock)
    outcome = _run(rpc, clock)
    assert len(rpc.calls) == 3
    assert outcome.n_error_retries == 2
    assert outcome.result.stop_reason == "error"
    assert outcome.error_message == "err three"


def test_identical_errors_fast_fail_before_the_cap():
    """A non-retryable provider error leaves a poisoned transcript that fails
    identically every time; spending the rest of the budget proving that only
    delays the trial's real ending."""
    clock = _Clock()
    rpc = _StubRpc([_err("same"), _err("same"), _err("same")], clock)
    outcome = _run(rpc, clock)
    assert len(rpc.calls) == 2, "should have stopped before the third attempt"
    assert outcome.n_error_retries == 1


def test_differing_errors_do_not_fast_fail():
    """The streak is CONSECUTIVE-identical, not any-repeat: two different
    failures are evidence of flakiness, which is exactly what retrying is
    for."""
    clock = _Clock()
    rpc = _StubRpc([_err("one"), _err("two"), _ok()], clock)
    outcome = _run(rpc, clock)
    assert len(rpc.calls) == 3
    assert outcome.n_error_retries == 2
    assert outcome.result.stop_reason == "agent_end"


# ── budget ───────────────────────────────────────────────────────────────


def test_retry_is_skipped_when_too_little_budget_remains():
    """A retry with seconds on the clock can't do useful work. Ending the
    trial normally at least preserves what the agent already wrote."""
    clock = _Clock()
    rpc = _StubRpc([_err()], clock)
    outcome = _run(rpc, clock, timeout=30.0)  # floor is 60s
    assert len(rpc.calls) == 1
    assert outcome.n_error_retries == 0
    assert outcome.error_message == "upstream 500"


def test_the_backoff_counts_against_the_floor():
    """Budget is checked against what would be left AFTER the backoff, since
    the backoff spends real trial time too. 63s remaining minus the 5s first
    backoff is 58s, under the 60s floor."""
    clock = _Clock()
    rpc = _StubRpc([_err()], clock)
    outcome = _run(rpc, clock, timeout=63.0)
    assert len(rpc.calls) == 1
    assert outcome.n_error_retries == 0


def test_retry_gets_the_remaining_budget_not_the_original_timeout():
    """Re-passing the full timeout is how a retry loop quietly doubles a
    trial's wall clock past the deadline the harness enforces."""
    clock = _Clock()
    rpc = _StubRpc([_err(), _ok()], clock, consume_sec=1000.0)
    _run(rpc, clock, timeout=3600.0)
    first_timeout = rpc.calls[0][1]
    retry_timeout = rpc.calls[1][1]
    assert first_timeout == 3600.0
    # 3600 spent down by the 1000s attempt and the 5s backoff.
    assert retry_timeout == pytest.approx(3600.0 - 1000.0 - 5.0)


def test_retries_never_outlive_an_externally_supplied_deadline():
    """The adapters pass the trial's own deadline -- the same instant pi was
    handed as LITTLE_CODER_DEADLINE_EPOCH_MS -- rather than letting the loop
    invent one from `timeout`."""
    clock = _Clock()
    rpc = _StubRpc([_err(), _ok()], clock, consume_sec=100.0)
    _run(rpc, clock, timeout=3600.0, deadline=clock.now() + 900.0)
    assert rpc.calls[1][1] == pytest.approx(900.0 - 100.0 - 5.0)


def test_backoff_grows_between_retries():
    clock = _Clock()
    start = clock.now()
    rpc = _StubRpc([_err("one"), _err("two"), _ok()], clock)
    _run(rpc, clock)
    assert clock.now() - start == pytest.approx(sum(rpc_client.ERROR_RETRY_BACKOFF_SEC))


# ── plumbing ─────────────────────────────────────────────────────────────


def test_log_callback_records_each_failed_attempt():
    clock = _Clock()
    lines = []
    rpc = _StubRpc([_err("one"), _ok()], clock)
    _run(rpc, clock, log=lines.append)
    joined = "\n".join(lines)
    assert "one" in joined
    assert "retry 1" in joined


def test_works_without_a_log_callback():
    clock = _Clock()
    rpc = _StubRpc([_err(), _ok()], clock)
    assert _run(rpc, clock).n_error_retries == 1


# ── adapter wiring ───────────────────────────────────────────────────────

_ADAPTERS = [
    Path(__file__).parent / "harbor_adapter" / "little_coder_agent.py",
    Path(__file__).parent / "tb_adapter" / "little_coder_agent.py",
]


@pytest.mark.parametrize("path", _ADAPTERS, ids=lambda p: p.parent.name)
def test_adapter_uses_the_shared_retry_and_records_it(path):
    """Textual, deliberately: importing either adapter pulls in harbor /
    terminal_bench, which aren't installed next to these tests (see
    test_harbor_adapter_timeout.py's importorskip). What this can still catch
    is the wiring silently going away -- an adapter reverting to a bare
    prompt_and_collect, or dropping the retry count from the record it
    writes, both of which would leave every test above passing against code
    nothing calls."""
    source = path.read_text()
    # Name, not call site: the Harbor adapter hands it to asyncio.to_thread
    # rather than calling it directly.
    assert "prompt_with_error_retry" in source
    assert "rpc.prompt_and_collect(" not in source, "should go through the retry helper"
    assert "n_error_retries" in source
    assert "preview_tool_result(" in source
    assert "[:400]" not in source, "raw slice should be gone from the log previews"
