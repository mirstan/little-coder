"""The bounded retry both adapters wrap prompt_and_collect() in.

The policy itself lives in rpc_client.prompt_with_error_retry (shared, so the
Harbor and TB adapters cannot drift apart on it) and is driven here against a
stubbed rpc and an injected clock -- no harbor, no terminal_bench, no real
seconds. test_rpc_terminal.py covers the same loop against a real fake-pi
subprocess; this file covers the policy's edges, which a scripted subprocess
cannot express cheaply.

The same file also covers prompt_with_mid_run_compaction, the outer loop
that calls that policy once per compaction cycle -- driven the same way,
against a stub that replays scripted events so the trigger can fire.

The adapters' own wiring (call the helpers, copy n_error_retries /
error_message / n_deliberate_compactions into the metadata they already
build) is guarded textually at the bottom, because importing either adapter
module needs a benchmark harness package that isn't installed alongside
these tests.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rpc_client  # noqa: E402
from rpc_client import (  # noqa: E402
    COMPACTION_CONTINUE_PROMPT,
    ERROR_RETRY_PROMPT,
    PromptResult,
    prompt_with_error_retry,
    prompt_with_mid_run_compaction,
)


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
    rather than vacuous. A scripted entry that is an Exception is raised
    instead of returned, and `alive` is what the helper's pre-retry
    liveness check sees.

    `busy_polls` scripts the pre-retry idle wait: that many get_state reads
    report pi still streaming before one reports it idle. The default 0
    keeps every pre-existing test's wait a single no-op read, so their
    budget assertions still measure only the attempts and backoffs.
    """

    def __init__(self, results, clock=None, consume_sec=0.0, alive=True,
                 busy_polls=0, state_error=None):
        self._results = list(results)
        self._clock = clock
        self._consume_sec = consume_sec
        self.alive = alive
        self._busy_polls = busy_polls
        self._state_error = state_error
        self.calls = []  # [(message, timeout)]
        self.state_calls = 0

    def prompt_and_collect(self, message, timeout=900, on_event=None):
        self.calls.append((message, timeout))
        if self._clock is not None:
            self._clock.t += self._consume_sec
        assert self._results, "prompt_and_collect called more often than scripted"
        nxt = self._results.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    def is_alive(self):
        return self.alive

    def get_state(self):
        self.state_calls += 1
        if self._state_error is not None:
            raise self._state_error
        busy = self.state_calls <= self._busy_polls
        return {"isStreaming": busy, "isCompacting": False}


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


def test_the_first_attempt_is_clamped_to_the_deadline_too():
    """Not just the retries: the adapters take their deadline before their
    own setup work, so `timeout` can already overshoot it before a retry is
    even in question."""
    clock = _Clock()
    rpc = _StubRpc([_ok()], clock)
    _run(rpc, clock, timeout=3600.0, deadline=clock.now() + 900.0)
    assert rpc.calls[0][1] == pytest.approx(900.0)


def test_a_first_attempt_already_past_the_deadline_gets_no_budget():
    """The clamp floors at zero rather than handing pi a negative timeout."""
    clock = _Clock()
    rpc = _StubRpc([_ok()], clock)
    _run(rpc, clock, timeout=3600.0, deadline=clock.now() - 10.0)
    assert rpc.calls[0][1] == 0.0


# ── a pi that went away ──────────────────────────────────────────────────


def test_a_dead_pi_is_not_re_prompted():
    """rpc_client's own derivation catches a death visible when the result
    was built; this catches one in the gap between attempts. Re-prompting
    then raises PiProcessExited out of the trial."""
    clock = _Clock()
    lines = []
    rpc = _StubRpc([_err()], clock, alive=False)
    outcome = _run(rpc, clock, log=lines.append)
    assert len(rpc.calls) == 1
    assert outcome.n_error_retries == 0
    assert outcome.error_message == "upstream 500"
    assert "pi process is gone" in "\n".join(lines)


def test_an_exception_from_a_retry_keeps_what_earlier_attempts_produced():
    """A retry is extra credit: the tool calls and tokens the attempts that
    DID finish already spent are worth more to the caller than the
    exception."""
    clock = _Clock()
    first = PromptResult(
        stop_reason="error", error_message="upstream 500", turn_count=3,
        tool_calls=[{"name": "read"}], assistant_text="partial work",
    )
    rpc = _StubRpc([first, rpc_client.PiProcessExited("pi died")], clock)
    outcome = _run(rpc, clock)
    assert len(rpc.calls) == 2, "the retry was issued and blew up"
    assert outcome.result.turn_count == 3
    assert outcome.result.tool_calls == [{"name": "read"}]
    assert outcome.result.assistant_text == "partial work"
    assert outcome.error_message == "upstream 500"


def test_a_swallowed_retry_exception_is_still_recorded():
    """Caught instead of raised is not the same as never happened: a rejected
    prompt or a dead pipe is a HARNESS fault, and `error_message` only ever
    carries the provider's verdict."""
    clock = _Clock()
    rpc = _StubRpc([_err(), rpc_client.PiProcessExited("pi died")], clock)
    outcome = _run(rpc, clock)
    assert "PiProcessExited" in outcome.retry_exception
    assert "pi died" in outcome.retry_exception
    assert outcome.error_message == "upstream 500", "the provider error is untouched"


def test_a_retry_that_left_pi_dead_no_longer_reports_a_retryable_error():
    """"error" is the value the whole helper treats as retryable, so leaving
    it on a session that is provably gone is the same contradiction the
    stop_reason derivation re-checks liveness to avoid."""
    clock = _Clock()
    rpc = _StubRpc([_err(), rpc_client.PiProcessExited("pi died")], clock,
                   alive=True)
    original_prompt = rpc.prompt_and_collect

    def _die(*a, **kw):
        # Alive at the pre-retry liveness check, dead by the time the retry
        # itself blows up -- the window the check cannot cover.
        try:
            return original_prompt(*a, **kw)
        except Exception:
            rpc.alive = False
            raise

    rpc.prompt_and_collect = _die
    outcome = _run(rpc, clock)
    assert len(rpc.calls) == 2
    assert outcome.result.stop_reason == "process_exit"
    assert outcome.result.error_message == ""


def test_a_retry_exception_with_pi_still_alive_keeps_the_error_verdict():
    """The correction above is scoped to a dead pi: a retry that raised for
    some other reason leaves a live session whose last verdict really was a
    provider error."""
    clock = _Clock()
    rpc = _StubRpc([_err(), RuntimeError("pi rejected prompt: busy")], clock,
                   alive=True)
    outcome = _run(rpc, clock)
    assert outcome.result.stop_reason == "error"
    assert "pi rejected prompt" in outcome.retry_exception


def test_an_exception_from_the_first_attempt_still_propagates():
    """Wrapping a call site in this helper must not swallow a failure the
    bare prompt_and_collect() it replaces would have raised."""
    clock = _Clock()
    rpc = _StubRpc([rpc_client.PiProcessExited("pi died")], clock)
    with pytest.raises(rpc_client.PiProcessExited):
        _run(rpc, clock)


# ── one result across every attempt ──────────────────────────────────────


def test_the_result_is_merged_across_attempts():
    """Returning only the last attempt's result silently discards every tool
    call, turn and token the earlier ones spent -- which the trial was
    charged for either way."""
    clock = _Clock()
    first = PromptResult(
        stop_reason="error", error_message="upstream 500", turn_count=2,
        tool_calls=[{"name": "read"}], assistant_text="first",
        compaction_events=1,
        usage={"input": 100, "output": 10, "cache_read": 0, "cache_write": 0,
               "cost": 0.5},
    )
    second = PromptResult(
        stop_reason="agent_end", turn_count=3, agent_ended=True, settled=True,
        tool_calls=[{"name": "write"}], assistant_text="second",
        compaction_events=2,
        usage={"input": 40, "output": 5, "cache_read": 0, "cache_write": 0,
               "cost": 0.25},
    )
    merged = _run(_StubRpc([first, second], clock), clock).result
    assert merged.turn_count == 5
    assert [t["name"] for t in merged.tool_calls] == ["read", "write"]
    assert merged.compaction_events == 3
    assert merged.usage["input"] == 140
    assert merged.usage["output"] == 15
    assert merged.usage["cost"] == pytest.approx(0.75)
    assert merged.assistant_text == "first\nsecond"
    assert merged.agent_ended is True


def test_the_merged_verdict_comes_from_the_last_attempt_alone():
    """stop_reason / error_message / settled say how the CALL ended; summing
    or latching them would report a recovered call as still failing."""
    clock = _Clock()
    merged = _run(_StubRpc([_err("upstream 500"), _ok()], clock), clock).result
    assert merged.stop_reason == "agent_end"
    assert merged.error_message == ""


def test_a_call_that_never_retried_returns_that_attempt_untouched():
    """The no-retry path must stay byte-identical to a bare
    prompt_and_collect(), not hand back a reconstructed copy."""
    clock = _Clock()
    only = _ok()
    assert _run(_StubRpc([only], clock), clock).result is only


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


# ── a pi that stayed busy ────────────────────────────────────────────────


def _busy(message="pi stayed busy across 5 readiness attempts"):
    return rpc_client.PiBusyError(message)


def test_a_busy_rejection_is_retried_rather_than_ending_the_trial():
    """The shape of the real TB2.0 train-fasttext failure: the retry landed
    while pi was mid-compaction, and the blanket exception handler then
    ended the trial on the first of its two available retries."""
    clock = _Clock()
    rpc = _StubRpc([_err(), _busy(), _ok()], clock)
    outcome = _run(rpc, clock)
    assert len(rpc.calls) == 3
    assert rpc.calls[2][0] == ERROR_RETRY_PROMPT
    assert outcome.result.stop_reason == "agent_end"


def test_a_recovered_busy_rejection_is_still_recorded():
    """Same reason the recovered provider error is: a trial that spent an
    attempt on a busy pi and then succeeded would otherwise leave no trace."""
    clock = _Clock()
    rpc = _StubRpc([_err(), _busy("mid-compaction"), _ok()], clock)
    outcome = _run(rpc, clock)
    assert "PiBusyError" in outcome.retry_exception
    assert "mid-compaction" in outcome.retry_exception
    assert outcome.error_message == "upstream 500", "the provider error is untouched"


def test_a_busy_rejected_attempt_still_counts_toward_the_cap():
    """Retrying a busy pi is bounded by exactly the same max_attempts as
    retrying an errored one -- an unbounded wait-and-resend loop is a worse
    failure than ending the trial."""
    clock = _Clock()
    rpc = _StubRpc([_err(), _busy(), _busy()], clock)
    outcome = _run(rpc, clock)
    assert len(rpc.calls) == 3
    assert "PiBusyError" in outcome.retry_exception


def test_a_busy_rejection_does_not_feed_the_identical_error_streak():
    """A busy rejection has no provider error text to compare, and resolves
    by waiting rather than by giving up -- folding it into the streak would
    fast-fail the loop on the one condition retrying actually fixes."""
    clock = _Clock()
    rpc = _StubRpc([_err("same"), _busy(), _busy(), _ok()], clock)
    outcome = _run(rpc, clock, max_attempts=4)
    assert len(rpc.calls) == 4
    assert outcome.result.stop_reason == "agent_end"


def test_a_busy_rejection_still_respects_the_budget_floor():
    clock = _Clock()
    lines = []
    # Two attempts at 1790s each plus the 5s backoff leaves 15s, under the
    # floor -- and the busy branch has to check it for itself, since it
    # never reaches the loop's own pre-backoff check.
    rpc = _StubRpc([_err(), _busy()], clock, consume_sec=1790.0)
    outcome = _run(rpc, clock, log=lines.append)
    assert len(rpc.calls) == 2
    assert "below the 60s floor" in "\n".join(lines)
    assert "PiBusyError" in outcome.retry_exception


def test_a_busy_rejection_from_the_first_attempt_still_propagates():
    """The first attempt's exceptions are the caller's, same as any other
    type -- only a RETRY is turned into a merged return."""
    clock = _Clock()
    rpc = _StubRpc([_busy()], clock)
    with pytest.raises(rpc_client.PiBusyError):
        _run(rpc, clock)


# ── waiting for pi to go idle ────────────────────────────────────────────


def test_a_retry_waits_for_pi_to_go_idle_before_re_prompting():
    """Re-sending into a compaction window is what produced the busy
    rejection in the first place; the wait is what makes the retry land."""
    clock = _Clock()
    rpc = _StubRpc([_err(), _ok()], clock, busy_polls=3)
    _run(rpc, clock)
    # The 5s backoff plus three 10s polls before the fourth read saw idle.
    assert rpc.calls[1][1] == pytest.approx(3600.0 - 5.0 - 30.0)


def test_the_idle_wait_stops_at_the_budget_floor_not_only_the_flat_cap():
    """Waiting is only worth doing while there is still a trial left to
    spend the budget on."""
    clock = _Clock()
    rpc = _StubRpc([], clock, busy_polls=10_000)
    start = clock.now()
    idle = rpc_client.wait_for_pi_idle(
        rpc, deadline=clock.now() + 300.0, now=clock.now, sleep=clock.sleep
    )
    assert idle is False
    # 300s of budget minus the 60s floor, far short of the 1800s cap.
    assert clock.now() - start == pytest.approx(240.0)


def test_the_idle_wait_gives_up_at_the_flat_cap_rather_than_hanging():
    """A pi wedged in isCompacting against a long deadline would otherwise
    burn the whole remaining budget on a wait that never ends."""
    clock = _Clock()
    rpc = _StubRpc([], clock, busy_polls=10_000)
    start = clock.now()
    idle = rpc_client.wait_for_pi_idle(
        rpc, deadline=clock.now() + 100_000.0, now=clock.now, sleep=clock.sleep
    )
    assert idle is False
    assert clock.now() - start == pytest.approx(rpc_client.PI_IDLE_WAIT_CAP_SEC)


def test_an_idle_pi_is_not_waited_on_at_all():
    clock = _Clock()
    rpc = _StubRpc([], clock)
    start = clock.now()
    assert rpc_client.wait_for_pi_idle(
        rpc, deadline=clock.now() + 3600.0, now=clock.now, sleep=clock.sleep
    ) is True
    assert clock.now() == start


def test_an_unreadable_state_keeps_waiting_rather_than_giving_up():
    """A state read that times out most often means pi is busy enough not to
    answer it; treating "unknown" as "idle" would send straight back into
    the window the wait exists to avoid."""
    clock = _Clock()
    lines = []
    rpc = _StubRpc([], clock, state_error=TimeoutError("no response"))
    idle = rpc_client.wait_for_pi_idle(
        rpc, deadline=clock.now() + 300.0, now=clock.now, sleep=clock.sleep,
        log=lines.append,
    )
    assert idle is False
    assert clock.now() - 1000.0 == pytest.approx(240.0)
    assert sum("could not read pi state" in line for line in lines) == 1, \
        "the unreadable state should be logged once, not once per poll"


def test_a_dead_pi_is_not_waited_on():
    clock = _Clock()
    rpc = _StubRpc([], clock, alive=False, busy_polls=10_000)
    start = clock.now()
    assert rpc_client.wait_for_pi_idle(
        rpc, deadline=clock.now() + 3600.0, now=clock.now, sleep=clock.sleep
    ) is False
    assert clock.now() == start


# ── deliberate mid-run compaction ────────────────────────────────────────


def _turn(tokens):
    """A turn_end carrying `tokens` of context, in pi's own usage shape."""
    return {"type": "turn_end", "message": {"usage": {
        "input": tokens, "output": 0, "cacheRead": 0, "cacheWrite": 0,
    }}}


class _CycleRpc:
    """Replays (events, result) pairs, one pair per prompt_and_collect.

    The events are handed to `on_event` exactly as the real drain does, which
    is what lets the compaction trigger fire from inside a cycle. Compact
    round-trips are scripted separately: each await_compact() consumes one
    entry, returning a dict or raising an Exception.

    A cycle entry that is itself an Exception is raised instead of replayed,
    the way _StubRpc does it -- that is how a prompt rejected outright (a
    busy pi) is expressed, since it produces no events and no result.
    """

    def __init__(self, cycles, clock=None, compact_results=(), alive=True):
        self._cycles = list(cycles)
        self._clock = clock
        self._compact_results = list(compact_results)
        self.alive = alive
        self.calls = []            # [(message, timeout)]
        self.compact_requests = []
        self.awaited = []

    def prompt_and_collect(self, message, timeout=900, on_event=None):
        self.calls.append((message, timeout))
        assert self._cycles, "prompt_and_collect called more often than scripted"
        nxt = self._cycles.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        events, result = nxt
        for ev in events:
            if on_event is not None:
                on_event(ev)
        return result

    def is_alive(self):
        return self.alive

    def get_state(self):
        return {"isStreaming": False, "isCompacting": False}

    def request_compact(self):
        rid = f"compact-{len(self.compact_requests)}"
        self.compact_requests.append(rid)
        return rid

    def await_compact(self, rid, timeout=600):
        self.awaited.append(rid)
        assert self._compact_results, "await_compact called more often than scripted"
        nxt = self._compact_results.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _run_compaction(rpc, clock, timeout=3600.0, context_window=262_144, **kw):
    return prompt_with_mid_run_compaction(
        rpc, "original prompt", timeout,
        context_window=context_window, sleep=clock.sleep, now=clock.now, **kw,
    )


def test_a_quiet_run_never_compacts_and_reads_as_a_bare_retry():
    clock = _Clock()
    rpc = _CycleRpc([([_turn(10_000)], _ok())], clock)
    outcome = _run_compaction(rpc, clock)
    assert rpc.compact_requests == []
    assert len(rpc.calls) == 1
    assert outcome.n_deliberate_compactions == 0
    assert outcome.result.stop_reason == "agent_end"


def test_no_context_window_leaves_the_trigger_permanently_disarmed():
    """The adapter's probe is best-effort, so "we don't know the window" has
    to degrade to plain prompt_with_error_retry rather than guessing one."""
    clock = _Clock()
    rpc = _CycleRpc([([_turn(500_000)], _ok())], clock)
    outcome = _run_compaction(rpc, clock, context_window=None)
    assert rpc.compact_requests == []
    assert outcome.n_deliberate_compactions == 0


@pytest.mark.parametrize("window", ["262144", 0, -1, True])
def test_a_malformed_context_window_disarms_rather_than_raising(window):
    """contextWindow is wire data from a pi build we don't control.

    The probe that reads it is best-effort precisely so a bad answer costs
    the compaction mechanism and nothing else; a TypeError here would
    instead end the trial before the first prompt was ever sent.
    """
    clock = _Clock()
    rpc = _CycleRpc([([_turn(500_000)], _ok())], clock)
    outcome = _run_compaction(rpc, clock, context_window=window)
    assert rpc.compact_requests == []
    assert outcome.n_deliberate_compactions == 0


def test_crossing_the_threshold_compacts_and_continues_the_same_session():
    """The whole point: the run pi aborted for the compaction is relabelled
    rather than reported as the agent finishing, and the trial continues."""
    clock = _Clock()
    rpc = _CycleRpc(
        [([_turn(230_000)], _ok()),
         ([_turn(60_000)], PromptResult(stop_reason="agent_end",
                                        assistant_text="finished"))],
        clock,
        compact_results=[{"tokensBefore": 230_000, "estimatedTokensAfter": 60_000}],
    )
    outcome = _run_compaction(rpc, clock)
    assert len(rpc.compact_requests) == 1
    assert rpc.awaited == rpc.compact_requests, "the compact response is awaited"
    assert len(rpc.calls) == 2
    assert rpc.calls[1][0] == COMPACTION_CONTINUE_PROMPT
    assert outcome.n_deliberate_compactions == 1
    # The LAST cycle's real verdict, not the intermediate "compacted" label.
    assert outcome.result.stop_reason == "agent_end"
    assert "finished" in outcome.result.assistant_text


def test_the_threshold_follows_a_smaller_context_window():
    """min(flat trigger, 84% of the window): a model with a window under the
    flat trigger would otherwise never compact before overflowing."""
    clock = _Clock()
    rpc = _CycleRpc(
        [([_turn(90_000)], _ok()), ([], _ok())],
        clock,
        compact_results=[{"estimatedTokensAfter": 20_000}],
    )
    _run_compaction(rpc, clock, context_window=100_000)
    assert len(rpc.compact_requests) == 1


def test_only_one_compact_request_is_in_flight_at_a_time():
    """A second request would have pi abort the run it started for the
    first."""
    clock = _Clock()
    rpc = _CycleRpc(
        [([_turn(230_000), _turn(240_000), _turn(250_000)], _ok()), ([], _ok())],
        clock,
        compact_results=[{"estimatedTokensAfter": 20_000}],
    )
    _run_compaction(rpc, clock)
    assert len(rpc.compact_requests) == 1


def test_a_compaction_that_regained_too_little_disarms_the_trigger():
    """Compacting again would spend another summarization proving the same
    thing: whatever fills the transcript is not what compaction removes."""
    clock = _Clock()
    rpc = _CycleRpc(
        [([_turn(230_000)], _ok()), ([_turn(240_000)], _ok())],
        clock,
        compact_results=[{"estimatedTokensAfter": 215_000}],
    )
    outcome = _run_compaction(rpc, clock)
    assert len(rpc.compact_requests) == 1, "the second crossing must not re-trigger"
    assert len(rpc.calls) == 2
    assert outcome.n_deliberate_compactions == 1


def test_the_regain_floor_follows_the_smaller_window_too():
    """The threshold moves with the window, so the regain check has to.

    Measured against the flat 220k trigger, a 100k-window trial that
    compacted 84k down to 83k would read as having regained 137k, re-arm,
    and compact again on the very next turn -- a summarization per turn for
    the rest of the trial.
    """
    clock = _Clock()
    rpc = _CycleRpc(
        [([_turn(90_000)], _ok()), ([_turn(95_000)], _ok())],
        clock,
        compact_results=[{"estimatedTokensAfter": 83_000}],
    )
    outcome = _run_compaction(rpc, clock, context_window=100_000)
    assert len(rpc.compact_requests) == 1, "the second crossing must not re-trigger"
    assert outcome.n_deliberate_compactions == 1


def test_a_missing_estimate_is_treated_as_too_little_regained():
    """pi marks estimatedTokensAfter optional; unmeasurable headroom is not
    evidence of headroom."""
    clock = _Clock()
    rpc = _CycleRpc(
        [([_turn(230_000)], _ok()), ([_turn(240_000)], _ok())],
        clock,
        compact_results=[{"tokensBefore": 230_000}],
    )
    _run_compaction(rpc, clock)
    assert len(rpc.compact_requests) == 1


def test_the_regain_floor_stays_satisfiable_on_a_small_window():
    """The floor is clamped to half the threshold, so it can always be met.

    A flat 30k floor measured against a 32 768-token model's 27 525 threshold
    asks for a negative context, so EVERY compaction reads as having regained
    too little and the trigger goes inert after firing once -- on six of the
    nine models in models.json, and on precisely the ones that overflow
    soonest. Here the compaction frees 93% of the window; it has to re-arm.
    """
    clock = _Clock()
    rpc = _CycleRpc(
        [([_turn(30_000)], _ok()), ([_turn(30_000)], _ok()), ([], _ok())],
        clock,
        compact_results=[{"tokensBefore": 30_000, "estimatedTokensAfter": 2_000},
                         {"tokensBefore": 30_000, "estimatedTokensAfter": 2_000}],
    )
    _run_compaction(rpc, clock, context_window=32_768)
    assert len(rpc.compact_requests) == 2, "the second crossing must re-trigger"


def test_a_big_window_keeps_the_flat_regain_floor():
    """The clamp binds only below ~71k; the tuned big-window bound is the
    flat 30k, and a compaction that left 215k of a 220k threshold still has
    to disarm."""
    clock = _Clock()
    rpc = _CycleRpc(
        [([_turn(230_000)], _ok()), ([_turn(240_000)], _ok())],
        clock,
        compact_results=[{"estimatedTokensAfter": 215_000}],
    )
    _run_compaction(rpc, clock, context_window=262_144)
    assert len(rpc.compact_requests) == 1


def test_a_busy_continuation_keeps_the_cycles_already_completed():
    """prompt_with_error_retry leaves its first attempt unguarded on purpose,
    but every post-compaction continuation is also an "attempt 1".

    A busy pi there used to raise straight out of the helper, ending the
    trial and discarding the compacted cycle's own text, tool calls and
    tokens -- the blanket-handler failure this whole helper exists to stop.
    """
    clock = _Clock()
    rpc = _CycleRpc(
        [([_turn(230_000)], PromptResult(stop_reason="agent_end",
                                         assistant_text="work done")),
         rpc_client.PiBusyError("pi stayed busy across 5 readiness attempts")],
        clock,
        compact_results=[{"estimatedTokensAfter": 20_000}],
    )
    outcome = _run_compaction(rpc, clock, timeout=36_000.0)
    assert "work done" in outcome.result.assistant_text, "the cycle's work survives"
    assert outcome.n_deliberate_compactions == 1
    assert "PiBusyError" in outcome.retry_exception


def test_a_busy_first_prompt_still_propagates_through_the_wrapper():
    """The trial's own first prompt keeps the bare prompt_and_collect
    contract: the wrapper must not swallow what a direct call would raise."""
    clock = _Clock()
    rpc = _CycleRpc(
        [rpc_client.PiBusyError("pi stayed busy across 5 readiness attempts")],
        clock,
    )
    with pytest.raises(rpc_client.PiBusyError):
        _run_compaction(rpc, clock)


def test_the_deliberate_compaction_cap_is_enforced():
    clock = _Clock()
    rpc = _CycleRpc(
        [([_turn(230_000)], _ok())] * 3,
        clock,
        compact_results=[{"estimatedTokensAfter": 20_000}] * 3,
    )
    lines = []
    outcome = _run_compaction(rpc, clock, max_compactions=2, log=lines.append)
    assert len(rpc.compact_requests) == 2
    assert len(rpc.calls) == 2
    assert outcome.n_deliberate_compactions == 2
    assert "2 deliberate compactions used" in "\n".join(lines)


@pytest.mark.parametrize("failure", [
    TimeoutError("compact never answered"),
    RuntimeError("pi rejected compact: Nothing to compact (session too small)"),
])
def test_a_failed_compaction_still_gets_its_continuation(failure):
    """session.compact() aborts the active run BEFORE any of these failures
    can happen, so the session is sitting idle mid-task either way -- ending
    the trial here would throw away the budget the abort just freed."""
    clock = _Clock()
    rpc = _CycleRpc(
        [([_turn(230_000)], _ok()),
         ([], PromptResult(stop_reason="agent_end", assistant_text="finished"))],
        clock,
        compact_results=[failure],
    )
    outcome = _run_compaction(rpc, clock)
    assert len(rpc.calls) == 2
    assert rpc.calls[1][0] == COMPACTION_CONTINUE_PROMPT
    assert outcome.result.stop_reason == "agent_end"
    assert outcome.n_deliberate_compactions == 1


def test_a_failed_compaction_disarms_further_attempts():
    clock = _Clock()
    rpc = _CycleRpc(
        [([_turn(230_000)], _ok()), ([_turn(240_000)], _ok())],
        clock,
        compact_results=[RuntimeError("pi rejected compact: Already compacted")],
    )
    _run_compaction(rpc, clock)
    assert len(rpc.compact_requests) == 1


@pytest.mark.parametrize("stop_reason", ["deadline", "process_exit", "error"])
def test_a_genuinely_terminal_cycle_is_not_continued(stop_reason):
    """"compacted" is only ever an agent_end we relabelled. These three are
    terminal conditions of their own and have nothing to do with the abort
    we asked for, so the loop must not paper over them with a continuation."""
    clock = _Clock()
    # Repeated because "error" is the one value the INNER loop retries on its
    # own; it exhausts its own brakes first and only then hands the verdict
    # out here, which is the case this is about.
    cycle = ([_turn(230_000)], PromptResult(stop_reason=stop_reason,
                                            error_message="upstream 500"))
    rpc = _CycleRpc([cycle] * 3, clock)
    outcome = _run_compaction(rpc, clock)
    assert rpc.awaited == [], "no point awaiting a compaction nothing will use"
    assert COMPACTION_CONTINUE_PROMPT not in [c[0] for c in rpc.calls]
    assert outcome.result.stop_reason == stop_reason


def test_results_are_merged_across_compaction_cycles():
    """A cycle the harness cut short still spent the trial's tokens and
    wrote the trial's files."""
    clock = _Clock()
    first = PromptResult(
        stop_reason="agent_end", turn_count=2, tool_calls=[{"name": "read"}],
        assistant_text="before",
        usage={"input": 100, "output": 10, "cache_read": 0, "cache_write": 0,
               "cost": 0.5},
    )
    second = PromptResult(
        stop_reason="agent_end", turn_count=3, tool_calls=[{"name": "write"}],
        assistant_text="after",
        usage={"input": 40, "output": 5, "cache_read": 0, "cache_write": 0,
               "cost": 0.25},
    )
    rpc = _CycleRpc(
        [([_turn(230_000)], first), ([], second)],
        clock,
        compact_results=[{"estimatedTokensAfter": 20_000}],
    )
    merged = _run_compaction(rpc, clock).result
    assert merged.turn_count == 5
    assert [t["name"] for t in merged.tool_calls] == ["read", "write"]
    assert merged.usage["input"] == 140
    assert merged.assistant_text == "before\nafter"


def test_error_retries_are_summed_across_cycles():
    """Each cycle runs its own inner retry loop; reporting only the last
    one's count would understate what the trial actually spent."""
    clock = _Clock()
    rpc = _CycleRpc(
        [([_turn(230_000)], _err("upstream 500")), ([_turn(1_000)], _ok()),
         ([], _err("upstream 503")), ([], _ok())],
        clock,
        compact_results=[{"estimatedTokensAfter": 20_000}],
    )
    outcome = _run_compaction(rpc, clock)
    assert outcome.n_error_retries == 2
    assert outcome.error_message == "upstream 503"


def test_the_continuation_prompt_leaves_a_finished_agent_an_exit():
    """Same escape hatch ERROR_RETRY_PROMPT carries, for the same reason:
    the nudge has to be able to lose."""
    assert "already complete" in COMPACTION_CONTINUE_PROMPT
    assert "stop" in COMPACTION_CONTINUE_PROMPT


def test_a_dead_pi_is_not_continued_after_a_compaction():
    clock = _Clock()
    rpc = _CycleRpc(
        [([_turn(230_000)], _ok())],
        clock,
        compact_results=[{"estimatedTokensAfter": 20_000}],
        alive=False,
    )
    lines = []
    outcome = _run_compaction(rpc, clock, log=lines.append)
    assert len(rpc.calls) == 1
    assert "pi process is gone" in "\n".join(lines)
    assert outcome.n_deliberate_compactions == 1


def test_a_spent_budget_is_not_continued_after_a_compaction():
    clock = _Clock()
    rpc = _CycleRpc(
        [([_turn(230_000)], _ok())],
        clock,
        compact_results=[{"estimatedTokensAfter": 20_000}],
    )
    lines = []
    outcome = _run_compaction(rpc, clock, timeout=30.0, log=lines.append)
    assert len(rpc.calls) == 1
    assert "below the 60s floor" in "\n".join(lines)
    assert outcome.result.stop_reason == "compacted"


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
    assert "prompt_with_mid_run_compaction" in source
    # And the wrapper INSTEAD of the helper underneath it, not alongside it:
    # an adapter reverting to prompt_with_error_retry would still retry
    # provider errors and still keep `rpc.prompt_and_collect(` absent, so
    # every other assertion here would pass while the trial silently lost
    # the compaction boundary this wiring exists to supply.
    assert "prompt_with_error_retry" not in source
    assert "rpc.prompt_and_collect(" not in source, "should go through the retry helper"
    assert "n_error_retries" in source
    assert "n_deliberate_compactions" in source
    assert "preview_tool_result(" in source
    assert "[:400]" not in source, "raw slice should be gone from the log previews"
