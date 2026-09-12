#!/usr/bin/env python3
"""A scriptable stand-in for the `pi` subprocess, for tests.

The real tests spawn pi and skip when node_modules/.bin/pi is absent, so none of
them exercise the JSONL event loop. EOF/deadline/crash behaviour cannot be
expressed against a real agent anyway: it needs a process that exits on cue.

Mode comes from FAKE_PI_MODE. Reads JSONL requests on stdin, emits JSONL on
stdout, exactly as rpc_client expects.
"""
import json, os, sys, time

# Canned get_session_stats data (docs/rpc.md's documented shape) -- distinct
# from any single turn_end's usage so tests can tell the two sources apart
# (session_stats is meant to be the complete, session-cumulative one).
SESSION_STATS_DATA = {
    "sessionFile": None,
    "sessionId": "fake-session",
    "userMessages": 1,
    "assistantMessages": 1,
    "toolCalls": 1,
    "toolResults": 1,
    "totalMessages": 3,
    "tokens": {"input": 500, "output": 100, "cacheRead": 50, "cacheWrite": 10, "total": 660},
    "cost": 0.12,
    "contextUsage": {"tokens": 660, "contextWindow": 200000, "percent": 0.33},
}

# Fixed per-turn usage stamped onto turn_end's message.usage -- small, made
# up numbers, just enough for tests to assert prompt_and_collect() sums them
# correctly across one or more turns.
TURN_USAGE = {"input": 100, "output": 20, "cacheRead": 10, "cacheWrite": 0,
              "cost": {"input": 0.0008, "output": 0.0002, "cacheRead": 0,
                        "cacheWrite": 0, "total": 0.001}}


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def emit_turn_end(usage=TURN_USAGE):
    emit({"type": "turn_end", "message": {"usage": usage}, "toolResults": []})


def read_prompt():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("type") == "prompt":
            return msg
    return None


def serve_requests(timeout=30):
    """After the canned prompt sequence, keep servicing requests -- currently
    just get_session_stats -- until stdin closes (the caller's rpc.close())
    or `timeout` elapses. Replaces a blind time.sleep(): a real pi process
    sits idle between prompts and would answer get_session_stats the same
    way, so this lets hermetic tests exercise PiRpc.session_stats() the same
    as they exercise prompt_and_collect()."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = sys.stdin.readline()
        if not line:
            return  # EOF -- caller closed stdin
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("type") == "get_session_stats":
            emit({"type": "response", "id": msg.get("id"), "command": "get_session_stats",
                  "success": True, "data": SESSION_STATS_DATA})
        elif msg.get("id"):
            emit({"type": "response", "id": msg.get("id"), "success": False,
                  "error": "fake_pi: unexpected request while idle in this mode"})


def main():
    mode = os.environ.get("FAKE_PI_MODE", "clean")

    if mode == "exit_before_ack":
        sys.stderr.write("fake_pi: dying before ack\n")
        sys.stderr.flush()
        os._exit(3)

    msg = read_prompt()
    if msg is None:
        return
    rid = msg.get("id")

    if mode == "crash_after_ack":
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        sys.stderr.write("fake_pi: boom\n")
        sys.stderr.flush()
        os._exit(1)

    if mode == "hang_after_ack":
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        time.sleep(3600)

    if mode == "end_then_exit":
        # agent_end and EOF in the same breath -- the ordering hazard
        emit({"type": "response", "id": rid, "success": True})
        emit_turn_end()
        emit({"type": "agent_end"})
        os._exit(0)

    if mode == "abort_then_followup":
        # Simulates a thinking-budget breach: the extension's ctx.abort()
        # makes pi emit agent_end mid-thought, but the follow-up nudge
        # message it queued survives the abort and pi immediately runs a
        # second, real turn on the same connection -- only settling (no
        # more continuations queued) once that second turn finishes.
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        emit({"type": "message_update",
              "assistantMessageEvent": {"type": "text_delta", "delta": "thinking too long..."}})
        emit({"type": "agent_end"})  # abort mid-thought, no turn_end
        emit({"type": "agent_start"})
        emit({"type": "message_update",
              "assistantMessageEvent": {"type": "text_delta", "delta": "recovered answer"}})
        emit_turn_end()
        emit({"type": "agent_end"})
        emit({"type": "agent_settled"})
        serve_requests()
        return

    if mode == "end_never_settles":
        # agent_end fires but pi never follows up with agent_settled and
        # never exits either -- pins the bounded settle_grace fallback so a
        # caller can't hang forever waiting for a settle signal that will
        # never come (e.g. an older pi build without agent_settled at all).
        emit({"type": "response", "id": rid, "success": True})
        emit_turn_end()
        emit({"type": "agent_end"})
        time.sleep(3600)
        return

    if mode == "end_then_write":
        emit({"type": "response", "id": rid, "success": True})
        emit_turn_end()
        emit({"type": "agent_end"})
        deadline = time.time() + 2
        i = 0
        while time.time() < deadline:
            with open(os.path.join(os.getcwd(), "late_write.txt"), "a") as fh:
                fh.write(f"{i}\n")
            i += 1
            time.sleep(0.2)
        emit({"type": "agent_settled"})
        serve_requests()
        return

    if mode == "busy_then_ready":
        # Simulates the readiness-retry race: the first send is rejected
        # because a previous turn is still winding down, and that previous
        # turn's own leftover agent_end lands in the queue before the
        # retry's real completion does, with the leftover written
        # immediately (before the retry is even sent). Regression fixture
        # for the stale-event-queue bug: without discarding events that
        # predate the retry's own acceptance, this leftover agent_end gets
        # mistaken for the new turn's completion.
        emit({"type": "response", "id": rid, "success": False,
              "error": "Agent is already processing"})
        emit({"type": "agent_end"})  # stale leftover from the "previous" turn
        msg2 = read_prompt()
        if msg2 is None:
            return
        rid2 = msg2.get("id")
        emit({"type": "response", "id": rid2, "success": True})
        emit({"type": "agent_start"})
        emit({"type": "message_update",
              "assistantMessageEvent": {"type": "text_delta", "delta": "real answer"}})
        emit_turn_end()
        emit({"type": "agent_end"})
        emit({"type": "agent_settled"})
        serve_requests()
        return

    if mode == "busy_then_late_stale":
        # Harder variant of busy_then_ready: what makes this harder is
        # ORDER, not delay -- the previous turn's leftover agent_end is
        # written AFTER the retry has already been sent (read_prompt()
        # below has already returned it), whereas busy_then_ready writes it
        # immediately after the rejection, before the retry is even sent. A
        # fix that discards stale events only once, right before resending
        # (rather than by watermarking the moment of the retry's own
        # acceptance), happens to still catch the immediate case but not
        # this one: the leftover lands in the queue after that one-time
        # clear and is never removed.
        emit({"type": "response", "id": rid, "success": False,
              "error": "Agent is already processing"})
        msg2 = read_prompt()  # the retry -- arrives while we're still "busy"
        emit({"type": "agent_end"})  # stale leftover, written after the retry was sent
        if msg2 is None:
            return
        rid2 = msg2.get("id")
        emit({"type": "response", "id": rid2, "success": True})
        emit({"type": "agent_start"})
        emit({"type": "message_update",
              "assistantMessageEvent": {"type": "text_delta", "delta": "real answer"}})
        emit_turn_end()
        emit({"type": "agent_end"})
        emit({"type": "agent_settled"})
        serve_requests()
        return

    if mode == "stray_end_then_clean_reuse":
        # A stray leftover can reach a session that is reused WITHOUT any
        # rejection ever happening -- PiRpc's own docstring advertises reuse
        # "across prompts within a session". Turn 1 completes normally, but
        # pi (in whatever internal state motivates this whole PR) emits a
        # second, stray agent_end right after its real one. The next prompt
        # is then accepted outright, no "already processing" involved. A fix
        # that only trims the queue on the readiness-retry path (gated on a
        # rejection having occurred) never runs here, so the stray agent_end
        # from turn 1 sits at the head of the queue and turn 2's own drain
        # returns on it immediately.
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        emit({"type": "message_update",
              "assistantMessageEvent": {"type": "text_delta", "delta": "first answer"}})
        emit_turn_end()
        emit({"type": "agent_end"})
        emit({"type": "agent_end"})  # stray duplicate, no rejection involved
        msg2 = read_prompt()
        if msg2 is None:
            return
        rid2 = msg2.get("id")
        emit({"type": "response", "id": rid2, "success": True})
        emit({"type": "agent_start"})
        emit({"type": "message_update",
              "assistantMessageEvent": {"type": "text_delta", "delta": "second answer"}})
        emit_turn_end()
        emit({"type": "agent_end"})
        emit({"type": "agent_settled"})
        serve_requests()
        return

    if mode == "mixed_usage":
        # Three turns: one with no "usage" key at all, one with a malformed
        # (non-dict) "usage", and one with valid usage -- pins that
        # prompt_and_collect() tolerates missing/malformed usage on some
        # turns (skipping just that turn's contribution) without raising and
        # crashing the whole trial over token accounting.
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        emit({"type": "message_update",
              "assistantMessageEvent": {"type": "text_delta", "delta": "first"}})
        emit({"type": "turn_end", "message": {}})  # no usage key
        emit({"type": "agent_start"})
        emit({"type": "message_update",
              "assistantMessageEvent": {"type": "text_delta", "delta": "second"}})
        emit({"type": "turn_end", "message": {"usage": "not-a-dict"}})  # malformed
        emit({"type": "agent_start"})
        emit({"type": "message_update",
              "assistantMessageEvent": {"type": "text_delta", "delta": "third"}})
        emit_turn_end()  # valid usage
        emit({"type": "agent_end"})
        emit({"type": "agent_settled"})
        serve_requests()
        return

    if mode == "null_message_usage":
        # Two turns whose "message" field itself is (or contains) JSON null,
        # not merely absent -- pins that turn_end handling uses an
        # isinstance() guard rather than dict.get(key, default), which only
        # substitutes the default when the key is MISSING, never when its
        # value is null. `.get("message", {}).get("usage")` raises
        # AttributeError on a None "message"; this must not crash the trial.
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        emit({"type": "message_update",
              "assistantMessageEvent": {"type": "text_delta", "delta": "first"}})
        emit({"type": "turn_end", "message": None})  # message itself is null
        emit({"type": "agent_start"})
        emit({"type": "message_update",
              "assistantMessageEvent": {"type": "text_delta", "delta": "second"}})
        emit({"type": "turn_end", "message": {"usage": None}})  # usage is null
        emit({"type": "agent_end"})
        emit({"type": "agent_settled"})
        serve_requests()
        return

    if mode == "settled_after_extra_event":
        # agent_settled arrives right after an unrelated event
        # (not immediately after agent_end). The old two-phase drain's
        # Phase 1 predicate only matched agent_end, so this agent_settled
        # was treated as an ordinary event and Phase 1 waited out the
        # entire remaining timeout for an agent_end that would never come.
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "turn_end"})
        emit({"type": "agent_end"})
        emit({"type": "usage_update", "usage": {"tokens": 123}})
        emit({"type": "agent_settled"})
        time.sleep(30)
        return

    if mode == "retry_then_settled":
        # An auto_retry_end between agent_end and
        # agent_settled must not be mistaken for "a continuation started"
        # (which would re-arm the full remaining timeout).
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "turn_end"})
        emit({"type": "agent_end"})
        emit({"type": "auto_retry_end"})
        emit({"type": "agent_settled"})
        time.sleep(30)
        return

    if mode == "abort_then_followup_then_exit":
        # Same sequence as abort_then_followup, but pi exits immediately
        # after agent_settled instead of sleeping -- combined with a slow
        # on_event in the test, this lets the reader thread race ahead and
        # set _eof before the consumer has drained the whole queue. The old
        # "if self._eof: break" guards discarded the queued recovery turn in
        # exactly this race.
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        emit({"type": "message_update",
              "assistantMessageEvent": {"type": "text_delta", "delta": "thinking too long..."}})
        emit({"type": "agent_end"})
        emit({"type": "agent_start"})
        emit({"type": "message_update",
              "assistantMessageEvent": {"type": "text_delta", "delta": "recovered answer"}})
        emit({"type": "turn_end"})
        emit({"type": "agent_end"})
        emit({"type": "agent_settled"})
        os._exit(0)

    if mode == "end_then_stray_then_hang":
        # The "stray must not re-arm" case: a duplicate
        # agent_end while SETTLING must not restart the settle window, and
        # must not be treated as renewed work either. pi then hangs forever
        # without ever emitting agent_settled.
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "turn_end"})
        emit({"type": "agent_end"})
        emit({"type": "agent_end"})  # stray duplicate while SETTLING
        time.sleep(3600)
        return

    if mode == "continuation_never_finishes":
        # A genuine continuation starts (agent_start) but never
        # finishes. The call must run the full outer timeout and report
        # stop_reason="deadline" -- NOT "agent_end", even though an
        # agent_end was seen earlier in this same call.
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "turn_end"})
        emit({"type": "agent_end"})
        emit({"type": "agent_start"})
        time.sleep(3600)
        return

    if mode == "chatty_then_hang":
        # Pins the absolute (not per-event) settle window: a continuous
        # stream of non-terminal events after agent_end must not extend
        # SETTLING past the settle_deadline computed on entry.
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "turn_end"})
        emit({"type": "agent_end"})
        while True:
            emit({"type": "queue_update", "queued": 1})
            time.sleep(0.05)

    # default: clean single turn with one tool call
    emit({"type": "response", "id": rid, "success": True})
    emit({"type": "agent_start"})
    emit({"type": "message_update",
          "assistantMessageEvent": {"type": "text_delta", "delta": "hello"}})
    emit({"type": "tool_execution_start", "toolCallId": "t1", "toolName": "read", "args": {"path": "x"}})
    emit({"type": "tool_execution_end", "toolCallId": "t1", "toolName": "read",
          "result": {"content": [{"type": "text", "text": "ok"}]}, "isError": False})
    emit_turn_end()
    emit({"type": "agent_end"})
    emit({"type": "agent_settled"})
    serve_requests()


if __name__ == "__main__":
    main()
