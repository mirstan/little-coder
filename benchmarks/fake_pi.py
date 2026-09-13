#!/usr/bin/env python3
"""A scriptable stand-in for the `pi` subprocess, for tests.

The real tests spawn pi and skip when node_modules/.bin/pi is absent, so none of
them exercise the JSONL event loop. EOF/deadline/crash behaviour cannot be
expressed against a real agent anyway: it needs a process that exits on cue.

Mode comes from FAKE_PI_MODE. Reads JSONL requests on stdin, emits JSONL on
stdout, exactly as rpc_client expects.
"""
import base64, json, os, sys, time

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


def _system_prompt_path_from_argv() -> str | None:
    """pi's real argv includes `--system-prompt <path>` whenever
    rpc_client._build_system_prompt() resolved one (rpc_client.py:205) --
    fake_pi.py is launched with the exact same argv via
    LITTLE_CODER_PI_BIN_OVERRIDE, so this is how a test can see what
    candidate text an agent invocation actually received."""
    for i, arg in enumerate(sys.argv):
        if arg == "--system-prompt" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def _write_solution_files():
    """FAKE_PI_WRITE_FILES: JSON {"relative/path.py": "base64 content", ...}.
    Writes each, relative to os.getcwd() (the exercise workdir -- PiRpc is
    constructed with cwd=str(work), aider_polyglot.py:671-686), and emits a
    realistic tool_execution_start/end pair per file so tool_calls looks real."""
    raw = os.environ.get("FAKE_PI_WRITE_FILES")
    if not raw:
        return
    files = json.loads(raw)
    for i, (rel_path, content_b64) in enumerate(files.items()):
        content = base64.b64decode(content_b64).decode("utf-8")
        target = os.path.join(os.getcwd(), rel_path)
        emit({"type": "tool_execution_start", "toolCallId": f"w{i}", "toolName": "write",
              "args": {"path": rel_path}})
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)  # matches the real pi write tool
        with open(target, "w") as fh:
            fh.write(content)
        emit({"type": "tool_execution_end", "toolCallId": f"w{i}", "toolName": "write",
              "result": {"content": [{"type": "text", "text": "ok"}]}, "isError": False})


def emit_turn_end(usage=TURN_USAGE, stop_reason="stop", error_message=None):
    message = {"usage": usage, "stopReason": stop_reason}
    if error_message is not None:
        message["errorMessage"] = error_message
    emit({"type": "turn_end", "message": message, "toolResults": []})


def emit_text(delta):
    """A turn's assistant text.

    Not decoration: rpc_client reads a turn that produced NO text and NO tool
    calls as the empty-completion error shape, so a fixture standing in for a
    turn that DID work has to actually say something, the way a real one
    would. Fixtures that predate that rule call this purely to stay
    non-empty; the ones deliberately exercising the empty shape don't.
    """
    emit({"type": "message_update",
          "assistantMessageEvent": {"type": "text_delta", "delta": delta}})


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
        emit_text("done")
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
        emit_text("done")
        emit_turn_end()
        emit({"type": "agent_end"})
        time.sleep(3600)
        return

    if mode == "end_then_write":
        emit({"type": "response", "id": rid, "success": True})
        emit_text("done")
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
        emit_text("done")
        emit({"type": "turn_end"})
        emit({"type": "agent_end"})
        emit({"type": "usage_update", "usage": {"tokens": 123}})
        emit({"type": "agent_settled"})
        time.sleep(30)
        return

    if mode == "solve_from_env":
        # Writes FAKE_PI_WRITE_FILES unconditionally, then finishes cleanly.
        # Exits promptly (no sleep(30)) -- these modes back a real
        # subprocess-per-exercise e2e test that runs many times per test
        # session, and a lingering sleep(30) child (harmless since
        # PiRpc.close() kills it regardless, but needless) has no purpose
        # here the way it does for the hang/crash regression fixtures above.
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        _write_solution_files()
        emit({"type": "turn_end"})
        emit({"type": "agent_end"})
        emit({"type": "agent_settled"})
        return

    if mode == "noop_then_solve":
        # Solves only on the SECOND invocation. Attempts are separate
        # PROCESSES -- aider_polyglot.py opens a fresh PiRpc (and therefore a
        # fresh fake_pi.py subprocess) per attempt, deliberately (its own
        # comment: reusing one session across attempts ballooned context and
        # wedged pi) -- so "which attempt is this" must be tracked
        # out-of-process via FAKE_PI_STATE_FILE, not an in-memory counter.
        state_file = os.environ["FAKE_PI_STATE_FILE"]
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        if os.path.exists(state_file):
            _write_solution_files()
        else:
            with open(state_file, "w") as fh:
                fh.write("attempt-1-done\n")
        emit({"type": "turn_end"})
        emit({"type": "agent_end"})
        emit({"type": "agent_settled"})
        return

    if mode == "fail_then_report_lesson_then_solve":
        # Attempt 1: fails (writes nothing). Attempt 2: emits a LESSON: line
        # via text_delta (mirroring a real model following aider_polyglot.py's
        # retry-prompt LESSON: instruction), then solves for real. Same
        # cross-process attempt tracking as noop_then_solve -- see its own
        # comment for why an in-memory counter can't work here.
        state_file = os.environ["FAKE_PI_STATE_FILE"]
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        if os.path.exists(state_file):
            emit({"type": "message_update",
                  "assistantMessageEvent": {"type": "text_delta",
                                             "delta": "LESSON: needed clearer guidance\n"}})
            _write_solution_files()
        else:
            with open(state_file, "w") as fh:
                fh.write("attempt-1-done\n")
        emit({"type": "turn_end"})
        emit({"type": "agent_end"})
        emit({"type": "agent_settled"})
        return

    if mode == "read_system_prompt_echo":
        # Copies the system-prompt file's content to FAKE_PI_ECHO_FILE, so a
        # test can assert a candidate's proposed text actually reached the
        # agent invocation -- the regression test for the exact bug this
        # whole live-eval rewrite exists to fix (the old frozen-data design
        # never let a candidate's text affect anything).
        echo_file = os.environ["FAKE_PI_ECHO_FILE"]
        system_prompt_path = _system_prompt_path_from_argv()
        with open(echo_file, "w") as fh:
            fh.write(open(system_prompt_path).read() if system_prompt_path else "")
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        emit({"type": "turn_end"})
        emit({"type": "agent_end"})
        emit({"type": "agent_settled"})
        return

    if mode == "solve_if_prompt_contains":
        # Solves only if the RECEIVED system prompt contains
        # FAKE_PI_MAGIC_TOKEN -- lets a test feed two candidates that differ
        # only in instruction text and assert they score differently,
        # without needing a real model to "decide" based on the text.
        token = os.environ["FAKE_PI_MAGIC_TOKEN"]
        system_prompt_path = _system_prompt_path_from_argv()
        prompt_text = open(system_prompt_path).read() if system_prompt_path else ""
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        if token in prompt_text:
            _write_solution_files()
        emit({"type": "turn_end"})
        emit({"type": "agent_end"})
        emit({"type": "agent_settled"})
        return

    if mode == "emit_non_text_delta":
        # Regression fixture for PromptResult.non_text_deltas (rpc_client.py):
        # emits a message_update whose assistantMessageEvent.type is NOT
        # "text_delta" (a plausible stand-in for a reasoning/thinking-content
        # delta -- NOT a confirmed real pi event name; static analysis of
        # pi's own compiled types could not pin the exact string, see
        # PromptResult.non_text_deltas' own docstring) before the normal
        # text_delta, proving the capture mechanism itself works end to end
        # without depending on knowing pi's real schema.
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        emit({"type": "message_update",
              "assistantMessageEvent": {"type": "thinking_delta", "delta": "reasoning about the problem..."}})
        emit({"type": "message_update",
              "assistantMessageEvent": {"type": "text_delta", "delta": "final answer"}})
        emit({"type": "turn_end"})
        emit({"type": "agent_end"})
        emit({"type": "agent_settled"})
        return

    if mode == "emit_more_than_max_non_text_deltas":
        # Regression fixture for rpc_client.py's _MAX_NON_TEXT_DELTAS
        # backstop: a pathological reasoning stream emitting more than the
        # cap must not grow PromptResult.non_text_deltas past it.
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        count = int(os.environ.get("FAKE_PI_NON_TEXT_DELTA_COUNT", "1"))
        for i in range(count):
            emit({"type": "message_update",
                  "assistantMessageEvent": {"type": "thinking_delta", "delta": f"chunk {i}"}})
        emit({"type": "turn_end"})
        emit({"type": "agent_end"})
        emit({"type": "agent_settled"})
        return

    if mode == "emit_multi_thinking_delta":
        # Regression fixture for live_eval.py's
        # _reasoning_excerpt_from_trajectory(): emits several thinking_delta
        # chunks (like a real reasoning stream, which arrives incrementally,
        # not as one blob -- see the real 162-chunk trace this fixture is
        # modeled on) interleaved with a tool call and text_delta content,
        # then writes the real solution so the exercise actually passes.
        # Proves both that the chunks get concatenated IN ORDER and that
        # reasoning content never leaks into transcript_excerpt (which must
        # only ever accumulate text_delta).
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        for chunk in ("Let me read the stub", " and the test file first.", " Now I understand the task."):
            emit({"type": "message_update", "assistantMessageEvent": {"type": "thinking_delta", "delta": chunk}})
        _write_solution_files()
        emit({"type": "message_update",
              "assistantMessageEvent": {"type": "text_delta", "delta": "Implemented and tests pass."}})
        emit({"type": "turn_end"})
        emit({"type": "agent_end"})
        emit({"type": "agent_settled"})
        return

    if mode == "emit_tool_error_then_solve":
        # Regression fixture for live_eval.py's summarized_transcript
        # (summarize_for_reflection): a recoverable tool failure mid-run,
        # followed by a real solve -- proves the error still reaches
        # reflection even though the attempt ultimately passes.
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        emit({"type": "tool_execution_start", "toolCallId": "e1", "toolName": "bash",
              "args": {"command": "chmod +x ./run.sh"}})
        emit({"type": "tool_execution_end", "toolCallId": "e1", "toolName": "bash",
              "result": {"content": [{"type": "text", "text": "chmod: run.sh: Permission denied"}]},
              "isError": True})
        _write_solution_files()
        emit({"type": "turn_end"})
        emit({"type": "agent_end"})
        emit({"type": "agent_settled"})
        return

    if mode == "emit_compactions_then_solve":
        # Regression fixture for aider_polyglot.py's compaction_total
        # accumulator and live_eval.py's compaction-aware scoring: emits two
        # compaction_end events (rpc_client.py's PromptResult.compaction_events
        # increments on this exact event type), then solves for real.
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        emit({"type": "compaction_end"})
        emit({"type": "compaction_end"})
        _write_solution_files()
        emit({"type": "turn_end"})
        emit({"type": "agent_end"})
        emit({"type": "agent_settled"})
        return

    if mode == "sleep_forever":
        # Like hang_after_ack, but the sleep duration is configurable so a
        # deadline/timeout test doesn't have to wait out a hardcoded 3600s.
        sleep_s = float(os.environ.get("FAKE_PI_SLEEP_S", "3600"))
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        time.sleep(sleep_s)
        return

    if mode == "retry_then_settled":
        # An auto_retry_end between agent_end and
        # agent_settled must not be mistaken for "a continuation started"
        # (which would re-arm the full remaining timeout).
        emit({"type": "response", "id": rid, "success": True})
        emit_text("done")
        emit({"type": "turn_end"})
        emit({"type": "agent_end"})
        emit({"type": "auto_retry_end", "success": True, "attempt": 1})
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
        emit_text("done")
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
        emit_text("done")
        emit({"type": "turn_end"})
        emit({"type": "agent_end"})
        while True:
            emit({"type": "queue_update", "queued": 1})
            time.sleep(0.05)

    if mode == "error_end":
        # The shape three of three error-truncated TB2.1 trials actually hit
        # (confirmed from their tb-finalize-guard notifications:
        # "turn_end stopReason=error hasText=false hasToolCalls=false"). pi
        # reports the provider error, settles, and the whole session is over
        # with most of the wall clock unspent. willRetry:false -- pi is NOT
        # handling this one itself, so it is ours to retry.
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        emit_turn_end(stop_reason="error", error_message="upstream 500 from provider")
        emit({"type": "agent_end", "willRetry": False})
        emit({"type": "agent_settled"})
        serve_requests()
        return

    if mode == "error_empty_completion":
        # Same silent ending, but nothing on the wire says "error": the turn
        # simply produced no text and no tool calls. Some providers report
        # this as a normal stop with an empty body, so the empty-completion
        # fingerprint has to catch it without an errorMessage to key on.
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        emit_turn_end(stop_reason="stop")
        emit({"type": "agent_end", "willRetry": False})
        emit({"type": "agent_settled"})
        serve_requests()
        return

    if mode == "error_end_then_exit":
        # error_end, except pi EXITS instead of settling. The whole reason
        # this fix exists: the agent_end puts the drain in SETTLING, so the
        # settle branch of the stop_reason derivation wins on ordering even
        # though the process is already gone, and the errored turn used to
        # be reported as a retryable "error" against a dead session.
        # Deliberately no emit_text(): a fixture that says something dodges
        # nothing here, but this shape is the measured one.
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        emit_turn_end(stop_reason="error", error_message="upstream 500 from provider")
        emit({"type": "agent_end", "willRetry": False})
        os._exit(0)

    if mode == "empty_end_then_exit":
        # Same race reached through the other synthesis path: a CLEAN
        # turn_end (stopReason "stop") that produced no text and no tool
        # calls, then exit. Nothing on the wire says "error" -- only the
        # empty-completion fingerprint fires, and it must defer to the dead
        # process exactly as the flagged path does.
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        emit_turn_end(stop_reason="stop")
        emit({"type": "agent_end", "willRetry": False})
        os._exit(0)

    if mode == "error_then_clean_same_call":
        # TWO agent_end events in ONE prompt_and_collect call: the first turn
        # errors, then a genuine continuation (agent_start -> ... ->
        # agent_end) completes cleanly. Pins last-agent-end-wins: a verdict
        # latched at the first agent_end would report "error" for a call that
        # demonstrably finished fine. No willRetry involved anywhere here, so
        # this isolates the ordering rule from the willRetry suppression that
        # error_but_will_retry covers.
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        emit_turn_end(stop_reason="error", error_message="transient provider error")
        emit({"type": "agent_end", "willRetry": False})
        emit({"type": "agent_start"})
        emit_text("recovered answer")
        emit_turn_end(stop_reason="stop")
        emit({"type": "agent_end", "willRetry": False})
        emit({"type": "agent_settled"})
        serve_requests()
        return

    if mode == "error_but_will_retry":
        # pi's own internal retry: the errored agent_end carries
        # willRetry:true, and pi immediately re-runs the turn successfully in
        # the SAME call. Counting this as our error too would have the caller
        # retry a turn pi was already retrying.
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        emit_turn_end(stop_reason="error", error_message="transient provider error")
        emit({"type": "agent_end", "willRetry": True})
        emit({"type": "auto_retry_start", "attempt": 1, "maxAttempts": 3,
              "delayMs": 0, "errorMessage": "transient provider error"})
        emit({"type": "agent_start"})
        emit({"type": "auto_retry_end", "success": True, "attempt": 1})
        emit_text("recovered answer")
        emit_turn_end(stop_reason="stop")
        emit({"type": "agent_end", "willRetry": False})
        emit({"type": "agent_settled"})
        serve_requests()
        return

    if mode == "error_then_continuation_hangs":
        # An errored turn, then a continuation that never finishes. The call
        # runs out the outer timeout, so stop_reason must be "deadline" --
        # "error" refines the agent_end branch only and must never displace a
        # loop-exit reason that says the budget itself ran out.
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        emit_turn_end(stop_reason="error", error_message="upstream 500 from provider")
        emit({"type": "agent_end", "willRetry": False})
        emit({"type": "agent_start"})
        time.sleep(3600)
        return

    if mode == "error_then_clean":
        # Two SEPARATE prompt_and_collect calls on the same session, the way
        # the adapters' retry loop drives it: prompt 1 ends in a provider
        # error, the re-prompt then completes normally.
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        emit_turn_end(stop_reason="error", error_message="upstream 500 from provider")
        emit({"type": "agent_end", "willRetry": False})
        emit({"type": "agent_settled"})
        msg2 = read_prompt()
        if msg2 is None:
            return
        rid2 = msg2.get("id")
        emit({"type": "response", "id": rid2, "success": True})
        emit({"type": "agent_start"})
        emit_text("second answer")
        emit_turn_end(stop_reason="stop")
        emit({"type": "agent_end", "willRetry": False})
        emit({"type": "agent_settled"})
        serve_requests()
        return

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
