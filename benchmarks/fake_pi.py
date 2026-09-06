#!/usr/bin/env python3
"""A scriptable stand-in for the `pi` subprocess, for tests.

The real tests spawn pi and skip when node_modules/.bin/pi is absent, so none of
them exercise the JSONL event loop. EOF/deadline/crash behaviour cannot be
expressed against a real agent anyway: it needs a process that exits on cue.

Mode comes from FAKE_PI_MODE. Reads JSONL requests on stdin, emits JSONL on
stdout, exactly as rpc_client expects.
"""
import base64, json, os, sys, time


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
        emit({"type": "turn_end"})
        emit({"type": "agent_end"})
        os._exit(0)

    if mode == "end_then_write":
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "turn_end"})
        emit({"type": "agent_end"})
        deadline = time.time() + 2
        i = 0
        while time.time() < deadline:
            with open(os.path.join(os.getcwd(), "late_write.txt"), "a") as fh:
                fh.write(f"{i}\n")
            i += 1
            time.sleep(0.2)
        emit({"type": "agent_settled"})
        time.sleep(30)
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
        emit({"type": "turn_end"})
        emit({"type": "agent_end"})
        emit({"type": "agent_settled"})
        time.sleep(30)
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
        emit({"type": "turn_end"})
        emit({"type": "agent_end"})
        emit({"type": "agent_settled"})
        time.sleep(30)
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
        emit({"type": "turn_end"})
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
        emit({"type": "turn_end"})
        emit({"type": "agent_end"})
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

    if mode == "sleep_forever":
        # Like hang_after_ack, but the sleep duration is configurable so a
        # deadline/timeout test doesn't have to wait out a hardcoded 3600s.
        sleep_s = float(os.environ.get("FAKE_PI_SLEEP_S", "3600"))
        emit({"type": "response", "id": rid, "success": True})
        emit({"type": "agent_start"})
        time.sleep(sleep_s)
        return

    # default: clean single turn with one tool call
    emit({"type": "response", "id": rid, "success": True})
    emit({"type": "agent_start"})
    emit({"type": "message_update",
          "assistantMessageEvent": {"type": "text_delta", "delta": "hello"}})
    emit({"type": "tool_execution_start", "toolCallId": "t1", "toolName": "read", "args": {"path": "x"}})
    emit({"type": "tool_execution_end", "toolCallId": "t1", "toolName": "read",
          "result": {"content": [{"type": "text", "text": "ok"}]}, "isError": False})
    emit({"type": "turn_end"})
    emit({"type": "agent_end"})
    emit({"type": "agent_settled"})
    time.sleep(30)


if __name__ == "__main__":
    main()
