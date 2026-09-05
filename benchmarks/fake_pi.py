#!/usr/bin/env python3
"""A scriptable stand-in for the `pi` subprocess, for tests.

The real tests spawn pi and skip when node_modules/.bin/pi is absent, so none of
them exercise the JSONL event loop. EOF/deadline/crash behaviour cannot be
expressed against a real agent anyway: it needs a process that exits on cue.

Mode comes from FAKE_PI_MODE. Reads JSONL requests on stdin, emits JSONL on
stdout, exactly as rpc_client expects.
"""
import json, os, sys, time


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


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
