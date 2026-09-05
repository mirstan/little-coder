"""JSON-RPC client for little-coder.

Spawns `pi --mode rpc --no-session` as a subprocess and speaks JSONL.
Provides:
  - prompt_and_collect(): send a user prompt, drain events, return
    assistant text + tool-call summary
  - extension_ui hook: override __LC_TB_SHELL__ input requests to proxy
    shell commands back to a caller-supplied handler (used by the
    Terminal-Bench adapter to route commands through TmuxSession)

Usage:
    from benchmarks.rpc_client import PiRpc
    rpc = PiRpc(model="llamacpp/qwen3.6-35b-a3b", cwd="/tmp/exercise")
    result = rpc.prompt_and_collect("Implement solve() in solution.py")
    rpc.close()
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).parent.parent
# LITTLE_CODER_PI_BIN_OVERRIDE: integration-testing hook only. Lets a
# host-side harness (e.g. `tb run`, which runs pi on the host per
# tb_adapter's own comment) point PI_BIN at a stand-in (fake_pi.py) without
# modifying any tracked file. Unset -> byte-identical to prior behavior.
#
# `or`, not os.environ.get(name, default): .get() returns "" (not the
# default) when the var is exported EMPTY -- a real risk from a harness
# script doing `export LITTLE_CODER_PI_BIN_OVERRIDE="$SOME_UNSET_VAR"`.
# Path("") == Path("."), whose .exists() is True, silently defeating the
# "pi not found" FileNotFoundError check below and surfacing later as an
# opaque Popen error instead (confirmed by review).
PI_BIN = Path(
    os.environ.get("LITTLE_CODER_PI_BIN_OVERRIDE")
    or str(REPO_ROOT / "node_modules" / ".bin" / "pi")
)
TB_SHELL_PREFIX = "__LC_TB_SHELL__:"


def _extension_paths() -> list[str]:
    """Enumerate absolute paths to every extension in the repo.

    Pi's auto-discovery only scans `<cwd>/.pi/extensions/` — when pi's cwd
    is an exercise directory (not the repo), our extensions would be
    invisible. Passing each via `-e <absolute_path>` makes them portable
    across any cwd.
    """
    ext_dir = REPO_ROOT / ".pi" / "extensions"
    if not ext_dir.is_dir():
        return []
    paths = []
    for child in sorted(ext_dir.iterdir()):
        if child.is_dir() and (child / "index.ts").exists():
            paths.append(str(child / "index.ts"))
    return paths


def _build_system_prompt() -> Path:
    """Resolve the file passed to pi's --system-prompt flag.

    If PRINCIPLES.md exists alongside AGENTS.md, concatenate them into a
    generated file (gitignored, rewritten on every call so edits to either
    source file are always picked up) and point at that instead. If
    PRINCIPLES.md is absent, behavior is unchanged: point straight at
    AGENTS.md, matching pre-existing behavior exactly.
    """
    agents_md = REPO_ROOT / "AGENTS.md"
    principles_md = REPO_ROOT / "PRINCIPLES.md"
    # Real bug, confirmed by review: the pre-existing "AGENTS.md missing ->
    # degrade gracefully" guard (the caller checks system_prompt_path.exists()
    # after this returns) was bypassed on THIS path -- with PRINCIPLES.md
    # present but AGENTS.md absent, the read_text() below raised an
    # uncaught FileNotFoundError inside PiRpc.__init__, killing the whole
    # benchmark run instead of falling back the same way the no-PRINCIPLES
    # path already does.
    if not principles_md.exists() or not agents_md.exists():
        return agents_md

    generated = REPO_ROOT / ".pi" / ".system-prompt.generated.md"
    content = agents_md.read_text() + "\n\n# Principles\n\n" + principles_md.read_text()
    try:
        generated.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write via tmp-file + rename: a plain write_text() truncates
        # the shared file in place first, so a PiRpc constructed
        # concurrently with another (parallel benchmark attempts, or a
        # before/after comparison run) could read a corrupted, half-written
        # system prompt -- real gap, confirmed by review. Falls back to
        # AGENTS.md alone, same as the no-PRINCIPLES.md path above, if the
        # write itself fails (e.g. a read-only .pi/) rather than raising an
        # uncaught OSError out of PiRpc.__init__.
        tmp = generated.with_name(f"{generated.name}.tmp-{os.getpid()}-{threading.get_ident()}")
        tmp.write_text(content)
        tmp.replace(generated)
    except OSError:
        return agents_md
    return generated


class PiProcessExited(RuntimeError):
    """pi exited before completing the request. Carries its stderr tail."""


@dataclass
class PromptResult:
    """Outcome of a single prompt_and_collect() call."""
    assistant_text: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # [{name, args, result_text, is_error}]
    agent_ended: bool = False
    compaction_events: int = 0
    turn_count: int = 0
    #: Why the call returned: "agent_end" (pi finished the turn), "deadline"
    #: (budget expired), or "process_exit" (pi died mid-run). Callers must not
    #: infer this from elapsed time -- a crash burns the full budget too,
    #: because stdout EOF used not to wake the drain.
    stop_reason: str = "agent_end"


class PiRpc:
    """One pi subprocess, reused across prompts within a session.

    The RPC client is single-threaded on the caller side: call
    prompt_and_collect() sequentially. A background reader thread
    demultiplexes events / responses / extension_ui_request messages.
    """

    def __init__(
        self,
        model: str,
        cwd: Optional[str] = None,
        *,
        benchmark: Optional[str] = None,
        allowed_tools: Optional[list[str]] = None,
        session_id: Optional[str] = None,
        tb_mode: bool = False,
        env: Optional[dict] = None,
        max_turns: Optional[int] = None,
        thinking: Optional[str] = None,
        tb_shell_handler: Optional[Callable[[dict], str]] = None,
    ):
        if not PI_BIN.exists():
            raise FileNotFoundError(f"pi CLI not found at {PI_BIN}. Run `npm install` in {REPO_ROOT}.")

        self._tb_shell_handler = tb_shell_handler

        full_env = dict(os.environ)
        if env:
            full_env.update(env)
        # Required api-key envs (pi requires SOMETHING even for local providers)
        full_env.setdefault("LLAMACPP_API_KEY", "noop")
        full_env.setdefault("OLLAMA_API_KEY", "noop")
        if benchmark:
            full_env["LITTLE_CODER_BENCHMARK"] = benchmark
        if allowed_tools:
            full_env["LITTLE_CODER_ALLOWED_TOOLS"] = ",".join(allowed_tools)
        if session_id:
            full_env["LITTLE_CODER_SESSION_ID"] = session_id
        if tb_mode:
            full_env["LITTLE_CODER_TB_MODE"] = "1"
        if max_turns:
            full_env["LITTLE_CODER_MAX_TURNS"] = str(max_turns)

        cmd = [str(PI_BIN), "--mode", "rpc", "--no-session", "--model", model]
        if thinking:
            cmd.extend(["--thinking", thinking])
        for ext in _extension_paths():
            cmd.extend(["-e", ext])
        # Pi's --tools flag filters the tool SCHEMAS presented to the model
        # (not just execution). This is the schema-level equivalent of
        # agent.py's _filtered_schemas(); tool-gating extension still
        # handles execution-level blocking for defense in depth.
        if allowed_tools:
            cmd.extend(["--tools", ",".join(allowed_tools)])
        # Use AGENTS.md (plus PRINCIPLES.md, if present) as THE system prompt,
        # not as appended Project Context. Pi's --system-prompt resolves an
        # existing path to file content (resource-loader.js::resolvePromptInput).
        # --no-context-files prevents AGENTS.md from also being auto-discovered
        # and double-appended under `# Project Context`. Effect: pi's hardcoded
        # "You are an expert coding assistant operating inside pi…" identity and
        # the "Pi documentation" block both go away; AGENTS.md (+ PRINCIPLES.md)
        # alone defines the agent.
        system_prompt_path = _build_system_prompt()
        if system_prompt_path.exists():
            cmd.extend(["--no-context-files", "--system-prompt", str(system_prompt_path)])
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd or str(REPO_ROOT),
            env=full_env,
            text=True,
            bufsize=1,  # line-buffered
        )

        # Demultiplexer state
        self._responses: dict[str, dict] = {}
        self._event_q: list[dict] = []
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._closed = False
        #: Set once pi's stdout reaches EOF, i.e. the process is going away.
        self._eof = False
        self._stderr_buf: list[str] = []
        # ctx.ui.notify messages from extensions — used by the benchmark
        # harnesses to count skill injections, thinking-budget fires,
        # quality-monitor corrections etc. Each entry: {"message", "notifyType"}.
        self._notifications: list[dict] = []

        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_reader.start()

    # ── Reader threads ───────────────────────────────────────────────────
    def _read_loop(self):
        # Use explicit readline() — `for line in stdout` buffers opaquely
        # and can delay event delivery well past newlines in pi's stream.
        assert self._proc.stdout is not None
        try:
            self._read_loop_body()
        finally:
            # try/finally, not "after the break": if the loop raises (e.g. the
            # pipe is torn down under it) waiters must still be woken, or they
            # block for the full timeout and a crash is indistinguishable from
            # a deadline.
            with self._cv:
                self._eof = True
                self._cv.notify_all()

    def _read_loop_body(self):
        while True:
            line = self._proc.stdout.readline()
            if not line:
                break  # EOF
            line = line.rstrip("\r\n")
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "extension_ui_request":
                # Handle inline — don't block the reader
                self._handle_ui_request(msg)
                continue
            with self._cv:
                if msg.get("type") == "response" and msg.get("id"):
                    # Stamped here, under the same lock and in the same
                    # message-stream position the response itself arrived in,
                    # so a caller can later tell exactly which queued events
                    # existed strictly before this response -- see
                    # prompt_and_collect()'s readiness-retry handling.
                    msg["_event_q_watermark"] = len(self._event_q)
                    self._responses[msg["id"]] = msg
                else:
                    self._event_q.append(msg)
                self._cv.notify_all()

    def _read_stderr(self):
        assert self._proc.stderr is not None
        while True:
            line = self._proc.stderr.readline()
            if not line:
                break
            self._stderr_buf.append(line.rstrip())

    # ── extension_ui_request handling ────────────────────────────────────
    def _handle_ui_request(self, req: dict):
        method = req.get("method")
        rid = req.get("id")
        if method == "input":
            title = req.get("title", "") or ""
            if title.startswith(TB_SHELL_PREFIX) and self._tb_shell_handler is not None:
                try:
                    payload = json.loads(title[len(TB_SHELL_PREFIX):])
                    result = self._tb_shell_handler(payload)
                except Exception as e:
                    result = f"Error in TB shell handler: {e}"
                self._send({"type": "extension_ui_response", "id": rid, "value": result})
                return
            # Unhandled input request — respond with empty string so pi unblocks
            self._send({"type": "extension_ui_response", "id": rid, "value": ""})
        elif method == "confirm":
            # Auto-accept (we're headless)
            self._send({"type": "extension_ui_response", "id": rid, "confirmed": True})
        elif method == "select":
            # Take the first option
            opts = req.get("options", [])
            self._send({
                "type": "extension_ui_response",
                "id": rid,
                "value": opts[0] if opts else "",
            })
        elif method == "editor":
            prefill = req.get("prefill", "") or ""
            self._send({"type": "extension_ui_response", "id": rid, "value": prefill})
        elif method == "notify":
            # Accumulate for the harness to persist (skill injections,
            # thinking-budget fires, quality-monitor corrections, etc.).
            with self._lock:
                self._notifications.append({
                    "message": req.get("message", ""),
                    "notifyType": req.get("notifyType", "info"),
                })
        # setStatus / setWidget / setTitle / set_editor_text are fire-and-forget

    # ── Send / recv ──────────────────────────────────────────────────────
    def _send(self, obj: dict):
        if self._proc.stdin is None or self._proc.stdin.closed:
            return
        try:
            self._proc.stdin.write(json.dumps(obj) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass

    def _await_response(self, rid: str, timeout: float = 600) -> dict:
        start = time.time()
        with self._cv:
            while rid not in self._responses:
                if self._eof:
                    break
                remaining = timeout - (time.time() - start)
                if remaining <= 0:
                    raise TimeoutError(f"pi did not respond to request {rid} within {timeout}s")
                self._cv.wait(timeout=remaining)
            if rid in self._responses:
                return self._responses.pop(rid)
        raise PiProcessExited(
            f"pi exited before acknowledging request {rid}; stderr:\n{self.stderr()}"
        )

    def _drain_events_until(self, predicate, timeout: float) -> list[dict]:
        """Drain events until `predicate(event)` returns True or timeout."""
        start = time.time()
        collected: list[dict] = []
        with self._cv:
            while True:
                while self._event_q:
                    ev = self._event_q.pop(0)
                    collected.append(ev)
                    if predicate(ev):
                        return collected
                if self._eof:
                    return collected      # queue drained above; pi is gone
                remaining = timeout - (time.time() - start)
                if remaining <= 0:
                    return collected
                self._cv.wait(timeout=remaining)

    # ── Public API ───────────────────────────────────────────────────────
    def prompt_and_collect(self, message: str, timeout: float = 900) -> PromptResult:
        """Send a prompt, drain events until agent_end, return summary.

        Retries the SEND (not the whole turn) a few times on "Agent is already
        processing" -- a real, reproducible race under higher --thinking
        effort: pi's agent_end event can fire before pi's internal state has
        actually settled back to idle, so a prompt sent immediately after a
        previous prompt_and_collect() returns can be rejected even though the
        caller correctly waited for agent_end. Not observed at pi's default
        thinking level; higher effort apparently widens whatever internal
        window this races on.
        """
        if self._closed:
            raise RuntimeError("prompt_and_collect() on a closed PiRpc")
        resp = None
        for readiness_attempt in range(5):
            rid = str(uuid.uuid4())
            self._send({"id": rid, "type": "prompt", "message": message})
            resp = self._await_response(rid, timeout=30)
            if resp.get("success"):
                break
            err = str(resp.get("error", ""))
            if "already processing" in err.lower() and readiness_attempt < 4:
                time.sleep(2 * (readiness_attempt + 1))
                continue
            raise RuntimeError(f"pi rejected prompt: {resp.get('error')}")

        # Trim any event still queued from before THIS response was recorded
        # -- not gated on readiness_attempt > 0, because the same corruption
        # doesn't require a rejection at all: PiRpc is documented as "reused
        # across prompts within a session", and pi emitting a stray/duplicate
        # agent_end after a turn's real one leaves it queued regardless of
        # whether the *next* send happened to be accepted outright or needed
        # a retry first. On a clean session this is a no-op: the watermark is
        # 0 when nothing stale is queued.
        #
        # Neither "clear before resending" nor "clear right after the ack"
        # is race-free: the former leaves a gap for pi to enqueue a stale
        # event between the clear and the ack (reproduced: pi can write a
        # turn's agent_end right up until the moment it frees itself to
        # accept the next one), and the latter can instead discard the real
        # turn's own events if the reader thread queues them before this
        # thread wakes from _await_response(). Trimming to the watermark
        # recorded at the exact moment THIS response was stored is race-free
        # in both directions: anything queued before that point in the
        # message stream cannot belong to a turn pi had not yet accepted
        # when it wrote the response, and anything queued at or after it is
        # preserved regardless of scheduling.
        watermark = resp.get("_event_q_watermark", 0)
        with self._cv:
            del self._event_q[:watermark]

        events = self._drain_events_until(
            lambda ev: ev.get("type") == "agent_end",
            timeout=timeout,
        )

        result = PromptResult()
        # Derived from what was observed, not from how long it took: a crash
        # burns the same wall-clock as a deadline.
        saw_agent_end = any(ev.get("type") == "agent_end" for ev in events)
        if saw_agent_end:
            result.stop_reason = "agent_end"
        elif self._eof or self._proc.poll() is not None:
            result.stop_reason = "process_exit"
        else:
            result.stop_reason = "deadline"
        pending: dict[str, dict] = {}
        for ev in events:
            t = ev.get("type")
            if t == "message_update":
                delta = ev.get("assistantMessageEvent", {})
                if delta.get("type") == "text_delta":
                    result.assistant_text += delta.get("delta", "")
            elif t == "tool_execution_start":
                pending[ev.get("toolCallId", "")] = {
                    "name": ev.get("toolName", ""),
                    "args": ev.get("args", {}),
                }
            elif t == "tool_execution_end":
                tcid = ev.get("toolCallId", "")
                entry = pending.pop(tcid, {"name": ev.get("toolName", ""), "args": {}})
                res = ev.get("result", {})
                content = res.get("content", [])
                entry["result_text"] = "\n".join(
                    c.get("text", "") for c in content if c.get("type") == "text"
                )
                entry["is_error"] = bool(ev.get("isError"))
                result.tool_calls.append(entry)
            elif t == "turn_end":
                result.turn_count += 1
            elif t == "compaction_end":
                result.compaction_events += 1
            elif t == "agent_end":
                result.agent_ended = True
        return result

    def new_session(self):
        rid = str(uuid.uuid4())
        self._send({"id": rid, "type": "new_session"})
        self._await_response(rid)

    def _settle_stderr(self, timeout: float = 1.0):
        """Let the existing stderr reader finish once pi is gone.

        Before the EOF fix, callers only reached stderr() after a full timeout,
        by which point the reader had long since drained. Returning promptly
        now races it, so wait briefly for the thread to finish.
        """
        if self._eof or self._proc.poll() is not None:
            reader = getattr(self, "_stderr_reader", None)
            if reader is not None and reader.is_alive():
                reader.join(timeout=timeout)

    def stderr(self) -> str:
        self._settle_stderr()
        return "\n".join(self._stderr_buf)

    def notifications(self) -> list[dict]:
        """Return accumulated ctx.ui.notify events (shallow-copied).

        Each entry: {"message": str, "notifyType": "info"|"warning"|"error"}.
        Used by the benchmark harness to persist per-task extension activity
        (skill / knowledge injections, thinking-budget fires, quality-monitor
        corrections, turn-cap aborts, evidence-compact bridges).
        """
        with self._lock:
            return list(self._notifications)

    def close(self, timeout: float = 5):
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
