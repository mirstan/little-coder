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
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).parent.parent
PI_BIN = REPO_ROOT / "node_modules" / ".bin" / "pi"
TB_SHELL_PREFIX = "__LC_TB_SHELL__:"

# capture_environment_snapshot()'s config sources -- module-level constants
# (not inlined) so tests can monkeypatch each one independently, matching how
# REPO_ROOT/PI_BIN are already overridden in tests.
_PI_SETTINGS_PATH = Path.home() / ".pi" / "agent" / "settings.json"
_LC_MODELS_SHIPPED_DEFAULT = REPO_ROOT / "models.json"
_OMLX_SETTINGS = Path.home() / ".omlx" / "settings.json"
_OMLX_MODEL_SETTINGS = Path.home() / ".omlx" / "model_settings.json"
_VENDOR_PATCH_TARGET = (
    REPO_ROOT / "node_modules" / "@earendil-works" / "pi-coding-agent" / "node_modules"
    / "@earendil-works" / "pi-ai" / "dist" / "api" / "openai-completions.js"
)
_VENDOR_PATCH_MARKER = "PI_REASONING_MAX_TOKENS"
_OMLX_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "repetition_penalty")


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
    #: Token usage summed across every `turn_end` seen during this call
    #: (see prompt_and_collect's aggregation loop). Crash-proof but
    #: incomplete: unlike session_stats(), it does not include tool-result
    #: usage or compaction/branch-summary generation tokens, and it only
    #: covers turns whose `turn_end` actually arrived (see PiRpc.session_stats
    #: docstring for why session_stats is preferred when available).
    usage: dict = field(default_factory=lambda: {
        "input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "cost": 0.0,
    })


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
        self._session_id = session_id

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
        # Use AGENTS.md as THE system prompt, not as appended Project Context.
        # Pi's --system-prompt resolves an existing path to file content
        # (resource-loader.js::resolvePromptInput). --no-context-files prevents
        # AGENTS.md from also being auto-discovered and double-appended under
        # `# Project Context`. Effect: pi's hardcoded "You are an expert coding
        # assistant operating inside pi…" identity and the "Pi documentation"
        # block both go away; AGENTS.md alone defines the agent.
        agents_md = REPO_ROOT / "AGENTS.md"
        if agents_md.exists():
            cmd.extend(["--no-context-files", "--system-prompt", str(agents_md)])
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

    def _drain_events_until(
        self,
        predicate,
        timeout: float,
        on_event: Optional[Callable[[dict], None]] = None,
    ) -> list[dict]:
        """Drain events until `predicate(event)` returns True or timeout.

        `on_event`, if given, is called synchronously for each event the
        instant it's popped off the queue -- i.e. in real time as pi emits
        them, not after the whole call returns. This is what lets a caller
        (e.g. the Harbor adapter) stream a live trajectory log instead of
        only writing a summary once prompt_and_collect finally returns.

        Caller-supplied code (`on_event` and `predicate`) runs OUTSIDE
        self._cv, deliberately: while this thread holds that lock the reader
        thread cannot queue anything, so a callback doing slow file I/O would
        stall event demultiplexing (and with it the tb_shell proxy, which the
        same reader thread services), and a callback that reached back into
        PiRpc -- notifications(), stderr(), another prompt -- would deadlock
        outright on the non-reentrant lock.

        Events are popped one at a time (not batch-snapshotted) specifically
        for exception safety: if on_event/predicate raises, `collected` (and
        with it every event from this call, including ones already handed to
        on_event) is discarded along with the exception -- but everything
        still sitting in self._event_q, including a possible agent_end, is
        untouched and available to the next call. An earlier batch-snapshot
        version cleared the whole queue up front and only requeued the
        unconsumed remainder on the predicate-match path, so a mid-batch
        exception silently dropped even the events that hadn't been through
        on_event yet.
        """
        start = time.time()
        collected: list[dict] = []
        while True:
            # Rechecked every iteration, not just while _event_q is empty --
            # if on_event is slow and the reader keeps appending faster than
            # we drain, _event_q could stay nonempty indefinitely and this
            # loop would never otherwise notice the deadline passed.
            if timeout - (time.time() - start) <= 0:
                return collected
            with self._cv:
                while not self._event_q:
                    if self._eof:
                        return collected      # queue drained above; pi is gone
                    remaining = timeout - (time.time() - start)
                    if remaining <= 0:
                        return collected
                    self._cv.wait(timeout=remaining)
                ev = self._event_q.pop(0)
            collected.append(ev)
            if on_event is not None:
                on_event(ev)
            if predicate(ev):
                return collected

    # ── Public API ───────────────────────────────────────────────────────
    def prompt_and_collect(
        self,
        message: str,
        timeout: float = 900,
        on_event: Optional[Callable[[dict], None]] = None,
        settle_grace: float = 20.0,
    ) -> PromptResult:
        """Send a prompt, drain events through pi's own idle signal, return summary.

        Retries the SEND (not the whole turn) a few times on "Agent is already
        processing" -- a real, reproducible race under higher --thinking
        effort: pi's agent_end event can fire before pi's internal state has
        actually settled back to idle, so a prompt sent immediately after a
        previous prompt_and_collect() returns can be rejected even though the
        caller correctly waited for agent_end. Not observed at pi's default
        thinking level; higher effort apparently widens whatever internal
        window this races on. Returning at agent_settled (see below) instead
        of at the first agent_end largely defuses this race too, since
        agent_settled means pi is genuinely idle -- but the retry stays as
        defense in depth.

        `on_event`: optional callback invoked in real time as each RPC event
        arrives (see _drain_events_until) -- lets a caller stream a live
        trajectory log instead of only seeing the aggregated PromptResult
        once this call finally returns.

        Two-phase drain, because `agent_end` is not always the end of the
        turn: an extension can react to an `agent_end` (e.g. the
        thinking-budget extension aborting a runaway thinking stream) by
        queuing a follow-up message and letting pi continue processing in
        the same run. pi's queued-follow-up continuation (and auto-retry /
        auto-compaction continuations) runs to completion and only then
        emits `agent_settled` -- pi's own purpose-built "nothing left
        queued, truly idle" signal. Draining only to the first `agent_end`
        (the old behaviour) returns before that continuation starts, which
        for a Harbor trial means the caller treats the abort as the whole
        run finishing and tears the process down mid-recovery.

        Phase 1 drains until `agent_end` (or timeout/EOF), exactly as
        before. If an `agent_end` was seen and pi hasn't exited, Phase 2
        drains for up to `min(settle_grace, remaining)` waiting for the
        very next event:
          - `agent_settled` -> pi is genuinely idle; done.
          - anything else -> a continuation turn started; append it and
            loop back to Phase 1 to drain it out too.
          - nothing arrives (grace expires) or EOF -> return anyway. This
            is a defensive fallback for a pi build that never emits
            agent_settled; it logs a warning since it means this function
            is falling back to the old, weaker termination signal.

        All phases append into one `events` list, so tool calls / text /
        turn_count from a continuation turn are aggregated into the same
        PromptResult as the initial turn. `stop_reason`/`agent_ended` stay
        keyed on whether any `agent_end` was seen across all phases, so
        that semantics for existing callers is unchanged.
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

        start = time.time()
        events: list[dict] = []
        saw_agent_end = False
        while True:
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                break
            phase1_events = self._drain_events_until(
                lambda ev: ev.get("type") == "agent_end",
                timeout=remaining,
                on_event=on_event,
            )
            events.extend(phase1_events)
            if not any(ev.get("type") == "agent_end" for ev in phase1_events):
                # Deadline or EOF, no agent_end this phase -- nothing left to
                # settle-wait for.
                break
            saw_agent_end = True

            if self._eof:
                break

            remaining = timeout - (time.time() - start)
            grace = min(settle_grace, remaining)
            if grace <= 0:
                break

            # Phase 2: wait for the very next event, bounded by the settle
            # grace window. `_drain_events_until` with an always-true
            # predicate returns as soon as exactly one event arrives (or
            # empty on timeout/EOF).
            settle_events = self._drain_events_until(
                lambda ev: True,
                timeout=grace,
                on_event=on_event,
            )
            if not settle_events:
                if not self._eof:
                    print(
                        f"WARNING: pi did not emit agent_settled within "
                        f"{settle_grace}s of agent_end (session "
                        f"{self._session_id!r}); falling back to agent_end "
                        f"as the terminal signal for this prompt",
                        file=sys.stderr,
                    )
                break
            events.extend(settle_events)
            if settle_events[-1].get("type") == "agent_settled":
                break
            if self._eof:
                break
            # Renewed activity (e.g. a queued follow-up's own agent_start) --
            # loop back to Phase 1 and drain the continuation turn too.

        result = PromptResult()
        # Derived from what was observed, not from how long it took: a crash
        # burns the same wall-clock as a deadline.
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
                usage = ev.get("message", {}).get("usage")
                if isinstance(usage, dict):
                    # Defensive .get(..., 0) + isinstance checks throughout:
                    # this is untrusted wire data from a pi build we don't
                    # control, and a malformed/missing field here must not
                    # crash a whole Harbor trial over token accounting.
                    for key, src in (
                        ("input", "input"), ("output", "output"),
                        ("cache_read", "cacheRead"), ("cache_write", "cacheWrite"),
                    ):
                        val = usage.get(src, 0)
                        if isinstance(val, (int, float)):
                            result.usage[key] += val
                    cost = usage.get("cost", 0)
                    # Documented shape is {"input", "output", "cacheRead",
                    # "cacheWrite", "total"}; tolerate a flat number too, in
                    # case a future/older pi build ever reports cost bare.
                    if isinstance(cost, dict):
                        total = cost.get("total", 0)
                        if isinstance(total, (int, float)):
                            result.usage["cost"] += total
                    elif isinstance(cost, (int, float)):
                        result.usage["cost"] += cost
            elif t == "compaction_end":
                result.compaction_events += 1
            elif t == "agent_end":
                result.agent_ended = True
        return result

    def new_session(self):
        rid = str(uuid.uuid4())
        self._send({"id": rid, "type": "new_session"})
        self._await_response(rid)

    def get_state(self, timeout: float = 20) -> dict:
        """Query pi's own resolved session state (e.g. thinkingLevel).

        More authoritative than reading pi's settings files ourselves for
        anything -- pi computes this value through its own full resolution
        chain (CLI flag, scoped-model overrides, machine-local defaults),
        so this can't drift from what pi actually does the way re-deriving
        the same logic in Python could.
        """
        rid = str(uuid.uuid4())
        self._send({"id": rid, "type": "get_state"})
        resp = self._await_response(rid, timeout=timeout)
        if not resp.get("success"):
            # Matches prompt_and_collect()'s handling: an error response has
            # no "data" at all, so returning {} here would be indistinguishable
            # from a legitimately empty (but successful) state.
            raise RuntimeError(f"pi rejected get_state: {resp.get('error')}")
        return resp.get("data", {})

    def session_stats(self, timeout: float = 10) -> Optional[dict]:
        """Query pi's own cumulative token/cost accounting for this session.

        More complete than summing `turn_end.message.usage` in
        prompt_and_collect(): per docs/rpc.md, `get_session_stats`'s
        `tokens`/`cost` include tool-result usage and compaction/branch-
        summary generation across the WHOLE session, not just assistant
        turns. Session-cumulative, not per-call -- if a caller ever issues
        more than one prompt_and_collect() per PiRpc session, this reflects
        the running total across all of them, not just the most recent one
        (the Harbor adapter calls once per trial, so the two agree there).

        Never raises: returns None on any failure -- pi process already
        dead, timeout, or an older pinned pi build that doesn't recognize
        `get_session_stats` at all -- so a caller can treat this exactly
        like the graceful-degradation pattern used elsewhere in this class
        (e.g. capture_environment_snapshot's best-effort reads).
        """
        if self._closed or self._proc.poll() is not None:
            return None
        try:
            rid = str(uuid.uuid4())
            self._send({"id": rid, "type": "get_session_stats"})
            resp = self._await_response(rid, timeout=timeout)
        except Exception:
            return None
        if not resp.get("success"):
            return None
        return resp.get("data")

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
        # The process being dead doesn't mean the reader thread has finished
        # draining its stdout pipe and appending the last few lines to
        # self._notifications/self._responses -- readline() sees EOF only
        # once the kernel pipe buffer is fully consumed, an independent race
        # from _proc.wait() returning. Without this join, a caller reading
        # notifications() immediately after close() could still miss events
        # from the tail of the stream. In practice EOF follows process exit
        # almost immediately, so 10s is very generous headroom, not a
        # normal-case wait. Bounded rather than unbounded on purpose: a
        # harness running 200+ exercises unattended must never be able to
        # hang forever here even if a reader thread is somehow wedged --
        # trading a rare, bounded, diagnostics-only gap (a few missed
        # trailing notify events) for that guarantee. If it ever actually
        # times out, that itself is surprising enough to be worth a loud
        # warning rather than silently swallowing the possibility.
        self._reader.join(timeout=10)
        if self._reader.is_alive():
            print(f"WARNING: PiRpc reader thread did not finish draining "
                  f"stdout within 10s of process exit (session {self._session_id!r}) "
                  f"-- trailing notifications/responses may be missing",
                  file=sys.stderr)
        self._stderr_reader.join(timeout=10)
        if self._stderr_reader.is_alive():
            print(f"WARNING: PiRpc stderr reader thread did not finish within "
                  f"10s of process exit (session {self._session_id!r})",
                  file=sys.stderr)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# ── Environment snapshot ────────────────────────────────────────────────────


def _read_json(path: Path) -> Optional[dict]:
    """Best-effort JSON read. None (not a raise) on any failure, including a
    missing file -- callers distinguish "file missing/unreadable" (None) from
    "file present but empty of the field we wanted" ({}). Also None if the
    file parses but isn't a JSON object (a list, a bare string, null) -- every
    caller here calls .get()/iterates the result, so a syntactically-valid
    but wrong-shape file must not masquerade as usable config and crash the
    whole run (confirmed live with a JSON array in place of an object)."""
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _as_dict(value) -> dict:
    """Coerce a JSON value that should be an object but might not be (an
    explicit `null`, or some other type) to {} rather than let a nested
    .get() raise. Config files are operator-edited and can have any shape
    that's still valid JSON."""
    return value if isinstance(value, dict) else {}


def _resolve_little_coder_models_file() -> tuple[Path, str]:
    """Mirror .pi/extensions/llama-cpp-provider/config.ts::resolveOverridePath()'s
    precedence exactly -- kept in sync by hand, there is no shared source of
    truth between this Python harness and that TS extension."""
    env_override = os.environ.get("LITTLE_CODER_MODELS_FILE")
    if env_override:
        return Path(env_override), "env:LITTLE_CODER_MODELS_FILE"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "little-coder" / "models.json", "env:XDG_CONFIG_HOME"
    return Path.home() / ".config" / "little-coder" / "models.json", "home_default"


def _find_model_max_tokens(provider: str, model_id: str) -> dict:
    """maxTokens resolution: the user-override file wins if it defines this
    model; otherwise fall back to the shipped default at REPO_ROOT/models.json
    (the same file llama-cpp-provider's own pkgRoot points at).

    NOTE: unlike llama-cpp-provider/config.ts's mergeProviders(), which
    replaces a provider's entire `models` array wholesale if the override
    file mentions that provider at all, this falls back per-model. Documented
    divergence, not fixed here -- matching the TS merge semantics exactly
    would need provider-level short-circuiting, a larger behavior change.
    """
    out = {"value": None, "source_file": None, "resolution": None}
    override_path, resolution = _resolve_little_coder_models_file()
    for path, res in ((override_path, resolution), (_LC_MODELS_SHIPPED_DEFAULT, "shipped_default")):
        data = _read_json(path)
        if data is None:
            continue
        providers = _as_dict(data.get("providers"))
        models = _as_dict(providers.get(provider)).get("models")
        if not isinstance(models, list):
            continue
        for entry in models:
            if isinstance(entry, dict) and entry.get("id") == model_id:
                # pi's own fillModelDefaults() (config.ts) applies maxTokens: 4096
                # when a model entry omits the field -- {...defaults, ...m}, so an
                # absent key still means 4096 in effect, not "unknown".
                out.update(value=entry.get("maxTokens", 4096), source_file=str(path), resolution=res)
                return out
    return out


def _resolve_thinking(cli_thinking: Optional[str]) -> dict:
    settings = _read_json(_PI_SETTINGS_PATH)
    pi_default = settings.get("defaultThinkingLevel") if settings else None
    if cli_thinking:
        resolved, source = cli_thinking, "cli"
    elif pi_default:
        resolved, source = pi_default, "pi_default_settings"
    else:
        resolved, source = None, "unresolved"
    return {
        "cli_value": cli_thinking,
        "pi_default_setting": pi_default,
        "resolved": resolved,
        "source": source,
        # Filled in later by the caller once a live PiRpc session exists and
        # can be asked via get_state() -- see PiRpc.get_state() above. This
        # static resolution has no session to query, so it starts unset.
        "confirmed_live": None,
    }


def _capture_omlx_sampling(model_id: str, errors: list[dict]) -> dict:
    out = {
        "provider": "omlx",
        "global_default": None,
        "per_model_override": None,
        "effective": None,
        "source_files": [str(_OMLX_SETTINGS), str(_OMLX_MODEL_SETTINGS)],
    }
    global_settings = _read_json(_OMLX_SETTINGS)
    if global_settings is None:
        errors.append({"source": "server_sampling", "error": f"unreadable or missing: {_OMLX_SETTINGS}"})
    else:
        raw = _as_dict(global_settings.get("sampling"))
        out["global_default"] = {k: raw[k] for k in _OMLX_SAMPLING_KEYS if k in raw}

    model_settings = _read_json(_OMLX_MODEL_SETTINGS)
    if model_settings is None:
        errors.append({"source": "server_sampling", "error": f"unreadable or missing: {_OMLX_MODEL_SETTINGS}"})
    else:
        models = _as_dict(model_settings.get("models"))
        if model_id in models:
            raw = _as_dict(models.get(model_id))
            out["per_model_override"] = {k: raw[k] for k in _OMLX_SAMPLING_KEYS if k in raw}

    if out["global_default"] is not None or out["per_model_override"] is not None:
        out["effective"] = {**(out["global_default"] or {}), **(out["per_model_override"] or {})}
    return out


def capture_environment_snapshot(model: str, *, cli_thinking: Optional[str] = None, agent: str = "pi") -> dict:
    """Best-effort snapshot of config that affects generation but isn't visible
    to the harness's own CLI args: the machine-local default thinking level
    pi falls back to when --thinking is unset, the model's maxTokens, the
    model server's sampling params (temperature/top_p/top_k/repetition_penalty
    -- omlx only for now; rapid-mlx's sampling flags are CLI-launch-time only
    with no queryable file, so that provider degrades to a note rather than a
    guess), and whether the PI_REASONING_MAX_TOKENS vendor patch to pi's own
    vendored openai-completions.js is present in THIS install (it does not
    survive `npm ci`, so a fresh worktree can silently lose it with no other
    indication).

    `agent` gates everything except `model`/`agent` themselves: provider/
    model_id parsing, thinking level, little-coder's own maxTokens config,
    omlx sampling, and pi's own vendored patch are all meaningless for a
    codex run (a codex model id like "gpt-5.1-codex-max" has no provider
    prefix at all -- not a malformed pi model string) and would otherwise
    read as misleadingly authoritative pi-flavored provenance for a run
    that never touches pi.

    Reflects on-disk config, not necessarily a currently-running server's
    already-loaded state -- a hand-edited settings file the server hasn't
    picked up yet (no restart) would still read as the new value here.

    Never raises. A missing or unreadable source is recorded as absent (None)
    with a breadcrumb in "errors", not a crashed run -- this must not be the
    reason a benchmark exercise fails.
    """
    if agent != "pi":
        return {
            "model": model,
            "agent": agent,
            "note": f"environment snapshot only covers agent='pi' today; agent={agent!r} not introspected",
            "errors": [],
        }

    provider, sep, model_id = model.partition("/")
    errors: list[dict] = []
    if not sep:
        errors.append({"source": "model", "error": f"no provider prefix in {model!r}"})

    try:
        thinking = _resolve_thinking(cli_thinking)
    except Exception as exc:
        errors.append({"source": "thinking", "error": f"{type(exc).__name__}: {exc}"})
        thinking = {"cli_value": cli_thinking, "pi_default_setting": None,
                    "resolved": None, "source": "error", "confirmed_live": None}
    try:
        max_tokens = _find_model_max_tokens(provider, model_id)
    except Exception as exc:
        errors.append({"source": "max_tokens", "error": f"{type(exc).__name__}: {exc}"})
        max_tokens = {"value": None, "source_file": None, "resolution": None}

    snapshot = {
        "model": model,
        "agent": agent,
        "provider": provider,
        "thinking": thinking,
        "max_tokens": max_tokens,
    }

    if provider == "omlx":
        try:
            snapshot["server_sampling"] = _capture_omlx_sampling(model_id, errors)
        except Exception as exc:
            errors.append({"source": "server_sampling", "error": f"{type(exc).__name__}: {exc}"})
            snapshot["server_sampling"] = {"note": "error while reading omlx config", "provider": provider}
    else:
        snapshot["server_sampling"] = {"note": "not introspectable for this provider", "provider": provider}

    try:
        patch_exists = _VENDOR_PATCH_TARGET.is_file()
        patch_applied = patch_exists and _VENDOR_PATCH_MARKER in _VENDOR_PATCH_TARGET.read_text(errors="replace")
    except Exception as exc:
        patch_exists = patch_applied = False
        errors.append({"source": "vendor_patch", "error": f"{type(exc).__name__}: {exc}"})
    try:
        target_file = str(_VENDOR_PATCH_TARGET.relative_to(REPO_ROOT))
    except ValueError:
        # Not under REPO_ROOT -- e.g. a test monkeypatched this constant to a
        # tmp_path fixture. Fall back to the absolute path rather than raise.
        target_file = str(_VENDOR_PATCH_TARGET)
    snapshot["vendor_patch"] = {
        "target_file": target_file,
        "exists": patch_exists,
        "applied": patch_applied,
    }

    snapshot["errors"] = errors
    return snapshot
