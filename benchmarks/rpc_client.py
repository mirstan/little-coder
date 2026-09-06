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
_pi_bin_override = os.environ.get("LITTLE_CODER_PI_BIN_OVERRIDE")
# .resolve(): real bug, confirmed by review -- on POSIX, subprocess.Popen
# with both a RELATIVE executable path and an explicit `cwd=` resolves that
# path against the CHILD's cwd, not the launcher's. A relative override
# (e.g. "./fake_pi.py") would exist from the launcher's own cwd at import
# time, then silently fail to launch once PiRpc is constructed with a
# task-specific cwd (an aider_polyglot exercise dir, etc). Resolving once
# here, against the launcher's actual cwd, makes the override
# cwd-independent from then on.
PI_BIN = Path(_pi_bin_override).resolve() if _pi_bin_override else REPO_ROOT / "node_modules" / ".bin" / "pi"
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
        try:
            tmp.replace(generated)
        except OSError:
            # Real gap, confirmed by review: if replace() itself fails after
            # write_text() already succeeded, tmp was left behind under
            # .pi/ forever -- clean it up before falling through to the
            # same AGENTS.md-only fallback as any other write failure here.
            tmp.unlink(missing_ok=True)
            raise
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
    #: DIAGNOSTIC, not yet a real feature: every assistantMessageEvent whose
    #: type is anything other than "text_delta" (e.g. a reasoning/thinking
    #: delta, if pi's underlying @earendil-works/pi-ai stream emits one for
    #: an extended-thinking model), captured verbatim. assistant_text only
    #: ever accumulates "text_delta" content -- any reasoning/thinking-delta
    #: stream a model like omlx/tiel-coder-oq4e produces at a high thinking
    #: level was previously silently dropped here with zero record it even
    #: existed. Static analysis of pi's own compiled types (ExtensionEvent
    #: in dist/core/extensions/types.d.ts) confirmed there is no separate
    #: top-level reasoning event -- if it streams at all, it comes through
    #: THIS SAME channel with a different `type` value -- but the exact
    #: value is defined in @earendil-works/pi-ai, which isn't installed as
    #: inspectable source, so it can't be pinned without guessing. This
    #: captures whatever actually shows up, verbatim, so the real event
    #: shape can be read off a live run instead of assumed.
    non_text_deltas: list[dict] = field(default_factory=list)


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
                else:
                    # See PromptResult.non_text_deltas' own docstring --
                    # diagnostic capture, not yet consumed by anything.
                    result.non_text_deltas.append(delta)
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
