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

import collections
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).parent.parent
# LITTLE_CODER_PI_BIN_OVERRIDE: integration-testing hook only. Lets a
# host-side harness (e.g. `tb run`, which runs pi on the host per
# tb_adapter's own comment) point PI_BIN at a stand-in (fake_pi.py) without
# modifying any tracked file.
#
# Tested for truthiness, not just presence: an EMPTY export (a harness
# script doing `export LITTLE_CODER_PI_BIN_OVERRIDE="$SOME_UNSET_VAR"`)
# would otherwise give Path("") == Path("."), whose .exists() is True,
# defeating the "pi not found" FileNotFoundError check below and surfacing
# later as an opaque Popen error.
_pi_bin_override = os.environ.get("LITTLE_CODER_PI_BIN_OVERRIDE")
# .resolve(): on POSIX, subprocess.Popen with both a RELATIVE executable
# path and an explicit `cwd=` resolves that path against the CHILD's cwd,
# not the launcher's. A relative override ("./fake_pi.py") exists from the
# launcher's cwd at import time, then fails to launch once PiRpc is
# constructed with a task-specific cwd. Resolving once here makes the
# override cwd-independent.
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
    # Both must exist, not just PRINCIPLES.md: the caller's "AGENTS.md
    # missing -> degrade gracefully" guard runs on the returned path, so a
    # missing AGENTS.md here would instead raise FileNotFoundError out of
    # PiRpc.__init__ and kill the whole benchmark run.
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
        # system prompt. Falls back to AGENTS.md alone, as the
        # no-PRINCIPLES.md path above does, if the
        # write itself fails (e.g. a read-only .pi/) rather than raising an
        # uncaught OSError out of PiRpc.__init__.
        tmp = generated.with_name(f"{generated.name}.tmp-{os.getpid()}-{threading.get_ident()}")
        tmp.write_text(content)
        try:
            tmp.replace(generated)
        except OSError:
            # A replace() failure after a successful write_text() would
            # strand tmp under .pi/ forever; clean it up before falling
            # through to the same AGENTS.md-only fallback.
            tmp.unlink(missing_ok=True)
            raise
    except OSError:
        return agents_md
    return generated


class PiProcessExited(RuntimeError):
    """pi exited before completing the request. Carries its stderr tail."""


#: In-memory backstop so non_text_deltas can't grow unbounded across an
#: attempt (a model streaming heavy reasoning emits thousands of events);
#: the real trimming policy stays downstream in aider_polyglot.py's
#: _cap_non_text_deltas(), which needs head+tail retention that isn't
#: knowable mid-stream. Generous enough that no normal attempt reaches it.
#:
#: Kept as a fixed HEAD (_NON_TEXT_DELTA_HEAD_KEEP) plus a rolling TAIL
#: rather than a head-only cap: both _cap_non_text_deltas() and
#: live_eval.py's _reasoning_excerpt_from_trajectory() want the LATEST
#: reasoning, and a run long enough to hit this backstop is exactly the one
#: whose true tail a head-only cap would discard.
_MAX_NON_TEXT_DELTAS = 5_000
#: Fixed prefix retained even once the rolling tail below is full --
#: mirrors aider_polyglot.py's own _cap_non_text_deltas() head+tail split
#: philosophy (never drop ALL early context, even under a hard budget).
_NON_TEXT_DELTA_HEAD_KEEP = 500


@dataclass
class PromptResult:
    """Outcome of a single prompt_and_collect() call."""
    assistant_text: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # [{name, args, result_text, is_error}]
    agent_ended: bool = False
    compaction_events: int = 0
    turn_count: int = 0
    #: Why the call returned: "agent_end" (pi finished the turn -- this
    #: covers both a genuine agent_settled and the bounded settle-window
    #: fallback described below; "agent_settled" is deliberately NOT a
    #: separate value here, see prompt_and_collect()'s docstring), "error"
    #: (pi finished the turn, but the final assistant completion was a
    #: provider error or came back empty -- see error_message), "deadline"
    #: (budget expired), or "process_exit" (pi died mid-run). Callers must not
    #: infer this from elapsed time.
    #:
    #: "error" is a refinement of "agent_end", not of the other two:
    #: "deadline"/"process_exit" always win over it, because those describe
    #: how the DRAIN ended and are strictly more informative about a
    #: truncated trial than what the last completion happened to contain.
    #: An invariant the derivation enforces in both directions, not one that
    #: falls out of branch order alone: "deadline" does win by ordering, but
    #: an `agent_end` immediately followed by EOF lands in the settle branch,
    #: so both error-synthesis paths re-check whether pi is already gone
    #: before firing. Without that check an errored or empty final turn from
    #: a process that no longer existed reported "error", and
    #: prompt_with_error_retry then re-prompted a dead session.
    #:
    #: NOTE: agent_ended=True with stop_reason="deadline" is a reachable and
    #: meaningful combination as of the ACTIVE/SETTLING rewrite: it means a
    #: continuation turn started (e.g. a queued follow-up after an abort) but
    #: never finished before the outer timeout expired. Do not read
    #: agent_ended alone as "the call completed" -- check stop_reason too.
    stop_reason: str = "agent_end"
    #: Every assistantMessageEvent whose type is anything other than
    #: "text_delta", captured verbatim -- assistant_text accumulates only
    #: "text_delta", so a model's reasoning stream reaches nothing else.
    #: Confirmed against a real run at thinking=high: the reasoning event
    #: type is "thinking_delta", carrying incremental text in the same
    #: "delta" key text_delta uses, which
    #: benchmarks/self_improve/live_eval.py's
    #: _reasoning_excerpt_from_trajectory() reassembles the same way
    #: assistant_text is built above. Other types seen in that run:
    #: thinking_start/thinking_end (bracket a reasoning block),
    #: toolcall_start/toolcall_delta/toolcall_end, text_start/text_end.
    #: Bounded to _MAX_NON_TEXT_DELTAS entries during collection.
    non_text_deltas: list[dict] = field(default_factory=list)
    #: Provider error text for stop_reason == "error", and only then: either
    #: the assistant message's own `errorMessage` from the wire, or the
    #: synthetic "empty completion" when pi reported no error but produced a
    #: turn with no text and no tool calls. "" for every other stop_reason.
    error_message: str = ""
    #: Token usage summed across every `turn_end` seen during this call
    #: (see prompt_and_collect's aggregation loop). Crash-proof but
    #: incomplete: unlike session_stats(), it does not include tool-result
    #: usage or compaction/branch-summary generation tokens, and it only
    #: covers turns whose `turn_end` actually arrived (see PiRpc.session_stats
    #: docstring for why session_stats is preferred when available).
    usage: dict = field(default_factory=lambda: {
        "input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "cost": 0.0,
    })
    #: True only when pi's own agent_settled event was actually observed
    #: (i.e. NOT via the settle-grace fallback). Purely observational --
    #: stop_reason intentionally still reports "agent_end" in both cases for
    #: backward compatibility (see stop_reason's docstring).
    settled: bool = False


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
        if max_turns is not None:
            # `is not None`, not truthiness: an explicit max_turns=0 (deliberate
            # "no cap") must still WRITE the env var so it clobbers any ambient
            # LITTLE_CODER_MAX_TURNS inherited from the caller's own environment
            # (full_env starts as a copy of os.environ, above) -- otherwise a
            # leaked wrapper-script value would silently survive an explicit
            # "no cap" request. turn-cap.ts's resolveTurnCap already handles the
            # string "0" correctly: Number("0") == 0, so a cap of 0 is applied.
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

        Single loop, two modes, because `agent_end` is not always the end of
        the turn: an extension can react to an `agent_end` (e.g. the
        thinking-budget extension aborting a runaway thinking stream) by
        queuing a follow-up message and letting pi continue processing in
        the same run. pi's queued-follow-up continuation (and auto-retry /
        auto-compaction continuations) runs to completion and only then
        emits `agent_settled` -- pi's own purpose-built "nothing left
        queued, truly idle" signal. Draining only to the first `agent_end`
        returns before that continuation starts, which for a Harbor trial
        means the caller treats the abort as the whole run finishing and
        tears the process down mid-recovery.

        An earlier two-phase version of this drain (Phase 1 to `agent_end`,
        Phase 2 one event to decide "settled vs. continuation") could not
        express "am I waiting for work, or waiting for the session to
        quiesce?" as a single predicate, and that ambiguity was four bugs:

          (a) Phase 1's predicate only matched `agent_end`, so if the very
              next event after looping back was `agent_settled` itself, it
              was treated as an ordinary event and Phase 1 waited out the
              *entire remaining timeout* for an `agent_end` that would never
              arrive.
          (b) an `if self._eof: break` right after Phase 1 raced the reader
              thread: if pi's process (and EOF) arrived before a slow
              `on_event` had drained an already-queued recovery turn, this
              guard discarded that entire queued turn instead of letting
              `_drain_events_until` -- the single authority on "queue empty
              AND pi gone" -- decide.
          (c) Phase 2 treated "any event that is not agent_settled" as
              "a continuation started" and looped back to Phase 1 with the
              FULL remaining timeout -- so a stray/duplicate `agent_end`
              (pi does this) or benign post-turn chatter (`auto_retry_end`,
              `queue_update`) re-armed the whole budget instead of just
              being noise inside the settle window.
          (d) `stop_reason` was derived from a latched "did we ever see an
              agent_end", so a run that mis-looped per (c) until the outer
              timeout expired still reported `stop_reason="agent_end"` --
              a truncated trial reading as a completed one downstream.

        This version replaces both phases with one loop carrying an explicit
        mode:

          ACTIVE   -- work is (or may be) in flight. Wait up to the FULL
                      remaining timeout for a terminal event. The predicate
                      matches `agent_end` OR `agent_settled` (fixes (a): a
                      settle seen here ends the call immediately, it is
                      never mistaken for "just another event").
          SETTLING -- an `agent_end` has been seen; wait, ONE EVENT AT A
                      TIME, for the session to quiesce, bounded by an
                      ABSOLUTE settle deadline computed once on entry:
                      `settle_deadline = min(now + settle_grace, start +
                      timeout)`. Recomputing this per-event (rather than
                      once) would let a chatty post-turn stream extend
                      settling indefinitely -- bounded only by the outer
                      timeout, the opposite of what settle_grace is for.

        The event popped while SETTLING decides what happens next:
          - `agent_settled` -> terminal; pi's real idle signal, honored
            wherever it appears in the stream.
          - a *renewed-work* marker (currently just `agent_start`) -> back to
            ACTIVE with the full remaining timeout; a genuine continuation
            turn (the abort-then-followup case this function's two-phase
            drain was originally built for).
          - a second, consecutive `agent_end` -> STAY in SETTLING (fixes
            (c)): a stray/duplicate end must never re-arm the timeout.
          - anything else (`auto_retry_end`, `queue_update`, `usage_update`,
            an event type pi hasn't invented yet) -> STAY in SETTLING. The
            settle deadline still bounds us, so this is cheap even when
            wrong.
          - settle deadline expires with nothing new -> break, emit the
            existing "pi did not emit agent_settled" warning (unchanged
            defensive fallback for a pi build that never emits it).

        Design choice: allowlist, not denylist, for "renewed work". The old
        rule -- "anything that isn't agent_settled means a continuation
        started" -- is a denylist that fails OPEN on every event type pi
        ever adds (bug (c)). Inverting it to an allowlist of markers that
        re-enter ACTIVE fails CLOSED instead: an unrecognized event just
        keeps settling. Failing closed is cheap specifically because of
        (a)'s fix -- even a wrongly-ignored continuation costs at most until
        the next `agent_settled`, never the whole timeout. The allowlist
        starts as `{"agent_start"}` only, matching every fixture here and
        pi's documented behaviour: a continuation turn always opens with
        `agent_start`. Residual risk, noted rather than hidden: a
        hypothetical pi build that resumes work WITHOUT emitting
        `agent_start` degrades to the old pre-two-phase behaviour (return at
        `agent_end` after `settle_grace`) plus the existing warning -- a
        degradation, not a hang.

        `_drain_events_until` is the single authority on "queue empty AND pi
        gone" (it drains everything already queued, in order, before ever
        consulting `self._eof`); this loop deliberately never adds its own
        `if self._eof: break` on top of that, because such a check observes
        only the flag, not the queue, and can discard events that are
        already sitting there waiting to be popped (bug (b)).

        `stop_reason` is derived from how the loop exited, not from a
        latched "did we ever see agent_end" (fixes (d)):
          - the outer `timeout` expired while in ACTIVE (work still in
            flight) -> "deadline", even if one or more `agent_end` events
            were seen earlier in this call;
          - exited on `agent_settled`, or on the bounded settle-window
            fallback after an `agent_end` -> "agent_end" (this exact string
            is kept for both cases -- `aider_polyglot.py::_stop_reason` and
            the polyglot metadata tests key on it; "agent_settled" is
            deliberately NOT introduced as a new value), unless the final
            completion errored, in which case -> "error" -- or
            "process_exit" when pi has already exited by then (see below);
          - otherwise, if `self._eof or self._proc.poll() is not None` ->
            "process_exit";
          - otherwise -> "deadline".

        "error" refines ONLY the `agent_end` branch above. Two shapes count,
        both observed in this repo's own Harbor trajectory logs:

          - the last assistant message carries `stopReason: "error"` (the
            shape three of three error-truncated TB2.1 trials actually hit:
            `turn_end stopReason=error hasText=false hasToolCalls=false`).
            `errorMessage`, when present, lands in `error_message`;
          - pi reports no error at all but the whole call produced a single
            turn with no tool calls and no assistant text -- the empty
            completion `aider_polyglot.py::_is_empty_response` already
            fingerprints. `error_message` is then the synthetic
            "empty completion".

        Two rules keep this from over-firing, both load-bearing:

          - LAST `agent_end` WINS. The verdict is recomputed at every
            `agent_end` in the call, never latched from an earlier one: a
            single call routinely sees several (continuations, pi's own
            retries), and a turn that errored before a later one completed
            cleanly must not poison the clean result. Within an agent run
            the same rule applies to `turn_end`: the most recent one
            decides, so a turn pi re-ran successfully overrides the failed
            attempt that preceded it.
          - `willRetry` SUPPRESSES. pi's `agent_end` carries a `willRetry`
            boolean; `true` means pi's own internal retry is about to fire,
            so the error is already being handled and must not also be
            reported as ours (the caller would otherwise retry a turn pi is
            concurrently retrying). `auto_retry_start` / `auto_retry_end`
            are handled for the same reason: an errored turn pi then retries
            itself is not an error we own unless that retry also fails. The
            flag is read off the LAST `agent_end` by plain assignment, never
            OR-latched across several: a sticky latch would keep suppressing
            after a later `agent_end` said `willRetry:false`. It gates the
            empty-completion fingerprint below as well as the flagged-error
            path -- pi retrying a turn itself makes an empty one just as
            much not-ours as an errored one.
          - A DEAD pi OUTRANKS BOTH. pi can emit `agent_end` and then exit
            within the same settle window, which still exits the loop
            through the settled/SETTLING branch even though the process is
            gone. Both error-synthesis paths therefore re-check `self._eof
            or self._proc.poll() is not None` and report "process_exit"
            (with no `error_message`, per that value's contract) instead:
            there is no session left for anyone to retry, and
            `prompt_with_error_retry` doing it anyway raised
            `PiProcessExited` out of a Harbor trial before its metadata was
            ever written. Scoped to those two paths only -- a clean
            `agent_end` followed by exit still reports "agent_end".

        All modes append into one `events` list, so tool calls / text /
        turn_count from a continuation turn are aggregated into the same
        PromptResult as the initial turn. `agent_ended` stays True if any
        `agent_end` was observed across the whole call -- including a
        continuation that never finished, see PromptResult.stop_reason's
        docstring for why that combination is meaningful, not contradictory.
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

        # Renewed-work markers: an event of one of these types popped while
        # SETTLING means a genuine continuation turn started, so we go back
        # to ACTIVE with the full remaining timeout. Allowlist, not denylist
        # -- see the docstring above for why that direction matters.
        _RENEWED_WORK_MARKERS = ("agent_start",)
        _TERMINAL_TYPES = ("agent_end", "agent_settled")

        start = time.time()
        events: list[dict] = []
        saw_agent_end = False
        settled = False
        mode = "ACTIVE"
        settle_deadline = 0.0
        # Only ever set True when ACTIVE exits because the outer timeout
        # itself ran out (not because pi went away with time still on the
        # clock) -- this is what lets stop_reason distinguish a genuine
        # deadline from an EOF, per the derivation below.
        active_timeout_expired = False

        while True:
            if mode == "ACTIVE":
                remaining = timeout - (time.time() - start)
                if remaining <= 0:
                    active_timeout_expired = True
                    break
                batch = self._drain_events_until(
                    lambda ev: ev.get("type") in _TERMINAL_TYPES,
                    timeout=remaining,
                    on_event=on_event,
                )
                events.extend(batch)
                last_type = batch[-1].get("type") if batch else None
                if last_type not in _TERMINAL_TYPES:
                    # Nothing terminal arrived this phase -- either the
                    # outer timeout ran out mid-wait, or pi went away
                    # (_drain_events_until returns "queue empty AND EOF"
                    # either way; disambiguate by re-checking the clock, not
                    # by adding our own separate _eof check).
                    if timeout - (time.time() - start) <= 0:
                        active_timeout_expired = True
                    break
                if last_type == "agent_settled":
                    settled = True
                    break
                # agent_end -- work may or may not be fully done. Enter
                # SETTLING with a fresh, absolute deadline.
                saw_agent_end = True
                mode = "SETTLING"
                settle_deadline = min(time.time() + settle_grace, start + timeout)
                continue

            # mode == "SETTLING" -- one event at a time, deliberately (see
            # the docstring's "SETTLING" description). `_drain_events_until`
            # with an always-true predicate returns as soon as exactly one
            # event arrives (or empty on timeout/EOF); a non-positive
            # `grace_remaining` is passed through as-is and returns empty
            # immediately, so no separate pre-check is needed here.
            grace_remaining = settle_deadline - time.time()
            settle_events = self._drain_events_until(
                lambda ev: True,
                timeout=grace_remaining,
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
            ev_type = settle_events[-1].get("type")
            if ev_type == "agent_settled":
                settled = True
                break
            if ev_type in _RENEWED_WORK_MARKERS:
                mode = "ACTIVE"
                continue
            # A duplicate/stray agent_end, or any other post-turn chatter
            # (auto_retry_end, queue_update, usage_update, ...) -- stay in
            # SETTLING. settle_deadline is untouched: it was fixed on entry
            # to SETTLING and is not extended by what we see while here.
            continue

        result = PromptResult()
        result.settled = settled

        # ── Error derivation state (see the docstring's "error" section) ──
        # The verdict, recomputed at EVERY agent_end and never latched: one
        # call can legitimately see several, and the last one decides.
        error_flagged = False
        error_message = ""
        # The most recent turn_end's own stopReason/errorMessage, i.e. the
        # state of the turn currently in flight. Last turn_end wins within an
        # agent run for the same reason last agent_end wins across runs.
        turn_stop_reason = ""
        turn_error_message = ""
        # Whether the LAST agent_end said pi is retrying this itself.
        # Assigned, never OR-ed: a sticky latch would go on suppressing the
        # empty-completion fingerprint after a later agent_end said false.
        last_will_retry = False

        pending: dict[str, dict] = {}
        # See PromptResult.non_text_deltas' own docstring and
        # _MAX_NON_TEXT_DELTAS' -- a fixed head plus a bounded ROLLING tail
        # (evicts its own oldest entry once full, O(1) amortized) rather
        # than a single head-only cap, so a pathological run long past this
        # backstop still keeps its true tail, not just its opening. Local
        # to this call (not on `result` itself) so PromptResult.non_text_deltas
        # stays a plain list for every other caller/consumer.
        non_text_delta_head: list[dict] = []
        non_text_delta_tail: collections.deque = collections.deque(
            maxlen=_MAX_NON_TEXT_DELTAS - _NON_TEXT_DELTA_HEAD_KEEP
        )
        for ev in events:
            t = ev.get("type")
            if t == "message_update":
                delta = ev.get("assistantMessageEvent", {})
                if delta.get("type") == "text_delta":
                    result.assistant_text += delta.get("delta", "")
                elif len(non_text_delta_head) < _NON_TEXT_DELTA_HEAD_KEEP:
                    non_text_delta_head.append(delta)
                else:
                    non_text_delta_tail.append(delta)
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
                msg = ev.get("message")
                # Overwritten (not OR-ed) on every turn_end: only the most
                # recent turn's outcome is in flight, so a turn pi re-ran
                # successfully clears the failed attempt before it.
                turn_stop_reason = str(msg.get("stopReason") or "") if isinstance(msg, dict) else ""
                turn_error_message = str(msg.get("errorMessage") or "") if isinstance(msg, dict) else ""
                usage = msg.get("usage") if isinstance(msg, dict) else None
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
            elif t == "auto_retry_start":
                # pi is re-issuing the failed provider call itself. Drop the
                # errored turn's verdict so the retry's own turn_end (or the
                # auto_retry_end below) decides -- without this, an error pi
                # successfully recovered from could still be the last
                # turn_end this loop saw before agent_end, since a retried
                # turn need not emit a second turn_end.
                turn_stop_reason = ""
                turn_error_message = str(ev.get("errorMessage") or "")
            elif t == "auto_retry_end":
                if ev.get("success"):
                    turn_stop_reason = ""
                    turn_error_message = ""
                else:
                    turn_stop_reason = "error"
                    turn_error_message = str(ev.get("finalError") or turn_error_message)
            elif t == "agent_end":
                result.agent_ended = True
                last_will_retry = bool(ev.get("willRetry"))
                if last_will_retry:
                    # pi's own retry is already queued for this failure --
                    # reporting it as ours too would have the caller retry a
                    # turn pi is concurrently retrying.
                    error_flagged = False
                    error_message = ""
                else:
                    error_flagged = turn_stop_reason == "error"
                    error_message = turn_error_message if error_flagged else ""

        result.non_text_deltas = non_text_delta_head + list(non_text_delta_tail)

        # Derived from how the loop exited, not from a latched "did we ever
        # see agent_end" -- see PromptResult.stop_reason's docstring for why
        # agent_ended=True with stop_reason="deadline" is reachable and
        # meaningful. Deliberately computed AFTER the aggregation loop above
        # (it needs that loop's error verdict and content totals), but the
        # loop-exit branches are unchanged and still take precedence: only
        # the agent_end branch can become "error" -- or, when pi is already
        # gone, "process_exit".
        if active_timeout_expired:
            result.stop_reason = "deadline"
        elif settled or mode == "SETTLING":
            # Either pi's real idle signal fired, or we exited the bounded
            # settle window (deadline or EOF) after having seen an
            # agent_end -- both report the same string for backward
            # compatibility; see the docstring above.
            empty_completion = (
                # The empty-completion shape, with nothing on the wire saying
                # so. Same fingerprint aider_polyglot.py::_is_empty_response
                # uses; kept here too because a provider can return an empty
                # body with stopReason "stop" rather than "error".
                not error_flagged
                and not last_will_retry
                and result.turn_count <= 1
                and not result.tool_calls
                and not result.assistant_text.strip()
            )
            if error_flagged or empty_completion:
                if self._eof or self._proc.poll() is not None:
                    # agent_end and EOF in the same settle window: this
                    # branch won on ordering, but there is no session left
                    # to retry and saying "error" sends the retry helper at
                    # a dead process. Scoped to the two error-synthesis
                    # paths on purpose -- a clean agent_end followed by exit
                    # keeps reporting "agent_end".
                    result.stop_reason = "process_exit"
                elif error_flagged:
                    result.stop_reason = "error"
                    result.error_message = error_message or "provider error"
                else:
                    result.stop_reason = "error"
                    result.error_message = "empty completion"
            else:
                result.stop_reason = "agent_end"
        elif self._eof or self._proc.poll() is not None:
            result.stop_reason = "process_exit"
        else:
            result.stop_reason = "deadline"

        if result.stop_reason == "error":
            print(
                f"WARNING: pi completion ended in a provider error (session "
                f"{self._session_id!r}): {result.error_message}",
                file=sys.stderr,
            )
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

    def is_alive(self) -> bool:
        """Whether a further prompt could still reach pi.

        The same EOF/exit-status pair stop_reason's "process_exit" is derived
        from, widened by `_closed`: a session this client already closed has
        nothing left to prompt either.
        """
        return not self._closed and not self._eof and self._proc.poll() is None

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


# ── Bounded retry on a provider-error completion ────────────────────────────

#: Total attempts, i.e. the first prompt plus at most two retries.
ERROR_RETRY_MAX_ATTEMPTS = 3
#: Backoff before retry 1 and retry 2. The last entry repeats if
#: ERROR_RETRY_MAX_ATTEMPTS is ever raised past len(this).
ERROR_RETRY_BACKOFF_SEC = (5.0, 15.0)
#: Don't start a retry unless this much budget would still be left after the
#: backoff -- a retry with seconds on the clock can't do useful work, and
#: ending the trial normally at least preserves whatever the agent already
#: wrote.
ERROR_RETRY_MIN_BUDGET_SEC = 60.0
#: Stop after this many CONSECUTIVE attempts failing with byte-identical
#: error text: a non-retryable provider error (a poisoned transcript, an
#: unsupported request) fails the same way every time, and spending the rest
#: of the retry budget on it only delays the trial's real ending.
ERROR_RETRY_IDENTICAL_LIMIT = 2
#: Short and neutral on purpose. The escape hatch in the last sentence
#: matters: without it this nudge can goad an agent that genuinely finished
#: into undoing its own completed work.
ERROR_RETRY_PROMPT = (
    "Your previous response ended with a provider error. The task is not "
    "complete — please continue working on it. If the task is actually "
    "already complete and verified, say so explicitly and stop."
)


@dataclass
class ErrorRetryOutcome:
    """What prompt_with_error_retry() ended up with."""
    #: One PromptResult covering every attempt: turn_count, tool_calls,
    #: usage and compaction_events summed, assistant_text joined, agent_ended
    #: OR-ed. stop_reason / error_message / settled describe the FINAL
    #: attempt alone -- they are verdicts on how the call ended, not totals.
    #: A call that never retried hands back that single attempt's own object
    #: unchanged, exactly as a bare prompt_and_collect() would.
    result: PromptResult
    #: Retries actually issued (0 when the first attempt didn't error).
    n_error_retries: int = 0
    #: The last provider error seen across ALL attempts, retained even when a
    #: later attempt then succeeded -- otherwise a recovered trial would
    #: record n_error_retries > 0 with no trace of what it recovered from.
    error_message: str = ""
    #: "TypeName: message" when a RETRY raised and was turned into the merged
    #: result instead of propagating, "" otherwise. Separate from
    #: `error_message`, which is a provider verdict: this one is a harness
    #: fault (a rejected prompt, a dead pipe) that the caller would otherwise
    #: have seen as a raised exception and now cannot see at all.
    retry_exception: str = ""


def _merged_result(acc: PromptResult, latest: PromptResult) -> PromptResult:
    """Fold one more attempt into the accumulated PromptResult.

    Returns a new object instead of mutating `acc`, so the first attempt's
    own result -- the object a never-retried call hands straight back to the
    caller -- is never rewritten underneath anyone still holding it.
    """
    usage = dict(acc.usage)
    for key, value in (latest.usage or {}).items():
        usage[key] = usage.get(key, 0) + value
    texts = [t for t in (acc.assistant_text, latest.assistant_text) if t]
    return PromptResult(
        assistant_text="\n".join(texts),
        tool_calls=list(acc.tool_calls) + list(latest.tool_calls),
        agent_ended=acc.agent_ended or latest.agent_ended,
        compaction_events=acc.compaction_events + latest.compaction_events,
        turn_count=acc.turn_count + latest.turn_count,
        usage=usage,
        # Verdicts about how the call ended, so the latest attempt's alone.
        stop_reason=latest.stop_reason,
        error_message=latest.error_message,
        settled=latest.settled,
    )


def prompt_with_error_retry(
    rpc: PiRpc,
    message: str,
    timeout: float,
    on_event: Optional[Callable[[dict], None]] = None,
    *,
    deadline: Optional[float] = None,
    retry_message: str = ERROR_RETRY_PROMPT,
    max_attempts: int = ERROR_RETRY_MAX_ATTEMPTS,
    backoff_sec: tuple = ERROR_RETRY_BACKOFF_SEC,
    min_remaining_sec: float = ERROR_RETRY_MIN_BUDGET_SEC,
    identical_error_limit: int = ERROR_RETRY_IDENTICAL_LIMIT,
    log: Optional[Callable[[str], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> ErrorRetryOutcome:
    """prompt_and_collect(), retried a bounded number of times on `"error"`.

    Motivation, measured: across five failed TB2.1 trials, three ended when a
    single errored completion made pi emit `agent_end` with 62-81% of the
    wall-clock budget still unspent. Nothing was wrong with the container or
    the work already done -- the session simply stopped. Re-prompting the
    SAME session (never a fresh one: a new session would discard the whole
    transcript the agent built, which is the only thing that makes continuing
    cheaper than starting over) recovers that budget.

    Retries only on `stop_reason == "error"`. "deadline" means the budget is
    already gone and "process_exit" means there is no session left to prompt,
    so neither is retryable here.

    `deadline` is a `now()`-scale absolute instant; it defaults to
    `now() + timeout`. Pass the trial's own deadline instead whenever one
    exists, so the retries can never outlive the budget the agent was
    actually given. Each retry is issued with the budget REMAINING against
    that deadline, never the original full `timeout` -- passing the full
    timeout again is how a retry loop quietly doubles a trial's wall clock.

    Three independent brakes, any of which ends the loop early:
    `max_attempts`, `identical_error_limit`, and the `min_remaining_sec`
    budget floor (checked against the budget that would be left AFTER the
    backoff, since the backoff spends real trial time too). A fourth check
    is not policy but fact: `rpc.is_alive()` just before the backoff, since
    pi can die BETWEEN attempts -- after the stop_reason derivation that
    would have reported "process_exit" has already run.

    Everything the attempts produced is merged into the single returned
    PromptResult (see ErrorRetryOutcome.result): a retry that then fails
    must not erase the tool calls and tokens the earlier ones spent. An
    exception out of a RETRY is caught for the same reason and turned into
    that merged result; the first attempt's is left to propagate, so
    wrapping a call site in this helper cannot swallow a failure the bare
    prompt_and_collect() would have raised. A caught one is still reported
    three ways, because it is a harness fault the caller can no longer see
    raised: on stderr, in `ErrorRetryOutcome.retry_exception`, and -- when it
    left pi dead -- as a "process_exit" stop_reason rather than a retryable
    "error".

    `sleep`/`now` are injected purely so tests can drive the whole policy
    without spending real seconds.
    """
    if deadline is None:
        deadline = now() + timeout

    def _log(text: str) -> None:
        if log is not None:
            log(text)

    attempts = 0
    n_retries = 0
    last_error = ""
    identical_streak = 0
    attempt_message = message
    merged: Optional[PromptResult] = None
    # Attempt 1 is clamped against the deadline too, not just the retries:
    # callers pass a deadline taken before their own setup work, so `timeout`
    # can already overshoot it on the very first prompt.
    attempt_timeout = min(timeout, max(0.0, deadline - now()))

    while True:
        attempts += 1
        if attempts == 1:
            result = rpc.prompt_and_collect(attempt_message, attempt_timeout, on_event)
        else:
            try:
                result = rpc.prompt_and_collect(attempt_message, attempt_timeout, on_event)
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                _log(
                    f"retry {n_retries} raised {detail} -- keeping what the "
                    f"first {attempts - 1} attempt(s) produced"
                )
                # Unconditionally, not only through `log` -- which defaults to
                # None. This is the one path that turns a raised failure into
                # an ordinary return, so a caller without a logger would
                # otherwise see no trace of it whatsoever.
                print(
                    f"WARNING: error retry {n_retries} raised {detail}; "
                    f"returning the first {attempts - 1} attempt(s) instead",
                    file=sys.stderr,
                )
                if not rpc.is_alive():
                    # The accumulated verdict still reads "error", the
                    # RETRYABLE value, for a session that is provably gone --
                    # the same contradiction the stop_reason derivation
                    # re-checks liveness to avoid. `replace`, not mutation:
                    # `merged` can still be attempt 1's own object.
                    merged = replace(
                        merged, stop_reason="process_exit", error_message=""
                    )
                return ErrorRetryOutcome(merged, n_retries, last_error, detail)
        merged = result if merged is None else _merged_result(merged, result)
        if result.stop_reason != "error":
            return ErrorRetryOutcome(merged, n_retries, last_error)

        err = result.error_message or "provider error"
        identical_streak = identical_streak + 1 if err == last_error else 1
        last_error = err
        _log(
            f"prompt attempt {attempts}/{max_attempts} ended in a provider "
            f"error: {err}"
        )

        if attempts >= max_attempts:
            _log(f"not retrying: {max_attempts} attempts already used")
            return ErrorRetryOutcome(merged, n_retries, last_error)
        if identical_streak >= identical_error_limit:
            _log(
                f"not retrying: {identical_streak} consecutive attempts failed "
                f"with the identical error, treating it as non-retryable"
            )
            return ErrorRetryOutcome(merged, n_retries, last_error)

        backoff = backoff_sec[min(n_retries, len(backoff_sec) - 1)] if backoff_sec else 0.0
        remaining_after_backoff = (deadline - now()) - backoff
        if remaining_after_backoff < min_remaining_sec:
            _log(
                f"not retrying: only {remaining_after_backoff:.0f}s would be "
                f"left after a {backoff:.0f}s backoff, below the "
                f"{min_remaining_sec:.0f}s floor"
            )
            return ErrorRetryOutcome(merged, n_retries, last_error)

        if not rpc.is_alive():
            # pi can exit between attempts, once the derivation that would
            # have said "process_exit" has already run on this result.
            # Prompting anyway raises PiProcessExited out of the trial.
            _log("not retrying: pi process is gone")
            return ErrorRetryOutcome(merged, n_retries, last_error)

        sleep(backoff)
        n_retries += 1
        attempt_message = retry_message
        attempt_timeout = max(0.0, deadline - now())
        _log(
            f"retry {n_retries} on the same session with "
            f"{attempt_timeout:.0f}s of remaining budget"
        )


# ── Log previews ────────────────────────────────────────────────────────────


def preview_tool_result(text: str, limit: int = 400) -> str:
    """Shorten a formatted tool result for a trajectory log, readably.

    A raw `text[:limit]` slice -- what all three call sites used to do --
    fails a log reader twice over: it lands mid-word, and it drops the
    trailing `[exit=... cwd=... timed_out=... backend=...]` footer that
    `_format_output()` appends, which is the single most useful line in the
    whole result (did the command actually succeed, and where did it run?).
    Every result longer than `limit` therefore lost exactly the part worth
    keeping.

    So: cut the BODY at the last newline or space before the budget, mark how
    much was dropped, and re-attach the footer verbatim. The footer is taken
    to be the final line when it is bracket-delimited -- the shape both
    adapters' `_format_output()` always produces, and the reason this is a
    line test rather than a search for "exit=". A result that is nothing BUT
    an over-limit footer is that same final line, and is returned whole.

    Budget: the whole preview -- kept body, marker, footer -- stays within
    `limit`, with the body giving up whatever room the other two need. The
    one exception is a footer wider than `limit` itself, which is still
    preserved in full: dropping it is the bug this exists to fix.
    """
    # Strip trailing newlines before anything else: `_format_output()` never
    # emits one, but a caller that does would otherwise make `rpartition`
    # yield an empty `last` ("[exit=0]\n".rpartition("\n")[2] == ""), which
    # fails the footer test below and lets the real footer get cut mid-line
    # by the body-truncation path -- the exact bug this function exists to
    # prevent.
    text = (text or "").rstrip("\n")
    if len(text) <= limit:
        return text

    head, _sep, last = text.rpartition("\n")
    footer = ""
    body = text
    if last.startswith("[") and last.endswith("]"):
        footer = last
        # rpartition, not a >1-line test: a footer that IS the whole input
        # leaves head == "" here, where a line-count test instead left it as
        # the body and cut it mid-footer -- the one thing this promises not
        # to do.
        body = head

    def _with_footer(kept: str) -> str:
        # `if kept` so a body that gave up all its budget to the footer
        # doesn't produce a preview opening on a blank line.
        if not footer:
            return kept
        return f"{kept}\n{footer}" if kept else footer

    # Reserve room for the marker so the common case stays within `limit`.
    # A fixed reserve, not the marker's exact length, because that length
    # depends on the omitted count, which depends on where we cut.
    marker_reserve = 40
    body_budget = max(0, limit - marker_reserve - (len(footer) + 1 if footer else 0))
    if len(body) <= body_budget:
        return _with_footer(body)

    cut = body[:body_budget]
    for sep in ("\n", " "):
        idx = cut.rfind(sep)
        if idx > 0:
            cut = cut[:idx]
            break
    # No boundary at all (one unbroken token wider than the budget) leaves
    # `cut` as the hard slice -- unavoidable, and still better than also
    # losing the footer.
    marker = f"… [+{len(body) - len(cut)} chars truncated]"
    out = f"{cut}\n{marker}" if cut else marker
    return _with_footer(out)


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
