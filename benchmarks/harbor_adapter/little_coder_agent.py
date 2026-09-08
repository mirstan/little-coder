"""Harbor (Terminal-Bench 2.0) adapter for little-coder.

Subclasses harbor.agents.base.BaseAgent (the TB 2.0 counterpart of TB 1.0's
terminal_bench.agents.base_agent.BaseAgent). The heavy lifting — pi RPC
subprocess, extension stack, ShellSession proxy — is shared with the TB 1.0
adapter via benchmarks/rpc_client.py::PiRpc.

The one moving part that differs from TB 1.0:

  TB 1.0:  agent gets a TmuxSession.send_keys(...) interface — sync.
  TB 2.0:  agent gets environment.exec(command, ...) — *async*.

My PiRpc reader thread invokes the shell-proxy callback synchronously when
an extension_ui_request with the __LC_TB_SHELL__ prefix arrives. To call
harbor's async env.exec from that sync context, we stash the event loop
in run() and use asyncio.run_coroutine_threadsafe().

Launch:

    harbor run \
      --dataset terminal-bench@2.0 \
      --agent-import-path benchmarks.harbor_adapter.little_coder_agent:LittleCoderAgent \
      --model llamacpp/qwen3.6-35b-a3b \
      --n-concurrent 1
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
import tomllib
import uuid
from pathlib import Path


def _read_version_from_package_json() -> str:
    """Read agent version from the repo's package.json at import time.

    Avoids hardcoded version drift between the adapter's version() return
    and the actual released tag. Falls back to "unknown" if the file is
    missing or malformed.
    """
    try:
        pkg = Path(__file__).resolve().parents[2] / "package.json"
        return json.load(open(pkg)).get("version", "unknown")
    except Exception:
        return "unknown"


_AGENT_VERSION = _read_version_from_package_json()


from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

# benchmarks/ isn't a package — let the importer resolve by sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rpc_client import PiRpc  # noqa: E402


DEFAULT_ALLOWED_TOOLS = ["ShellSession", "ShellSessionCwd", "ShellSessionReset"]
DEFAULT_MODEL = "llamacpp/qwen3.6-35b-a3b"

# Fallback when the real per-task timeout can't be derived (see
# _resolve_trial_timeout_sec) -- e.g. a manual/local run with no Harbor
# trial_dir, or a future Harbor version with a different cache layout.
DEFAULT_PROMPT_TIMEOUT_SEC = 3600.0
# Safety margin below Harbor's own enforced timeout: leaves headroom for
# our own graceful cutoff (and finalize-warn's wall-clock nudge) to land
# before Harbor's external asyncio.wait_for kills the process outright.
DEADLINE_SAFETY_MARGIN = 0.9
HARBOR_TASK_CACHE = Path.home() / ".cache" / "harbor" / "tasks"


def _resolve_trial_timeout_sec(logs_dir: Path | None) -> float:
    """Best-effort derivation of this trial's real Harbor-enforced timeout
    (task.toml's [agent].timeout_sec x the job's timeout_multiplier), so our
    internal deadline tracks Harbor's actual budget instead of a blind guess.

    Harbor gives a custom agent no direct/supported way to read this value --
    it's a private Trial attribute enforced purely via an external
    asyncio.wait_for() one level above the agent call. Every step here
    opportunistically reads files Harbor already writes/caches for other
    reasons (the trial's own config.json, and its local task-download cache).
    Falls back to DEFAULT_PROMPT_TIMEOUT_SEC on any failure -- missing files,
    unexpected shape, ambiguous cache match, etc. -- rather than raising.
    """
    try:
        if logs_dir is None:
            return DEFAULT_PROMPT_TIMEOUT_SEC
        trial_dir = logs_dir.parent
        config = json.loads((trial_dir / "config.json").read_text())
        multiplier = float(config["timeout_multiplier"])
        # A legacy name@version trial config's task dict has "path" (a bare
        # task name, e.g. "configure-git-webserver"); a newer org/name
        # package dataset's has "name" instead (namespaced, e.g.
        # "terminal-bench/overfull-hbox") and no "path" key at all -- using
        # the wrong key unconditionally raised KeyError, silently swallowed
        # by this function's own broad except-fallback, so every trial under
        # a package dataset silently used DEFAULT_PROMPT_TIMEOUT_SEC instead
        # of its real per-task budget (confirmed: caused overfull-hbox to be
        # hard-killed by Harbor's own 2250s enforcement while this function
        # still thought it had 3600s left). Confirmed both shapes live on
        # disk, not assumed.
        task_name = config["task"].get("name") or config["task"]["path"]
        # Strip any "<org>/" namespace prefix -- both cache layouts key their
        # task directory by the bare name only (see the two glob patterns
        # below), never the namespaced form.
        task_name = task_name.rsplit("/", 1)[-1]
        # Harbor caches a task's files differently depending on which
        # registry resolved the dataset: a legacy name@version dataset (e.g.
        # "terminal-bench@2.0") caches at <hash>/<task_name>/task.toml, while
        # a newer org/name package dataset (e.g. "terminal-bench/terminal-
        # bench-2-1") caches one level deeper, at
        # packages/<org>/<task_name>/<content-hash>/task.toml. Try both --
        # confirmed by inspecting both live on disk, not assumed.
        matches = list(HARBOR_TASK_CACHE.glob(f"*/{task_name}/task.toml"))
        matches += list(HARBOR_TASK_CACHE.glob(f"packages/*/{task_name}/*/task.toml"))
        if not matches:
            return DEFAULT_PROMPT_TIMEOUT_SEC
        # Ambiguous match (e.g. both an old and a new cache exist for the
        # same task name, as happens right after switching dataset versions)
        # -- prefer the most recently written one over an arbitrary pick.
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        toml_data = tomllib.loads(matches[0].read_text())
        base_timeout_sec = float(toml_data["agent"]["timeout_sec"])
        return base_timeout_sec * multiplier * DEADLINE_SAFETY_MARGIN
    except Exception:
        return DEFAULT_PROMPT_TIMEOUT_SEC

# Same line-dedup + ANSI-strip + truncation used by the TB 1.0 adapter so
# output-format consistency is preserved across benchmarks.
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
MAX_LINES = 200


def _strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s or "")


def _format_output(stdout: str, stderr: str, code: int, cwd: str, timed_out: bool) -> str:
    raw = (stdout or "") + (("\n[stderr]\n" + stderr) if stderr else "")
    cleaned = _strip_ansi(raw).replace("\r", "")
    lines = cleaned.split("\n")
    # dedup
    deduped, last, dup = [], None, 0
    for ln in lines:
        if ln == last:
            dup += 1; continue
        if dup > 0:
            deduped.append(f"  [... {dup} duplicate line(s) collapsed ...]")
        dup = 0
        deduped.append(ln)
        last = ln
    if dup > 0:
        deduped.append(f"  [... {dup} duplicate line(s) collapsed ...]")
    # truncate
    truncated = False
    if len(deduped) > MAX_LINES:
        head, tail = MAX_LINES // 2, MAX_LINES // 4
        skipped = len(deduped) - head - tail
        deduped = deduped[:head] + [f"  [... {skipped} lines truncated ...]"] + deduped[-tail:]
        truncated = True
    body = "\n".join(deduped)
    bits = [f"exit={code}", f"cwd={cwd}", f"timed_out={'true' if timed_out else 'false'}"]
    if truncated: bits.append("output_truncated=true")
    bits.append("backend=harbor-env")
    footer = "[" + " ".join(bits) + "]"
    return f"{body}\n{footer}" if body else footer


class _HarborShellProxy:
    """Stateful shell proxy over harbor's BaseEnvironment.exec().

    harbor's env.exec() is stateless — each call is a fresh shell. To give
    little-coder's ShellSession tool the persistent-cwd / persistent-env
    semantics it expects, we track cwd in the proxy and prepend `cd <cwd>`
    to each command. `pwd` is echoed after the user command so we can
    capture the possibly-updated cwd for the next call.
    """

    def __init__(self, environment: BaseEnvironment, loop: asyncio.AbstractEventLoop, logger: logging.Logger):
        self.env = environment
        self.loop = loop
        self.logger = logger
        self.cwd = "/app"  # TB 2.0 convention — overridden by first `pwd`

    async def _exec_async(self, command: str, timeout: int) -> str:
        sentinel = f"__LC_END_{uuid.uuid4().hex[:8]}__"
        wrapped = f"cd {self.cwd} 2>/dev/null; {{ {command} ; }} ; __rc=$? ; printf '\\n{sentinel}:%d:' $__rc ; pwd"
        try:
            result = await self.env.exec(command=wrapped, timeout_sec=timeout)
        except asyncio.TimeoutError:
            return _format_output("", "command timed out", -1, self.cwd, True)
        except Exception as e:
            return _format_output("", f"env.exec error: {e}", -1, self.cwd, False)

        out = result.stdout or ""
        err = result.stderr or ""
        # Peel sentinel to recover exit code + new cwd
        marker = out.rfind(sentinel + ":")
        code = result.return_code if result.return_code is not None else 0
        if marker >= 0:
            tail = out[marker + len(sentinel) + 1:]
            parts = tail.split(":", 1)
            try: code = int(parts[0])
            except (ValueError, IndexError): pass
            if len(parts) > 1:
                cwd_line = parts[1].lstrip("\r\n").split("\n")
                if cwd_line and cwd_line[0].strip():
                    self.cwd = cwd_line[0].strip()
            out = out[:marker].rstrip()
        return _format_output(out, err, code, self.cwd, False)

    def run(self, command: str, timeout: int) -> str:
        """Sync entry point called by PiRpc's reader thread."""
        fut = asyncio.run_coroutine_threadsafe(self._exec_async(command, timeout), self.loop)
        try:
            return fut.result(timeout=timeout + 30)
        except Exception as e:
            return _format_output("", f"shell proxy error: {e}", -1, self.cwd, False)

    def reset(self) -> str:
        self.cwd = "/app"
        return f"shell reset (cwd → /app)"


class LittleCoderAgent(BaseAgent):
    """Harbor (TB 2.0) adapter for little-coder (v0.1.0+ pi port)."""

    SUPPORTS_ATIF = False

    @staticmethod
    def name() -> str:
        return "little-coder"

    def version(self) -> str | None:
        return _AGENT_VERSION

    async def setup(self, environment: BaseEnvironment) -> None:
        # little-coder runs pi on the host; no in-container setup needed.
        # The environment is used only for command proxying during run().
        pass

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        model = self.model_name or DEFAULT_MODEL
        session_id = f"hb-{uuid.uuid4().hex[:10]}"

        loop = asyncio.get_running_loop()
        proxy = _HarborShellProxy(environment, loop, self.logger)

        def tb_shell_handler(payload: dict) -> str:
            op = payload.get("op")
            if op == "run":
                return proxy.run(payload.get("command", ""), int(payload.get("timeout", 30)))
            if op == "reset":
                return proxy.reset()
            return f"Error: unknown ShellSession op '{op}'"

        prompt = (
            "You are solving a Terminal-Bench 2.0 task inside a Linux container.\n"
            "The ONLY way to interact with the container is the ShellSession tool; "
            "its cwd persists between calls (tracked by the adapter). Any shell "
            "command is available — no command whitelist applies in this benchmark "
            "(Docker is the isolation boundary, not a shell filter), so use `cd`, "
            "compilers, daemonizing tools (setsid/nohup), package managers, etc. "
            "directly as needed, the same as any other terminal session.\n"
            "Default working directory is /app. `cd <path>` (standalone or as the "
            "first part of a `&&` chain) persists across every subsequent "
            "ShellSession call.\n"
            "File tools like Read/Write/Edit are NOT available — use shell commands "
            "(cat, sed -i, heredoc 'cat > file <<EOF') through ShellSession instead.\n\n"
            "Approach: briefly research the task first (inspect the relevant files, "
            "commands, or error output to understand what's actually being asked), "
            "form a short plan, then implement a quick first-pass solution rather "
            "than exhaustively enumerating options before writing anything. Get a "
            "working attempt in place early, verify it, and refine from there if "
            "time remains.\n\n"
            "Verification: when the instructions describe a specific client-side "
            "interface (e.g. `ssh`/`git clone user@host:...`, a particular port, "
            "protocol, or auth method), test through that exact path rather than a "
            "simplified local substitute — a substitute that skips part of the real "
            "flow (e.g. a local clone instead of an SSH one) can falsely appear to "
            "pass while the actual grading exercises a code path you never tested. "
            "If the instructions say you don't need to worry about some part of the "
            "setup (like login credentials), that usually still means the server "
            "side of it must be functional and reachable, just that you're not "
            "responsible for the client's half.\n\n"
            f"TASK:\n{instruction}\n\n"
            "When the task is complete, stop calling tools and say 'done'."
        )

        log_path = self.logs_dir / "little_coder.log"
        log_fh = log_path.open("w") if self.logs_dir else None

        # little_coder.log (above) is only written once prompt_and_collect
        # finally returns, so `tail -f` on it shows nothing for the whole
        # trial. This second file is flushed turn-by-turn via PiRpc's
        # on_event callback (see rpc_client.py's prompt_and_collect) so a
        # trial can actually be watched live, not just post-mortemed.
        live_log_path = self.logs_dir / "little_coder.live.log"
        live_log_fh = live_log_path.open("w") if self.logs_dir else None
        pending_text: list[str] = []

        def on_event(ev: dict) -> None:
            if live_log_fh is None:
                return
            t = ev.get("type")
            if t == "message_update":
                delta = ev.get("assistantMessageEvent", {})
                if delta.get("type") == "text_delta":
                    pending_text.append(delta.get("delta", ""))
                return
            if t == "tool_execution_start":
                if pending_text:
                    live_log_fh.write("".join(pending_text) + "\n")
                    pending_text.clear()
                live_log_fh.write(
                    f">> {ev.get('toolName', '')}({ev.get('args', {})})\n"
                )
                live_log_fh.flush()
            elif t == "tool_execution_end":
                res = ev.get("result", {})
                content = res.get("content", [])
                text = "\n".join(
                    c.get("text", "") for c in content if c.get("type") == "text"
                )
                live_log_fh.write(f"<< {text[:400]}\n")
                live_log_fh.flush()
            elif t == "agent_end":
                if pending_text:
                    live_log_fh.write("".join(pending_text) + "\n")
                    pending_text.clear()
                live_log_fh.write("=== agent_end ===\n")
                live_log_fh.flush()

        effective_timeout_sec = _resolve_trial_timeout_sec(self.logs_dir)
        self.logger.info(
            f"LittleCoderAgent: effective trial timeout={effective_timeout_sec:.0f}s"
        )
        deadline_epoch_ms = int((time.time() + effective_timeout_sec) * 1000)

        try:
            # PiRpc spawns pi --mode rpc and wires the shell proxy. The reader
            # thread invokes tb_shell_handler synchronously; the handler
            # bridges to this event loop via run_coroutine_threadsafe.
            rpc = await asyncio.to_thread(
                PiRpc,
                model=model,
                cwd=str(Path.cwd()),
                benchmark="terminal_bench",
                allowed_tools=DEFAULT_ALLOWED_TOOLS,
                session_id=session_id,
                tb_mode=True,
                # 40 was too tight for tasks needing several iterative
                # debug-retrain cycles (e.g. train-fasttext hit 41/40 turns
                # one call away from its correct final fix -- confirmed via
                # agent_result.metadata.n_turns in the trial's result.json).
                max_turns=80,
                tb_shell_handler=tb_shell_handler,
                # permission-gate's SAFE_PREFIXES whitelist is meant to guard
                # a real user's own machine during interactive use; its own
                # header comment already documents the opt-out for exactly
                # this context: "'accept-all' mode all commands pass
                # (benchmark runs set this explicitly)". Docker is the actual
                # isolation boundary for a TB trial, not the whitelist, and
                # every other TB agent (bare pi, codex) already runs here
                # with unrestricted tool access -- so withholding it only
                # from little-coder was an unfair, unintentional handicap,
                # not a deliberate safety choice. Observed directly: fix-git
                # blocked on `cd`/`git -C` (no way to work outside /app),
                # prove-plus-comm blocked on `coqc` (wrote a correct proof,
                # couldn't compile it), configure-git-webserver blocked on
                # `setsid`/`nc`/`socat`/`crontab` (no way to daemonize the
                # server the task needed running) -- three different tools
                # across three unrelated tasks, not a pattern fixable by
                # allow-listing one command at a time.
                env={
                    "LITTLE_CODER_PERMISSION_MODE": "accept-all",
                    "LITTLE_CODER_DEADLINE_EPOCH_MS": str(deadline_epoch_ms),
                },
            )
            try:
                result = await asyncio.to_thread(
                    rpc.prompt_and_collect,
                    prompt,
                    effective_timeout_sec,
                    on_event,
                )
                stop_reason = getattr(result, "stop_reason", "unknown")
                if log_fh:
                    # Distinguishes a crashed pi from a model that ran long;
                    # both used to look identical here.
                    log_fh.write(f"=== stop_reason: {stop_reason} ===\n")
                    log_fh.write(f"=== assistant text ===\n{result.assistant_text}\n\n")
                    for tc in result.tool_calls:
                        log_fh.write(f">> {tc['name']}({tc.get('args', {})})\n")
                        preview = (tc.get("result_text", "") or "")[:400]
                        log_fh.write(f"<< {preview}\n")
                    notes = rpc.notifications() if hasattr(rpc, "notifications") else []
                    if notes:
                        log_fh.write(f"\n=== pi notifications ({len(notes)}) ===\n")
                        for n in notes:
                            log_fh.write(f"[{n.get('notifyType','info')}] {n.get('message','')}\n")
                    stderr = rpc.stderr()
                    if stderr:
                        log_fh.write(f"\n=== pi stderr ===\n{stderr}\n")
                # Harbor's AgentContext: populate what we can. Token usage
                # isn't currently plumbed through pi-ai; leave None.
                context.metadata = {
                    "stop_reason": stop_reason,
                    "n_tool_calls": len(result.tool_calls),
                    "n_turns": result.turn_count,
                    "n_compactions": result.compaction_events,
                    "n_notifications": len(rpc.notifications()) if hasattr(rpc, "notifications") else 0,
                    "little_coder_version": self.version(),
                    "benchmark": "terminal_bench_2.0",
                }
            finally:
                await asyncio.to_thread(rpc.close, 3)
        except Exception as e:
            self.logger.error(f"LittleCoderAgent run failed: {e}")
            if log_fh:
                log_fh.write(f"\nAGENT ERROR: {e}\n")
            raise
        finally:
            if log_fh:
                log_fh.flush()
                log_fh.close()
            if live_log_fh:
                # pending_text is only flushed inline on tool_execution_start
                # / agent_end. A trial that ends on "deadline" or
                # "process_exit" (no agent_end event -- routine: finalize-warn
                # itself notes trials regularly end this way) would otherwise
                # lose whatever trailing assistant text was still buffered.
                if pending_text:
                    live_log_fh.write("".join(pending_text) + "\n")
                    pending_text.clear()
                live_log_fh.flush()
                live_log_fh.close()
