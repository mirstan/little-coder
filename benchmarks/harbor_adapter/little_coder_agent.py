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


def _fallback_timeout_info() -> dict:
    return {
        "cache_layout": None,
        "resolution": "fallback-default",
        "selected_task_toml": None,
        "base_timeout_sec": None,
        "multiplier": None,
        "effective_timeout_sec": DEFAULT_PROMPT_TIMEOUT_SEC,
    }


_MAX_CANDIDATES_RECORDED = 10


def _resolve_pkg_candidates(matches: list[Path], base_resolution: str):
    """Resolve >=1 surviving package-cache task.toml candidates by reading
    each one's [agent].timeout_sec instead of trusting mtime (every real
    package-cache task.toml on disk has an EPOCH mtime, so mtime carries no
    signal here at all -- see this module's cache-layout docstring).

    Returns (match_path, resolution, ambiguous, candidates_considered,
    candidate_task_tomls, candidate_timeouts_sec). Raises on a malformed
    candidate; the caller's broad except degrades that to the default
    timeout like any other unexpected shape.
    """
    timeouts = [float(tomllib.loads(p.read_text())["agent"]["timeout_sec"]) for p in matches]
    considered = len(matches)
    if considered == 1 or len(set(timeouts)) == 1:
        resolution = base_resolution if considered == 1 else f"{base_resolution}-agreed"
        return matches[0], resolution, False, considered, None, None
    # Disagreement: an under-estimate merely finalizes early; an
    # over-estimate risks the trial getting hard-killed by Harbor mid-write
    # -- so the minimum is the safer pick.
    min_idx = min(range(considered), key=lambda i: timeouts[i])
    return (
        matches[min_idx],
        base_resolution,
        True,
        considered,
        [str(p) for p in matches[:_MAX_CANDIDATES_RECORDED]],
        timeouts[:_MAX_CANDIDATES_RECORDED],
    )


def _resolve_trial_timeout_info(logs_dir: Path | None) -> dict:
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

    Returns a provenance dict (see _fallback_timeout_info for the shape) so a
    caller can log/record exactly which file was used and how it was found,
    instead of just the bare float.

    Harbor caches a task's files differently depending on which registry
    resolved the dataset, and the two layouts must NEVER be mixed within one
    lookup: a legacy name@version dataset (e.g. "terminal-bench@2.0") caches
    at <hash>/<task_name>/task.toml (task.toml's mtime is the real download
    time), while a newer org/name package dataset (e.g. "terminal-bench/
    terminal-bench-2-1") caches one level deeper, at packages/<org>/
    <task_name>/<content-hash>/task.toml -- and every package-cache task.toml
    on disk has an EPOCH mtime (Harbor's package extractor doesn't restore
    timestamps). A naive "union both globs, newest mtime wins" therefore
    always prefers the legacy file whenever a task name happens to exist in
    both layouts, silently reading the wrong dataset generation's timeout for
    any task that shares a name with a predecessor from the other layout. The
    trial's own config.json shape (task.name+ref vs task.path) tells us
    unambiguously which layout THIS trial belongs to -- use that instead of
    mtime.
    """
    try:
        if logs_dir is None:
            return _fallback_timeout_info()
        trial_dir = logs_dir.parent
        config = json.loads((trial_dir / "config.json").read_text())
        multiplier = float(config["timeout_multiplier"])
        task = config["task"]
        # A legacy name@version trial config's task dict has "path" (a bare
        # task name, e.g. "configure-git-webserver"); a newer org/name
        # package dataset's has "name" instead (namespaced, e.g.
        # "terminal-bench/some-task") and no "path" key at all -- using the
        # wrong key unconditionally raised KeyError, silently swallowed by
        # this function's own broad except-fallback, so every trial under a
        # package dataset silently used DEFAULT_PROMPT_TIMEOUT_SEC instead of
        # its real per-task budget. That's a real risk, not just a wrong
        # number: a task under-timed this way can be hard-killed by Harbor's
        # own enforcement while this function still thinks it has budget
        # left. Both shapes are read from config.json directly, not assumed.
        task_name = task.get("name")
        if task_name is not None:
            # Package shape. task_name is typically namespaced
            # ("<org>/<bare>"); the package cache directory is keyed by
            # (org, bare) separately, plus a content-hash directory equal to
            # task["ref"]'s sha256 hex when present -- giving an exact,
            # unambiguous path with no heuristics needed.
            org, _, bare = task_name.rpartition("/")
            org = org or "*"
            ref = task.get("ref") or ""
            match_path = None
            candidates_considered = 1
            ambiguous = False
            candidate_task_tomls = None
            candidate_timeouts_sec = None
            if ref.startswith("sha256:"):
                ref_hex = ref.split(":", 1)[1]
                candidate = HARBOR_TASK_CACHE / "packages" / org / bare / ref_hex / "task.toml"
                if org != "*" and candidate.exists():
                    match_path = candidate
                    resolution = "exact-ref"
            if match_path is None and ref.startswith("sha256:"):
                # Wildcard org, exact content hash: task_name had no
                # "<org>/" prefix (a bare package-shape name), which used to
                # force org="*" and skip the exact-ref fast path above
                # entirely -- even though the content hash alone already
                # identifies a real, unambiguous directory regardless of
                # which org it lives under. This outranks the
                # generation-mixing fallback glob below because the hash
                # match is exact.
                ref_hex = ref.split(":", 1)[1]
                glob_matches = sorted(
                    HARBOR_TASK_CACHE.glob(f"packages/*/{bare}/{ref_hex}/task.toml")
                )
                if glob_matches:
                    (
                        match_path, resolution, ambiguous, candidates_considered,
                        candidate_task_tomls, candidate_timeouts_sec,
                    ) = _resolve_pkg_candidates(glob_matches, "exact-ref-glob")
            if match_path is None:
                # Ref missing/mismatched (e.g. cache evicted and re-fetched
                # under a different content hash) -- fall back to a glob
                # restricted to the package layout ONLY. Never fall through
                # to the legacy glob here; that would silently read a
                # different dataset generation's metadata.
                pkg_matches = sorted(HARBOR_TASK_CACHE.glob(f"packages/{org}/{bare}/*/task.toml"))
                if not pkg_matches:
                    return _fallback_timeout_info()
                # Package-cache mtimes may all be epoch, and (when org=="*")
                # this glob also spans every org sharing the bare name -- so
                # mtime is never trusted here; _resolve_pkg_candidates reads
                # each candidate's real timeout_sec instead.
                (
                    match_path, resolution, ambiguous, candidates_considered,
                    candidate_task_tomls, candidate_timeouts_sec,
                ) = _resolve_pkg_candidates(pkg_matches, "fallback-glob")
            layout = "package"
        else:
            # Legacy shape.
            bare = task["path"].rsplit("/", 1)[-1]
            legacy_matches = [
                p for p in HARBOR_TASK_CACHE.glob(f"*/{bare}/task.toml")
                if "packages" not in p.relative_to(HARBOR_TASK_CACHE).parts
            ]
            if not legacy_matches:
                return _fallback_timeout_info()
            legacy_matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            match_path = legacy_matches[0]
            resolution = "glob"
            layout = "legacy"
            candidates_considered = 1
            ambiguous = False
            candidate_task_tomls = None
            candidate_timeouts_sec = None

        toml_data = tomllib.loads(match_path.read_text())
        base_timeout_sec = float(toml_data["agent"]["timeout_sec"])
        effective_timeout_sec = base_timeout_sec * multiplier * DEADLINE_SAFETY_MARGIN
        info = {
            "cache_layout": layout,
            "resolution": resolution,
            "selected_task_toml": str(match_path),
            "base_timeout_sec": base_timeout_sec,
            "multiplier": multiplier,
            "effective_timeout_sec": effective_timeout_sec,
            "candidates_considered": candidates_considered,
            "ambiguous": ambiguous,
        }
        if ambiguous:
            info["candidate_task_tomls"] = candidate_task_tomls
            info["candidate_timeouts_sec"] = candidate_timeouts_sec
        return info
    except Exception:
        return _fallback_timeout_info()


def _resolve_trial_timeout_sec(logs_dir: Path | None) -> float:
    """Thin wrapper around _resolve_trial_timeout_info for callers that only
    need the effective timeout float."""
    return _resolve_trial_timeout_info(logs_dir)["effective_timeout_sec"]

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
            "Approach: briefly investigate the task first (inspect the relevant files, "
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
        # Turn boundary counter for the markers below. One prompt_and_collect
        # call can now legitimately span several agent_end events (an
        # abort-then-recover continuation, an auto-retry, ...), so an
        # unqualified "=== agent_end ===" marker (the original version of
        # this closure, and the misdiagnosis evidence cited by this PR's own
        # first commit) is worse than before: a reader tailing the live log
        # would stop at the FIRST agent_end and miss every turn after it.
        # Track turn boundaries explicitly instead -- agent_settled is the
        # only line a reader should treat as "the trial's turn is actually
        # done".
        turn_counter = 0

        def on_event(ev: dict) -> None:
            nonlocal turn_counter
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
            elif t == "agent_start":
                turn_counter += 1
                live_log_fh.write(f"=== turn {turn_counter} start ===\n")
                live_log_fh.flush()
            elif t == "agent_end":
                if pending_text:
                    live_log_fh.write("".join(pending_text) + "\n")
                    pending_text.clear()
                live_log_fh.write(f"=== turn {turn_counter} agent_end (may continue) ===\n")
                live_log_fh.flush()
            elif t == "agent_settled":
                if pending_text:
                    live_log_fh.write("".join(pending_text) + "\n")
                    pending_text.clear()
                live_log_fh.write("=== agent_settled — trial turn complete ===\n")
                live_log_fh.flush()

        timeout_info = _resolve_trial_timeout_info(self.logs_dir)
        effective_timeout_sec = timeout_info["effective_timeout_sec"]
        if timeout_info["resolution"] == "fallback-default":
            self.logger.info(
                f"LittleCoderAgent: timeout resolved from fallback-default: "
                f"-> {effective_timeout_sec:.0f}s"
            )
        else:
            self.logger.info(
                "LittleCoderAgent: timeout resolved from "
                f"{timeout_info['cache_layout']}/{timeout_info['resolution']}: "
                f"{timeout_info['selected_task_toml']} "
                f"base={timeout_info['base_timeout_sec']:.0f}s "
                f"x{timeout_info['multiplier']} -> {effective_timeout_sec:.0f}s"
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
                # No turn cap: 40 was too tight (train-fasttext hit 41/40,
                # one call from its correct final fix), so it was raised to
                # 80 -- which mteb-leaderboard then hit at 80/80, again one
                # investigation away from a real answer (it had already built
                # a correct top-30 leaderboard table and was fetching the
                # required historical snapshot when the cap fired). Any fixed
                # N is a guess at how many turns a task needs, and task
                # difficulty varies enormously -- this is whack-a-mole, not a
                # fix. Wall-clock (finalize-warn's deadline trigger,
                # _resolve_trial_timeout_sec above) is the more principled
                # boundary: it's what Harbor itself actually enforces as the
                # grading limit, and matches the vendor's own published
                # Terminal-Bench methodology (flat 3h timeout, no turn limit
                # at all). Local inference has no per-token cost, so the
                # downside of no cap -- a genuinely stuck trial burning its
                # full wall-clock budget instead of aborting early and
                # cheaply -- is an acceptable trade against truncating
                # trials that are still making real progress. Explicit 0
                # (falsy, same as omitting the kwarg -- see rpc_client.py's
                # `if max_turns:` check) rather than leaving it unset, so the
                # "no cap" choice reads as deliberate, not an oversight.
                max_turns=0,
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
