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
import os
import re
import subprocess
import sys
import time
import tomllib
import uuid
from pathlib import Path

# Repo root, derived the same way _read_version_from_package_json() finds
# package.json -- benchmarks/harbor_adapter/little_coder_agent.py is two
# levels below it.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_version_from_package_json() -> str:
    """Read agent version from the repo's package.json at import time.

    Avoids hardcoded version drift between the adapter's version() return
    and the actual released tag. Falls back to "unknown" if the file is
    missing or malformed.
    """
    try:
        pkg = _REPO_ROOT / "package.json"
        return json.load(open(pkg)).get("version", "unknown")
    except Exception:
        return "unknown"


def _read_code_sha() -> str:
    """Best-effort short git SHA of this repo's HEAD, computed once at
    module IMPORT time -- not per-trial. Import time is exactly when this
    process's copy of the adapter code was frozen, so a Harbor job that has
    been running for hours on stale code still reports its OWN stale SHA
    here, unaffected by anything landing on disk afterward. That's the
    missing positive signal from the stale-code incident (dfa706b's
    max_turns fix landed mid-job at 2026-09-08 22:26, but the already-running
    job kept enforcing the pre-fix behavior for hours with nothing recording
    which code was actually active). Falls back to "unknown" for a non-git
    checkout or missing git binary, mirroring
    _read_version_from_package_json()'s fallback.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        sha = out.stdout.strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


_AGENT_VERSION = _read_version_from_package_json()
_CODE_SHA = _read_code_sha()
try:
    _ADAPTER_MTIME = Path(__file__).resolve().stat().st_mtime
except Exception:
    _ADAPTER_MTIME = None


from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

# benchmarks/ isn't a package — let the importer resolve by sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rpc_client import PiRpc, capture_environment_snapshot  # noqa: E402


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
    # over-estimate gets the trial hard-killed by Harbor mid-write (the
    # original incident's failure mode) -- so the minimum is the safer pick.
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
    timestamps), so a naive "union both globs, newest mtime wins" always
    prefers the legacy file whenever a task name happens to exist in both
    layouts (confirmed: all TB2.1 tasks with a same-named TB2.0 predecessor
    hit this, e.g. caffe-cifar-10 resolving 1200s instead of 3600s,
    crack-7z-hash resolving 900s instead of 1800s). The trial's own
    config.json shape (task.name+ref vs task.path) tells us unambiguously
    which layout THIS trial belongs to -- use that instead of mtime.
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
        # "terminal-bench/overfull-hbox") and no "path" key at all -- using
        # the wrong key unconditionally raised KeyError, silently swallowed
        # by this function's own broad except-fallback, so every trial under
        # a package dataset silently used DEFAULT_PROMPT_TIMEOUT_SEC instead
        # of its real per-task budget (confirmed: caused overfull-hbox to be
        # hard-killed by Harbor's own 2250s enforcement while this function
        # still thought it had 3600s left). Confirmed both shapes live on
        # disk, not assumed.
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


def _derive_benchmark_label(logs_dir: Path | None) -> str:
    """Best-effort derivation of this trial's own dataset identity, for
    context.metadata["benchmark"] -- replacing the hardcoded
    "terminal_bench_2.0" (wrong now that harbor_pilot.sh defaults to TB2.1;
    Codex finding [medium]).

    Reads the same trial config.json _resolve_trial_timeout_info reads
    (confirmed on disk: both the package shape -- task.name/ref/source --
    and the legacy shape -- task.path/git_url/git_commit_id/source -- carry
    a "source" key giving the dataset identity directly, e.g.
    "terminal-bench/terminal-bench-2-1"). Legacy datasets can be re-pinned to
    different commits under the same name@version string, so a short
    git_commit_id is appended when present to disambiguate; package datasets
    are already uniquely identified by "source" alone (the content-hash
    "ref" is a task-cache lookup key, not part of the dataset's own
    identity).

    Never raises -- a label derivation failure must not fail a trial; falls
    back to "terminal_bench_unknown". This only changes what's reported in
    Harbor's own result metadata; the LITTLE_CODER_BENCHMARK env var
    (consumed by benchmark-profiles for model-profile selection) is a
    separate, unaffected mechanism and always stays "terminal_bench".
    """
    try:
        if logs_dir is None:
            return "terminal_bench_unknown"
        trial_dir = logs_dir.parent
        config = json.loads((trial_dir / "config.json").read_text())
        task = config["task"]
        source = task.get("source")
        if not source:
            return "terminal_bench_unknown"
        if task.get("name") is not None:
            # Package shape -- source alone is the dataset identity.
            return source
        # Legacy shape -- append a short commit id when present.
        commit = task.get("git_commit_id")
        if commit:
            return f"{source}@{commit[:12]}"
        return source
    except Exception:
        return "terminal_bench_unknown"


def _build_environment_snapshot(
    model: str,
    *,
    max_turns: int,
    ambient_max_turns_env: str | None,
    timeout_info: dict,
) -> dict:
    """Assemble the per-trial environment_snapshot.json payload (Plan 5,
    fix D): rpc_client.capture_environment_snapshot()'s existing pi-config
    introspection, plus the config values this adapter itself resolved --
    the active turn cap, the ambient env var seen at process entry, this
    process's own code identity, and the FULL timeout-provenance dict from
    _resolve_trial_timeout_info() (not just the effective float) so a
    reader can tell exactly which task.toml/cache-layout produced it.

    Split out from run() so it's unit-testable without spinning up a real
    PiRpc/environment. Never raises on its own logic; capture_environment_
    snapshot() already guarantees no-raise for its half.
    """
    snapshot = capture_environment_snapshot(model)
    snapshot["max_turns"] = max_turns
    snapshot["ambient_max_turns_env"] = ambient_max_turns_env
    snapshot["little_coder_version"] = _AGENT_VERSION
    snapshot["code_sha"] = _CODE_SHA
    snapshot["adapter_file"] = str(Path(__file__).resolve())
    snapshot["adapter_mtime"] = _ADAPTER_MTIME
    snapshot["timeout_provenance"] = timeout_info
    return snapshot

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


def _resolve_token_usage(result_usage: dict, turn_count: int, stats: dict | None) -> dict:
    """Pick a token-usage source and compute the Harbor AgentContext mapping.

    Plan 7 (token-usage-tracking): pi computes full per-turn `Usage` and a
    session-cumulative `get_session_stats`, but Harbor's `result.json` fields
    were left null since nothing read either. `stats` (PiRpc.session_stats()'s
    return value) is preferred when available -- it's session-complete,
    including tool-result usage and compaction/branch-summary generation
    tokens (docs/rpc.md), which summing `turn_end` events alone misses.
    `result_usage` (PromptResult.usage, itself summed from `turn_end` events)
    is the fallback for when pi already died (process_exit/deadline paths) or
    is running an older build without `get_session_stats` -- exactly the
    trials where token counts matter most, per the plan's rejected
    alternatives.

    Pulled out as a standalone function (rather than inlined in run()) so it
    can be unit-tested without needing a live PiRpc/AgentContext/harbor
    environment -- pure function of already-fetched data.

    Returns a dict with "n_input_tokens", "n_cache_tokens", "n_output_tokens",
    "cost_usd" (Harbor AgentContext field values) plus "raw" (the flat
    input/output/cache_read/cache_write/cost breakdown) and "token_source"
    ("session_stats" | "event_sum" | "unavailable") for context.metadata.
    """
    def _num(v):
        # bool is an int subclass -- exclude it so a stray True doesn't
        # silently count as 1. Anything else non-numeric (a string, a dict,
        # None, ...) coerces to 0 rather than propagating a TypeError into
        # `input_tok + cache_read + cache_write` or `cost > 0` below, which
        # would otherwise fail an OTHERWISE-SUCCESSFUL Harbor trial purely
        # over token telemetry -- untrusted wire data (stats is pi's
        # get_session_stats response; result_usage is itself summed from
        # equally-untrusted turn_end events) must not do that.
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else 0

    if isinstance(stats, dict) and isinstance(stats.get("tokens"), dict):
        tok = stats["tokens"]
        input_tok = _num(tok.get("input", 0))
        output_tok = _num(tok.get("output", 0))
        cache_read = _num(tok.get("cacheRead", 0))
        cache_write = _num(tok.get("cacheWrite", 0))
        cost = _num(stats.get("cost", 0))
        # token_source reflects that a session_stats payload WAS available
        # and had a "tokens" dict -- even if every field inside it turned out
        # to be non-numeric junk that _num() coerced to 0. Don't silently
        # reclassify that as event_sum/unavailable; the source was real.
        token_source = "session_stats"
    else:
        input_tok = _num(result_usage.get("input", 0))
        output_tok = _num(result_usage.get("output", 0))
        cache_read = _num(result_usage.get("cache_read", 0))
        cache_write = _num(result_usage.get("cache_write", 0))
        cost = _num(result_usage.get("cost", 0))
        # No turn_end ever fired (e.g. pi crashed before completing a single
        # turn) -- there is nothing meaningful to have summed, so this is
        # genuinely "unavailable" rather than a real (zero) event sum.
        # Distinct from a provider that legitimately reports zero usage for
        # turns that DID complete -- that case is still "event_sum", just
        # with zero-valued fields.
        token_source = "event_sum" if turn_count > 0 else "unavailable"

    return {
        "n_input_tokens": input_tok + cache_read + cache_write,
        "n_cache_tokens": cache_read,
        "n_output_tokens": output_tok,
        "cost_usd": cost if cost > 0 else None,
        "token_source": token_source,
        "raw": {
            "input": input_tok,
            "output": output_tok,
            "cache_read": cache_read,
            "cache_write": cache_write,
            "cost": cost,
        },
    }


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

        # No turn cap: 40 was too tight (train-fasttext hit 41/40, one call
        # from its correct final fix), so it was raised to 80 -- which
        # mteb-leaderboard then hit at 80/80, again one investigation away
        # from a real answer (it had already built a correct top-30
        # leaderboard table and was fetching the required historical
        # snapshot when the cap fired). Any fixed N is a guess at how many
        # turns a task needs, and task difficulty varies enormously -- this
        # is whack-a-mole, not a fix. Wall-clock (finalize-warn's deadline
        # trigger, _resolve_trial_timeout_sec above) is the more principled
        # boundary: it's what Harbor itself actually enforces as the grading
        # limit, and matches the vendor's own published Terminal-Bench
        # methodology (flat 3h timeout, no turn limit at all). Local
        # inference has no per-token cost, so the downside of no cap -- a
        # genuinely stuck trial burning its full wall-clock budget instead
        # of aborting early and cheaply -- is an acceptable trade against
        # truncating trials that are still making real progress. Explicit 0
        # rather than leaving it unset, so the "no cap" choice reads as
        # deliberate, not an oversight -- and, since rpc_client.py now
        # writes the env var on `is not None` rather than truthiness (Plan
        # 5, fix A1), this 0 is guaranteed to actually reach the subprocess
        # rather than being silently skipped.
        #
        # Hoisted to a named local (rather than inlined as the kwarg below)
        # so this log line and the PiRpc kwarg read the exact same value and
        # can never diverge (Plan 5, fix C).
        max_turns = 0
        ambient_max_turns_env = os.environ.get("LITTLE_CODER_MAX_TURNS")
        self.logger.info(
            "LittleCoderAgent: config provenance "
            f"max_turns={max_turns} "
            f"ambient_LITTLE_CODER_MAX_TURNS={ambient_max_turns_env!r} "
            f"code_sha={_CODE_SHA} "
            f"adapter_file={__file__} adapter_mtime={_ADAPTER_MTIME}"
        )

        # Per-trial environment_snapshot.json (Plan 5, fix D): best-effort,
        # must never fail a trial. Executes before the PiRpc-construction
        # try/except below so it (and the config-provenance log line above)
        # still land even if PiRpc itself fails to construct (e.g. PI_BIN
        # missing) -- exactly the diagnostics that failure needs most.
        if self.logs_dir:
            try:
                snapshot = _build_environment_snapshot(
                    model,
                    max_turns=max_turns,
                    ambient_max_turns_env=ambient_max_turns_env,
                    timeout_info=timeout_info,
                )
                (self.logs_dir / "environment_snapshot.json").write_text(
                    json.dumps(snapshot, indent=2, default=str)
                )
            except Exception as e:
                self.logger.warning(f"LittleCoderAgent: environment snapshot failed: {e}")

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
                max_turns=max_turns,
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

                # Token usage: prefer pi's own cumulative get_session_stats
                # (includes tool-result usage and compaction/branch-summary
                # generation -- see PiRpc.session_stats' docstring) and fall
                # back to the per-turn event sum on PromptResult.usage when
                # stats are unavailable (pi already dead on the
                # process_exit/deadline paths, or an older pi build without
                # the get_session_stats command). Queried here, before
                # rpc.close(), because a closed/dead process can't answer it.
                stats = await asyncio.to_thread(rpc.session_stats)
                tokens = _resolve_token_usage(result.usage, result.turn_count, stats)

                # Harbor's AgentContext (harbor/models/agent/context.py) field
                # semantics are underspecified beyond their doc comments, so
                # this mapping is spelled out explicitly:
                #   n_input_tokens: field doc says "including cache" -> add
                #     both cache directions in, not just reads.
                #   n_cache_tokens: report cache READ hits (the reused-token
                #     savings) here; cache_write is mirrored into metadata
                #     separately since Harbor has no dedicated field for it.
                #   cost_usd: local providers (llama.cpp, omlx) report 0 --
                #     None reads as "not priced", matching how other Harbor
                #     agents report an unpriced run, rather than "free".
                context.n_input_tokens = tokens["n_input_tokens"]
                context.n_cache_tokens = tokens["n_cache_tokens"]
                context.n_output_tokens = tokens["n_output_tokens"]
                context.cost_usd = tokens["cost_usd"]

                context.metadata = {
                    "stop_reason": stop_reason,
                    "n_tool_calls": len(result.tool_calls),
                    "n_turns": result.turn_count,
                    "n_compactions": result.compaction_events,
                    "n_notifications": len(rpc.notifications()) if hasattr(rpc, "notifications") else 0,
                    "little_coder_version": self.version(),
                    # Derived from the trial's own config.json rather than
                    # hardcoded (Plan 5, fix E) -- the pilot's default
                    # dataset moved to TB2.1 (see harbor_pilot.sh) and the
                    # old literal no longer matched reality.
                    "benchmark": _derive_benchmark_label(self.logs_dir),
                    "token_usage": tokens["raw"],
                    "token_source": tokens["token_source"],
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
