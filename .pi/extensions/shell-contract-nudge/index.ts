import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { SHELL_TOOLS, scan, splitCommandChain, stripHeredocBodies } from "../_shared/shell-write.ts";
import { harnessIntervention } from "../_shared/intervention.ts";
import { allowedToolSet, toolsAvailable } from "../_shared/allowed-tools.ts";

// shell-contract-nudge: steering-only mitigation for two real hazards baked
// into this harness's `bash -c "<full raw command text>"` contract (the
// vendored pi package runs every bash/ShellSession/ShellStart call this way,
// detached, with a short grace period before the harness tears down that
// process's stdout/stderr pipes):
//
// 1. A backgrounded job (`some_long_job &`) that still shares the tool
//    call's stdout/stderr pipes gets SIGPIPE'd/killed shortly after the tool
//    call returns, because the harness destroys those pipes on a short timer
//    regardless of whether a backgrounded grandchild is still writing to
//    them. `.pi/extensions/bg-shell` registers a `ShellStart` tool that
//    solves this properly with a real, persistent job registry — the model
//    just doesn't always know to reach for it instead of a bare `&`.
//
// 2. The full command text is a literal, unscrubbed argv element of the
//    `bash -c` wrapper process. A later `pkill -f <pattern>` (or `killall`)
//    call can therefore match and kill the WRAPPER PROCESS OF ITS OWN
//    CURRENT COMMAND — not just the intended target — whenever `<pattern>`
//    happens to appear in that command's own text.
//
// Neither hazard can be fixed from here (the `bash -c` contract lives in the
// vendored pi package, not this repo), so this extension only steers: detect
// the risky shape, always log a diagnostic, and — once per session per
// pattern — send a steering nudge. It never blocks or edits the call; a
// one-shot background job or a legitimate `pkill -f` can both be exactly
// what's intended.

// ---------------------------------------------------------------------------
// Bare-`&` detection
// ---------------------------------------------------------------------------
//
// Built directly on `scan` (the shared quote/escape-tracking character walk)
// rather than on `splitCommandChain`, so this works correctly regardless of
// whether a separate, concurrent change elsewhere makes `splitCommandChain`
// itself start cutting on bare `&` as a chain operator — see `subSegments`
// below for how the two designs coexist either way.
//
// Accepted, not-fixed limitation: this scanner has no `#`-comment awareness
// and no arithmetic-context awareness (`$(( a & b ))` is bitwise-AND, not
// backgrounding) — both can still false-positive here. Advisory-only, so a
// false positive costs nothing but a spurious nudge; low value to fix.

/**
 * Indices of unquoted `&` in `strippedCmd` (already heredoc-stripped) that are
 * genuine backgrounding operators — not part of `&&`, `&>`, `>&`, `|&`, `<&`,
 * or `;&`/`;;&`.
 */
function bareAmpIndices(strippedCmd: string): number[] {
  const out: number[] = [];
  scan(strippedCmd, (ch, i, quoted) => {
    if (quoted || ch !== "&") return;
    const prev = strippedCmd[i - 1];
    const next = strippedCmd[i + 1];
    if (next === "&" || prev === "&") return; // && (either char of the pair)
    if (next === ">") return; // &> / &>>
    if (prev === ">") return; // >& fd-dup out (2>&1, >&2, >&-)
    if (prev === "<") return; // <& fd-dup in  (read x <&3, exec 0<&-)
    if (prev === "|") return; // |&
    if (prev === ";") return; // ;& and ;;& case-statement fallthrough
    out.push(i);
  });
  return out;
}

/**
 * Shared sub-segmentation: split `raw` on `splitCommandChain`'s chain
 * operators, then further split each of those segments on its own bare-`&`
 * indices. Used by both the wait-suppression below and the pkill/killall
 * detector, so e.g. `long_job & pkill -f watchdog` is seen as two segments by
 * both.
 *
 * Deliberately order-independent of whether `splitCommandChain` itself ever
 * starts cutting on bare `&`: if/when it does, this sub-split simply finds
 * nothing left to cut on a given segment — same end result either way.
 */
function subSegments(raw: string): string[] {
  const out: string[] = [];
  for (const seg of splitCommandChain(raw)) {
    let start = 0;
    for (const i of bareAmpIndices(seg)) {
      out.push(seg.slice(start, i));
      start = i + 1;
    }
    out.push(seg.slice(start));
  }
  return out.filter((s) => s.trim().length > 0);
}

// A later top-level segment beginning with `wait` is the safe, common
// `job1 & job2 & wait` idiom (parallel launch then block until all finish —
// nothing outlives the tool call), so a candidate `&` is suppressed when one
// exists after it.
const WAIT_RE = /^wait(\s|$)/;

function followedByWait(strippedCmd: string, ampIndex: number): boolean {
  return subSegments(strippedCmd.slice(ampIndex + 1)).some((seg) => WAIT_RE.test(seg.trim()));
}

// Suppresses a candidate `&` whose immediately preceding text already
// redirects both stdout and stderr away from the tool's shared pipes — this
// also fixes a self-inflicted problem: the nudge's own recommended fallback
// text (see BARE_AMP_NUDGE below) contains a literal bare `&`, and would
// otherwise re-trigger this same diagnostic forever.
//
// Accepted gap: a quoted filename right before the `&` (e.g.
// `> "my log" 2>&1 &`) can miss this and go unsuppressed — accepted, since
// the cost is only a spurious advisory, never a functional problem.
const BOTH_STREAMS_REDIRECTED_RE = /(?:&>>?\s*\S+|>>?\s*\S+\s+2>>?\s*(?:&1|\S+))\s*$/;

function bothStreamsRedirectedBefore(strippedCmd: string, ampIndex: number): boolean {
  return BOTH_STREAMS_REDIRECTED_RE.test(strippedCmd.slice(0, ampIndex));
}

/** True when `raw` backgrounds a job with a bare `&` that shares this tool call's pipes. */
export function hasBareBackgroundAmpersand(raw: string): boolean {
  const stripped = stripHeredocBodies(raw);
  return bareAmpIndices(stripped).some(
    (i) => !followedByWait(stripped, i) && !bothStreamsRedirectedBefore(stripped, i),
  );
}

// ---------------------------------------------------------------------------
// pkill/killall detection
// ---------------------------------------------------------------------------

// `-f`, `-fx`, `-xf`, `-fi`, … — any short-flag cluster that includes `f`.
// `pkill`'s `-f` (match against the full argv, not just the process name) is
// specifically what makes it able to see the current command's own wrapper
// text; `pkill somename` alone (no `-f`) only matches on the short process
// name and doesn't have this hazard.
const PKILL_FULL_FLAG_RE = /^-[A-Za-z]*f[A-Za-z]*$/;

// Wrapper commands that just re-exec their remaining argv as the real
// command — skipped (along with their own flags) so e.g. `sudo pkill -f x`
// or `env FOO=bar nice pkill -f x` still resolve to `pkill` underneath.
const WRAPPERS = new Set([
  "sudo",
  "doas",
  "exec",
  "env",
  "nice",
  "ionice",
  "nohup",
  "stdbuf",
  "time",
  "command",
]);

/**
 * True when `segment` invokes `pkill -f`/`pkill --full` or `killall`, seeing
 * through a leading run of wrapper commands (sudo/env/nice/timeout/…),
 * `VAR=val` assignments, and `command`'s existence-check form.
 *
 * Deliberately word-based rather than a full argv parse — this only feeds an
 * advisory nudge, never a gate, so a missed exotic invocation is low-stakes.
 *
 * Accepted gap, not fixed: a wrapper's separated value-flag (space between
 * flag and value, e.g. `nice -n 10 pkill`, `sudo -u www pkill`) may cause the
 * walk to stop one word too early and miss the real command — rare,
 * advisory-only.
 */
function segmentInvokesPkillOrKillall(segment: string): boolean {
  const words = segment.trim().split(/\s+/).filter(Boolean);
  let i = 0;
  while (i < words.length) {
    const name = words[i].replace(/^.*\//, ""); // drop a leading path, e.g. /usr/bin/pkill
    if (/^\w+=/.test(name)) {
      i++; // VAR=val leading assignment (with or without `env`)
      continue;
    }
    if (name === "command" && (words[i + 1] === "-v" || words[i + 1] === "-V")) {
      return false; // pure existence check, runs nothing — do not flag
    }
    if (name === "timeout") {
      i++;
      while (i < words.length && words[i].startsWith("-")) i++; // timeout's own flags
      i++; // timeout's mandatory DURATION operand
      continue;
    }
    if (WRAPPERS.has(name)) {
      i++;
      while (i < words.length && words[i].startsWith("-")) i++; // wrapper's own flags
      continue;
    }
    break; // reached the real command word
  }
  const name = i < words.length ? words[i].replace(/^.*\//, "") : "";
  if (!name) return false;
  if (name === "killall") return true;
  return (
    name === "pkill" &&
    words.slice(i + 1).some((w) => PKILL_FULL_FLAG_RE.test(w) || w === "--full")
  );
}

/**
 * True when any segment of `cmd` runs `pkill -f`/`pkill --full` or `killall`.
 * Segmented via `subSegments` (not plain `splitCommandChain`) so a bare-`&`
 * backgrounded job followed by a pkill call (e.g.
 * `long_job & pkill -f watchdog`) is seen as two separate segments.
 */
export function hasPkillOrKillall(cmd: string): boolean {
  return subSegments(stripHeredocBodies(cmd)).some(segmentInvokesPkillOrKillall);
}

// ---------------------------------------------------------------------------
// Extension wiring
// ---------------------------------------------------------------------------

let nudgedBareAmp = false;
let nudgedPkill = false;

function bareAmpNudgeText(allowed: Set<string> | undefined): string {
  const shellStartAvailable = toolsAvailable(["ShellStart", "ShellList", "ShellLog"], allowed);
  const redirectSentence =
    "If you must background inline instead, redirect both streams to a file — " +
    "`nohup cmd > job.log 2>&1 &` — so the job stops sharing this tool call's pipes.";

  if (!shellStartAvailable) {
    return (
      "Heads up: a bare trailing `&` backgrounds a job on this harness's shared " +
      "stdout/stderr pipes — those pipes get torn down shortly after this tool " +
      "call returns, and the backgrounded job can be SIGPIPE'd/killed once that " +
      `happens, even though it's still running. ${redirectSentence}`
    );
  }

  return (
    "Heads up: a bare trailing `&` backgrounds a job on this harness's shared " +
    "stdout/stderr pipes — those pipes get torn down shortly after this tool " +
    "call returns, and the backgrounded job can be SIGPIPE'd/killed once that " +
    "happens, even though it's still running. For anything that needs to keep " +
    "running past this tool call, use the ShellStart tool (bg-shell's job " +
    "registry) instead of a bare `&` — it keeps the job alive and lets you " +
    `check on it later with ShellList/ShellLog. ${redirectSentence}`
  );
}

const PKILL_NUDGE =
  "Heads up: this harness runs your command as a literal, unscrubbed argv " +
  "element of a `bash -c \"<command>\"` wrapper process. `pkill -f <pattern>` " +
  "(and `killall`) matches against full command lines, so if `<pattern>` " +
  "appears anywhere in the current command's own text, it can match and kill " +
  "that wrapper process itself — not just the intended target. Prefer a more " +
  "specific pattern that can't match your own invocation, or kill by PID " +
  "(e.g. from `pgrep`, or a PID you saved earlier) instead of `-f` matching " +
  "on full command text.";

export default function (pi: ExtensionAPI) {
  pi.on("session_start", async () => {
    nudgedBareAmp = false;
    nudgedPkill = false;
  });

  pi.on("tool_call", async (event, ctx) => {
    const toolName = String((event as any).toolName ?? "");
    if (!SHELL_TOOLS.has(toolName)) return;
    const cmd = ((event as any).input ?? {})?.command;
    if (typeof cmd !== "string" || !cmd) return;

    // bg-shell's ShellStart starts a job in its own persistent job registry —
    // jobs there outlive the tool call by design, so the pipe-teardown hazard
    // this branch warns about doesn't apply to it. The pkill/killall branch
    // below still applies to ShellStart: the self-pkill hazard is about the
    // wrapper process's own argv text, unrelated to which registry mechanism
    // started the job.
    if (toolName !== "ShellStart" && hasBareBackgroundAmpersand(cmd)) {
      ctx.ui.notify(
        `shell-contract-nudge: "${toolName}" call backgrounds a job with a bare "&" — ` +
          "its stdout/stderr pipes get torn down shortly after this call returns.",
        "info",
      );
      if (!nudgedBareAmp) {
        let sent = false;
        try {
          pi.sendUserMessage(bareAmpNudgeText(allowedToolSet()), { deliverAs: "steer" });
          sent = true;
        } catch {
          // SDK without sendUserMessage, or the send otherwise failed — leave
          // the latch false so a later offending call can try again, mirroring
          // finalize-warn/quality-monitor's own "don't burn the one-shot latch
          // on a nudge that was never delivered" convention.
        }
        // harnessIntervention fires only on a successfully-delivered steer —
        // mirroring tb-finalize-guard's own send-then-intervene ordering — so
        // it's paired 1:1 with a genuine redirect, not every detection.
        if (sent) {
          nudgedBareAmp = true;
          harnessIntervention(
            ctx,
            `"${toolName}" call backgrounds a job with a bare "&" — steering the model ` +
              "toward ShellStart/a fully-redirected background invocation instead.",
          );
        }
      }
    }

    if (hasPkillOrKillall(cmd)) {
      ctx.ui.notify(
        `shell-contract-nudge: "${toolName}" call runs pkill -f/killall — full-command-line ` +
          "matching can hit this call's own bash -c wrapper process.",
        "info",
      );
      if (!nudgedPkill) {
        let sent = false;
        try {
          pi.sendUserMessage(PKILL_NUDGE, { deliverAs: "steer" });
          sent = true;
        } catch {
          // See the bare-`&` branch above — don't burn the latch on a nudge
          // that never actually sent.
        }
        if (sent) {
          nudgedPkill = true;
          harnessIntervention(
            ctx,
            `"${toolName}" call runs pkill -f/killall — steering the model toward a more ` +
              "specific pattern or PID-based kill instead.",
          );
        }
      }
    }

    // Advisory only: never block or rewrite the call, and no return value —
    // a one-shot background job or a legitimate pkill -f can be exactly
    // right, so this extension only ever informs.
  });
}
