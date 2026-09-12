import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { SHELL_TOOLS, splitCommandChain } from "../_shared/shell-write.ts";

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

/**
 * True when `segment` (already isolated from `&&`/`||`/`;`/`|`/newline by
 * `splitCommandChain`) ends in a bare, unquoted, backgrounding `&` — as
 * opposed to being part of an `&&` operator.
 *
 * Relies on `splitCommandChain` having already consumed every `&&` (and
 * every other chain operator) into a cut, so a segment it hands back can
 * only still end in `&` if that `&` was a lone character in the original
 * text. The `endsWith("&&")` check below is a second, redundant guard for
 * that same case — kept because it's nearly free and makes the invariant
 * self-evident at the call site rather than resting entirely on
 * `splitCommandChain`'s internals.
 */
function endsWithBareAmpersand(segment: string): boolean {
  const trimmed = segment.trimEnd();
  if (!trimmed.endsWith("&")) return false;
  if (trimmed.endsWith("&&")) return false;
  return true;
}

/** True when any segment of `cmd`'s `;`/`&&`/`||`/`|`-chain backgrounds a job with a bare trailing `&`. */
export function hasBareBackgroundAmpersand(cmd: string): boolean {
  return splitCommandChain(cmd).some(endsWithBareAmpersand);
}

// `-f`, `-fx`, `-xf`, `-fi`, … — any short-flag cluster that includes `f`.
// `pkill`'s `-f` (match against the full argv, not just the process name) is
// specifically what makes it able to see the current command's own wrapper
// text; `pkill somename` alone (no `-f`) only matches on the short process
// name and doesn't have this hazard.
const PKILL_FULL_FLAG_RE = /^-[A-Za-z]*f[A-Za-z]*$/;

/**
 * True when any segment of `cmd` runs `pkill -f`/`pkill --full` or `killall`
 * (a leading `sudo`/`exec` is skipped so `sudo pkill -f x` still matches).
 * Deliberately word-based rather than a full argv parse — this only feeds an
 * advisory nudge, never a gate, so a missed exotic invocation is low-stakes.
 */
export function hasPkillOrKillall(cmd: string): boolean {
  for (const segment of splitCommandChain(cmd)) {
    const words = segment.split(/\s+/).filter(Boolean);
    let i = 0;
    while (words[i] === "sudo" || words[i] === "exec") i++;
    const name = words[i]?.replace(/^.*\//, ""); // drop a leading path, e.g. /usr/bin/pkill
    if (!name) continue;
    if (name === "killall") return true;
    if (
      name === "pkill" &&
      words.slice(i + 1).some((w) => PKILL_FULL_FLAG_RE.test(w) || w === "--full")
    ) {
      return true;
    }
  }
  return false;
}

let nudgedBareAmp = false;
let nudgedPkill = false;

const BARE_AMP_NUDGE =
  "Heads up: a bare trailing `&` backgrounds a job on this harness's shared " +
  "stdout/stderr pipes — those pipes get torn down shortly after this tool " +
  "call returns, and the backgrounded job can be SIGPIPE'd/killed once that " +
  "happens, even though it's still running. For anything that needs to keep " +
  "running past this tool call, use the ShellStart tool (bg-shell's job " +
  "registry) instead of a bare `&` — it keeps the job alive and lets you " +
  "check on it later with ShellList/ShellLog.";

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

    if (hasBareBackgroundAmpersand(cmd)) {
      ctx.ui.notify(
        `shell-contract-nudge: "${toolName}" call backgrounds a job with a bare "&" — ` +
          "its stdout/stderr pipes get torn down shortly after this call returns.",
        "info",
      );
      if (!nudgedBareAmp) {
        try {
          pi.sendUserMessage(BARE_AMP_NUDGE, { deliverAs: "steer" });
          nudgedBareAmp = true;
        } catch {
          // SDK without sendUserMessage, or the send otherwise failed — leave
          // the latch false so a later offending call can try again, mirroring
          // finalize-warn/quality-monitor's own "don't burn the one-shot latch
          // on a nudge that was never delivered" convention.
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
        try {
          pi.sendUserMessage(PKILL_NUDGE, { deliverAs: "steer" });
          nudgedPkill = true;
        } catch {
          // See the bare-`&` branch above — don't burn the latch on a nudge
          // that never actually sent.
        }
      }
    }

    // Advisory only: never block or rewrite the call, and no return value —
    // a one-shot background job or a legitimate pkill -f can be exactly
    // right, so this extension only ever informs.
  });
}
