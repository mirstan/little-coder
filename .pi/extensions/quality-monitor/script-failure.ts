// Trial-wide counter for a pattern `near_duplicate_loop`'s content-clustering
// definitionally cannot catch: a script gets rewritten, run, and fails --
// differently each time -- with real gaps (~10-30 turns) between events, so
// most pairs never share the fuzzy tracker's 8-turn window at all
// (overfull-hbox: 8 such events, only 2 caught by near_duplicate_loop).
//
// Monotonic, not a streak or rolling window: the claim is "how many times has
// this happened at all this trial", true regardless of what ran in between,
// and needs no window-size tuning the way a rolling span would.

import { splitFooter } from "../truncated-view/truncation.ts";
import { footerExit } from "../syntax-check/helpers.ts";
import { readResult } from "./failure-signature.ts";

const INTERPRETER_RE = /\b(?:perl|python3?|ruby|node|bash|sh)\b/i;
const SCRIPT_EXT_RE = /\.(?:pl|pm|t|py|rb|js|sh)\b/i;

function commandOf(input: unknown): string | undefined {
  const command = (input as { command?: unknown } | null | undefined)?.command;
  return typeof command === "string" ? command : undefined;
}

/**
 * Split a shell command chain into its top-level segments, at `;`, `|`, and
 * newlines only -- separators where every segment unconditionally runs and
 * the LAST one's exit is what the whole command reports (plain bash, no
 * `pipefail`). Deliberately does NOT split on `&&`/`||`: those short-circuit,
 * so the exit code can belong to an EARLIER segment (the one that stopped
 * the chain) rather than the last one written -- splitting on them risks the
 * opposite mistake this function exists to prevent, excluding a script whose
 * own failure is exactly what aborted a trailing `&& next_step`. Quote-aware
 * so a separator character quoted inside a script's own argument doesn't
 * fracture the chain: `perl -e 'print "a;b"'` stays one segment.
 */
function commandSegments(command: string): string[] {
  const segments: string[] = [];
  let current = "";
  let quote: '"' | "'" | null = null;
  for (let i = 0; i < command.length; i++) {
    const ch = command[i];
    if (quote) {
      current += ch;
      if (ch === quote) quote = null;
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      current += ch;
      continue;
    }
    // `&&`/`||` are short-circuit operators, not a split point (see this
    // function's own doc) -- consumed as a literal pair so a lone `|`
    // immediately following another `|` is never mistaken for the
    // unconditional single-pipe separator below.
    if ((ch === "&" && command[i + 1] === "&") || (ch === "|" && command[i + 1] === "|")) {
      current += ch + command[i + 1];
      i++;
      continue;
    }
    if (ch === ";" || ch === "|" || ch === "\n") {
      segments.push(current);
      current = "";
      continue;
    }
    current += ch;
  }
  segments.push(current);
  return segments;
}

/**
 * A shell command whose own FINAL segment both names a common interpreter
 * and targets a common script extension. Deliberately a small local table,
 * not a cross-extension import of syntax-check/helpers.ts's checkerFor: pi's
 * tool_result handlers for different extensions fire independently in load
 * order (quality-monitor loads alphabetically before syntax-check), so this
 * handler sees the raw result before syntax-check's own marker is ever
 * appended to it.
 *
 * Restricted to the chain's final `;`/`|`/newline-delimited segment because
 * ShellSession reports one exit code for the whole command, and for those
 * separators that code is always the LAST segment's: `python3 solve.py;
 * grep expected missing.txt` exits on grep's status, so matching anywhere
 * earlier in the chain would misattribute an unrelated trailing command's
 * failure to a script that may have run cleanly. A `&&`/`||`-joined
 * subcommand is not split further (see `commandSegments`), so it stays
 * eligible wherever it sits within its enclosing segment.
 */
export function looksLikeScriptRun(input: unknown): boolean {
  const command = commandOf(input);
  if (command === undefined) return false;
  const segments = commandSegments(command);
  const final = segments[segments.length - 1] ?? "";
  return INTERPRETER_RE.test(final) && SCRIPT_EXT_RE.test(final);
}

export interface ScriptFailureDetection {
  count: number;
  /** Second and final message for this trial. */
  escalated: boolean;
}

const THRESHOLD_1 = 3;
const THRESHOLD_2 = 6;

/**
 * Counts script write-run-fail events across the whole trial, independent of
 * `FuzzyLoopTracker`'s content clustering. Steer-only, like the other two
 * quality-monitor watchdogs: a script failing repeatedly cannot prove the
 * NEXT rewrite is wrong, only that the current approach isn't converging.
 *
 * Three things are excluded from the count so it isn't counting the same
 * event twice or a non-event at all: `exit === -1` (ShellSession's own
 * timeout/proxy-no-response/sentinel code, not a script failure), a verbatim
 * repeat of the immediately preceding COUNTED command (the same attempt seen
 * again, and already the verbatim loop-breaker's own case), and a chain whose
 * matched interpreter+extension token isn't the chain's own final segment
 * (see `looksLikeScriptRun`).
 */
export class ScriptFailureTracker {
  private count = 0;
  private lastCountedCommand: string | null = null;
  /** Highest threshold whose message has actually been delivered. */
  private notifiedThreshold = 0;

  /**
   * Fold one tool result into the trial-wide count, then report whichever
   * threshold is due a message right now -- computed fresh from `count` and
   * `notifiedThreshold` on every call, not mutated the instant a threshold is
   * crossed. A crossing whose message never actually gets sent (the turn it
   * happened on is later classified non-ok, so the caller never reaches
   * `sendUserMessage`) must not be lost: only `markNotified()` advances
   * `notifiedThreshold`, so an unmarked crossing keeps being reported on
   * every subsequent `record()` call until it is -- mirroring
   * `FailureSignatureTracker`'s / `FuzzyLoopTracker`'s own
   * due()-computed-fresh + explicit markNotified()-only-when-sent split,
   * rather than consuming the notification the instant it is detected.
   */
  record(obs: { input: unknown; text: string; isError: boolean }): ScriptFailureDetection | null {
    if (!looksLikeScriptRun(obs.input)) return null;

    const { footer } = splitFooter(obs.text);
    if (footerExit(footer) === -1) return null;

    // minTokens 0: this tracker only cares whether the command failed, not
    // whether the result said enough to identify which failure it was --
    // that's readResult's `hasContent` floor, built for a different watchdog.
    const { failed } = readResult(obs.text, obs.isError, 0);
    if (!failed) return null;

    const command = commandOf(obs.input) ?? null;
    // A verbatim repeat of the immediately preceding counted command is the
    // same attempt seen twice (e.g. re-run after an unrelated call in
    // between), not a new distinct failure.
    if (command !== null && command === this.lastCountedCommand) return null;

    this.count++;
    this.lastCountedCommand = command;
    return this.due();
  }

  /**
   * Consume the message slot for whichever threshold `count` currently
   * satisfies. Called by the caller only once a detection has actually been
   * delivered (see the class doc) -- never from inside `record()` itself.
   */
  markNotified(): void {
    if (this.count >= THRESHOLD_2) this.notifiedThreshold = THRESHOLD_2;
    else if (this.count >= THRESHOLD_1) this.notifiedThreshold = THRESHOLD_1;
  }

  private due(): ScriptFailureDetection | null {
    if (this.count >= THRESHOLD_2 && this.notifiedThreshold < THRESHOLD_2) {
      return { count: this.count, escalated: true };
    }
    if (this.count >= THRESHOLD_1 && this.notifiedThreshold < THRESHOLD_1) {
      return { count: this.count, escalated: false };
    }
    return null;
  }
}
