// Trial-wide counter for a pattern `near_duplicate_loop`'s content-clustering
// definitionally cannot catch: a script gets rewritten, run, and fails --
// differently each time -- with real gaps (~10-30 turns) between events, so
// most pairs never share the fuzzy tracker's 8-turn window at all
// (overfull-hbox: 8 such events, only 2 caught by near_duplicate_loop).
//
// Monotonic, not a streak or rolling window: the claim is "how many times has
// this happened at all this trial", true regardless of what ran in between,
// and needs no window-size tuning the way a rolling span would.

import { readResult } from "./failure-signature.ts";

const INTERPRETER_RE = /\b(?:perl|python3?|ruby|node|bash|sh)\b/i;
const SCRIPT_EXT_RE = /\.(?:pl|pm|t|py|rb|js|sh)\b/i;

/**
 * A shell command that both names a common interpreter and targets a common
 * script extension. Deliberately a small local table, not a cross-extension
 * import of syntax-check/helpers.ts's checkerFor: pi's tool_result handlers
 * for different extensions fire independently in load order (quality-monitor
 * loads alphabetically before syntax-check), so this handler sees the raw
 * result before syntax-check's own marker is ever appended to it.
 */
export function looksLikeScriptRun(input: unknown): boolean {
  const command = (input as { command?: unknown } | null | undefined)?.command;
  return typeof command === "string" && INTERPRETER_RE.test(command) && SCRIPT_EXT_RE.test(command);
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
 */
export class ScriptFailureTracker {
  private count = 0;
  private notifiedThreshold = 0;

  /**
   * Fold one tool result into the trial-wide count. Returns a detection only
   * on the turn that first crosses THRESHOLD_1 or THRESHOLD_2 -- every other
   * turn, including ones that also match and fail, returns null so the model
   * isn't re-notified every single time.
   */
  record(obs: { input: unknown; text: string; isError: boolean }): ScriptFailureDetection | null {
    if (!looksLikeScriptRun(obs.input)) return null;
    // minTokens 0: this tracker only cares whether the command failed, not
    // whether the result said enough to identify which failure it was --
    // that's readResult's `hasContent` floor, built for a different watchdog.
    const { failed } = readResult(obs.text, obs.isError, 0);
    if (!failed) return null;

    this.count++;
    if (this.count >= THRESHOLD_2 && this.notifiedThreshold < THRESHOLD_2) {
      this.notifiedThreshold = THRESHOLD_2;
      return { count: this.count, escalated: true };
    }
    if (this.count >= THRESHOLD_1 && this.notifiedThreshold < THRESHOLD_1) {
      this.notifiedThreshold = THRESHOLD_1;
      return { count: this.count, escalated: false };
    }
    return null;
  }
}
