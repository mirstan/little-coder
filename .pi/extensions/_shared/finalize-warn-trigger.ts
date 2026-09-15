// Shared, pure trigger condition for finalize-warn's "save now" nudge:
// WARN_REMAINING turns left before the turn-cap would abort the run, OR
// WARN_REMAINING_MS left before a known wall-clock deadline.
//
// finalize-warn and tb-finalize-guard both need this condition:
// tb-finalize-guard independently re-derives whether finalize-warn's nudge
// would have fired this turn rather than reading finalize-warn's private
// state, since finalize-warn exposes no latch for that (see
// tb-finalize-guard/index.ts for why). Both extensions import the constants
// and the condition from this one module so they cannot drift apart.
//
// This dir has no `index.ts` on purpose — the launcher's extension discovery
// requires one, so `_shared` is skipped and stays a plain library.

export const WARN_REMAINING = 5; // turns
export const WARN_REMAINING_MS = 10 * 60 * 1000; // wall-clock headroom before deadline

export interface FinalizeWarnWouldFireArgs {
  /** 1-indexed turn count within the current agent run. */
  turnsThisRun: number;
  /** Turn cap for the current run; 0 (or any non-positive value) = no cap. */
  capForRun: number;
  /** Absolute epoch-ms deadline for the current run; 0 = no deadline known. */
  deadlineForRun: number;
  /** Injectable clock, for testability. Defaults to Date.now. */
  now?: () => number;
}

/**
 * True at the exact turn finalize-warn's own turn-count-OR-wall-clock
 * trigger condition fires. Pure and stateless — callers own the one-shot
 * latch (finalize-warn's `warnedThisRun`, tb-finalize-guard's `armed`).
 */
export function finalizeWarnWouldFire({
  turnsThisRun,
  capForRun,
  deadlineForRun,
  now = Date.now,
}: FinalizeWarnWouldFireArgs): boolean {
  const turnTrigger =
    capForRun > WARN_REMAINING && turnsThisRun === capForRun - WARN_REMAINING + 1;

  const remainingMs = deadlineForRun > 0 ? deadlineForRun - now() : Infinity;
  const timeTrigger = deadlineForRun > 0 && remainingMs <= WARN_REMAINING_MS;

  return turnTrigger || timeTrigger;
}
