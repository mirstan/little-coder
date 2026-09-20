// Shared, stateless "how much of this trial's wall-clock budget is already
// spent" — the fraction tb-finalize-guard's Trigger D fires its progress
// milestones on.
//
// Kept apart from deadline.ts because the two answer different questions:
// deadline.ts answers "how long until I must stop", which needs only the
// far end of the interval, and every extension built on it is indifferent
// to when the trial began. A fraction needs both ends, so it needs a
// trial-start instant nothing else reads.
//
// The start instant comes from the party that owns the budget (both TB
// adapters, at the same instant they derive the deadline from) rather than
// from when this extension process happened to start: the harbor adapter
// snapshots and probes the container before pi exists, so the two instants
// are not the same and the difference would silently mis-scale every
// milestone.
//
// This dir has no `index.ts` on purpose — see finalize-warn-trigger.ts.

/**
 * Ascending. Callers scan for the LARGEST crossed-but-unspent milestone, so
 * one long turn that skips a milestone produces one message, not a backlog.
 */
export const PROGRESS_MILESTONES = [0.5, 0.75] as const;

// Same precedence and validation as resolveDeadlineEpochMs, deliberately:
// the two values are one interval and a reader that resolved its ends by
// different rules could produce a fraction from two unrelated clocks. See
// deadline.ts for why "set" means non-empty and why Number(), not
// parseInt(). The event override mirrors deadline.ts's own unused-but-
// symmetric one — nothing sets it today.
export function resolveBudgetStartEpochMs(event: unknown): number {
  const raw = process.env.LITTLE_CODER_BUDGET_START_EPOCH_MS;
  if (raw !== undefined && raw.trim() !== "") {
    const n = Number(raw);
    return Number.isInteger(n) && n > 0 ? n : 0;
  }
  const opts: any = (event as any)?.systemPromptOptions ?? {};
  const evStart = Number(opts?.littleCoder?.budgetStartEpochMs);
  return Number.isInteger(evStart) && evStart > 0 ? evStart : 0;
}

export interface BudgetFractionUsedArgs {
  /** Absolute epoch-ms instant the trial's budget started; 0 = unknown. */
  startForRun: number;
  /** Absolute epoch-ms deadline for the current run; 0 = no deadline known. */
  deadlineForRun: number;
  /** Injectable clock, for testability. Defaults to Date.now. */
  now?: () => number;
}

/**
 * Fraction of [start, deadline] elapsed at `now`, or undefined when either
 * end is unknown or the interval is non-positive — callers must then stand
 * down rather than assume anything, since an adapter that sets neither var
 * (GAIA, aider_polyglot, interactive pi) is not running against a budget at
 * all.
 *
 * Not clamped: past the deadline it exceeds 1, which keeps every milestone
 * comparison monotone.
 */
export function budgetFractionUsed({
  startForRun,
  deadlineForRun,
  now = Date.now,
}: BudgetFractionUsedArgs): number | undefined {
  if (startForRun <= 0 || deadlineForRun <= 0) return undefined;
  const totalMs = deadlineForRun - startForRun;
  if (totalMs <= 0) return undefined;
  return (now() - startForRun) / totalMs;
}
