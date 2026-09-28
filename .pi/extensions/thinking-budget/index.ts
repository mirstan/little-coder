import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { harnessIntervention } from "../_shared/intervention.ts";
import { resolveDeadlineEpochMs } from "../_shared/deadline.ts";
import { resolveTurnCap } from "../_shared/turn-cap.ts";
import { envNumber } from "../_shared/env-number.ts";
import { WARN_REMAINING, WARN_REMAINING_MS } from "../_shared/finalize-warn-trigger.ts";

// pi's thinking-level union (not re-exported from the package root). Mirrors
// settings-manager's ThinkingLevel; structurally assignable to pi's own type.
type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

// Port of the thinking-budget cap + partial-trace reuse logic from
// providers.py. little-coder's Python implementation aborts the stream
// mid-flight when thinking tokens cross the budget, re-injects the partial
// trace as assistant context, and retries with thinking disabled. Pi's
// AgentSession doesn't expose mid-stream abort-and-replace, so we approximate
// it: count thinking tokens, and on breach disable thinking + queue a
// commit-to-an-implementation nudge, then abort the over-long turn.
//
// ── Issue #8, second reproduction (1.4.3) ───────────────────────────────────
// The v1.0.0 fix deferred the recovery (`setThinkingLevel("off")` +
// `sendUserMessage`) to a `turn_end` handler, after a `setImmediate` yield, and
// ran it against the module-scope `pi`. But `ctx.abort()` makes pi's `agent_end`
// run auto-retry / auto-compaction (both enabled in .pi/settings.json), which
// REPLACES the session (dispose() → ExtensionRunner.invalidate()). The
// setImmediate yield is exactly what let that replacement land *before* the
// deferred recovery, so the recovery touched a now-stale `pi` and threw
// ("This extension ctx is stale after session replacement or reload"). Net
// effect: thinking was never turned off (the next step kept thinking) and the
// follow-up never reached the model (the agent appeared to stop) — the #8
// symptom, on a different mechanism than the original.
//
// Fix: do the whole recovery SYNCHRONOUSLY inside `message_update`, BEFORE
// `ctx.abort()`, while `pi` is still live and the session hasn't been replaced.
// No `turn_end` handler, no `setImmediate` — nothing runs against a stale ref.
//
//   1. Count thinking_delta tokens during message_update.
//   2. On breach: capture the current thinking level, flip thinking to "off",
//      queue the commit nudge as a follow-up, surface one harness-intervention
//      line, THEN ctx.abort().
//   3. Keep thinking off across the restart turn(s): `forcedOff` re-asserts
//      "off" on every turn_start until the user submits a genuinely new prompt
//      (`input` event), at which point the prior level is restored so the next
//      task can think again. (A new task should not inherit "off" just because
//      a previous one over-thought.)
//
// ── Trajectory follow-up: deadline/turn-cap-aware adaptation ────────────────
// A fixed `thinking:high` budget was applied to a local model later observed
// running at ~2.5 tokens/sec: it burned 5 turns and then died mid-generation
// on a single ~70-minute request, with nothing in the loop deadline-aware.
// Two new, independent mechanisms address that, neither touching forcedOff:
//
//   1. Turn-boundary adaptive budget (resolveAdaptiveBudget, applied at
//      turn_start): shrinks or zeroes the *token* budget threshold that the
//      breach check below already reads, based on wall-clock headroom and
//      recent turn durations. The threshold itself is re-derived fresh every
//      turn_start, so it self-corrects as conditions change; a threshold of
//      zero additionally forces the level to "off" (adaptiveOff), which is a
//      real latch and carries its own restore-on-input path, kept separate
//      from forcedOff's.
//   2. Per-turn wall-clock guard (in message_update, before the
//      thinking_delta-only early-return): the mechanism that actually
//      interrupts a hung in-flight generation, since a shrunk token budget
//      alone can't stop a request the model is already mid-way through.
//      On breach it runs the same recovery sequence as a token-budget breach
//      (including forcedOff/priorLevel) — same remedy, different trigger. It
//      releases differently though, since it is the only trigger reachable in
//      a single-prompt run: see forcedOffByWallClockGuard.
//
// Both breach triggers stand down inside finalize-warn's endgame window
// (inFinalizeWarnWindow), where an abort would cut off the final answer.
//
// ── Trajectory follow-up: the abort spiral ──────────────────────────────────
// A trial burned ~50 minutes in a self-sustaining abort loop with near-zero
// forward progress, from three things compounding:
//
//   - The guard clocked from turn_start, so the ~118s it took to re-prefill
//     the suffix an abort had just discarded counted as generation time.
//   - Its window shrank with the remaining trial time (900s → 375s across the
//     trial) while the model's throughput collapsed at large context (~25
//     tok/s against the 50-70 it normally does), so turns that were producing
//     output perfectly well, just slowly, were aborted for being slow.
//   - Every guard trip latches forcedOff, zeroing the adaptive budget — and
//     that model's config pinned thinking on regardless of the level, so the
//     next turn's first thinking token breached a budget of 0 instantly and
//     bought another 118s prefill for nothing.
//
// Three additions, all reachable only through the two existing triggers:
//
//   3. The guard's clock starts at the first streamed delta, not turn_start,
//      so prefill is excluded (see turnFirstDeltaAt).
//   4. Its window is floored at the time the currently-granted budget plus one
//      concrete action actually costs at observed throughput (resolveGuardMs).
//   5. A futility rule and a circuit breaker (see consecutiveNoContentAborts)
//      stop the loop where the throughput floor cannot: aborting can't make a
//      model honor a thinking level it is ignoring, and two fruitless aborts
//      in a row mean the remedy isn't reaching it at all.

const DEFAULT_BUDGET = 4096;

// Adaptive-budget tuning (mechanism 1). Both env-overridable: they're tuned
// from the single observed failure above, not a principled derivation.
const DEFAULT_ADAPT_MIN_TURNS = 4;
const DEFAULT_ADAPT_OUTSIZED_FRACTION = 0.15;

// Wall-clock guard tuning (mechanism 2), same rationale.
const DEFAULT_GUARD_REMAINING_FRACTION = 0.25;
const DEFAULT_GUARD_HARD_CAP_MS = 15 * 60 * 1000;

// Throughput-floor tuning. minActionTokens is the headroom past the thinking
// budget a turn needs in order to actually *do* something once it has finished
// deliberating — without it the floor would grant exactly enough time to think
// and none to act. The slack factor absorbs the chars→tokens heuristic's error
// (dense code and non-Latin text both skew len/3.5).
const DEFAULT_GUARD_MIN_ACTION_TOKENS = 1024;
const DEFAULT_GUARD_THROUGHPUT_SLACK = 1.25;

// Bars a throughput sample must clear to be used at all. A short burst of
// deltas measures the provider's chunking, not the model's rate, and would
// read as an absurdly high chars/sec — which would collapse the floor to
// nothing exactly when it is needed.
const SAMPLE_MIN_SPAN_MS = 5000;
const SAMPLE_MIN_CHARS = 500;

// Consecutive content-free extension aborts before both breach triggers stand
// down. Two rather than one: a single fruitless abort is ordinary, a second in
// a row means the recovery is not reaching the model and further aborts only
// buy another prefill.
const BREAKER_ABORT_THRESHOLD = 2;

// Every knob below treats a configured `<= 0` as "disabled", which is why
// they read through _shared/env-number.ts rather than a local resolver that
// folds 0 back into the default.

// Per-run rolling state.
let thinkingChars = 0;
let budgetForTurn = DEFAULT_BUDGET;
let aborted = false;
// True from a budget breach until the next genuine user input. While set, we
// re-assert thinking "off" at the start of every turn so the restart turn (and
// any follow-on turns of the same task) can't silently come back with thinking
// re-enabled by the post-replacement profile resolution.
let forcedOff = false;
// The thinking level in effect when we first forced it off, restored on the
// next user input so a new task is unaffected.
let priorLevel: ThinkingLevel | undefined;
// Which trigger set `forcedOff`. The wall-clock guard is the only trigger that
// can fire in a single-prompt run (the harbor adapter issues one
// rpc.prompt_and_collect per trial), and `input`/`session_start` — the only
// releases — never come again there, so a guard trip would otherwise disable
// thinking for the whole remaining trial however healthy it turns out to be.
// Guard-flavored forcedOff therefore self-clears at turn_start once headroom
// is back, mirroring adaptiveOff: both are reactions to deadline pressure, and
// the deadline is re-read every turn anyway. A token-budget breach keeps the
// original release rule — that one says "this task over-thought," which no
// amount of remaining clock disproves.
let forcedOffByWallClockGuard = false;

// The adaptive equivalent of forcedOff/priorLevel, kept deliberately separate:
// a zero adaptive budget means "no headroom to think right now," which is a
// different condition from "this task over-thought and was cut off," and the
// two can end at different moments. Holding one pair of variables for both
// would make a breach that lands during an adaptive-off window capture "off"
// as the level to restore later.
let adaptiveOff = false;
let adaptivePriorLevel: ThinkingLevel | undefined;
// True for the remainder of the turn in which adaptiveOff newly latched.
// pi snapshots config.reasoning per agent run and only refreshes it in
// prepareNextTurn, which runs after turn_end (pi-agent-core agent.js /
// agent-loop.js) — so the "off" that turn_start just set does not reach the
// turn already in flight, while the zeroed budget does. charsToTokens rounds
// up, so the first thinking char of that turn reads as a breach and promotes
// the self-restoring adaptive latch into the permanent forcedOff one. A breach
// here means "the level change is still pending," not "the model over-thought."
let adaptiveOffJustLatchedThisTurn = false;

// The raw budget resolved from the profile/env at before_agent_start —
// `budgetForTurn`'s adaptive-adjusted value is re-derived from this every
// turn_start, never from its own previous (possibly already-shrunk) value.
let baseBudgetForRun = DEFAULT_BUDGET;
// Deadline/turn-cap visibility, resolved the same way finalize-warn and
// turn-cap already do. `capForRun` feeds the wall-clock guard's
// finalize-warn-window check (the turn-count half of it); the adaptive math
// below still doesn't use it, since turn-cap's own extension owns turn-count
// exhaustion.
let deadlineForRun = 0;
let capForRun = 0;
// Turn counter for this agent run, 1-indexed at the top of turn_start exactly
// as finalize-warn counts, so the guard's "are we inside finalize-warn's
// endgame window" test compares like with like.
let turnsThisRun = 0;
// One-shot each, so a trigger that keeps standing down on every delta of a
// long final turn explains itself once instead of flooding the transcript.
// Separate flags so one trigger's stand-down never swallows the other's line.
let guardSuppressionNotified = false;
let budgetSuppressionNotified = false;
// Guard knobs, resolved per run so a 0 in either one disables the guard
// outright rather than silently collapsing its window to zero (which would
// abort on the first delta of every turn).
let guardFractionForRun = DEFAULT_GUARD_REMAINING_FRACTION;
let guardHardCapMsForRun = DEFAULT_GUARD_HARD_CAP_MS;
let guardEnabledForRun = false;
// The wall-clock window between "now" and the deadline as first observed for
// this run. Used by resolveAdaptiveBudget's outsized-single-turn check, which
// needs a FIXED denominator — unlike the turns-remaining check, which
// deliberately uses the fresher, shrinking remaining-time figure instead.
// Re-captured only when the resolved deadline value itself changes (tracked
// by capturedDeadlineForTotal): re-capturing on every before_agent_start
// would shrink the denominator each time one fired, so the same turn duration
// would read as a progressively larger share of the budget and halve the
// token budget purely as a side effect of a new prompt arriving.
let totalBudgetMsForRun = Infinity;
let capturedDeadlineForTotal: number | undefined;
// Wall-clock turn timing, for both mechanisms: turnStartedAt anchors the
// message_update guard's elapsed-this-turn check; lastTurnMs/avgTurnMs feed
// resolveAdaptiveBudget.
//
// Session-scoped, not per-run: turn durations measure this model's throughput
// on this machine, which doesn't reset just because the user started a second
// task in the same session — carrying the history over means a later prompt
// starts with a real throughput estimate instead of zero data. (This is a
// simplification, not a bug fix: a breach-recovery restart delivers its
// follow-up through agent.followUp() → agent.continue(), which never re-fires
// before_agent_start, so per-run scoping was not actually wiping this state
// mid-recovery.)
let turnStartedAt: number | undefined;
let avgTurnMs = 0;
let lastTurnMs = 0;
let turnsObserved = 0;

// Guard knobs for the throughput floor, resolved per run like the two above.
let guardMinActionTokensForRun = DEFAULT_GUARD_MIN_ACTION_TOKENS;
let guardSlackForRun = DEFAULT_GUARD_THROUGHPUT_SLACK;

// Per-turn stream observation. The first streamed delta is the prefill/
// generation boundary: message_update cannot fire during prefill at all (the
// stream's first event follows the first SSE chunk), so "time since first
// delta" is an exact generation clock rather than an estimate — which is what
// lets the guard stop charging a re-prefill to the model.
let turnFirstDeltaAt: number | undefined;
let turnLastDeltaAt: number | undefined;
let turnDeltaChars = 0;
// Thinking deltas deliberately excluded: a turn that only deliberated and was
// then cut off produced nothing the circuit breaker should credit as progress.
let turnNonThinkingChars = 0;
// Session-scoped, not per-run, for the same reason avgTurnMs above is: it
// measures this model on this machine, which a new prompt doesn't change.
let ewmaCharsPerSec = 0;

// Circuit-breaker state. `lastTurnEndedInExtensionAbort` can't be read via
// the existing `aborted` flag: agent_start fires immediately before
// turn_start on the recovery path (agent-loop.js) and clears `aborted`
// there, so turn_start would read false on every retry.
let lastTurnEndedInExtensionAbort = false;
let consecutiveNoContentAborts = 0;
let breakerNotified = false;
// One-shot per run, like the two suppression flags above.
let futilityNotified = false;

function charsToTokens(chars: number): number {
  // Matches local/context_manager.estimate_tokens (len/3.5)
  return Math.ceil(chars / 3.5);
}

function resetTurnStreamState(): void {
  turnFirstDeltaAt = undefined;
  turnLastDeltaAt = undefined;
  turnDeltaChars = 0;
  turnNonThinkingChars = 0;
}

/**
 * Observed generation rate in characters per second, or 0 for "no estimate".
 *
 * The turn in flight is preferred over the rolling estimate because the thing
 * that moves throughput most is context length, which is a property of this
 * turn rather than of the run — a pre-compaction EWMA describes a different
 * context than the one currently generating. The EWMA covers only the window
 * before the current turn has produced a sample worth trusting.
 */
function charsPerSecEstimate(): number {
  if (turnFirstDeltaAt !== undefined && turnLastDeltaAt !== undefined) {
    const spanMs = turnLastDeltaAt - turnFirstDeltaAt;
    if (spanMs >= SAMPLE_MIN_SPAN_MS && turnDeltaChars >= SAMPLE_MIN_CHARS) {
      return (turnDeltaChars / spanMs) * 1000;
    }
  }
  return ewmaCharsPerSec;
}

function foldThroughputSample(): void {
  if (turnFirstDeltaAt === undefined || turnLastDeltaAt === undefined) return;
  const spanMs = turnLastDeltaAt - turnFirstDeltaAt;
  if (spanMs < SAMPLE_MIN_SPAN_MS || turnDeltaChars < SAMPLE_MIN_CHARS) return;
  const sample = (turnDeltaChars / spanMs) * 1000;
  ewmaCharsPerSec = ewmaCharsPerSec > 0 ? 0.5 * ewmaCharsPerSec + 0.5 * sample : sample;
}

/**
 * Turn-boundary adaptive budget (mechanism 1). Pure and stateless: the caller
 * (turn_start) re-derives every field fresh from the clock and its own
 * rolling turn-duration bookkeeping each time. Returns 0 to mean "no headroom
 * to think at all"; acting on that — including the latch needed to restore the
 * level afterwards — is the caller's job.
 */
export function resolveAdaptiveBudget(params: {
  baseBudget: number;
  remainingMs: number;
  avgTurnMs: number;
  lastTurnMs: number;
  now: number;
  totalBudgetMs: number;
}): number {
  const { baseBudget, remainingMs, avgTurnMs, lastTurnMs, totalBudgetMs } = params;

  // No deadline configured (caller passes Infinity — see totalBudgetMsForRun
  // above): treat as "plenty of time," never shrink or disable thinking
  // based on a clock that isn't actually governing this run.
  if (!Number.isFinite(remainingMs)) return baseBudget;

  const turnEstimateMs = Math.max(avgTurnMs, lastTurnMs, 1);
  const predictedTurnsRemaining = remainingMs / turnEstimateMs;
  const minTurns = envNumber("LITTLE_CODER_THINKING_ADAPT_MIN_TURNS", DEFAULT_ADAPT_MIN_TURNS);
  // The `minTurns > 0` half is load-bearing past the deadline: remainingMs
  // goes negative there and turnEstimateMs is always >= 1, so a bare
  // `predictedTurnsRemaining < 0` would still hold and zero the budget on a
  // knob the operator explicitly disabled.
  if (minTurns > 0 && predictedTurnsRemaining < minTurns) {
    return 0;
  }

  if (Number.isFinite(totalBudgetMs) && totalBudgetMs > 0) {
    const turnFraction = lastTurnMs / totalBudgetMs;
    const outsizedFraction = envNumber(
      "LITTLE_CODER_THINKING_ADAPT_OUTSIZED_FRACTION",
      DEFAULT_ADAPT_OUTSIZED_FRACTION,
    );
    // Without `outsizedFraction > 0`, disabling this knob would invert it:
    // any completed turn makes turnFraction positive, so `> 0` holds on every
    // turn after the first and the budget halves permanently.
    if (outsizedFraction > 0 && turnFraction > outsizedFraction) return Math.floor(baseBudget / 2);
  }

  return baseBudget;
}

/**
 * The wall-clock guard's threshold as of this moment (mechanism 2). Pure and
 * stateless in the same way resolveAdaptiveBudget is: every input is re-derived
 * by the caller from the clock and the current turn's observed throughput.
 *
 * The floor is the whole point. The base window is a fraction of the remaining
 * trial time, so it shrinks as the trial proceeds — and it shrinks fastest
 * exactly when context has grown large enough to halve generation throughput,
 * which is when turns legitimately need MORE time, not less. Left alone it
 * aborts turns that are producing output perfectly well and throws that output
 * away. The floor refuses to abort before the turn has had the time its own
 * granted budget plus one concrete action actually costs at the rate we are
 * observing. It only ever raises the threshold, so a stale-slow estimate can
 * delay an abort but never bring one forward.
 */
export function resolveGuardMs(params: {
  remainingMs: number;
  guardFraction: number;
  guardHardCapMs: number;
  budgetForTurn: number;
  charsPerSec: number;
  minActionTokens: number;
  slack: number;
}): number {
  const { remainingMs, guardFraction, guardHardCapMs, budgetForTurn } = params;
  const { charsPerSec, minActionTokens, slack } = params;
  const baseGuardMs = Math.min(remainingMs * guardFraction, guardHardCapMs);

  // Rate form of charsToTokens — the same /3.5 heuristic without its rounding,
  // which belongs to counting a total rather than dividing by a duration.
  const estTokensPerSec = charsPerSec / 3.5;
  if (!(estTokensPerSec > 0) || minActionTokens <= 0 || slack <= 0) return baseGuardMs;

  const minUsefulTokens = Math.max(budgetForTurn, 0) + minActionTokens;
  const throughputFloorMs = (minUsefulTokens / estTokensPerSec) * 1000 * slack;
  return Math.max(baseGuardMs, throughputFloorMs);
}

/**
 * True while this run is inside finalize-warn's endgame window, by either of
 * finalize-warn's own triggers. Both breach paths in message_update stand down
 * here: near the end of a run the adaptive budget has usually resolved to 0, so
 * aborting on the first thinking token would cut off the model's final answer —
 * the abort would cost more than the over-thinking it prevents.
 *
 * With no deadline the wall-clock half is Infinity and only the turn-count half
 * can hold, so a deadline-less run is unaffected unless a turn cap is set.
 */
function inFinalizeWarnWindow(now: number): boolean {
  const remainingMs = deadlineForRun > 0 ? deadlineForRun - now : Infinity;
  return (
    remainingMs <= WARN_REMAINING_MS ||
    (capForRun > WARN_REMAINING && turnsThisRun >= capForRun - WARN_REMAINING + 1)
  );
}

// setThinkingLevel / getThinkingLevel are guarded: a stale-ctx throw must never
// escape (pi reports an uncaught extension throw as a hard "Extension error"),
// and older SDK builds may lack the getter.
function safeGetThinkingLevel(pi: ExtensionAPI): ThinkingLevel | undefined {
  try {
    return typeof pi.getThinkingLevel === "function" ? pi.getThinkingLevel() : undefined;
  } catch {
    return undefined;
  }
}

function safeSetThinkingLevel(pi: ExtensionAPI, level: ThinkingLevel): void {
  try {
    pi.setThinkingLevel(level);
  } catch {
    // Stale ctx / unsupported — leave the level alone rather than crash the run.
  }
}

function notifyBreakerOnce(ctx: any): void {
  if (breakerNotified) return;
  breakerNotified = true;
  harnessIntervention(
    ctx,
    `${BREAKER_ABORT_THRESHOLD} consecutive interventions produced no output — standing down ` +
      "both abort triggers until a turn makes real progress, since retrying only re-prefills.",
  );
}

// The exact recovery sequence both breach triggers (token-budget below, and
// the wall-clock guard) run: capture + force off, queue the follow-up, notify,
// THEN abort — in that order, and synchronously, for the reason in the Issue
// #8 header above (ctx.abort() replaces the session; anything deferred past it
// runs against a stale `pi`).
function runBreachRecovery(
  pi: ExtensionAPI,
  ctx: any,
  followUpMessage: string,
  notifyMessage: string,
  triggeredByWallClockGuard: boolean,
): void {
  // Breaker bookkeeping first: ctx.abort() below replaces the session, and
  // turn_start reads both of these before resetting the per-turn counters.
  lastTurnEndedInExtensionAbort = true;
  if (turnNonThinkingChars === 0) consecutiveNoContentAborts++;

  if (!forcedOff) {
    if (adaptiveOff) {
      // Take over the adaptive latch's captured level rather than reading the
      // current one: adaptive-off has already set the level to "off", so
      // asking now would record "off" as what to restore, and the user's real
      // level would be lost for the rest of the session.
      priorLevel = adaptivePriorLevel;
      adaptiveOff = false;
      adaptivePriorLevel = undefined;
    } else {
      priorLevel = safeGetThinkingLevel(pi);
    }
    forcedOff = true;
    forcedOffByWallClockGuard = triggeredByWallClockGuard;
  }
  safeSetThinkingLevel(pi, "off");
  try {
    // Not awaited: this must all land before ctx.abort() below.
    pi.sendUserMessage(followUpMessage, { deliverAs: "followUp" });
  } catch {
    // SDK without sendUserMessage — abort still forces the turn to end.
  }
  harnessIntervention(ctx, notifyMessage);
  ctx.abort();
}

export default function (pi: ExtensionAPI) {
  // A new session (startup, /clear, resume, reload) is a clean slate — clear
  // everything, including the forced-off window. The recovery restart does NOT
  // fire session_start (it's a follow-up within the same session), so this
  // never clobbers the re-assertion. Also stops module-scoped state leaking
  // across sessions in-process.
  pi.on("session_start", async () => {
    thinkingChars = 0;
    aborted = false;
    forcedOff = false;
    forcedOffByWallClockGuard = false;
    priorLevel = undefined;
    // Bare reset, no restore: a new session re-resolves the thinking level
    // from the profile, so there is nothing of the old one left to put back.
    adaptiveOff = false;
    adaptivePriorLevel = undefined;
    adaptiveOffJustLatchedThisTurn = false;
    turnStartedAt = undefined;
    avgTurnMs = 0;
    lastTurnMs = 0;
    turnsObserved = 0;
    totalBudgetMsForRun = Infinity;
    capturedDeadlineForTotal = undefined;
    resetTurnStreamState();
    ewmaCharsPerSec = 0;
    lastTurnEndedInExtensionAbort = false;
    consecutiveNoContentAborts = 0;
    breakerNotified = false;
    futilityNotified = false;
  });

  // Hard reset of per-turn counters between agent runs. `forcedOff` /
  // `priorLevel` are deliberately NOT reset here: agent_start ALSO fires for
  // the recovery restart turn, and clearing the force there would let thinking
  // come straight back on — exactly the bug. They are cleared on `input`
  // (a genuinely new user task) or `session_start`.
  //
  // The stream-observation counters are deliberately NOT reset here either:
  // agent_start fires immediately before turn_start on the recovery path, so
  // clearing them would destroy the just-aborted turn's throughput sample
  // before turn_start's EWMA fold could use it — and a slow turn that got
  // aborted is precisely the sample the estimator most needs. It would also
  // wipe `turnNonThinkingChars` out from under the circuit breaker's own
  // reset check in turn_start.
  pi.on("agent_start", async () => {
    thinkingChars = 0;
    aborted = false;
  });

  // A genuinely new user prompt ends both "thinking is off" windows: restore
  // the level the user actually had, so a new task never inherits a
  // restriction imposed on the previous one.
  pi.on("input", async (event) => {
    // pi.sendUserMessage() — ours AND every other extension's nudge — routes
    // through prompt(), which emits this event with source "extension" before
    // queueing the message (agent-session.js). Only a genuine new prompt,
    // "interactive" (typed) or "rpc" (the harbor adapter's driver), should end
    // these windows. Without this filter our own breach-recovery follow-up
    // synchronously undoes the very state it is being sent to deliver, and
    // finalize-warn's near-deadline nudge re-enables thinking at the worst
    // possible moment. Filtering out "extension" specifically, rather than
    // allow-listing the other two, keeps a missing source (older harness, test
    // fixture) treated as a genuine prompt.
    //
    // `streamingBehavior` is the second half of the same filter, and catches
    // what `source` alone cannot: pi sets it on the input event only when
    // isStreaming is true (agent-session.js), i.e. the user typed something
    // while the agent was mid-turn. That is a steer of the current task, not a
    // new one, so it arrives as source "interactive" and would otherwise
    // restore thinking in the middle of the very task a breach just forced it
    // off for — issue #8's symptom through a different door.
    if ((event as any)?.source === "extension" || (event as any)?.streamingBehavior !== undefined) {
      return;
    }

    // Restore before clearing, never a bare reset: clearing alone would leave
    // the level physically "off" with no record of what it had been, and the
    // next turn_start would then latch "off" as the level to restore — sticking
    // thinking off for the rest of the session.
    if (adaptiveOff) {
      if (adaptivePriorLevel !== undefined) safeSetThinkingLevel(pi, adaptivePriorLevel);
      adaptiveOff = false;
      adaptivePriorLevel = undefined;
    }
    if (forcedOff) {
      if (priorLevel !== undefined) safeSetThinkingLevel(pi, priorLevel);
      forcedOff = false;
      forcedOffByWallClockGuard = false;
      priorLevel = undefined;
    }
    thinkingChars = 0;
    aborted = false;
    // Discards only the idle gap between this prompt and the previous turn —
    // human thinking time, which is not a measurement of model throughput.
    turnStartedAt = undefined;
    resetTurnStreamState();
    // A genuinely new task is a fresh start, not a continuation of whatever
    // abort streak the previous task left behind — without this, two
    // content-free aborts on one task would leave the breaker holding over
    // into an unrelated task that follows it.
    lastTurnEndedInExtensionAbort = false;
    consecutiveNoContentAborts = 0;
    breakerNotified = false;
  });

  pi.on("before_agent_start", async (event) => {
    const opts: any = (event as any).systemPromptOptions ?? {};
    const lc = opts.littleCoder ?? {};
    const profileBudget = Number(lc.thinkingBudget);
    const envBudget = Number(process.env.LITTLE_CODER_THINKING_BUDGET);
    baseBudgetForRun =
      (Number.isFinite(profileBudget) && profileBudget > 0 && profileBudget) ||
      (Number.isFinite(envBudget) && envBudget > 0 && envBudget) ||
      DEFAULT_BUDGET;
    budgetForTurn = baseBudgetForRun;

    deadlineForRun = resolveDeadlineEpochMs(event);
    capForRun = resolveTurnCap(event);
    if (deadlineForRun !== capturedDeadlineForTotal) {
      capturedDeadlineForTotal = deadlineForRun;
      totalBudgetMsForRun = deadlineForRun > 0 ? deadlineForRun - Date.now() : Infinity;
    }

    guardFractionForRun = envNumber(
      "LITTLE_CODER_THINKING_GUARD_REMAINING_FRACTION",
      DEFAULT_GUARD_REMAINING_FRACTION,
    );
    guardHardCapMsForRun = envNumber(
      "LITTLE_CODER_THINKING_GUARD_HARD_CAP_MS",
      DEFAULT_GUARD_HARD_CAP_MS,
    );
    guardEnabledForRun = guardFractionForRun > 0 && guardHardCapMsForRun > 0;
    guardMinActionTokensForRun = envNumber(
      "LITTLE_CODER_THINKING_GUARD_MIN_ACTION_TOKENS",
      DEFAULT_GUARD_MIN_ACTION_TOKENS,
    );
    guardSlackForRun = envNumber(
      "LITTLE_CODER_THINKING_GUARD_THROUGHPUT_SLACK",
      DEFAULT_GUARD_THROUGHPUT_SLACK,
    );

    // Per-run, deliberately not session-scoped like the timing state above:
    // both exist to mirror finalize-warn's own per-run bookkeeping.
    turnsThisRun = 0;
    guardSuppressionNotified = false;
    budgetSuppressionNotified = false;
    futilityNotified = false;
  });

  pi.on("turn_start", async () => {
    turnsThisRun++;
    const now = Date.now();
    // turnStartedAt is undefined on the first turn of a session and on the
    // first turn after a new user prompt — nothing worth clocking in either
    // case (the latter would measure the human's idle time, not the model's).
    if (turnStartedAt !== undefined) {
      lastTurnMs = now - turnStartedAt;
      turnsObserved++;
      avgTurnMs += (lastTurnMs - avgTurnMs) / turnsObserved;
    }
    turnStartedAt = now;

    // Both of these read the turn that just ENDED, so they have to run before
    // the per-turn counters below are cleared.
    foldThroughputSample();
    // The breaker only holds a streak of turns THIS extension aborted for
    // producing nothing. Any other outcome breaks that streak: real content
    // (even from a turn the guard also ended up aborting for taking too
    // long overall — runBreachRecovery only increments for a turn with zero
    // content in the first place, so a content-producing abort should still
    // clear here) is direct evidence of progress; and a turn that ended on
    // its OWN terms without needing our intervention at all — whether or
    // not it happened to produce content — means the abort mechanism itself
    // wasn't even exercised that turn, so it can't be "the second of a
    // pair" with the abort before it.
    if (turnNonThinkingChars > 0 || !lastTurnEndedInExtensionAbort) {
      consecutiveNoContentAborts = 0;
      breakerNotified = false;
    }
    lastTurnEndedInExtensionAbort = false;
    resetTurnStreamState();

    thinkingChars = 0;
    aborted = false;
    adaptiveOffJustLatchedThisTurn = false;

    // Mechanism 1: re-derive the effective budget fresh from the clock every
    // turn, independent of forcedOff — see resolveAdaptiveBudget above.
    // Resolved before the forcedOff handling below, which now consults it.
    const remainingMs = deadlineForRun > 0 ? deadlineForRun - now : Infinity;
    budgetForTurn = resolveAdaptiveBudget({
      baseBudget: baseBudgetForRun,
      remainingMs,
      avgTurnMs,
      lastTurnMs,
      now,
      totalBudgetMs: totalBudgetMsForRun,
    });

    // Release a wall-clock-guard-flavored forcedOff once the same headroom
    // check that governs adaptiveOff says the pressure is gone — see
    // forcedOffByWallClockGuard above for why only this flavor self-releases.
    // Restoring only at a non-zero budget is what keeps this clear of the
    // pending-level-change hazard adaptiveOffJustLatchedThisTurn describes: the
    // turn it takes effect on has a real threshold, not an instantly-met one.
    if (forcedOff && forcedOffByWallClockGuard && budgetForTurn > 0) {
      if (priorLevel !== undefined) safeSetThinkingLevel(pi, priorLevel);
      forcedOff = false;
      forcedOffByWallClockGuard = false;
      priorLevel = undefined;
    }

    // Re-assert "off" for the restart turn (and any follow-on turns of the same
    // task). After the session replacement triggered by the abort, the new
    // run can otherwise resolve thinking back to the profile default.
    if (forcedOff) safeSetThinkingLevel(pi, "off");

    // A zero budget means the deadline no longer affords any thinking at all.
    // The token counter alone can't enforce that: it only reacts once the
    // model has already emitted thinking deltas, so the level has to actually
    // go to "off". That makes this a latch — it needs the same
    // capture-and-restore care as forcedOff — rather than the stateless
    // re-derivation the rest of mechanism 1 gets. Skipped entirely while
    // forcedOff owns the level, so the two never fight over it.
    if (!forcedOff) {
      if (budgetForTurn <= 0) {
        if (!adaptiveOff) {
          adaptivePriorLevel = safeGetThinkingLevel(pi);
          adaptiveOff = true;
          adaptiveOffJustLatchedThisTurn = true;
        }
        safeSetThinkingLevel(pi, "off");
      } else if (adaptiveOff) {
        // Headroom came back (a fast turn pulled the estimate down).
        if (adaptivePriorLevel !== undefined) safeSetThinkingLevel(pi, adaptivePriorLevel);
        adaptiveOff = false;
        adaptivePriorLevel = undefined;
      }
    }
  });

  pi.on("message_update", async (event, ctx) => {
    const ev: any = (event as any).assistantMessageEvent;

    // Stream observation, ahead of both breach triggers because both now read
    // it. Only the three `*_delta` members of AssistantMessageEvent carry a
    // string `delta` (`*_start` carries none, `*_end` carries `content` /
    // `toolCall`), so this type-tests rather than enumerating the names.
    const deltaText: string = typeof ev?.delta === "string" ? ev.delta : "";
    if (deltaText.length > 0) {
      const deltaAt = Date.now();
      turnFirstDeltaAt ??= deltaAt;
      turnLastDeltaAt = deltaAt;
      turnDeltaChars += deltaText.length;
      if (ev.type !== "thinking_delta") turnNonThinkingChars += deltaText.length;
    }

    const breakerHolding = consecutiveNoContentAborts >= BREAKER_ABORT_THRESHOLD;

    // ── Mechanism 2: per-turn wall-clock guard ──────────────────────────────
    // Runs before the thinking_delta-only early-return below (and before the
    // `!ev`/type checks entirely) so it also catches a hang during plain,
    // non-thinking generation — a model can hang emitting ordinary text just
    // as easily as it can hang "thinking." A shrunk token budget from
    // mechanism 1 can't interrupt a request already in flight; this can.
    //
    // Residual limitation: a provider that stalls with literally zero
    // streamed deltas never reaches this handler at all, so this guard can't
    // fire for that case — only the adapter's own outer request timeout
    // covers it. pi's vendored HTTP layer has a 300s idle timeout
    // (DEFAULT_HTTP_IDLE_TIMEOUT_MS, http-dispatcher.js) as the sole backstop
    // there, so "stuck" stays bounded while this guard handles only "slow".
    // Requires a deadline: with none, this would abort any turn merely longer
    // than the hard cap, which is normal in interactive use and has no budget
    // to protect. Deadline-less runs get no guard, deliberately.
    //
    // Gated on turnFirstDeltaAt rather than turnStartedAt: before the first
    // delta the model is prefilling, and a re-prefill of the context an abort
    // just discarded ran ~118s in the observed spiral — charging that to the
    // model is what made each abort guarantee the next one.
    if (!aborted && turnFirstDeltaAt !== undefined && guardEnabledForRun && deadlineForRun > 0) {
      const now = Date.now();
      const generationElapsedMs = now - turnFirstDeltaAt;
      const remainingMs = deadlineForRun - now;
      const guardMs = resolveGuardMs({
        remainingMs,
        guardFraction: guardFractionForRun,
        guardHardCapMs: guardHardCapMsForRun,
        budgetForTurn,
        charsPerSec: charsPerSecEstimate(),
        minActionTokens: guardMinActionTokensForRun,
        slack: guardSlackForRun,
      });
      // While the breaker holds, the guard is raised rather than disarmed: at
      // 25 tok/s the configured maxTokens alone would let one turn run ~53
      // minutes, so standing down completely would replace a spiral with a
      // single turn that eats the rest of the trial. "Run to completion, but
      // never into the finalize window" is the bound.
      const breakerGuardMs = breakerHolding
        ? Math.max(guardMs, remainingMs - WARN_REMAINING_MS)
        : guardMs;
      if (generationElapsedMs > guardMs) {
        if (inFinalizeWarnWindow(now)) {
          if (!guardSuppressionNotified) {
            guardSuppressionNotified = true;
            harnessIntervention(
              ctx,
              "turn wall-clock guard standing down: inside the finalize-warn window, " +
                "letting the model finish its answer uninterrupted.",
            );
          }
        } else if (generationElapsedMs <= breakerGuardMs) {
          notifyBreakerOnce(ctx);
        } else {
          aborted = true;
          runBreachRecovery(
            pi,
            ctx,
            "[turn wall-clock guard] This turn has been generating for too long — " +
              "stop and take one concrete action now: call a tool, write code, or give your answer.",
            `this turn ran past its wall-clock guard (~${Math.round(guardMs / 1000)}s) — forcing it to act.`,
            true,
          );
          return;
        }
      }
    }

    if (!ev) return;
    if (ev.type !== "thinking_delta") return;
    thinkingChars += deltaText.length;
    if (aborted) return;
    // The level "off" that latched this turn hasn't reached the in-flight turn
    // yet, but its zeroed budget has — see adaptiveOffJustLatchedThisTurn.
    if (adaptiveOffJustLatchedThisTurn) return;
    const tokens = charsToTokens(thinkingChars);
    if (tokens <= budgetForTurn) return;

    // Same stand-down the wall-clock guard gets, for the same reason: this
    // path is the one that actually fires near a deadline, where the adaptive
    // budget has resolved to 0 and the first thinking token of the final
    // answer would otherwise abort it.
    if (inFinalizeWarnWindow(Date.now())) {
      if (!budgetSuppressionNotified) {
        budgetSuppressionNotified = true;
        harnessIntervention(
          ctx,
          "thinking-budget cap standing down: inside the finalize-warn window, " +
            "letting the model finish its answer uninterrupted.",
        );
      }
      return;
    }

    if (breakerHolding) {
      notifyBreakerOnce(ctx);
      return;
    }

    // Futility rule. Thinking is already supposed to be off and the model is
    // emitting thinking deltas regardless — a chatTemplateKwargs entry that
    // pins `enable_thinking` to a literal `true` makes setThinkingLevel("off")
    // a no-op, and the model config is not something the harness can fix from
    // here. The remedy this path has to offer has already been applied and
    // ignored, so aborting cannot improve anything; it only discards the turn
    // and buys another full re-prefill. The wall-clock guard stays the sole
    // abort authority for such a turn.
    //
    // The turn that SET either latch is already excluded upstream (`aborted`
    // for forcedOff, adaptiveOffJustLatchedThisTurn for adaptiveOff), so this
    // needs no "which turn set it" tracking. It is not a latch of its own
    // either: it lapses whenever forcedOff/adaptiveOff do.
    if (forcedOff || adaptiveOff) {
      if (!futilityNotified) {
        futilityNotified = true;
        harnessIntervention(
          ctx,
          "thinking-budget cap standing down: thinking is already off and the model is " +
            "thinking anyway, so aborting would only cost a re-prefill.",
        );
      }
      return;
    }

    // Breach. Do the entire recovery now, while `pi` is still live — BEFORE
    // ctx.abort() triggers the session replacement that would make `pi` stale.
    aborted = true;
    runBreachRecovery(
      pi,
      ctx,
      "[thinking budget exceeded] Stop deliberating and make concrete progress now — " +
        "call a tool, write code, or give your answer.",
      "the model has thought long enough — forcing it to act.",
      false,
    );
  });
}
