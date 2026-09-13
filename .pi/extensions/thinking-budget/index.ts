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
//      On breach it runs the exact same recovery sequence as a token-budget
//      breach (including forcedOff/priorLevel) — same remedy, different
//      trigger.

const DEFAULT_BUDGET = 4096;

// Adaptive-budget tuning (mechanism 1). Both env-overridable: they're tuned
// from the single observed failure above, not a principled derivation.
const DEFAULT_ADAPT_MIN_TURNS = 4;
const DEFAULT_ADAPT_OUTSIZED_FRACTION = 0.15;

// Wall-clock guard tuning (mechanism 2), same rationale.
const DEFAULT_GUARD_REMAINING_FRACTION = 0.25;
const DEFAULT_GUARD_HARD_CAP_MS = 15 * 60 * 1000;

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

// The adaptive equivalent of forcedOff/priorLevel, kept deliberately separate:
// a zero adaptive budget means "no headroom to think right now," which is a
// different condition from "this task over-thought and was cut off," and the
// two can end at different moments. Holding one pair of variables for both
// would make a breach that lands during an adaptive-off window capture "off"
// as the level to restore later.
let adaptiveOff = false;
let adaptivePriorLevel: ThinkingLevel | undefined;

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
// One-shot, so a guard that keeps standing down on every delta of a long
// final turn explains itself once instead of flooding the transcript.
let guardSuppressionNotified = false;
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

function charsToTokens(chars: number): number {
  // Matches local/context_manager.estimate_tokens (len/3.5)
  return Math.ceil(chars / 3.5);
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
): void {
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
  }
  safeSetThinkingLevel(pi, "off");
  try {
    // Not awaited: this must all land before ctx.abort() below. prompt()
    // rethrows though, so the rejection needs its own handler — the sync
    // catch never sees it, and Node's default is to crash on it.
    void Promise.resolve(pi.sendUserMessage(followUpMessage, { deliverAs: "followUp" })).catch(
      () => {},
    );
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
    priorLevel = undefined;
    // Bare reset, no restore: a new session re-resolves the thinking level
    // from the profile, so there is nothing of the old one left to put back.
    adaptiveOff = false;
    adaptivePriorLevel = undefined;
    turnStartedAt = undefined;
    avgTurnMs = 0;
    lastTurnMs = 0;
    turnsObserved = 0;
    totalBudgetMsForRun = Infinity;
    capturedDeadlineForTotal = undefined;
  });

  // Hard reset of per-turn counters between agent runs. `forcedOff` /
  // `priorLevel` are deliberately NOT reset here: agent_start ALSO fires for
  // the recovery restart turn, and clearing the force there would let thinking
  // come straight back on — exactly the bug. They are cleared on `input`
  // (a genuinely new user task) or `session_start`.
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
    if ((event as any)?.source === "extension") return;

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
      priorLevel = undefined;
    }
    thinkingChars = 0;
    aborted = false;
    // Discards only the idle gap between this prompt and the previous turn —
    // human thinking time, which is not a measurement of model throughput.
    turnStartedAt = undefined;
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

    // Per-run, deliberately not session-scoped like the timing state above:
    // both exist to mirror finalize-warn's own per-run bookkeeping.
    turnsThisRun = 0;
    guardSuppressionNotified = false;
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

    thinkingChars = 0;
    aborted = false;
    // Re-assert "off" for the restart turn (and any follow-on turns of the same
    // task). After the session replacement triggered by the abort, the new
    // run can otherwise resolve thinking back to the profile default.
    if (forcedOff) safeSetThinkingLevel(pi, "off");

    // Mechanism 1: re-derive the effective budget fresh from the clock every
    // turn, independent of forcedOff — see resolveAdaptiveBudget above.
    const remainingMs = deadlineForRun > 0 ? deadlineForRun - now : Infinity;
    budgetForTurn = resolveAdaptiveBudget({
      baseBudget: baseBudgetForRun,
      remainingMs,
      avgTurnMs,
      lastTurnMs,
      now,
      totalBudgetMs: totalBudgetMsForRun,
    });

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
    // covers it. (pi's vendored HTTP layer also has its own ~300s idle
    // timeout as a backstop, so that residual case is bounded, just not by
    // this extension.)
    // Requires a deadline: with none, this would abort any turn merely longer
    // than the hard cap, which is normal in interactive use and has no budget
    // to protect. Deadline-less runs get no guard, deliberately.
    if (!aborted && turnStartedAt !== undefined && guardEnabledForRun && deadlineForRun > 0) {
      const now = Date.now();
      const elapsedThisTurn = now - turnStartedAt;
      const remainingMs = deadlineForRun - now;
      // Stand down for the whole of finalize-warn's endgame window — both of
      // its triggers, not just wall-clock. With a turn cap configured,
      // finalize-warn can fire on turn count with plenty of clock left, and
      // aborting the model's final-answer turn is precisely what this
      // suppression exists to prevent.
      const inFinalizeWarnWindow =
        remainingMs <= WARN_REMAINING_MS ||
        (capForRun > WARN_REMAINING && turnsThisRun >= capForRun - WARN_REMAINING + 1);
      const guardMs = Math.min(remainingMs * guardFractionForRun, guardHardCapMsForRun);
      if (elapsedThisTurn > guardMs) {
        if (inFinalizeWarnWindow) {
          if (!guardSuppressionNotified) {
            guardSuppressionNotified = true;
            harnessIntervention(
              ctx,
              "turn wall-clock guard standing down: inside the finalize-warn window, " +
                "letting the model finish its answer uninterrupted.",
            );
          }
        } else {
          aborted = true;
          runBreachRecovery(
            pi,
            ctx,
            "[turn wall-clock guard] This turn has been generating for too long — " +
              "stop and take one concrete action now: call a tool, write code, or give your answer.",
            `this turn ran past its wall-clock guard (~${Math.round(guardMs / 1000)}s) — forcing it to act.`,
          );
          return;
        }
      }
    }

    const ev: any = (event as any).assistantMessageEvent;
    if (!ev) return;
    if (ev.type !== "thinking_delta") return;
    const delta = typeof ev.delta === "string" ? ev.delta : "";
    thinkingChars += delta.length;
    if (aborted) return;
    const tokens = charsToTokens(thinkingChars);
    if (tokens <= budgetForTurn) return;

    // Breach. Do the entire recovery now, while `pi` is still live — BEFORE
    // ctx.abort() triggers the session replacement that would make `pi` stale.
    aborted = true;
    runBreachRecovery(
      pi,
      ctx,
      "[thinking budget exceeded] Stop deliberating and make concrete progress now — " +
        "call a tool, write code, or give your answer.",
      "the model has thought long enough — forcing it to act.",
    );
  });
}
