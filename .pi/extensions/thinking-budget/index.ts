import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { harnessIntervention } from "../_shared/intervention.ts";
import { resolveDeadlineEpochMs } from "../_shared/deadline.ts";
import { resolveTurnCap } from "../_shared/turn-cap.ts";

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
//      recent turn durations. Re-derived fresh every turn_start, so it's
//      self-correcting and needs no restore-on-input logic — unlike
//      forcedOff, it never sets its own persistent latch.
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

function envNumber(name: string, fallback: number): number {
  const n = Number(process.env[name]);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

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

// The raw budget resolved from the profile/env at before_agent_start —
// `budgetForTurn`'s adaptive-adjusted value is re-derived from this every
// turn_start, never from its own previous (possibly already-shrunk) value.
let baseBudgetForRun = DEFAULT_BUDGET;
// Deadline/turn-cap visibility, resolved the same way finalize-warn and
// turn-cap already do. `capForRun` isn't consumed by the adaptive math below
// (turn-cap's own extension owns turn-count exhaustion) — resolved here only
// for parity with those extensions' before_agent_start bookkeeping.
let deadlineForRun = 0;
let capForRun = 0;
// The wall-clock window between "now" and the deadline as first observed for
// this run (captured once at before_agent_start; Infinity if no deadline).
// Used by resolveAdaptiveBudget's outsized-single-turn check, which needs a
// fixed denominator — unlike the turns-remaining check, which deliberately
// uses the fresher, shrinking remaining-time figure instead.
let totalBudgetMsForRun = Infinity;
// Wall-clock turn timing, for both mechanisms: turnStartedAt anchors the
// message_update guard's elapsed-this-turn check; lastTurnMs/avgTurnMs feed
// resolveAdaptiveBudget.
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
 * rolling turn-duration bookkeeping each time, which is what makes this
 * self-correcting without a restore-on-input latch like forcedOff's.
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
  if (predictedTurnsRemaining < envNumber("LITTLE_CODER_THINKING_ADAPT_MIN_TURNS", DEFAULT_ADAPT_MIN_TURNS)) {
    return 0;
  }

  if (Number.isFinite(totalBudgetMs) && totalBudgetMs > 0) {
    const turnFraction = lastTurnMs / totalBudgetMs;
    const outsizedFraction = envNumber(
      "LITTLE_CODER_THINKING_ADAPT_OUTSIZED_FRACTION",
      DEFAULT_ADAPT_OUTSIZED_FRACTION,
    );
    if (turnFraction > outsizedFraction) return Math.floor(baseBudget / 2);
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
    priorLevel = safeGetThinkingLevel(pi);
    forcedOff = true;
  }
  safeSetThinkingLevel(pi, "off");
  try {
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
    priorLevel = undefined;
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

  // A genuinely new user prompt ends the "forced off" window: restore the
  // level the user actually had before the breach. Programmatic follow-ups
  // (our nudge) do not emit an `input` event, so the restart turn stays off.
  pi.on("input", async () => {
    if (forcedOff) {
      if (priorLevel !== undefined) safeSetThinkingLevel(pi, priorLevel);
      forcedOff = false;
      priorLevel = undefined;
    }
    thinkingChars = 0;
    aborted = false;
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
    totalBudgetMsForRun = deadlineForRun > 0 ? deadlineForRun - Date.now() : Infinity;

    turnStartedAt = undefined;
    avgTurnMs = 0;
    lastTurnMs = 0;
    turnsObserved = 0;
  });

  pi.on("turn_start", async () => {
    const now = Date.now();
    // turnStartedAt is undefined only for the very first turn of a run — no
    // prior turn to clock yet.
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
    if (!aborted && turnStartedAt !== undefined) {
      const now = Date.now();
      const elapsedThisTurn = now - turnStartedAt;
      const remainingMs = deadlineForRun > 0 ? deadlineForRun - now : Infinity;
      const guardMs = Math.min(
        remainingMs * envNumber("LITTLE_CODER_THINKING_GUARD_REMAINING_FRACTION", DEFAULT_GUARD_REMAINING_FRACTION),
        envNumber("LITTLE_CODER_THINKING_GUARD_HARD_CAP_MS", DEFAULT_GUARD_HARD_CAP_MS),
      );
      if (elapsedThisTurn > guardMs) {
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
