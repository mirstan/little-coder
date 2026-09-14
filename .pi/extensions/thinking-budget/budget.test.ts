import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import setupExtension, { resolveAdaptiveBudget, resolveGuardMs } from "./index.ts";

// Exercise the char→token conversion (matches local/context_manager.py)
function charsToTokens(chars: number): number {
  return Math.ceil(chars / 3.5);
}

describe("thinking budget token estimation", () => {
  it("converts chars to tokens via /3.5", () => {
    expect(charsToTokens(0)).toBe(0);
    expect(charsToTokens(3)).toBe(1);
    expect(charsToTokens(7)).toBe(2);
    expect(charsToTokens(3500)).toBe(1000);
  });
  it("4096 tokens ~ 14336 chars (the v1.5.0 default budget)", () => {
    expect(charsToTokens(14336)).toBe(4096);
    expect(charsToTokens(14337)).toBeGreaterThan(4096);
  });
});

// ── Issue #8 regression coverage (second reproduction, 1.4.3) ───────────────
// The bug: recovery (setThinkingLevel("off") + sendUserMessage) was deferred to
// a `turn_end` handler that ran against the module-scope `pi` AFTER ctx.abort()
// triggered a session replacement → stale `pi` → throw → thinking never turned
// off + follow-up never sent.
//
// The fix: do the whole recovery synchronously in `message_update`, BEFORE
// ctx.abort(), while `pi` is still live. These tests pin that choreography:
//   - no `turn_end` handler exists (nothing can run against a stale pi),
//   - setThinkingLevel + sendUserMessage are ordered strictly before abort,
//   - thinking is re-asserted off across the restart turn,
//   - the prior level is restored on the next genuine user input.

interface Handler {
  (event: any, ctx: any): Promise<unknown> | unknown;
}

function makeHarness(initialLevel = "high") {
  const calls: string[] = []; // ordered log across pi + ctx
  const followUps: string[] = [];
  const notifies: string[] = [];
  let level = initialLevel;
  const handlers: Record<string, Handler[]> = {};
  // Declared before `pi` because the sendUserMessage mock below passes it to
  // the input handlers it fires.
  const ctx = {
    abort() {
      calls.push("abort");
    },
    ui: {
      notify(m: string) {
        notifies.push(m);
        calls.push("notify");
      },
    },
  };
  const pi = {
    handlers,
    on(name: string, h: Handler) {
      (handlers[name] ??= []).push(h);
    },
    getThinkingLevel() {
      return level;
    },
    setThinkingLevel(l: string) {
      level = l;
      calls.push(`set:${l}`);
    },
    sendUserMessage(m: string) {
      // Mirrors pi 0.83.0: sendUserMessage -> prompt() emits `input` with
      // source "extension" BEFORE queueing the message (agent-session.js
      // emits at the top of prompt(), above the isStreaming followUp
      // branch). Not awaiting reproduces the real hazard exactly — the input
      // handler has no awaits, so it runs to completion synchronously inside
      // this call, mid-recovery.
      for (const h of handlers["input"] ?? []) void h({ type: "input", text: m, source: "extension" }, ctx);
      followUps.push(m);
      calls.push("send");
    },
  };
  return {
    pi,
    ctx,
    calls,
    followUps,
    notifies,
    level: () => level,
    setLevelExternally: (l: string) => {
      level = l;
    },
  };
}

async function fire(pi: any, name: string, event: any, ctx: any) {
  for (const h of pi.handlers[name] ?? []) await h(event, ctx);
}

function thinkingDelta(s: string) {
  return { assistantMessageEvent: { type: "thinking_delta", delta: s } };
}

function textDelta(s: string) {
  return { assistantMessageEvent: { type: "text_delta", delta: s } };
}

// Always begin from a clean session — resets the extension's module-scoped
// state so cases don't leak `forcedOff` / `priorLevel` into one another (and
// mirrors real startup: session_start always precedes the first agent run).
async function startRun(h: ReturnType<typeof makeHarness>) {
  await fire(h.pi, "session_start", {}, h.ctx);
  await fire(h.pi, "agent_start", {}, h.ctx);
  await fire(h.pi, "before_agent_start", { systemPromptOptions: {} }, h.ctx);
  await fire(h.pi, "turn_start", {}, h.ctx);
}

describe("thinking-budget recovery (issue #8)", () => {
  beforeEach(() => {
    process.env.LITTLE_CODER_THINKING_BUDGET = "10"; // tiny budget for short strings
  });
  afterEach(() => {
    delete process.env.LITTLE_CODER_THINKING_BUDGET;
  });

  it("registers NO turn_end handler (recovery must not run against a stale pi)", () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    expect(h.pi.handlers["turn_end"]).toBeUndefined();
  });

  it("on breach, runs the full recovery BEFORE abort and exactly once", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);

    await fire(h.pi, "message_update", thinkingDelta("x".repeat(1000)), h.ctx);

    // setThinkingLevel("off") and sendUserMessage both happen before abort.
    expect(h.calls).toEqual(["set:off", "send", "notify", "abort"]);
    expect(h.level()).toBe("off");
    expect(h.followUps).toHaveLength(1);
    expect(h.followUps[0]).toMatch(/thinking budget exceeded/i);
    expect(h.notifies[0]).toMatch(/harness intervention:.*thought long enough/i);
  });

  it("does not double-abort across multiple bursts in the same turn", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(h.pi, "message_update", thinkingDelta("x".repeat(1000)), h.ctx);
    await fire(h.pi, "message_update", thinkingDelta("y".repeat(1000)), h.ctx);
    await fire(h.pi, "message_update", thinkingDelta("z".repeat(1000)), h.ctx);

    expect(h.calls.filter((c) => c === "abort")).toHaveLength(1);
    expect(h.followUps).toHaveLength(1);
  });

  it("does not fire under budget", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(h.pi, "message_update", thinkingDelta("ok"), h.ctx); // 2 chars < 10 tokens
    expect(h.calls).toEqual([]);
    expect(h.level()).toBe("high");
  });

  it("re-asserts thinking off on the restart turn even if pi re-enables it", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(h.pi, "message_update", thinkingDelta("x".repeat(1000)), h.ctx); // breach → off

    // Simulate the post-abort session replacement re-resolving thinking to the
    // profile default. The bug was that this stuck; the fix re-asserts off.
    h.setLevelExternally("high");
    await fire(h.pi, "agent_start", {}, h.ctx); // restart run after the followUp
    await fire(h.pi, "before_agent_start", { systemPromptOptions: {} }, h.ctx);
    await fire(h.pi, "turn_start", {}, h.ctx);

    expect(h.level()).toBe("off");
  });

  it("restores the prior thinking level on the next genuine user input", async () => {
    const h = makeHarness("medium");
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(h.pi, "message_update", thinkingDelta("x".repeat(1000)), h.ctx); // breach
    expect(h.level()).toBe("off");

    // A new user prompt ends the forced-off window and restores the level.
    await fire(h.pi, "input", { text: "next task", source: "interactive" }, h.ctx);
    expect(h.level()).toBe("medium");

    // And the force is cleared: a subsequent turn does NOT re-disable thinking.
    await fire(h.pi, "turn_start", {}, h.ctx);
    expect(h.level()).toBe("medium");
  });

  it("a fresh task (no prior breach) is never forced off", async () => {
    const h = makeHarness("low");
    setupExtension(h.pi as any);
    await fire(h.pi, "input", { text: "task", source: "interactive" }, h.ctx);
    await startRun(h);
    await fire(h.pi, "message_update", thinkingDelta("ok"), h.ctx);
    expect(h.level()).toBe("low");
    expect(h.calls).toEqual([]);
  });
});

describe("thinking-budget resolution", () => {
  afterEach(() => {
    delete process.env.LITTLE_CODER_THINKING_BUDGET;
  });

  it("a profile budget wins over the env budget", async () => {
    process.env.LITTLE_CODER_THINKING_BUDGET = "10";
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "session_start", {}, h.ctx);
    await fire(h.pi, "agent_start", {}, h.ctx);
    // profile budget 100 tokens (~350 chars) overrides env's 10.
    await fire(
      h.pi,
      "before_agent_start",
      { systemPromptOptions: { littleCoder: { thinkingBudget: 100 } } },
      h.ctx,
    );
    await fire(h.pi, "turn_start", {}, h.ctx);
    // 200 chars ≈ 58 tokens — under the 100-token profile budget, over env's 10.
    await fire(h.pi, "message_update", thinkingDelta("x".repeat(200)), h.ctx);
    expect(h.calls).toEqual([]);
  });
});

// ── Mechanism 1: turn-boundary adaptive budget ──────────────────────────────
// resolveAdaptiveBudget is pure/stateless, so it's exercised directly with
// explicit inputs rather than through the extension harness.
describe("resolveAdaptiveBudget", () => {
  afterEach(() => {
    delete process.env.LITTLE_CODER_THINKING_ADAPT_MIN_TURNS;
    delete process.env.LITTLE_CODER_THINKING_ADAPT_OUTSIZED_FRACTION;
  });

  const NOW = 1_700_000_000_000;

  it.each([
    {
      name: "no deadline configured -> baseBudget unchanged",
      params: {
        baseBudget: 4096,
        remainingMs: Infinity,
        avgTurnMs: 5000,
        lastTurnMs: 600_000, // would be "outsized" if a deadline existed
        now: NOW,
        totalBudgetMs: Infinity,
      },
      expected: 4096,
    },
    {
      name: "healthy: plenty of predicted turns, no outsized turn -> baseBudget unchanged",
      params: {
        baseBudget: 4096,
        remainingMs: 600_000, // 10 min left
        avgTurnMs: 30_000, // 30s/turn -> ~20 turns fit
        lastTurnMs: 30_000,
        now: NOW,
        totalBudgetMs: 3_600_000, // 1h original window
      },
      expected: 4096,
    },
    {
      name: "few predicted turns remain -> thinking off (0)",
      params: {
        baseBudget: 4096,
        remainingMs: 60_000, // 1 min left
        avgTurnMs: 30_000,
        lastTurnMs: 30_000, // -> predicts 2 more turns, under the 4-turn floor
        now: NOW,
        totalBudgetMs: 3_600_000,
      },
      expected: 0,
    },
    {
      name: "single turn ate an outsized share of the original budget -> halved",
      params: {
        baseBudget: 4096,
        remainingMs: 2_500_000, // ~41.7 min left -> predicts ~4.16 turns, still healthy
        avgTurnMs: 200_000,
        lastTurnMs: 600_000, // 10 min, ~16.7% of the 1h original window
        now: NOW,
        totalBudgetMs: 3_600_000,
      },
      expected: 2048,
    },
  ])("$name", ({ params, expected }) => {
    expect(resolveAdaptiveBudget(params)).toBe(expected);
  });

  // Each knob's 0 must mean "disabled", not "compare against 0" — which for
  // the outsized check would invert it into permanent halving, since any
  // completed turn makes turnFraction positive.
  it("an outsized fraction of 0 disables halving even when turnFraction > 0", () => {
    process.env.LITTLE_CODER_THINKING_ADAPT_OUTSIZED_FRACTION = "0";
    expect(
      resolveAdaptiveBudget({
        baseBudget: 4096,
        remainingMs: 2_500_000,
        avgTurnMs: 200_000,
        lastTurnMs: 600_000, // ~16.7% of the window: halved at the default 0.15
        now: NOW,
        totalBudgetMs: 3_600_000,
      }),
    ).toBe(4096);
  });

  // Reachable only past the deadline, where remainingMs goes negative and a
  // bare `predictedTurnsRemaining < minTurns` would hold against a 0 floor.
  it("a min-turns floor of 0 disables the floor even past the deadline", () => {
    process.env.LITTLE_CODER_THINKING_ADAPT_MIN_TURNS = "0";
    expect(
      resolveAdaptiveBudget({
        baseBudget: 4096,
        remainingMs: -1000,
        avgTurnMs: 1000,
        lastTurnMs: 1000,
        now: NOW,
        totalBudgetMs: 3_600_000,
      }),
    ).toBe(4096);
  });

  it("thresholds are overridable via env vars", () => {
    // remainingMs / turnEstimateMs predicts exactly 5 more turns: healthy
    // under the default 4-turn floor, but a tightened override (6) now
    // classifies it as too few turns remaining.
    process.env.LITTLE_CODER_THINKING_ADAPT_MIN_TURNS = "6";
    expect(
      resolveAdaptiveBudget({
        baseBudget: 4096,
        remainingMs: 150_000,
        avgTurnMs: 30_000,
        lastTurnMs: 30_000,
        now: NOW,
        totalBudgetMs: 3_600_000,
      }),
    ).toBe(0);
  });
});

// ── P1: the guard's throughput floor ────────────────────────────────────────
// resolveGuardMs is pure/stateless like resolveAdaptiveBudget, so it's
// exercised directly. The base window shrinks with the remaining trial time,
// which is backwards: it tightens fastest exactly when a grown context has
// halved throughput and turns legitimately need longer.
describe("resolveGuardMs", () => {
  // 100 chars/sec ÷ 3.5 = 28.571 tok/sec, the rate measured off a real
  // completion in the trial this floor was added for (9793 tokens / 509.37s).
  const base = {
    remainingMs: 1_200_000, // 20 min left -> a 300s base window at 25%
    guardFraction: 0.25,
    guardHardCapMs: 900_000,
    budgetForTurn: 32768,
    charsPerSec: 100,
    minActionTokens: 1024,
    slack: 1.25,
  };

  it("falls back to the base window with no throughput estimate yet", () => {
    expect(resolveGuardMs({ ...base, charsPerSec: 0 })).toBe(300_000);
  });

  // (32768 + 1024) tokens / 28.571 tok/s = 1182.72s, x1.25 slack = 1478.4s.
  // The budget alone would be 1432s — omitting minActionTokens buys the turn
  // exactly enough time to finish thinking and none to act on it.
  it("floors the window at the time the granted budget actually costs", () => {
    expect(resolveGuardMs(base) / 1000).toBeCloseTo(1478.4, 1);
  });

  // The late-game regime: once the adaptive budget is 0 the floor is only the
  // action headroom, so it cannot let a single turn eat the endgame.
  it("collapses to the action headroom alone at a zero budget", () => {
    const guardMs = resolveGuardMs({ ...base, budgetForTurn: 0, remainingMs: 100_000 });
    expect(guardMs / 1000).toBeCloseTo(44.8, 1);
  });

  it("never lowers the window below the base one", () => {
    // A fast model: the floor computes well under the base window.
    expect(resolveGuardMs({ ...base, charsPerSec: 100_000 })).toBe(300_000);
  });

  it("a minActionTokens of 0 disables the floor entirely", () => {
    expect(resolveGuardMs({ ...base, minActionTokens: 0 })).toBe(300_000);
  });

  it("the slack factor scales the floor", () => {
    const single = resolveGuardMs({ ...base, slack: 1 });
    expect(resolveGuardMs({ ...base, slack: 2 })).toBeCloseTo(single * 2, 5);
  });
});

// ── Mechanism 2: per-turn wall-clock guard ──────────────────────────────────
// This is the mechanism that actually interrupts a hung in-flight generation
// (mechanism 1's shrunk token budget can't stop a request already underway).
describe("thinking-budget wall-clock guard", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    // A tiny hard cap so the test doesn't need to simulate 15 real minutes.
    process.env.LITTLE_CODER_THINKING_GUARD_HARD_CAP_MS = "1000";
    // The guard only runs against a real deadline. An hour out from the
    // fake clock's 0 leaves the finalize-warn window (last 10 minutes) well
    // clear, so these cases exercise the guard itself rather than its
    // stand-down path, and keeps 25% of remaining time far above the 1000ms
    // hard cap so the cap is what binds.
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(60 * 60 * 1000);
  });
  afterEach(() => {
    vi.useRealTimers();
    delete process.env.LITTLE_CODER_THINKING_GUARD_HARD_CAP_MS;
    delete process.env.LITTLE_CODER_THINKING_BUDGET;
    delete process.env.LITTLE_CODER_DEADLINE_EPOCH_MS;
  });

  it("fires the exact recovery sequence, in order, before abort, on breach", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);

    // The guard clocks generation, not the turn, so a first delta is what
    // starts its window — see the prefill-exclusion cases below.
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);
    vi.advanceTimersByTime(1500); // past the 1000ms hard cap
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);

    expect(h.calls).toEqual(["set:off", "send", "notify", "abort"]);
    expect(h.level()).toBe("off");
    expect(h.followUps[0]).toMatch(/wall-clock/i);
    expect(h.notifies[0]).toMatch(/harness intervention:.*wall-clock/i);
  });

  // This is the test that fails if the guard is placed AFTER the
  // `ev.type !== "thinking_delta"` early-return: a non-thinking delta type
  // (here, a plain text delta) must still trip the guard.
  it("fires for a non-thinking delta type too, not just thinking_delta", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);

    const textDelta = { assistantMessageEvent: { type: "text_delta", delta: "hello there" } };
    await fire(h.pi, "message_update", textDelta, h.ctx);
    vi.advanceTimersByTime(1500);
    await fire(h.pi, "message_update", textDelta, h.ctx);

    expect(h.calls).toEqual(["set:off", "send", "notify", "abort"]);
    expect(h.level()).toBe("off");
  });

  it("does not fire while comfortably within the guard window", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);

    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);
    vi.advanceTimersByTime(200); // well under the 1000ms hard cap
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);

    expect(h.calls).toEqual([]);
  });

  // Prefill is dead time the model spends re-reading context, not generating,
  // and it ran ~118s per cycle in the spiral this guard was reworked for.
  // message_update cannot fire during it, so the first delta is an exact
  // boundary — everything before it must be outside the guard's window.
  it("excludes prefill: a long wait before the first delta does not trip it", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);

    vi.advanceTimersByTime(118_000); // a full re-prefill, >> the 1000ms cap
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);

    expect(h.calls).toEqual([]);

    // ...and the window then runs from that first delta, not from turn_start.
    vi.advanceTimersByTime(1500);
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);

    expect(h.calls).toEqual(["set:off", "send", "notify", "abort"]);
  });
});

// ── input-source filtering ──────────────────────────────────────────────────
// pi routes pi.sendUserMessage() through prompt(), which emits `input` with
// source "extension" before queueing — so every extension's nudge (ours,
// finalize-warn's, tb-finalize-guard's) lands in this handler mid-flight. The
// harness mock above reproduces that. Treating one as a new user prompt undoes
// the forced-off state while the recovery that set it is still running.
describe("thinking-budget input-source filtering", () => {
  beforeEach(() => {
    process.env.LITTLE_CODER_THINKING_BUDGET = "10";
  });
  afterEach(() => {
    delete process.env.LITTLE_CODER_THINKING_BUDGET;
  });

  it("our own recovery follow-up does not restore thinking mid-recovery", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);

    await fire(h.pi, "message_update", thinkingDelta("x".repeat(1000)), h.ctx);

    // No set:high between set:off and send — that interleaving is the bug.
    expect(h.calls).toEqual(["set:off", "send", "notify", "abort"]);
    expect(h.level()).toBe("off");

    // The forced-off window survived the follow-up: a session replacement
    // re-enabling thinking is still overridden on the restart turn.
    h.setLevelExternally("high");
    await fire(h.pi, "turn_start", {}, h.ctx);
    expect(h.level()).toBe("off");
  });

  it("an extension-sourced input after a breach leaves the level alone", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(h.pi, "message_update", thinkingDelta("x".repeat(1000)), h.ctx);
    expect(h.level()).toBe("off");

    await fire(h.pi, "input", { text: "some other extension's nudge", source: "extension" }, h.ctx);
    expect(h.level()).toBe("off");

    // Still latched, so a genuine prompt can still restore.
    await fire(h.pi, "input", { text: "new task", source: "interactive" }, h.ctx);
    expect(h.level()).toBe("high");
  });

  it("an rpc-sourced input after a breach DOES restore (the harbor adapter's driver)", async () => {
    const h = makeHarness("medium");
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(h.pi, "message_update", thinkingDelta("x".repeat(1000)), h.ctx);
    expect(h.level()).toBe("off");

    await fire(h.pi, "input", { text: "next task", source: "rpc" }, h.ctx);
    expect(h.level()).toBe("medium");
  });

  it("an input with no source is treated as genuine, not filtered", async () => {
    const h = makeHarness("low");
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(h.pi, "message_update", thinkingDelta("x".repeat(1000)), h.ctx);

    await fire(h.pi, "input", { text: "next task" }, h.ctx);
    expect(h.level()).toBe("low");
  });
});

// ── Mechanism 1's latch: adaptiveOff / adaptivePriorLevel ───────────────────
// A zero adaptive budget has to actually set the level to "off", which makes
// it a latch needing the same restore care as forcedOff — and a separate one,
// since the two windows end at different moments.
describe("thinking-budget adaptive-off latch", () => {
  const DEADLINE = 60 * 60 * 1000; // 1h out from the fake clock's 0

  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(DEADLINE);
  });
  afterEach(() => {
    vi.useRealTimers();
    delete process.env.LITTLE_CODER_DEADLINE_EPOCH_MS;
  });

  // Turn 1 at t=0, then one ~16.7min turn: the min-turns floor now predicts
  // only ~2.6 turns fit in the remaining 43min, so the budget resolves to 0.
  async function latchAdaptiveOff(h: ReturnType<typeof makeHarness>) {
    await startRun(h);
    vi.advanceTimersByTime(1_000_000);
    await fire(h.pi, "turn_start", {}, h.ctx);
  }

  it("latches thinking off at turn_start when headroom is exhausted", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await latchAdaptiveOff(h);
    expect(h.level()).toBe("off");
  });

  it("restores the real prior level on a genuine input, and re-latches that level (not off)", async () => {
    const h = makeHarness("high");
    setupExtension(h.pi as any);
    await latchAdaptiveOff(h);
    expect(h.level()).toBe("off");

    await fire(h.pi, "input", { text: "next task", source: "interactive" }, h.ctx);
    expect(h.level()).toBe("high");

    // Headroom is still exhausted, so this re-latches. A bare reset on input
    // (clear without restore) would have left the level "off" here and
    // captured "off" as the level to restore — sticking thinking off forever.
    await fire(h.pi, "turn_start", {}, h.ctx);
    expect(h.level()).toBe("off");

    await fire(h.pi, "input", { text: "third task", source: "interactive" }, h.ctx);
    expect(h.level()).toBe("high");
  });

  it("an extension-sourced input neither restores nor clears the latch", async () => {
    const h = makeHarness("high");
    setupExtension(h.pi as any);
    await latchAdaptiveOff(h);

    await fire(h.pi, "input", { text: "a nudge", source: "extension" }, h.ctx);
    expect(h.level()).toBe("off");

    await fire(h.pi, "input", { text: "real task", source: "interactive" }, h.ctx);
    expect(h.level()).toBe("high");
  });

  it("a breach during adaptive-off inherits the real prior level, not off", async () => {
    const h = makeHarness("high");
    setupExtension(h.pi as any);
    await latchAdaptiveOff(h);
    expect(h.level()).toBe("off");

    // A turn on from the latch, so the level "off" has actually reached the
    // model and a breach means over-thinking rather than a pending change —
    // without this the just-latched suppression below would skip the check.
    vi.advanceTimersByTime(1_000_000);
    await fire(h.pi, "turn_start", {}, h.ctx);

    // Budget is 0, so any thinking at all breaches. forcedOff takes ownership.
    await fire(h.pi, "message_update", thinkingDelta("hm"), h.ctx);
    expect(h.level()).toBe("off");

    // If the handoff captured "off" instead of the adaptive latch's "high",
    // this restores to "off" and thinking is lost for the session.
    await fire(h.pi, "input", { text: "next task", source: "interactive" }, h.ctx);
    expect(h.level()).toBe("high");
  });

  // The latch's whole design is self-restoring, but it was unreachable: the
  // zeroed budget binds on the turn already in flight while the "off" it was
  // paired with only binds on the next one, so the first thinking char of that
  // turn breached and promoted the transient latch into permanent forcedOff.
  it("does not escalate into permanent forcedOff on the turn it latches", async () => {
    const h = makeHarness("high");
    setupExtension(h.pi as any);
    await latchAdaptiveOff(h);
    expect(h.level()).toBe("off");

    // Pre-fix this breached (tokens=1 > budget=0) and ran the full recovery.
    await fire(h.pi, "message_update", thinkingDelta("hm"), h.ctx);
    expect(h.calls).not.toContain("abort");
    expect(h.followUps).toHaveLength(0);

    // And because forcedOff never took over, the latch can still self-restore
    // once a fast turn pulls the estimate back down. Pre-fix this stayed "off"
    // for the rest of the session: the adaptive block is gated on !forcedOff.
    vi.advanceTimersByTime(100);
    await fire(h.pi, "turn_start", {}, h.ctx);
    expect(h.level()).toBe("high");
  });

  it("unlatches and restores when headroom recovers", async () => {
    const h = makeHarness("high");
    setupExtension(h.pi as any);
    await latchAdaptiveOff(h);
    expect(h.level()).toBe("off");

    // A fast turn pulls the rolling average down far enough that the
    // predicted turns remaining clears the floor again.
    vi.advanceTimersByTime(100);
    await fire(h.pi, "turn_start", {}, h.ctx);
    expect(h.level()).toBe("high");
  });
});

// ── Guard gating: deadline requirement, finalize-warn stand-down, knobs ─────
describe("thinking-budget wall-clock guard gating", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    process.env.LITTLE_CODER_THINKING_GUARD_HARD_CAP_MS = "1000";
  });
  afterEach(() => {
    vi.useRealTimers();
    delete process.env.LITTLE_CODER_THINKING_GUARD_HARD_CAP_MS;
    delete process.env.LITTLE_CODER_THINKING_GUARD_REMAINING_FRACTION;
    delete process.env.LITTLE_CODER_DEADLINE_EPOCH_MS;
    delete process.env.LITTLE_CODER_MAX_TURNS;
  });

  it("is fully inert with no deadline configured (interactive use)", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);

    vi.advanceTimersByTime(1500);
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);

    expect(h.calls).toEqual([]);
    expect(h.level()).toBe("high");
  });

  it("stands down inside the wall-clock finalize-warn window, notifying exactly once", async () => {
    // 5 minutes out: already inside finalize-warn's 10-minute window.
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(5 * 60 * 1000);
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);

    await fire(h.pi, "message_update", thinkingDelta("w"), h.ctx); // starts the generation clock
    vi.advanceTimersByTime(1500);
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);
    await fire(h.pi, "message_update", thinkingDelta("y"), h.ctx);
    await fire(h.pi, "message_update", thinkingDelta("z"), h.ctx);

    expect(h.calls).toEqual(["notify"]);
    expect(h.notifies[0]).toMatch(/standing down.*finalize-warn window/i);
    expect(h.level()).toBe("high");
  });

  it("stands down inside the turn-count finalize-warn window, notifying exactly once", async () => {
    // Cap 10 with WARN_REMAINING 5: the window opens at turn 6, while an
    // hour of wall-clock remains — the case a wall-clock-only check misses.
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(60 * 60 * 1000);
    process.env.LITTLE_CODER_MAX_TURNS = "10";
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h); // turn 1
    for (let i = 0; i < 5; i++) await fire(h.pi, "turn_start", {}, h.ctx); // turns 2-6

    await fire(h.pi, "message_update", thinkingDelta("w"), h.ctx); // starts the generation clock
    vi.advanceTimersByTime(1500);
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);
    await fire(h.pi, "message_update", thinkingDelta("y"), h.ctx);

    expect(h.calls).toEqual(["notify"]);
    expect(h.notifies[0]).toMatch(/standing down.*finalize-warn window/i);
    expect(h.level()).toBe("high");
  });

  it("still fires normally on the turn just before the turn-count window opens", async () => {
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(60 * 60 * 1000);
    process.env.LITTLE_CODER_MAX_TURNS = "10";
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h); // turn 1
    for (let i = 0; i < 4; i++) await fire(h.pi, "turn_start", {}, h.ctx); // turns 2-5

    await fire(h.pi, "message_update", thinkingDelta("w"), h.ctx); // starts the generation clock
    vi.advanceTimersByTime(1500);
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);

    expect(h.calls).toEqual(["set:off", "send", "notify", "abort"]);
    expect(h.notifies[0]).toMatch(/wall-clock guard/i);
  });

  it.each([
    ["LITTLE_CODER_THINKING_GUARD_REMAINING_FRACTION", "fraction"],
    ["LITTLE_CODER_THINKING_GUARD_HARD_CAP_MS", "hard cap"],
  ])("a %s of 0 disables the guard even with a valid deadline", async (varName) => {
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(60 * 60 * 1000);
    process.env[varName] = "0";
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);

    vi.advanceTimersByTime(1500);
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);

    expect(h.calls).toEqual([]);
  });
});

// ── The wall-clock guard's forcedOff releases on headroom, not only on input ─
// The guard needs a deadline, and the only thing that sets one is the harbor
// adapter — which issues exactly one rpc.prompt_and_collect per trial. So in
// the one context where the guard can fire, neither of forcedOff's releases
// (session_start, a genuine input) ever comes again, and a single trip used to
// disable thinking for every remaining turn of the trial.
describe("thinking-budget wall-clock guard forcedOff release", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    process.env.LITTLE_CODER_THINKING_GUARD_HARD_CAP_MS = "1000";
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(60 * 60 * 1000);
  });
  afterEach(() => {
    vi.useRealTimers();
    delete process.env.LITTLE_CODER_THINKING_GUARD_HARD_CAP_MS;
    delete process.env.LITTLE_CODER_DEADLINE_EPOCH_MS;
    delete process.env.LITTLE_CODER_THINKING_ADAPT_MIN_TURNS;
  });

  async function tripGuard(h: ReturnType<typeof makeHarness>) {
    await startRun(h);
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx); // starts the generation clock
    vi.advanceTimersByTime(1500); // past the 1000ms hard cap
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);
  }

  it("restores thinking at the next turn_start once headroom is back", async () => {
    const h = makeHarness("high");
    setupExtension(h.pi as any);
    await tripGuard(h);
    expect(h.level()).toBe("off");

    // An hour of deadline remains and the turn that tripped was 1.5s, so the
    // headroom check that governs adaptiveOff says there is no problem now.
    // Pre-fix the level stayed "off" here and for every turn after it.
    await fire(h.pi, "turn_start", {}, h.ctx);
    expect(h.level()).toBe("high");

    // Twenty more healthy turns: thinking stays on, and nothing re-forces it.
    for (let i = 0; i < 20; i++) {
      vi.advanceTimersByTime(100);
      await fire(h.pi, "turn_start", {}, h.ctx);
    }
    expect(h.level()).toBe("high");
  });

  it("keeps thinking off while the headroom problem persists", async () => {
    // A floor no run can clear keeps resolveAdaptiveBudget pinned at 0, which
    // is the "deadline pressure is still on" signal the release consults.
    process.env.LITTLE_CODER_THINKING_ADAPT_MIN_TURNS = String(1e9);
    const h = makeHarness("high");
    setupExtension(h.pi as any);
    await tripGuard(h);

    await fire(h.pi, "turn_start", {}, h.ctx);
    expect(h.level()).toBe("off");
  });

  it("a token-budget breach's forcedOff does NOT self-release", async () => {
    // Only the guard flavor self-releases: a token-budget breach means this
    // task over-thought, which remaining clock does not disprove.
    process.env.LITTLE_CODER_THINKING_BUDGET = "10";
    const h = makeHarness("high");
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(h.pi, "message_update", thinkingDelta("x".repeat(1000)), h.ctx);
    expect(h.level()).toBe("off");

    vi.advanceTimersByTime(100);
    await fire(h.pi, "turn_start", {}, h.ctx);
    expect(h.level()).toBe("off");
    delete process.env.LITTLE_CODER_THINKING_BUDGET;
  });
});

// ── Both breach triggers stand down in the finalize-warn window ─────────────
// Near the deadline resolveAdaptiveBudget resolves to 0 by design, so the
// token-budget path fires on the first thinking token of what is very likely
// the model's final-answer turn. The guard already stood down there; this path
// did not.
describe("thinking-budget cap finalize-warn stand-down", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    // Pins the adaptive budget at 0 without needing a slow-turn setup, which
    // is exactly the state a near-deadline run is in.
    process.env.LITTLE_CODER_THINKING_ADAPT_MIN_TURNS = String(1e9);
  });
  afterEach(() => {
    vi.useRealTimers();
    delete process.env.LITTLE_CODER_THINKING_ADAPT_MIN_TURNS;
    delete process.env.LITTLE_CODER_DEADLINE_EPOCH_MS;
  });

  // Turn 1 latches adaptive-off; turn 2 is where the budget check is live
  // again (turn 1 is suppressed as the just-latched turn).
  async function twoTurnsAtZeroBudget(h: ReturnType<typeof makeHarness>) {
    await startRun(h);
    vi.advanceTimersByTime(10);
    await fire(h.pi, "turn_start", {}, h.ctx);
  }

  it("does not abort a zero-budget turn inside the window, notifying once", async () => {
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(5 * 60 * 1000); // inside the 10min window
    const h = makeHarness("high");
    setupExtension(h.pi as any);
    await twoTurnsAtZeroBudget(h);

    await fire(h.pi, "message_update", thinkingDelta("hm"), h.ctx);
    await fire(h.pi, "message_update", thinkingDelta("more"), h.ctx);
    await fire(h.pi, "message_update", thinkingDelta("still more"), h.ctx);

    // Pre-fix: ["set:off", "set:off", "send", "notify", "abort"] — the final
    // answer cut off two thinking tokens in.
    expect(h.calls).not.toContain("abort");
    expect(h.followUps).toHaveLength(0);
    expect(h.notifies).toHaveLength(1);
    expect(h.notifies[0]).toMatch(/standing down.*finalize-warn window/i);
  });

  it("stands down on the turn-count trigger too, with clock to spare", async () => {
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(60 * 60 * 1000);
    process.env.LITTLE_CODER_MAX_TURNS = "10"; // window opens at turn 6
    const h = makeHarness("high");
    setupExtension(h.pi as any);
    await startRun(h); // turn 1
    for (let i = 0; i < 5; i++) await fire(h.pi, "turn_start", {}, h.ctx); // turns 2-6

    await fire(h.pi, "message_update", thinkingDelta("hm"), h.ctx);

    expect(h.calls).not.toContain("abort");
    expect(h.notifies[0]).toMatch(/standing down.*finalize-warn window/i);
    delete process.env.LITTLE_CODER_MAX_TURNS;
  });

  // Outside the finalize-warn window a zero-budget breach used to abort. It
  // no longer does, because reaching this point at all means the level is
  // already "off" and the model thought anyway — see the futility-rule cases
  // below for why that makes an abort strictly lossy.
  it("stands down on a zero-budget breach outside the window too (futility)", async () => {
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(60 * 60 * 1000); // far from any window
    const h = makeHarness("high");
    setupExtension(h.pi as any);
    await twoTurnsAtZeroBudget(h);

    await fire(h.pi, "message_update", thinkingDelta("hm"), h.ctx);

    expect(h.calls).not.toContain("abort");
    expect(h.followUps).toHaveLength(0);
    expect(h.notifies[0]).toMatch(/thinking is already off and the model is thinking anyway/i);
  });
});

// ── P2: the futility rule ──────────────────────────────────────────────────
// A model config can pin thinking on regardless of the level (a literal
// `"enable_thinking": true` in chatTemplateKwargs makes setThinkingLevel("off")
// a no-op). Once that is happening, the token path's only remedy has already
// been applied and ignored, so each further abort discards a turn and buys a
// full re-prefill for nothing — 11 such cycles in the trial that motivated
// this. The condition is exactly "a latch already says off", not a guess at
// whether the model is honoring it.
describe("thinking-budget cap futility rule", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    process.env.LITTLE_CODER_THINKING_BUDGET = "10";
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(60 * 60 * 1000);
  });
  afterEach(() => {
    vi.useRealTimers();
    delete process.env.LITTLE_CODER_THINKING_BUDGET;
    delete process.env.LITTLE_CODER_DEADLINE_EPOCH_MS;
  });

  it("does not re-abort a forcedOff turn that keeps thinking, notifying once", async () => {
    const h = makeHarness("high");
    setupExtension(h.pi as any);
    await startRun(h);

    // Turn 1 over-thinks and is aborted normally: forcedOff latches.
    await fire(h.pi, "message_update", thinkingDelta("x".repeat(1000)), h.ctx);
    expect(h.calls.filter((c) => c === "abort")).toHaveLength(1);

    // The recovery restart. A token-budget forcedOff does not self-release, so
    // turn 2 runs with the level pinned off and a zero budget.
    await fire(h.pi, "agent_start", {}, h.ctx);
    await fire(h.pi, "turn_start", {}, h.ctx);

    // The model thinks anyway, repeatedly. Pre-fix each burst cost an abort.
    await fire(h.pi, "message_update", thinkingDelta("y".repeat(1000)), h.ctx);
    await fire(h.pi, "message_update", thinkingDelta("z".repeat(1000)), h.ctx);

    expect(h.calls.filter((c) => c === "abort")).toHaveLength(1);
    expect(h.followUps).toHaveLength(1);
    expect(h.notifies.filter((n) => /thinking is already off/i.test(n))).toHaveLength(1);
  });

  it("leaves the budget path armed when neither latch says off", async () => {
    // The guard flavor of forcedOff self-releases once headroom is back, and
    // the budget path must come back armed with it — the futility rule is a
    // condition on current state, not a latch of its own.
    process.env.LITTLE_CODER_THINKING_GUARD_HARD_CAP_MS = "1000";
    const h = makeHarness("high");
    setupExtension(h.pi as any);
    await startRun(h);

    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);
    vi.advanceTimersByTime(1500);
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx); // guard trip
    expect(h.level()).toBe("off");

    vi.advanceTimersByTime(100);
    await fire(h.pi, "turn_start", {}, h.ctx); // headroom is back: forcedOff releases
    expect(h.level()).toBe("high");

    await fire(h.pi, "message_update", thinkingDelta("q".repeat(1000)), h.ctx);

    expect(h.calls.filter((c) => c === "abort")).toHaveLength(2);
    expect(h.followUps[1]).toMatch(/thinking budget exceeded/i);
    delete process.env.LITTLE_CODER_THINKING_GUARD_HARD_CAP_MS;
  });
});

// ── Mid-stream steering is not a new task ──────────────────────────────────
// pi sets streamingBehavior on the input event only while isStreaming is true
// (agent-session.js), so its presence is an exact discriminator for "the user
// typed at a running agent" — which arrives as source "interactive" and used
// to slip through the extension-source filter.
describe("thinking-budget mid-stream steer filtering", () => {
  beforeEach(() => {
    process.env.LITTLE_CODER_THINKING_BUDGET = "10";
  });
  afterEach(() => {
    delete process.env.LITTLE_CODER_THINKING_BUDGET;
  });

  it("a steer during a forced-off task neither restores nor clears the latch", async () => {
    const h = makeHarness("high");
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(h.pi, "message_update", thinkingDelta("x".repeat(1000)), h.ctx);
    expect(h.level()).toBe("off");

    // Pre-fix this restored "high" mid-task — issue #8's symptom again.
    await fire(
      h.pi,
      "input",
      { text: "also fix the typo", source: "interactive", streamingBehavior: "steer" },
      h.ctx,
    );
    expect(h.level()).toBe("off");

    // Still latched, so the restart turn is still held off...
    h.setLevelExternally("high");
    await fire(h.pi, "turn_start", {}, h.ctx);
    expect(h.level()).toBe("off");

    // ...and a genuinely new prompt still restores.
    await fire(h.pi, "input", { text: "new task", source: "interactive" }, h.ctx);
    expect(h.level()).toBe("high");
  });

  it("filters a steer regardless of source", async () => {
    const h = makeHarness("medium");
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(h.pi, "message_update", thinkingDelta("x".repeat(1000)), h.ctx);

    await fire(h.pi, "input", { text: "steer", source: "rpc", streamingBehavior: "interrupt" }, h.ctx);
    expect(h.level()).toBe("off");
  });
});

// sendUserMessage returns void (ExtensionAPI in types.d.ts) and the runtime
// binding attaches its own .catch — so the only failure this call can present
// to us is a synchronous throw, which is what the try/catch around it covers.
describe("thinking-budget recovery when sendUserMessage throws", () => {
  beforeEach(() => {
    process.env.LITTLE_CODER_THINKING_BUDGET = "10";
  });
  afterEach(() => {
    delete process.env.LITTLE_CODER_THINKING_BUDGET;
  });

  it("still forces thinking off and aborts", async () => {
    const h = makeHarness("high");
    h.pi.sendUserMessage = () => {
      throw new Error("no sendUserMessage on this SDK");
    };
    setupExtension(h.pi as any);
    await startRun(h);

    await fire(h.pi, "message_update", thinkingDelta("x".repeat(1000)), h.ctx);

    expect(h.calls).toEqual(["set:off", "notify", "abort"]);
    expect(h.level()).toBe("off");
  });
});

// ── The abort spiral, end to end ────────────────────────────────────────────
// A real trial burned ~50 minutes in a loop of its own making: 11 instant
// token-budget aborts (thinking was pinned on by the model config, so the
// budget was 0 and the first thinking token breached it) interleaved with 8
// wall-clock aborts whose window shrank 900s → 375s as the trial's remaining
// time did, each one paying ~118s to re-prefill the context the last abort
// discarded. P1 and P2 attack different halves of that, so they get separate
// regression cases rather than one "the trial survives" test.
describe("thinking-budget abort spiral", () => {
  const HOUR = 60 * 60 * 1000;

  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
  });
  afterEach(() => {
    vi.useRealTimers();
    for (const name of [
      "LITTLE_CODER_DEADLINE_EPOCH_MS",
      "LITTLE_CODER_THINKING_BUDGET",
      "LITTLE_CODER_THINKING_GUARD_HARD_CAP_MS",
      "LITTLE_CODER_THINKING_GUARD_MIN_ACTION_TOKENS",
      "LITTLE_CODER_THINKING_ADAPT_MIN_TURNS",
    ]) {
      delete process.env[name];
    }
  });

  const aborts = (h: ReturnType<typeof makeHarness>) => h.calls.filter((c) => c === "abort").length;

  // A turn generating steadily at `chars / stepMs` — the shape of the real
  // slow-but-healthy completion (9793 tokens over 509s at ~100 chars/sec).
  async function generate(
    h: ReturnType<typeof makeHarness>,
    opts: { steps: number; stepMs: number; chars: number; kind?: typeof thinkingDelta },
  ) {
    const kind = opts.kind ?? thinkingDelta;
    for (let i = 0; i < opts.steps; i++) {
      vi.advanceTimersByTime(opts.stepMs);
      await fire(h.pi, "message_update", kind("x".repeat(opts.chars)), h.ctx);
    }
  }

  // Regime (a): budget > 0, the mid-game window before the first guard trip
  // collapses it. Here the throughput floor alone is what saves the turn — the
  // futility rule cannot help, since neither latch says "off" yet.
  describe("P1: the throughput floor, while the budget is still positive", () => {
    // 25 min of trial. Turn 1 runs 300s, leaving 20 min — a 300s base window,
    // against a turn that needs 390s of generation to finish.
    async function slowButProductiveTurn(h: ReturnType<typeof makeHarness>) {
      process.env.LITTLE_CODER_THINKING_BUDGET = "32768";
      process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(1_500_000);
      setupExtension(h.pi as any);
      await startRun(h);
      vi.advanceTimersByTime(300_000);
      await fire(h.pi, "turn_start", {}, h.ctx);
      // 390s of generation at ~100 chars/sec, the observed degraded rate.
      await generate(h, { steps: 40, stepMs: 10_000, chars: 1000 });
    }

    it("lets a 390s turn finish inside a 300s base window", async () => {
      const h = makeHarness();
      await slowButProductiveTurn(h);

      expect(h.calls).toEqual([]);
    });

    // The same turn with the floor switched off, which is also the pre-fix
    // behavior: the shrinking base window catches it about 240s in.
    it("aborts that same turn once the floor is disabled", async () => {
      process.env.LITTLE_CODER_THINKING_GUARD_MIN_ACTION_TOKENS = "0";
      const h = makeHarness();
      await slowButProductiveTurn(h);

      expect(aborts(h)).toBe(1);
      expect(h.followUps[0]).toMatch(/wall-clock guard/i);
    });
  });

  // Regime (b): budget pinned at 0, which is where the trial actually spent
  // its 50 minutes. The floor collapses to the action headroom here and cannot
  // save the turn on its own — P2 is what ends this one.
  it("P2 ends the spiral in the zero-budget regime", async () => {
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(HOUR);
    process.env.LITTLE_CODER_THINKING_GUARD_HARD_CAP_MS = "100000";
    // Pins the adaptive budget at 0 for every turn, as the endgame does.
    process.env.LITTLE_CODER_THINKING_ADAPT_MIN_TURNS = String(1e9);
    const h = makeHarness("high");
    setupExtension(h.pi as any);
    await startRun(h);

    // Turn 1: thinking arrives despite the level being off. The just-latched
    // suppression covers this one turn; the guard then trips on it.
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);
    vi.advanceTimersByTime(150_000);
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);
    expect(aborts(h)).toBe(1);

    // Turn 2, on the retry path. The model keeps thinking with the level off
    // and a budget of 0 — pre-fix, every one of these was an instant abort and
    // another ~118s re-prefill.
    await fire(h.pi, "agent_start", {}, h.ctx);
    await fire(h.pi, "turn_start", {}, h.ctx);
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);
    expect(aborts(h)).toBe(1);
    expect(h.notifies.some((n) => /thinking is already off/i.test(n))).toBe(true);

    vi.advanceTimersByTime(150_000);
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);
    expect(aborts(h)).toBe(2); // the second fruitless abort trips the breaker

    // Turn 3: the ~509s completion that actually finished in the real trial,
    // 40 turns and ~50 minutes after it first tried to.
    await fire(h.pi, "agent_start", {}, h.ctx);
    await fire(h.pi, "turn_start", {}, h.ctx);
    await generate(h, { steps: 50, stepMs: 10_000, chars: 1000 });
    await generate(h, { steps: 5, stepMs: 10_000, chars: 1000, kind: textDelta });

    expect(aborts(h)).toBe(2); // bounded at 2, not 19
    expect(h.notifies.some((n) => /standing down.*until a turn makes real progress/i.test(n))).toBe(
      true,
    );
  });

  // The estimator's best samples come from turns that were aborted for being
  // slow, and agent_start fires immediately before turn_start on the retry
  // path — so resetting the stream counters there would throw away exactly the
  // measurements the floor depends on.
  it("carries a turn's throughput sample across agent_start", async () => {
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(HOUR);
    process.env.LITTLE_CODER_THINKING_GUARD_HARD_CAP_MS = "6000";
    process.env.LITTLE_CODER_THINKING_BUDGET = "32768";
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);

    // Turn 1: 6000 chars over 5s — a qualifying sample at ~1200 chars/sec.
    await fire(h.pi, "message_update", textDelta("x".repeat(1000)), h.ctx);
    await generate(h, { steps: 5, stepMs: 1000, chars: 1000, kind: textDelta });
    expect(h.calls).toEqual([]);

    await fire(h.pi, "agent_start", {}, h.ctx);
    await fire(h.pi, "turn_start", {}, h.ctx);

    // Turn 2 has no sample of its own yet, so the floor runs off the carried
    // estimate and 7s of generation stays well inside it.
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);
    vi.advanceTimersByTime(7000);
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);
    expect(h.calls).toEqual([]);

    // A new session clears the estimate, and the identical turn now trips the
    // bare 6000ms cap — which is what turn 2 above would have done without it.
    await startRun(h);
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);
    vi.advanceTimersByTime(7000);
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);
    expect(aborts(h)).toBe(1);
  });
});

// ── P2: the circuit breaker ─────────────────────────────────────────────────
// Two consecutive extension aborts that produced no content mean the recovery
// is not reaching the model, and every further abort just buys another
// re-prefill. The reset condition is the subtle part: `aborted` cannot drive
// it, because agent_start fires immediately before turn_start on the retry
// path and clears that flag, so turn_start would read false every time and
// reset the count before it could ever reach the threshold.
describe("thinking-budget circuit breaker", () => {
  const HOUR = 60 * 60 * 1000;

  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    process.env.LITTLE_CODER_THINKING_GUARD_HARD_CAP_MS = "1000";
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(HOUR);
  });
  afterEach(() => {
    vi.useRealTimers();
    delete process.env.LITTLE_CODER_THINKING_GUARD_HARD_CAP_MS;
    delete process.env.LITTLE_CODER_DEADLINE_EPOCH_MS;
  });

  const aborts = (h: ReturnType<typeof makeHarness>) => h.calls.filter((c) => c === "abort").length;

  // One content-free turn: thinking only, then past the 1000ms cap.
  async function contentFreeGuardTrip(h: ReturnType<typeof makeHarness>) {
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);
    vi.advanceTimersByTime(1500);
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);
  }

  // pi's own recovery ordering (agent-loop.js): agent_start, then turn_start.
  async function retryTurn(h: ReturnType<typeof makeHarness>) {
    await fire(h.pi, "agent_start", {}, h.ctx);
    await fire(h.pi, "turn_start", {}, h.ctx);
  }

  async function tripTwice(h: ReturnType<typeof makeHarness>) {
    setupExtension(h.pi as any);
    await startRun(h);
    await contentFreeGuardTrip(h);
    await retryTurn(h);
    await contentFreeGuardTrip(h);
    await retryTurn(h);
  }

  it("stands both triggers down after two content-free aborts, notifying once", async () => {
    const h = makeHarness("high");
    await tripTwice(h);
    expect(aborts(h)).toBe(2);

    // Third turn: the same profile that aborted twice now does not.
    await contentFreeGuardTrip(h);
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);

    expect(aborts(h)).toBe(2);
    const breakerNotices = h.notifies.filter((n) =>
      /standing down.*until a turn makes real progress/i.test(n),
    );
    expect(breakerNotices).toHaveLength(1);
  });

  // The stand-down is bounded, not a disarm: with maxTokens at 80,000 and a
  // model running ~25 tok/s, a fully disarmed guard would let one turn run
  // ~53 minutes and eat the rest of the trial.
  it("still aborts rather than let a turn run into the finalize window", async () => {
    const h = makeHarness("high");
    await tripTwice(h);

    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);
    // Far past the 1000ms cap but still ~33 minutes clear of the deadline, so
    // the finalize-warn stand-down is not what is doing the work here.
    vi.advanceTimersByTime(1_600_000);
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);

    expect(aborts(h)).toBe(3);
    expect(h.followUps[2]).toMatch(/wall-clock guard/i);
  });

  it("releases once a turn produces real content", async () => {
    const h = makeHarness("high");
    await tripTwice(h);

    // The third turn is immune, and this time the model actually acts.
    await contentFreeGuardTrip(h);
    expect(aborts(h)).toBe(2);
    await fire(h.pi, "message_update", textDelta("writing the file now"), h.ctx);

    // A turn that ended on its own terms with content clears the count.
    await fire(h.pi, "turn_start", {}, h.ctx);
    await contentFreeGuardTrip(h);

    expect(aborts(h)).toBe(3);
  });

  // The regression for the reset condition itself. Reading `aborted` here
  // instead of `lastTurnEndedInExtensionAbort` reinstates the spiral: each
  // agent_start below clears it, so the count resets every turn and the
  // breaker never trips, leaving the third turn to abort like the first two.
  it("counts aborts across the agent_start on the retry path", async () => {
    const h = makeHarness("high");
    await tripTwice(h);

    await contentFreeGuardTrip(h);

    expect(aborts(h)).toBe(2);
  });

  // A turn can stream real content and still get guard-aborted for taking
  // too long overall (P1 is purely elapsed-time based, independent of
  // whether content already flowed) — that is still evidence of progress,
  // not a fruitless retry, and must clear the streak the same as a turn
  // that ends on its own terms. Gating the reset on "and wasn't aborted
  // too" (as opposed to content alone) would leave a prior no-content
  // streak stuck through this turn, one abort away from tripping the
  // breaker on what is actually a productive turn.
  it("clears the streak when an aborted turn had already produced real content", async () => {
    const h = makeHarness("high");
    setupExtension(h.pi as any);
    await startRun(h);
    await contentFreeGuardTrip(h);
    await retryTurn(h);
    expect(aborts(h)).toBe(1);

    // Second turn streams real content, then still runs past the cap.
    await fire(h.pi, "message_update", textDelta("writing the file now"), h.ctx);
    vi.advanceTimersByTime(1500);
    await fire(h.pi, "message_update", textDelta("still going"), h.ctx);
    expect(aborts(h)).toBe(2);
    await retryTurn(h);

    // Third turn is content-free again. This is where an unreset streak
    // would already sit at 2, but a single content-free abort must still
    // abort normally rather than hit the breaker.
    await contentFreeGuardTrip(h);
    expect(aborts(h)).toBe(3);
    await retryTurn(h);

    // Fourth turn, also content-free: if the second turn's content hadn't
    // cleared the streak, this would be the SECOND of an unbroken pair
    // (turns 3 and 4) and the breaker would hold instead of aborting. It
    // must instead be treated as only the second of a pair starting at
    // turn 3, i.e. still just the ordinary case, so this one aborts too.
    await contentFreeGuardTrip(h);
    expect(aborts(h)).toBe(4);
    const breakerNotices = h.notifies.filter((n) =>
      /standing down.*until a turn makes real progress/i.test(n),
    );
    expect(breakerNotices).toHaveLength(0);
  });

  // A genuine new task (not our own breach-recovery follow-up, filtered via
  // `source`/`streamingBehavior` — see the `input` handler) must not inherit
  // an unrelated prior task's abort streak. Without resetting the breaker
  // state here, two content-free aborts on one task would leave the next,
  // completely unrelated task starting with the breaker already primed.
  it("resets the breaker on a genuinely new task, not just a retry", async () => {
    const h = makeHarness("high");
    await tripTwice(h);
    expect(aborts(h)).toBe(2);

    await fire(h.pi, "input", { text: "a new, unrelated task", source: "interactive" }, h.ctx);
    await fire(h.pi, "turn_start", {}, h.ctx);
    await contentFreeGuardTrip(h);

    // First content-free abort of the new task — must not trip the breaker.
    expect(aborts(h)).toBe(3);
    const breakerNotices = h.notifies.filter((n) =>
      /standing down.*until a turn makes real progress/i.test(n),
    );
    expect(breakerNotices).toHaveLength(0);
  });
});
