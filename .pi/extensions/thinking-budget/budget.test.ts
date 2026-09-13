import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import setupExtension, { resolveAdaptiveBudget } from "./index.ts";

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

    vi.advanceTimersByTime(1500);

    await fire(
      h.pi,
      "message_update",
      { assistantMessageEvent: { type: "text_delta", delta: "hello there" } },
      h.ctx,
    );

    expect(h.calls).toEqual(["set:off", "send", "notify", "abort"]);
    expect(h.level()).toBe("off");
  });

  it("does not fire while comfortably within the guard window", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);

    vi.advanceTimersByTime(200); // well under the 1000ms hard cap

    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);

    expect(h.calls).toEqual([]);
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

  it("still aborts a zero-budget breach outside the window", async () => {
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(60 * 60 * 1000); // far from any window
    const h = makeHarness("high");
    setupExtension(h.pi as any);
    await twoTurnsAtZeroBudget(h);

    await fire(h.pi, "message_update", thinkingDelta("hm"), h.ctx);

    expect(h.calls).toContain("abort");
    expect(h.followUps[0]).toMatch(/thinking budget exceeded/i);
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
