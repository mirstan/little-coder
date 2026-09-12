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
      followUps.push(m);
      calls.push("send");
    },
  };
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
    await fire(h.pi, "input", { text: "next task" }, h.ctx);
    expect(h.level()).toBe("medium");

    // And the force is cleared: a subsequent turn does NOT re-disable thinking.
    await fire(h.pi, "turn_start", {}, h.ctx);
    expect(h.level()).toBe("medium");
  });

  it("a fresh task (no prior breach) is never forced off", async () => {
    const h = makeHarness("low");
    setupExtension(h.pi as any);
    await fire(h.pi, "input", { text: "task" }, h.ctx);
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
  });
  afterEach(() => {
    vi.useRealTimers();
    delete process.env.LITTLE_CODER_THINKING_GUARD_HARD_CAP_MS;
    delete process.env.LITTLE_CODER_THINKING_BUDGET;
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
