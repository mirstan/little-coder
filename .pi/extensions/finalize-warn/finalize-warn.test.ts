import { describe, it, expect, afterEach } from "vitest";
import setupExtension from "./index.ts";

interface Handler {
  (event: any, ctx: any): Promise<unknown> | unknown;
}

function makeHarness() {
  const notifies: string[] = [];
  const sent: { text: string; options: any }[] = [];
  const handlers: Record<string, Handler[]> = {};
  const pi = {
    handlers,
    on(name: string, h: Handler) {
      (handlers[name] ??= []).push(h);
    },
    sendUserMessage(text: string, options: any) {
      sent.push({ text, options });
    },
  };
  const ctx = {
    ui: {
      notify(m: string) {
        notifies.push(m);
      },
    },
  };
  return { pi, ctx, notifies, sent };
}

async function fire(pi: any, name: string, event: any, ctx: any) {
  for (const h of pi.handlers[name] ?? []) await h(event, ctx);
}

async function runTurns(h: ReturnType<typeof makeHarness>, n: number) {
  for (let i = 0; i < n; i++) await fire(h.pi, "turn_start", {}, h.ctx);
}

describe("finalize-warn", () => {
  afterEach(() => {
    delete process.env.LITTLE_CODER_MAX_TURNS;
    delete process.env.LITTLE_CODER_BENCHMARK;
    delete process.env.LITTLE_CODER_DEADLINE_EPOCH_MS;
  });

  it("does nothing when no cap or deadline is configured", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "before_agent_start", {}, h.ctx);
    await runTurns(h, 100);
    expect(h.sent).toEqual([]);
  });

  it("does nothing when the cap is too small to give real headroom (cap <= WARN_REMAINING)", async () => {
    process.env.LITTLE_CODER_MAX_TURNS = "5";
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "before_agent_start", {}, h.ctx);
    await runTurns(h, 5);
    expect(h.sent).toEqual([]);
  });

  it("warns once, exactly WARN_REMAINING (5) turns before the cap, with the GAIA message when benchmark=gaia", async () => {
    process.env.LITTLE_CODER_MAX_TURNS = "40";
    process.env.LITTLE_CODER_BENCHMARK = "gaia";
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "before_agent_start", {}, h.ctx);
    await runTurns(h, 36); // turn 36 = cap(40) - WARN_REMAINING(5) + 1
    expect(h.sent).toHaveLength(1);
    expect(h.sent[0].text).toMatch(/Answer: <value>/);
    expect(h.sent[0].text).not.toMatch(/`/); // plain text, no backtick formatting
    expect(h.sent[0].options).toEqual({ deliverAs: "followUp" });

    await runTurns(h, 10); // keeps running past the cap point; still just 1 warn
    expect(h.sent).toHaveLength(1);
  });

  it("uses the generic fallback message when no benchmark is set", async () => {
    process.env.LITTLE_CODER_MAX_TURNS = "40";
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "before_agent_start", {}, h.ctx);
    await runTurns(h, 36);
    expect(h.sent).toHaveLength(1);
    expect(h.sent[0].text).not.toMatch(/Answer:/);
    expect(h.sent[0].text).not.toMatch(/ShellSession/);
  });

  it("uses the terminal_bench-specific message when LITTLE_CODER_BENCHMARK=terminal_bench", async () => {
    process.env.LITTLE_CODER_MAX_TURNS = "40";
    process.env.LITTLE_CODER_BENCHMARK = "terminal_bench";
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "before_agent_start", {}, h.ctx);
    await runTurns(h, 36);
    expect(h.sent).toHaveLength(1);
    expect(h.sent[0].text).toMatch(/ShellSession/);
    expect(h.sent[0].text).not.toMatch(/Answer:/);
  });

  it("resets per agent run", async () => {
    process.env.LITTLE_CODER_MAX_TURNS = "40";
    process.env.LITTLE_CODER_BENCHMARK = "gaia";
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "before_agent_start", {}, h.ctx);
    await runTurns(h, 36);
    expect(h.sent).toHaveLength(1);

    await fire(h.pi, "before_agent_start", {}, h.ctx); // new run
    await runTurns(h, 36);
    expect(h.sent).toHaveLength(2);
  });

  it("fires on wall-clock deadline even when turn count is nowhere near the cap", async () => {
    process.env.LITTLE_CODER_MAX_TURNS = "80";
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(Date.now() + 60_000); // 60s out, under the 10-min threshold
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "before_agent_start", {}, h.ctx);
    await runTurns(h, 1);
    expect(h.sent).toHaveLength(1);
  });

  it("does not fire on wall-clock grounds when plenty of time remains", async () => {
    process.env.LITTLE_CODER_MAX_TURNS = "80";
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(Date.now() + 3_600_000); // 1h out
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "before_agent_start", {}, h.ctx);
    await runTurns(h, 10);
    expect(h.sent).toEqual([]);
  });

  it("fires just inside the 10-minute wall-clock threshold, not just outside it", async () => {
    process.env.LITTLE_CODER_MAX_TURNS = "80";
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(Date.now() + 9 * 60_000 + 59_000); // 9m59s out
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "before_agent_start", {}, h.ctx);
    await runTurns(h, 1);
    expect(h.sent).toHaveLength(1);
  });

  it("does not fire just outside the 10-minute wall-clock threshold", async () => {
    process.env.LITTLE_CODER_MAX_TURNS = "80";
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(Date.now() + 10 * 60_000 + 1_000); // 10m01s out
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "before_agent_start", {}, h.ctx);
    await runTurns(h, 1);
    expect(h.sent).toEqual([]);
  });

  it("does not double-fire when both the turn-count and wall-clock triggers are true on the same turn", async () => {
    process.env.LITTLE_CODER_MAX_TURNS = "40";
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(Date.now() + 60_000);
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "before_agent_start", {}, h.ctx);
    await runTurns(h, 36); // also the turn-count trigger point
    expect(h.sent).toHaveLength(1);
    await runTurns(h, 10);
    expect(h.sent).toHaveLength(1);
  });
});
