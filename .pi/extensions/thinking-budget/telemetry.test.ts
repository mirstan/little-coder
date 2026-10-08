import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import setupExtension from "./index.ts";

interface Handler {
  (event: any, ctx: any): Promise<unknown> | unknown;
}

function makeHarness(initialLevel = "high") {
  const calls: string[] = [];
  const entries: any[] = [];
  let level = initialLevel;
  const handlers: Record<string, Handler[]> = {};
  const ctx = {
    abort() {
      calls.push("abort");
    },
    ui: {
      notify(_m: string) {
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
      for (const h of handlers["input"] ?? []) void h({ type: "input", text: m, source: "extension" }, ctx);
      calls.push("send");
    },
    appendEntry(customType: string, data: any) {
      entries.push({ customType, ...data });
      calls.push(`telemetry:${data.kind}`);
    },
  };
  return { pi, ctx, calls, entries };
}

async function fire(pi: any, name: string, event: any, ctx: any) {
  for (const h of pi.handlers[name] ?? []) await h(event, ctx);
}

function thinkingDelta(s: string) {
  return { assistantMessageEvent: { type: "thinking_delta", delta: s } };
}

function toolcallDelta(contentIndex: number, s: string, name = "ShellSession") {
  return {
    assistantMessageEvent: {
      type: "toolcall_delta",
      contentIndex,
      delta: s,
      partial: { content: { [contentIndex]: { type: "toolCall", name } } },
    },
  };
}

async function startRun(h: ReturnType<typeof makeHarness>) {
  await fire(h.pi, "session_start", {}, h.ctx);
  await fire(h.pi, "agent_start", {}, h.ctx);
  await fire(h.pi, "before_agent_start", { systemPromptOptions: {} }, h.ctx);
  await fire(h.pi, "turn_start", {}, h.ctx);
}

const ENV = [
  "LITTLE_CODER_THINKING_BUDGET",
  "LITTLE_CODER_TOOLCALL_MAX_CHARS",
  "LITTLE_CODER_THINKING_GUARD_HARD_CAP_MS",
  "LITTLE_CODER_DEADLINE_EPOCH_MS",
  "LITTLE_CODER_THINKING_ADAPT_MIN_TURNS",
  "LITTLE_CODER_MAX_TURNS",
  "LITTLE_CODER_TELEMETRY",
];

describe("thinking-budget telemetry", () => {
  let saved: Record<string, string | undefined>;
  beforeEach(() => {
    saved = Object.fromEntries(ENV.map((k) => [k, process.env[k]]));
    for (const k of ENV) delete process.env[k];
    process.env.LITTLE_CODER_TELEMETRY = "1";
    vi.useFakeTimers();
    vi.setSystemTime(0);
  });
  afterEach(() => {
    vi.useRealTimers();
    for (const k of ENV) {
      if (saved[k] === undefined) delete process.env[k];
      else process.env[k] = saved[k];
    }
  });

  it("records a token-budget abort before the abort itself, and nothing under the kill switch", async () => {
    process.env.LITTLE_CODER_THINKING_BUDGET = "10";
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(h.pi, "message_update", thinkingDelta("x".repeat(1000)), h.ctx);
    expect(h.calls).toEqual(["telemetry:guard_abort", "set:off", "send", "notify", "abort"]);
    expect(h.entries).toEqual([
      { customType: "lc-telemetry", v: 1, kind: "guard_abort", trigger: "thinking_budget", thinkingChars: 1000, budgetTokens: 10 },
    ]);

    delete process.env.LITTLE_CODER_TELEMETRY;
    const quiet = makeHarness();
    setupExtension(quiet.pi as any);
    await startRun(quiet);
    await fire(quiet.pi, "message_update", thinkingDelta("x".repeat(1000)), quiet.ctx);
    expect(quiet.calls).toEqual(["set:off", "send", "notify", "abort"]);
    expect(quiet.entries).toEqual([]);
  });

  it("records a runaway tool-call abort with the call's size", async () => {
    process.env.LITTLE_CODER_TOOLCALL_MAX_CHARS = "100";
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);
    for (const chunk of [50, 50, 1]) {
      await fire(h.pi, "message_update", toolcallDelta(0, "A".repeat(chunk)), h.ctx);
    }
    expect(h.calls).toEqual(["telemetry:guard_abort", "send", "notify", "abort"]);
    expect(h.entries).toEqual([
      { customType: "lc-telemetry", v: 1, kind: "guard_abort", trigger: "toolcall_cap", tool: "ShellSession", argChars: 101, capChars: 100 },
    ]);
  });

  it("records a wall-clock abort with generation time and the window", async () => {
    process.env.LITTLE_CODER_THINKING_GUARD_HARD_CAP_MS = "1000";
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(60 * 60 * 1000);
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);
    vi.advanceTimersByTime(1500);
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);
    expect(h.calls).toEqual(["telemetry:guard_abort", "set:off", "send", "notify", "abort"]);
    expect(h.entries).toEqual([
      { customType: "lc-telemetry", v: 1, kind: "guard_abort", trigger: "wall_clock", generationMs: 1500, guardMs: 1000 },
    ]);
  });

  it("records the finalize-window stand-down once", async () => {
    process.env.LITTLE_CODER_THINKING_ADAPT_MIN_TURNS = String(1e9);
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(5 * 60 * 1000);
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);
    vi.advanceTimersByTime(10);
    await fire(h.pi, "turn_start", {}, h.ctx);
    for (const s of ["hm", "more", "still more"]) await fire(h.pi, "message_update", thinkingDelta(s), h.ctx);
    expect(h.entries).toEqual([
      { customType: "lc-telemetry", v: 1, kind: "guard_stand_down", trigger: "thinking_budget", reason: "finalize_window" },
    ]);
    expect(h.calls).not.toContain("abort");
  });

  it("records the futility stand-down once after a forced-off restart", async () => {
    process.env.LITTLE_CODER_THINKING_BUDGET = "10";
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(60 * 60 * 1000);
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(h.pi, "message_update", thinkingDelta("x".repeat(1000)), h.ctx);
    await fire(h.pi, "agent_start", {}, h.ctx);
    await fire(h.pi, "turn_start", {}, h.ctx);
    await fire(h.pi, "message_update", thinkingDelta("y".repeat(1000)), h.ctx);
    await fire(h.pi, "message_update", thinkingDelta("z".repeat(1000)), h.ctx);
    expect(h.entries.map((e) => [e.kind, e.trigger, e.reason])).toEqual([
      ["guard_abort", "thinking_budget", undefined],
      ["guard_stand_down", "thinking_budget", "futility"],
    ]);
  });

  it("records the circuit-breaker stand-down once", async () => {
    process.env.LITTLE_CODER_THINKING_GUARD_HARD_CAP_MS = "1000";
    process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(60 * 60 * 1000);
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);
    const trip = async () => {
      await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);
      vi.advanceTimersByTime(1500);
      await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);
    };
    const retry = async () => {
      await fire(h.pi, "agent_start", {}, h.ctx);
      await fire(h.pi, "turn_start", {}, h.ctx);
    };
    await trip();
    await retry();
    await trip();
    await retry();
    await trip();
    await fire(h.pi, "message_update", thinkingDelta("x"), h.ctx);
    expect(h.entries.map((e) => [e.kind, e.trigger, e.reason])).toEqual([
      ["guard_abort", "wall_clock", undefined],
      ["guard_abort", "wall_clock", undefined],
      ["guard_stand_down", "wall_clock", "breaker"],
    ]);
  });
});
