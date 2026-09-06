import { describe, it, expect, afterEach } from "vitest";
import setupExtension from "./index.ts";

interface Handler {
  (event: any, ctx: any): Promise<unknown> | unknown;
}

function makeHarness() {
  const notifies: string[] = [];
  const aborted: boolean[] = [];
  const handlers: Record<string, Handler[]> = {};
  const pi = {
    handlers,
    on(name: string, h: Handler) {
      (handlers[name] ??= []).push(h);
    },
  };
  const ctx = {
    abort() {
      aborted.push(true);
    },
    ui: {
      notify(m: string) {
        notifies.push(m);
      },
    },
  };
  return { pi, ctx, notifies, aborted };
}

async function fire(pi: any, name: string, event: any, ctx: any) {
  for (const h of pi.handlers[name] ?? []) await h(event, ctx);
}

describe("turn-cap", () => {
  afterEach(() => {
    delete process.env.LITTLE_CODER_MAX_TURNS;
  });

  it("does nothing when no cap is configured", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "before_agent_start", {}, h.ctx);
    for (let i = 0; i < 10; i++) await fire(h.pi, "turn_start", {}, h.ctx);
    expect(h.aborted).toEqual([]);
  });

  it("aborts exactly on the turn after the cap (env var)", async () => {
    process.env.LITTLE_CODER_MAX_TURNS = "3";
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "before_agent_start", {}, h.ctx);
    await fire(h.pi, "turn_start", {}, h.ctx); // 1
    await fire(h.pi, "turn_start", {}, h.ctx); // 2
    expect(h.aborted).toEqual([]);
    await fire(h.pi, "turn_start", {}, h.ctx); // 3
    expect(h.aborted).toEqual([]);
    await fire(h.pi, "turn_start", {}, h.ctx); // 4 > cap
    expect(h.aborted).toEqual([true]);
    expect(h.notifies[0]).toMatch(/turn limit \(3\)/);
  });

  it("systemPromptOptions.littleCoder.maxTurns overrides the env var", async () => {
    process.env.LITTLE_CODER_MAX_TURNS = "40";
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "before_agent_start", { systemPromptOptions: { littleCoder: { maxTurns: 2 } } }, h.ctx);
    await fire(h.pi, "turn_start", {}, h.ctx); // 1
    await fire(h.pi, "turn_start", {}, h.ctx); // 2
    expect(h.aborted).toEqual([]);
    await fire(h.pi, "turn_start", {}, h.ctx); // 3 > cap of 2
    expect(h.aborted).toEqual([true]);
  });

  it("resets the counter on a new agent run", async () => {
    process.env.LITTLE_CODER_MAX_TURNS = "2";
    const h = makeHarness();
    setupExtension(h.pi as any);
    await fire(h.pi, "before_agent_start", {}, h.ctx);
    await fire(h.pi, "turn_start", {}, h.ctx);
    await fire(h.pi, "turn_start", {}, h.ctx);
    await fire(h.pi, "turn_start", {}, h.ctx); // aborts
    expect(h.aborted).toEqual([true]);

    await fire(h.pi, "before_agent_start", {}, h.ctx); // new run
    await fire(h.pi, "turn_start", {}, h.ctx);
    await fire(h.pi, "turn_start", {}, h.ctx);
    expect(h.aborted).toEqual([true]); // still just the one from before
  });
});
