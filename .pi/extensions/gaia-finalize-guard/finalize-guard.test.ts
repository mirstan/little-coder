import { describe, it, expect, beforeEach, afterEach } from "vitest";
import setupExtension from "./index.ts";

interface Handler {
  (event: any, ctx: any): Promise<unknown> | unknown;
}

function makeHarness() {
  const calls: string[] = [];
  const sent: { text: string; options: any }[] = [];
  const notifies: string[] = [];
  const handlers: Record<string, Handler[]> = {};
  const state = { sendThrows: false };
  const pi = {
    handlers,
    state,
    on(name: string, h: Handler) {
      (handlers[name] ??= []).push(h);
    },
    sendUserMessage(text: string, options: any) {
      if (state.sendThrows) throw new Error("SDK does not support sendUserMessage");
      sent.push({ text, options });
      calls.push("send");
    },
  };
  const ctx = {
    ui: {
      notify(m: string) {
        notifies.push(m);
        calls.push("notify");
      },
    },
  };
  return { pi, ctx, calls, sent, notifies, state };
}

async function fire(pi: any, name: string, event: any, ctx: any) {
  for (const h of pi.handlers[name] ?? []) await h(event, ctx);
}

function assistantTurn(opts: { text?: string; toolCalls?: number; stopReason?: string }) {
  const content: any[] = [];
  if (opts.text !== undefined) content.push({ type: "text", text: opts.text });
  for (let i = 0; i < (opts.toolCalls ?? 0); i++) {
    content.push({ type: "toolCall", name: "websearch", arguments: {} });
  }
  return { message: { content, stopReason: opts.stopReason } };
}

async function startRun(h: ReturnType<typeof makeHarness>, maxTurns?: number) {
  await fire(h.pi, "session_start", {}, h.ctx);
  await fire(
    h.pi,
    "before_agent_start",
    { systemPromptOptions: maxTurns ? { littleCoder: { maxTurns } } : {} },
    h.ctx,
  );
}

describe("gaia-finalize-guard", () => {
  beforeEach(() => {
    process.env.LITTLE_CODER_BENCHMARK = "gaia";
  });
  afterEach(() => {
    delete process.env.LITTLE_CODER_BENCHMARK;
    delete process.env.LITTLE_CODER_MAX_TURNS;
  });

  it("does nothing outside GAIA", async () => {
    process.env.LITTLE_CODER_BENCHMARK = "terminal_bench";
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(h.pi, "turn_end", assistantTurn({ text: "Let me check that." }), h.ctx);
    expect(h.calls).toEqual([]);
  });

  it("does nothing when the turn made a tool call", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(
      h.pi,
      "turn_end",
      assistantTurn({ text: "Let me check that.", toolCalls: 1 }),
      h.ctx,
    );
    expect(h.calls).toEqual([]);
  });

  it("does nothing when the reply ends with an Answer: line", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(h.pi, "turn_end", assistantTurn({ text: "Reasoning...\nAnswer: 42" }), h.ctx);
    expect(h.calls).toEqual([]);
  });

  it("accepts a Final Answer: line too (matches gaia_scorer.py's regex)", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(h.pi, "turn_end", assistantTurn({ text: "Final Answer: backtick" }), h.ctx);
    expect(h.calls).toEqual([]);
  });

  it("accepts an indented Answer: line (mirrors the scorer's line.strip())", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(h.pi, "turn_end", assistantTurn({ text: "  Answer: 42" }), h.ctx);
    expect(h.calls).toEqual([]);
  });

  it("nudges on a bare 'Answer:' with no value (scorer would score it wrong anyway)", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(h.pi, "turn_end", assistantTurn({ text: "Reasoning...\nAnswer:" }), h.ctx);
    expect(h.sent).toHaveLength(1);
  });

  it("nudges when the value is on a different physical line than Answer: (scorer requires same line)", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(h.pi, "turn_end", assistantTurn({ text: "Reasoning...\nAnswer:\n42" }), h.ctx);
    expect(h.sent).toHaveLength(1);
  });

  it("does nothing on a genuinely empty response (quality-monitor's territory)", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(h.pi, "turn_end", assistantTurn({ text: "" }), h.ctx);
    await fire(h.pi, "turn_end", assistantTurn({ text: "   " }), h.ctx);
    expect(h.calls).toEqual([]);
  });

  it("skips aborted and errored turns", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(
      h.pi,
      "turn_end",
      assistantTurn({ text: "Let me check that.", stopReason: "aborted" }),
      h.ctx,
    );
    await fire(
      h.pi,
      "turn_end",
      assistantTurn({ text: "Let me check that.", stopReason: "error" }),
      h.ctx,
    );
    expect(h.calls).toEqual([]);
  });

  it("nudges via steer + notifies when the turn trails off with no tool call and no Answer: line", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(
      h.pi,
      "turn_end",
      assistantTurn({ text: "Let me fetch it to get the zip codes." }),
      h.ctx,
    );
    expect(h.calls).toEqual(["send", "notify"]);
    expect(h.notifies[0]).toMatch(/harness intervention:.*without a tool call or an Answer/i);
    expect(h.sent[0].options).toEqual({ deliverAs: "steer" });
    expect(h.sent[0].text).toMatch(/didn't call a tool or end with/i);
  });

  it("fires at most once per run", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(h.pi, "turn_end", assistantTurn({ text: "Let me look into this." }), h.ctx);
    await fire(h.pi, "turn_end", assistantTurn({ text: "Let me look further." }), h.ctx);
    expect(h.sent).toHaveLength(1);
  });

  it("resets per session, so a new task can nudge again", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(h.pi, "turn_end", assistantTurn({ text: "Let me look into this." }), h.ctx);
    expect(h.sent).toHaveLength(1);

    await startRun(h);
    await fire(h.pi, "turn_end", assistantTurn({ text: "Let me look into that too." }), h.ctx);
    expect(h.sent).toHaveLength(2);
  });

  it("does not nudge once the run is already at turn-cap (the nudge would be aborted before the model sees it)", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h, 3);
    await fire(h.pi, "turn_start", {}, h.ctx);
    await fire(h.pi, "turn_start", {}, h.ctx);
    await fire(h.pi, "turn_start", {}, h.ctx); // turnsThisRun === capForRun (3)
    await fire(h.pi, "turn_end", assistantTurn({ text: "Let me look into this." }), h.ctx);
    expect(h.calls).toEqual([]);
  });

  it("still nudges with headroom below turn-cap", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h, 10);
    await fire(h.pi, "turn_start", {}, h.ctx);
    await fire(h.pi, "turn_end", assistantTurn({ text: "Let me look into this." }), h.ctx);
    expect(h.sent).toHaveLength(1);
  });

  it("nudges when an earlier line merely looks like Answer: but real prose follows it", async () => {
    // A mid-reasoning line that happens to match the convention (a draft, a
    // placeholder) must not suppress the nudge when the turn actually trails
    // off afterward — checking "any line matches" instead of "the last
    // non-blank line matches" would wrongly treat this as finalized.
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(
      h.pi,
      "turn_end",
      assistantTurn({
        text:
          "My plan:\nAnswer: <put final value here>\n" +
          "But I still need to verify with a search.",
      }),
      h.ctx,
    );
    expect(h.sent).toHaveLength(1);
  });

  it("does not nudge when the Answer: line is genuinely last, even with earlier reasoning", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);
    await fire(
      h.pi,
      "turn_end",
      assistantTurn({ text: "Some reasoning.\nMore reasoning.\nAnswer: 42" }),
      h.ctx,
    );
    expect(h.calls).toEqual([]);
  });

  it("does not claim the one-shot slot or notify when sendUserMessage throws, so a later turn can still try", async () => {
    const h = makeHarness();
    setupExtension(h.pi as any);
    await startRun(h);

    h.state.sendThrows = true;
    await fire(h.pi, "turn_end", assistantTurn({ text: "Let me look into this." }), h.ctx);
    // No notification shown for a nudge that was never actually delivered,
    // and the one-shot slot wasn't burned on a failed attempt.
    expect(h.notifies).toEqual([]);
    expect(h.sent).toEqual([]);

    h.state.sendThrows = false;
    await fire(h.pi, "turn_end", assistantTurn({ text: "Let me look further." }), h.ctx);
    expect(h.sent).toHaveLength(1);
  });
});
