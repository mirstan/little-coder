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

// A turn with plain text and/or tool calls (generic tool, not a shell tool —
// used for Trigger A's shape checks where the specific tool doesn't matter).
function assistantTurn(opts: { text?: string; toolCalls?: number; stopReason?: string }) {
  const content: any[] = [];
  if (opts.text !== undefined) content.push({ type: "text", text: opts.text });
  for (let i = 0; i < (opts.toolCalls ?? 0); i++) {
    content.push({ type: "toolCall", name: "websearch", arguments: {} });
  }
  return { message: { content, stopReason: opts.stopReason } };
}

// A turn that ran zero or more ShellSession commands — used for Trigger B's
// write-classification checks. Deliberately has no text and no non-shell
// tool call, so it never matches Trigger A's shape (a ShellSession call
// means toolCallCount > 0, so branch 1 never applies; stopReason is left
// undefined, so branch 2 never applies either).
function shellTurn(commands: string[]) {
  const content = commands.map((command) => ({
    type: "toolCall",
    name: "ShellSession",
    arguments: { command },
  }));
  return { message: { content, stopReason: undefined } };
}

async function startRun(h: ReturnType<typeof makeHarness>, maxTurns?: number) {
  await fire(
    h.pi,
    "before_agent_start",
    { systemPromptOptions: maxTurns ? { littleCoder: { maxTurns } } : {} },
    h.ctx,
  );
}

async function newSession(h: ReturnType<typeof makeHarness>, maxTurns?: number) {
  await fire(h.pi, "session_start", {}, h.ctx);
  await startRun(h, maxTurns);
}

async function turn(h: ReturnType<typeof makeHarness>, event: any) {
  await fire(h.pi, "turn_start", {}, h.ctx);
  await fire(h.pi, "turn_end", event, h.ctx);
}

const REAL_NOW = Date.now;
let fakeNow = 1_700_000_000_000;

function setDeadlineMinutesFromNow(minutes: number) {
  process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(fakeNow + minutes * 60 * 1000);
}

describe("tb-finalize-guard", () => {
  beforeEach(() => {
    process.env.LITTLE_CODER_BENCHMARK = "terminal_bench";
    fakeNow = 1_700_000_000_000;
    Date.now = () => fakeNow;
  });
  afterEach(() => {
    delete process.env.LITTLE_CODER_BENCHMARK;
    delete process.env.LITTLE_CODER_MAX_TURNS;
    delete process.env.LITTLE_CODER_DEADLINE_EPOCH_MS;
    Date.now = REAL_NOW;
  });

  describe("outside terminal_bench", () => {
    it("does nothing at all — not even the instrumentation log", async () => {
      process.env.LITTLE_CODER_BENCHMARK = "gaia";
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      await turn(h, assistantTurn({ text: "Let me check that." }));
      expect(h.calls).toEqual([]);
    });
  });

  describe("Trigger A — early voluntary quit", () => {
    it("does nothing on a turn that made a tool call", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      await turn(h, assistantTurn({ text: "Let me check that.", toolCalls: 1 }));
      expect(h.sent).toHaveLength(0);
    });

    it("does nothing on an aborted turn", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      await turn(h, assistantTurn({ text: "Let me check that.", stopReason: "aborted" }));
      expect(h.sent).toHaveLength(0);
    });

    it("does nothing when no deadline is known", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await newSession(h);
      await turn(h, assistantTurn({ text: "Let me check that." }));
      expect(h.sent).toHaveLength(0);
    });

    it("fires on non-empty text with zero tool calls, with plenty of budget left", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      await turn(h, assistantTurn({ text: "I think that's everything." }));
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].options).toEqual({ deliverAs: "steer" });
      expect(h.sent[0].text).toMatch(/stopped without calling a tool/i);
      expect(h.notifies.some((n) => /harness intervention:/i.test(n))).toBe(true);
    });

    it("does not fire on empty content with stopReason 'error' (provider/transport failure, unsteerable)", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      await turn(h, assistantTurn({ stopReason: "error" }));
      expect(h.sent).toHaveLength(0);
    });

    it("does not fire on empty content with a non-error stopReason", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      await turn(h, assistantTurn({}));
      expect(h.sent).toHaveLength(0);
    });

    it("an error-stopReason turn does not consume a Trigger A fire", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      await turn(h, assistantTurn({ stopReason: "error" })); // would-be fire #1, but unsteerable
      await turn(h, assistantTurn({ stopReason: "error" })); // would-be fire #2, but unsteerable
      expect(h.sent).toHaveLength(0);
      // Both of MAX_TRIGGER_A_FIRES's fires are still available for a real quit.
      await turn(h, assistantTurn({ text: "One." }));
      await turn(h, assistantTurn({ text: "Two." }));
      expect(h.sent).toHaveLength(2);
    });

    it("is suppressed once remaining budget drops below the 20-minute floor", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(15); // > finalize-warn's 10min, < this guard's 20min floor
      await newSession(h);
      await turn(h, assistantTurn({ text: "I think that's everything." }));
      expect(h.sent).toHaveLength(0);
    });

    it("fires right at the 20-minute floor", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(20);
      await newSession(h);
      await turn(h, assistantTurn({ text: "I think that's everything." }));
      expect(h.sent).toHaveLength(1);
    });

    it("is suppressed once the run is already at turn-cap", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await fire(h.pi, "session_start", {}, h.ctx);
      await startRun(h, 3);
      await fire(h.pi, "turn_start", {}, h.ctx);
      await fire(h.pi, "turn_start", {}, h.ctx);
      await fire(h.pi, "turn_start", {}, h.ctx); // turnsThisRun === capForRun (3)
      await fire(h.pi, "turn_end", assistantTurn({ text: "I think that's everything." }), h.ctx);
      expect(h.sent).toHaveLength(0);
    });

    it("still fires with headroom below turn-cap", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await fire(h.pi, "session_start", {}, h.ctx);
      await startRun(h, 10);
      await fire(h.pi, "turn_start", {}, h.ctx);
      await fire(h.pi, "turn_end", assistantTurn({ text: "I think that's everything." }), h.ctx);
      expect(h.sent).toHaveLength(1);
    });

    it("fires at most twice per session", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      await turn(h, assistantTurn({ text: "One." }));
      await turn(h, assistantTurn({ text: "Two." }));
      await turn(h, assistantTurn({ text: "Three." }));
      expect(h.sent).toHaveLength(2);
    });

    it("latch survives before_agent_start (no re-arming across runs in one session)", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      await turn(h, assistantTurn({ text: "One." })); // fire #1 this session
      expect(h.sent).toHaveLength(1);

      // A new run within the SAME session (before_agent_start again, no session_start).
      setDeadlineMinutesFromNow(30);
      await startRun(h);
      await turn(h, assistantTurn({ text: "Two." })); // fire #2 this session
      await turn(h, assistantTurn({ text: "Three." })); // would be #3 — blocked
      expect(h.sent).toHaveLength(2);
    });

    it("resets on session_start, allowing fresh fires for a new task", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      await turn(h, assistantTurn({ text: "One." }));
      await turn(h, assistantTurn({ text: "Two." }));
      expect(h.sent).toHaveLength(2); // session limit reached

      setDeadlineMinutesFromNow(30);
      await newSession(h); // session_start resets the latch
      await turn(h, assistantTurn({ text: "Three." }));
      expect(h.sent).toHaveLength(3);
    });

    it("does not burn the fire count or notify when sendUserMessage throws", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);

      h.state.sendThrows = true;
      await turn(h, assistantTurn({ text: "One." }));
      expect(h.sent).toEqual([]);
      expect(h.notifies.some((n) => /harness intervention:/i.test(n))).toBe(false);

      h.state.sendThrows = false;
      await turn(h, assistantTurn({ text: "Two." }));
      expect(h.sent).toHaveLength(1);
    });
  });

  describe("Trigger B — post-finalize-warn non-compliance", () => {
    // capForRun=10, WARN_REMAINING=5 -> finalize-warn's own turn trigger
    // fires when turnsThisRun === capForRun - WARN_REMAINING + 1 === 6.
    async function startArmableRun(h: ReturnType<typeof makeHarness>) {
      await fire(h.pi, "session_start", {}, h.ctx);
      await startRun(h, 10);
    }

    it("does not arm (and never fires) before finalize-warn's own condition would be met", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await startArmableRun(h);
      // Turns 1-5: well before the arm point (turn 6). Feed it many
      // consecutive no-write turns — should never fire since it's not armed.
      for (let i = 0; i < 5; i++) {
        await turn(h, shellTurn(["ls -la"]));
      }
      expect(h.sent).toHaveLength(0);
    });

    it("fires after 2 consecutive turns with no non-scratch write once armed", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await startArmableRun(h);
      for (let i = 0; i < 5; i++) await turn(h, shellTurn(["ls -la"])); // turns 1-5, not armed
      await turn(h, shellTurn(["ls -la"])); // turn 6 — arms here; this turn itself is not judged
      expect(h.sent).toHaveLength(0);
      await turn(h, shellTurn(["ls -la"])); // turn 7 — 1st non-compliant turn post-arm
      expect(h.sent).toHaveLength(0);
      await turn(h, shellTurn(["cat /app/output.txt"])); // turn 8 — 2nd non-compliant turn post-arm
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].options).toEqual({ deliverAs: "steer" });
      expect(h.sent[0].text).toMatch(/still have not written your answer/i);
    });

    it("does not fire if a non-scratch write occurs within the 2-turn window", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await startArmableRun(h);
      for (let i = 0; i < 5; i++) await turn(h, shellTurn(["ls -la"])); // turns 1-5
      await turn(h, shellTurn(["ls -la"])); // turn 6 — arms
      await turn(h, shellTurn(["ls -la"])); // turn 7 — 1 non-compliant turn
      await turn(h, shellTurn(["echo done > /app/result.txt"])); // turn 8 — compliant, resets counter
      await turn(h, shellTurn(["ls -la"])); // turn 9 — only 1 non-compliant turn since reset
      expect(h.sent).toHaveLength(0);
    });

    it("a write-shaped command targeting only /tmp does not clear the counter", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await startArmableRun(h);
      for (let i = 0; i < 5; i++) await turn(h, shellTurn(["ls -la"])); // turns 1-5
      await turn(h, shellTurn(["ls -la"])); // turn 6 — arms
      await turn(h, shellTurn(["echo scratch > /tmp/scratch.txt"])); // turn 7 — tmp-only write, still counts as non-compliant
      await turn(h, shellTurn(["echo scratch2 > /tmp/scratch2.txt"])); // turn 8 — 2nd non-compliant turn
      expect(h.sent).toHaveLength(1);
    });

    it("is single-shot — does not fire again after already firing", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await startArmableRun(h);
      for (let i = 0; i < 5; i++) await turn(h, shellTurn(["ls -la"])); // turns 1-5
      await turn(h, shellTurn(["ls -la"])); // turn 6 — arms
      await turn(h, shellTurn(["ls -la"])); // turn 7
      await turn(h, shellTurn(["ls -la"])); // turn 8 — fires
      expect(h.sent).toHaveLength(1);
      await turn(h, shellTurn(["ls -la"])); // turn 9
      await turn(h, shellTurn(["ls -la"])); // turn 10
      expect(h.sent).toHaveLength(1); // still just the one
    });

    it("does not burn the latch or notify when sendUserMessage throws", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await startArmableRun(h);
      for (let i = 0; i < 5; i++) await turn(h, shellTurn(["ls -la"])); // turns 1-5
      await turn(h, shellTurn(["ls -la"])); // turn 6 — arms
      await turn(h, shellTurn(["ls -la"])); // turn 7

      h.state.sendThrows = true;
      await turn(h, shellTurn(["ls -la"])); // turn 8 — would fire, but send throws
      expect(h.sent).toEqual([]);
      expect(h.notifies.some((n) => /harness intervention:/i.test(n))).toBe(false);

      h.state.sendThrows = false;
      await turn(h, shellTurn(["ls -la"])); // turn 9 — still non-compliant, retries
      expect(h.sent).toHaveLength(1);
    });
  });

  describe("precedence between the two triggers", () => {
    it("does not double-send when a turn matches Trigger A while Trigger B is armed", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await fire(h.pi, "session_start", {}, h.ctx);
      await startRun(h, 10);
      for (let i = 0; i < 6; i++) await turn(h, shellTurn(["ls -la"])); // turns 1-6, arms at 6
      // Turn 7: a toolless text quit — matches Trigger A's shape.
      await turn(h, assistantTurn({ text: "I believe I'm done." }));
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].text).toMatch(/stopped without calling a tool/i);
    });
  });

  describe("shared instrumentation logging", () => {
    it("logs stopReason and content shape on every terminal_bench turn_end regardless of trigger state", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await newSession(h); // no deadline set — neither trigger can fire
      await turn(h, assistantTurn({ text: "Just some text.", toolCalls: 1 }));
      const diag = h.notifies.filter((n) => n.includes("tb-finalize-guard: turn_end"));
      expect(diag).toHaveLength(1);
      expect(diag[0]).toMatch(/stopReason=undefined/);
      expect(diag[0]).toMatch(/hasText=true/);
      expect(diag[0]).toMatch(/hasToolCalls=true/);
      expect(diag[0]).toMatch(/isEmpty=false/);
    });

    it("logs isEmpty=true for a genuinely empty turn", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await newSession(h);
      await turn(h, assistantTurn({}));
      const diag = h.notifies.filter((n) => n.includes("tb-finalize-guard: turn_end"));
      expect(diag).toHaveLength(1);
      expect(diag[0]).toMatch(/isEmpty=true/);
    });

    it("does not log for non-terminal_bench sessions", async () => {
      process.env.LITTLE_CODER_BENCHMARK = "gaia";
      const h = makeHarness();
      setupExtension(h.pi as any);
      await newSession(h);
      await turn(h, assistantTurn({ text: "Just some text.", toolCalls: 1 }));
      expect(h.notifies.filter((n) => n.includes("tb-finalize-guard: turn_end"))).toHaveLength(0);
    });
  });
});
