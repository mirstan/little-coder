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
// `thinking`, when present, is pushed as a thinking content block ahead of
// the text entry — used to simulate an aborted turn whose only content was
// reasoning tokens (invisible to contentShape's text/toolCall scan).
function assistantTurn(opts: {
  text?: string;
  toolCalls?: number;
  stopReason?: string;
  thinking?: string;
}) {
  const content: any[] = [];
  if (opts.thinking !== undefined) content.push({ type: "thinking", thinking: opts.thinking });
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

// A turn whose only tool calls are ShellSend (writing to an already-running
// job's stdin) — used to check that ShellSend counts as evidence-of-work for
// Trigger B even though it's excluded from SHELL_TOOLS.
function shellSendTurn(texts: string[]) {
  const content = texts.map((text) => ({
    type: "toolCall",
    name: "ShellSend",
    arguments: { text },
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

async function settle(h: ReturnType<typeof makeHarness>) {
  await fire(h.pi, "agent_settled", {}, h.ctx);
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

    it("a cp to a non-scratch path counts as compliant (6.2a: cp/mv/install blind spot)", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await startArmableRun(h);
      for (let i = 0; i < 5; i++) await turn(h, shellTurn(["ls -la"])); // turns 1-5
      await turn(h, shellTurn(["ls -la"])); // turn 6 — arms
      await turn(h, shellTurn(["ls -la"])); // turn 7 — 1 non-compliant turn
      await turn(h, shellTurn(["cp deliverable.txt /app/out/"])); // turn 8 — compliant, resets counter
      await turn(h, shellTurn(["ls -la"])); // turn 9 — only 1 non-compliant turn since reset
      expect(h.sent).toHaveLength(0);
    });

    it("a ShellSend-only turn whose text writes a non-scratch path counts as compliant (6.2b: ShellSend blind spot)", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await startArmableRun(h);
      for (let i = 0; i < 5; i++) await turn(h, shellTurn(["ls -la"])); // turns 1-5
      await turn(h, shellTurn(["ls -la"])); // turn 6 — arms
      await turn(h, shellTurn(["ls -la"])); // turn 7 — 1 non-compliant turn
      await turn(h, shellSendTurn(["cp deliverable.txt /app/out/"])); // turn 8 — ShellSend-only, compliant
      await turn(h, shellTurn(["ls -la"])); // turn 9 — only 1 non-compliant turn since reset
      expect(h.sent).toHaveLength(0);
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

    it("fires one turn below the cap", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await startArmableRun(h); // maxTurns=10
      for (let i = 0; i < 5; i++) await turn(h, shellTurn(["ls -la"])); // turns 1-5
      await turn(h, shellTurn(["ls -la"])); // turn 6 — arms
      await turn(h, shellTurn(["echo done > /app/result.txt"])); // turn 7 — compliant, resets counter
      await turn(h, shellTurn(["ls -la"])); // turn 8 — 1st non-compliant turn since reset
      await turn(h, shellTurn(["ls -la"])); // turn 9 — 2nd non-compliant turn; 9 >= 10 is false
      expect(h.sent).toHaveLength(1);
    });

    it("is suppressed exactly at the cap (a queued nudge turn-cap would abort before delivery)", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await startArmableRun(h); // maxTurns=10
      for (let i = 0; i < 5; i++) await turn(h, shellTurn(["ls -la"])); // turns 1-5
      await turn(h, shellTurn(["ls -la"])); // turn 6 — arms
      await turn(h, shellTurn(["echo done > /app/result.txt"])); // turn 7 — compliant, resets counter
      await turn(h, shellTurn(["echo done2 > /app/result2.txt"])); // turn 8 — compliant again, shifts by one turn
      await turn(h, shellTurn(["ls -la"])); // turn 9 — 1st non-compliant turn since reset
      await turn(h, shellTurn(["ls -la"])); // turn 10 — 2nd non-compliant turn; 10 >= 10 is true
      expect(h.sent).toHaveLength(0);
    });

    it("does not burn the one-shot latch when suppressed by the cap — a later run in the same session still fires", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await startArmableRun(h); // maxTurns=10
      for (let i = 0; i < 5; i++) await turn(h, shellTurn(["ls -la"])); // turns 1-5
      await turn(h, shellTurn(["ls -la"])); // turn 6 — arms
      await turn(h, shellTurn(["echo done > /app/result.txt"])); // turn 7 — compliant, resets counter
      await turn(h, shellTurn(["echo done2 > /app/result2.txt"])); // turn 8 — compliant again, shifts by one turn
      await turn(h, shellTurn(["ls -la"])); // turn 9 — 1st non-compliant turn since reset
      await turn(h, shellTurn(["ls -la"])); // turn 10 — 2nd non-compliant turn; suppressed by the cap
      expect(h.sent).toHaveLength(0);

      // A new run within the SAME session (before_agent_start again, no
      // session_start), with a normal (non-capped-out) turn budget. If the
      // suppression above had wrongly consumed the one-shot latch, nothing
      // below would fire.
      await startRun(h, 10);
      for (let i = 0; i < 5; i++) await turn(h, shellTurn(["ls -la"])); // turns 1-5
      await turn(h, shellTurn(["ls -la"])); // turn 6 — arms
      await turn(h, shellTurn(["ls -la"])); // turn 7 — 1st non-compliant turn
      await turn(h, shellTurn(["ls -la"])); // turn 8 — 2nd non-compliant turn; fires
      expect(h.sent).toHaveLength(1);
    });

    it("no cap means no suppression (capForRun === 0 must not be treated as an exceeded cap)", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      // Deadline set before the run starts (before_agent_start resolves it
      // from the env var). 8 minutes is within finalize-warn's own 10-minute
      // wall-clock window, but well under Trigger A's 20-minute floor, so
      // Trigger A cannot interfere.
      setDeadlineMinutesFromNow(8);
      await fire(h.pi, "session_start", {}, h.ctx);
      await startRun(h); // no maxTurns -> capForRun === 0
      await turn(h, shellTurn(["ls -la"])); // turn 1 — arms via the wall-clock trigger
      await turn(h, shellTurn(["ls -la"])); // turn 2 — 1st non-compliant turn
      await turn(h, shellTurn(["ls -la"])); // turn 3 — 2nd non-compliant turn; fires
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

  describe("Trigger C — dead run: errored or empty final message on agent_settled", () => {
    it("fires on stopReason error", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await newSession(h);
      await turn(h, assistantTurn({ text: "oops", stopReason: "error" }));
      await settle(h);
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].options).toEqual({ deliverAs: "steer" });
      expect(h.sent[0].text).toMatch(/previous turn ended with an error or an empty response/i);
      expect(h.sent[0].text).toMatch(/task is not complete/i);
      expect(h.notifies.some((n) => /harness intervention:/i.test(n))).toBe(true);
      // No deadline/cap configured in this test -> recovery-framed, not urgency-framed.
      expect(h.sent[0].text).toMatch(/ample time/i);
      expect(h.sent[0].text).not.toMatch(/very little time left/i);
    });

    it("fires on empty message (no text, no tool calls)", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await newSession(h);
      await turn(h, assistantTurn({}));
      await settle(h);
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].text).toMatch(/previous turn ended with an error or an empty response/i);
    });

    it("fires regardless of budget — no deadline known, still fires", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await newSession(h); // no deadline set at all
      await turn(h, assistantTurn({ stopReason: "error" }));
      await settle(h);
      expect(h.sent).toHaveLength(1);
    });

    it("clean settle does not fire", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await newSession(h);
      await turn(h, assistantTurn({ text: "All done.", toolCalls: 1, stopReason: "stop" }));
      await settle(h);
      expect(h.sent).toHaveLength(0);
    });

    it("does not fire for non-terminal_bench sessions", async () => {
      process.env.LITTLE_CODER_BENCHMARK = "gaia";
      const h = makeHarness();
      setupExtension(h.pi as any);
      await newSession(h);
      await turn(h, assistantTurn({ stopReason: "error" }));
      await settle(h);
      expect(h.sent).toHaveLength(0);
    });

    it("honors the fire cap of 2 — a 3rd qualifying settled run does not fire", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await newSession(h);
      await turn(h, assistantTurn({ stopReason: "error" }));
      await settle(h);
      await startRun(h); // a new run — the fire cap is session-scoped, not run-scoped
      await turn(h, assistantTurn({ stopReason: "error" }));
      await settle(h);
      expect(h.sent).toHaveLength(2);
      await startRun(h);
      await turn(h, assistantTurn({ stopReason: "error" }));
      await settle(h);
      expect(h.sent).toHaveLength(2);
    });

    it("resets on session_start, allowing fresh fires for a new task", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await newSession(h);
      await turn(h, assistantTurn({ stopReason: "error" }));
      await settle(h);
      await startRun(h);
      await turn(h, assistantTurn({ stopReason: "error" }));
      await settle(h);
      expect(h.sent).toHaveLength(2);

      await newSession(h); // session_start resets the counter
      await turn(h, assistantTurn({ stopReason: "error" }));
      await settle(h);
      expect(h.sent).toHaveLength(3);
    });

    it("does not burn the fire cap or notify when sendUserMessage throws", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await newSession(h);

      h.state.sendThrows = true;
      await turn(h, assistantTurn({ stopReason: "error" }));
      await settle(h);
      expect(h.sent).toEqual([]);
      expect(h.notifies.some((n) => /harness intervention:/i.test(n))).toBe(false);

      h.state.sendThrows = false;
      await startRun(h); // start a new run
      await turn(h, assistantTurn({ stopReason: "error" }));
      await settle(h);
      expect(h.sent).toHaveLength(1);
    });

    it("does not fire on an aborted run whose final message is thinking-only (thinking-budget abort)", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await newSession(h);
      await turn(h, assistantTurn({ thinking: "let me think about this...", stopReason: "aborted" }));
      await settle(h);
      expect(h.sent).toHaveLength(0);
      // The abort did not burn a fire: a genuinely dead run later still fires.
      await startRun(h);
      await turn(h, assistantTurn({ stopReason: "error" }));
      await settle(h);
      expect(h.sent).toHaveLength(1);
    });

    it("uses the urgency finalize message when the run is also near the wall-clock deadline", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      // Inside finalize-warn's 10-minute window, so turn_start arms Trigger B's latch.
      setDeadlineMinutesFromNow(5);
      await newSession(h);
      await turn(h, assistantTurn({ stopReason: "error" }));
      await settle(h);
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].text).toMatch(/very little time left/i);
      expect(h.sent[0].text).not.toMatch(/ample time/i);
    });

    it("uses the urgency finalize message when the run is inside the turn-cap warn window", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await fire(h.pi, "session_start", {}, h.ctx);
      await startRun(h, 10); // finalize-warn's turn window opens at turn 6 (WARN_REMAINING=5)
      for (let i = 0; i < 6; i++) await turn(h, shellTurn(["ls -la"])); // turns 1-6, arms Trigger B's latch at turn 6
      await turn(h, assistantTurn({ stopReason: "error" })); // turn 7 — still below the cap itself
      await settle(h);
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].text).toMatch(/very little time left/i);
    });

    it("is suppressed when the run ended at its turn-cap (would grant a capped-out run a fresh turn budget)", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await fire(h.pi, "session_start", {}, h.ctx);
      await startRun(h, 3);
      await turn(h, shellTurn(["ls -la"])); // turn 1
      await turn(h, shellTurn(["ls -la"])); // turn 2
      await turn(h, assistantTurn({ stopReason: "error" })); // turn 3 == capForRun
      await settle(h);
      expect(h.sent).toHaveLength(0);
    });

    it("still fires with headroom below the turn-cap", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await fire(h.pi, "session_start", {}, h.ctx);
      await startRun(h, 10);
      await turn(h, assistantTurn({ stopReason: "error" })); // turn 1 of 10
      await settle(h);
      expect(h.sent).toHaveLength(1);
    });

    it("Trigger A/B/C fire caps are independent of each other", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await fire(h.pi, "session_start", {}, h.ctx);
      await startRun(h, 10);

      // Exhaust Trigger A's session cap (2 fires).
      await turn(h, assistantTurn({ text: "One." })); // turn 1
      await turn(h, assistantTurn({ text: "Two." })); // turn 2
      expect(h.sent).toHaveLength(2);

      // Trigger C fires on a settled run with an errored final message,
      // unaffected by Trigger A's now-exhausted cap.
      await turn(h, assistantTurn({ stopReason: "error" })); // turn 3
      await settle(h);
      expect(h.sent).toHaveLength(3);

      // A fresh run, still within the same session — Trigger C fires again.
      await startRun(h, 10);
      await turn(h, assistantTurn({ stopReason: "error" })); // turn 1 of the new run
      await settle(h);
      expect(h.sent).toHaveLength(4);

      // Trigger C's own session-scoped cap is now exhausted; a 3rd settled
      // error run does not fire.
      await startRun(h, 10);
      await turn(h, assistantTurn({ stopReason: "error" })); // turn 1 of yet another run
      await settle(h);
      expect(h.sent).toHaveLength(4);

      // Trigger A's cap is still exhausted, unaffected by Trigger C's fires.
      await turn(h, assistantTurn({ text: "Three." })); // turn 2 of this run
      expect(h.sent).toHaveLength(4);

      // Drive turns up to Trigger B's arm point (turn 6 for capForRun=10)
      // with no-op shell turns, then two non-compliant turns to fire B —
      // proving B's arming/latch was untouched by A's and C's exhausted caps.
      await turn(h, shellTurn(["ls -la"])); // turn 3
      await turn(h, shellTurn(["ls -la"])); // turn 4
      await turn(h, shellTurn(["ls -la"])); // turn 5
      await turn(h, shellTurn(["ls -la"])); // turn 6 — arms
      await turn(h, shellTurn(["ls -la"])); // turn 7 — 1st non-compliant turn
      await turn(h, shellTurn(["ls -la"])); // turn 8 — 2nd non-compliant turn; fires B
      expect(h.sent).toHaveLength(5);
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
