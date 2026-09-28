import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { execSync } from "node:child_process";
import { existsSync, unlinkSync } from "node:fs";
import setupExtension, {
  buildTriggerAMessage,
  buildTriggerCRecoveryMessage,
  buildTriggerDMessage,
  buildTriggerEMessage,
  parseByteLimit,
  parseDeliverablePath,
  looksLikeTestInvocation,
} from "./index.ts";
import { resolveFinalizeMessage } from "../_shared/finalize-message.ts";
import { INITIAL_SNAPSHOT_APP_DIR } from "../_shared/snapshot-paths.ts";
import { TB_PROXY_PREFIX } from "../_shared/tb-proxy.ts";

interface Handler {
  (event: any, ctx: any): Promise<unknown> | unknown;
}

function makeHarness() {
  const calls: string[] = [];
  const sent: { text: string; options: any }[] = [];
  const notifies: string[] = [];
  // Every title `ctx.ui.input` was called with, in call order — lets a test
  // inspect the actual `__LC_TB_SHELL__` payload (and so the actual command
  // string) a proxy call carried, not just that a call happened.
  const inputs: string[] = [];
  const handlers: Record<string, Handler[]> = {};
  // Queued `ui.input` replies for Trigger E's proxy calls; an empty queue is
  // a proxy that gave nothing usable.
  const state = { sendThrows: false, tbProxyResponses: [] as (string | null)[] };
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
      async input(title: string, _initial?: string) {
        inputs.push(title);
        calls.push("proxy-input");
        return state.tbProxyResponses.length > 0 ? (state.tbProxyResponses.shift() ?? null) : null;
      },
    },
  };
  return { pi, ctx, calls, sent, notifies, inputs, state };
}

/** A `wc -c`-shaped proxy response with a well-formed exit-0 footer. */
function wcResult(bytes: number, path: string, cwd = "/app"): string {
  return `${bytes} ${path}\n[exit=0 cwd=${cwd} timed_out=false]`;
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

// Trigger E parses its limit/path from the task prompt at before_agent_start
// — every other trigger's `startRun`/`newSession` fire with no `prompt` at
// all, which is exactly the "no confident match" shape those triggers stay
// silent under.
async function newSessionWithPrompt(h: ReturnType<typeof makeHarness>, prompt: string) {
  await fire(h.pi, "session_start", {}, h.ctx);
  await fire(h.pi, "before_agent_start", { systemPromptOptions: {}, prompt }, h.ctx);
}

// turn() synthesizes one tool_result per toolCall block in the message's
// array order before firing turn_end. A test that wants to exercise
// genuinely out-of-array-order sibling completion should fire tool_result
// directly itself before calling turn().
async function turn(h: ReturnType<typeof makeHarness>, event: any) {
  await fire(h.pi, "turn_start", {}, h.ctx);
  const content = event?.message?.content;
  if (Array.isArray(content)) {
    for (const block of content) {
      if (block?.type !== "toolCall") continue;
      await fire(
        h.pi,
        "tool_result",
        { toolName: block.name, input: block.arguments ?? {} },
        h.ctx,
      );
    }
  }
  await fire(h.pi, "turn_end", event, h.ctx);
}

// Named `settle` for historical/test-readability reasons (Trigger C used to
// fire on `agent_settled`) -- it now fires the `agent_end` event the
// extension actually listens on, since agent_settled proved undeliverable
// against the real Python harness (see index.ts's Trigger C comment).
async function settle(h: ReturnType<typeof makeHarness>) {
  await fire(h.pi, "agent_end", {}, h.ctx);
}

const REAL_NOW = Date.now;
let fakeNow = 1_700_000_000_000;

function setDeadlineMinutesFromNow(minutes: number) {
  process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(fakeNow + minutes * 60 * 1000);
}

// A trial budget of `totalMinutes` with `fractionUsed` of it already gone as
// of fakeNow. Both ends are resolved at before_agent_start, so this has to
// be called before the run starts; advancing fakeNow afterwards is what
// moves the trial through its milestones.
function setBudget(totalMinutes: number, fractionUsed: number) {
  const totalMs = totalMinutes * 60 * 1000;
  const start = fakeNow - Math.round(totalMs * fractionUsed);
  process.env.LITTLE_CODER_BUDGET_START_EPOCH_MS = String(start);
  process.env.LITTLE_CODER_DEADLINE_EPOCH_MS = String(start + totalMs);
}

function advanceMinutes(minutes: number) {
  fakeNow += minutes * 60 * 1000;
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
    // Left unset by default everywhere else in this suite, so every other
    // test also covers the no-baseline path a TB1.0 trial actually runs.
    delete process.env.LITTLE_CODER_INITIAL_SNAPSHOT;
    delete process.env.LITTLE_CODER_BUDGET_START_EPOCH_MS;
    delete process.env.LITTLE_CODER_TB_MODE;
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
      // Asks for more than existence: a present, well-formed file can still
      // have wrong content, or correct content plus extra scaffolding a
      // strict grader flags. Literal substrings rather than a `.*`-stitched
      // prose regex -- the clause's exact wording is pinned by the builder
      // unit tests below, and this only needs to prove the delivered message
      // is the demanding one.
      expect(h.sent[0].text).toContain("re-verify adversarially");
      expect(h.sent[0].text).toContain("could actually produce that result");
      // Cleanup is scoped to self-created files, never pre-existing content --
      // an unscoped "remove what's extra" could convert a passing trial into
      // a failing one.
      expect(h.sent[0].text).toMatch(/remove only ones you created yourself/i);
      expect(h.sent[0].text).toMatch(/never anything that was already there/i);
      expect(h.sent[0].text).toMatch(/most literal reading/i);
      // The nudge fires on a toolless text turn; it must not be answerable
      // with another one, or the second (and last) fire burns on the same
      // pattern with the model never touching the container.
      expect(h.sent[0].text).toMatch(/use\s+ShellSession/i);
      expect(h.notifies.some((n) => /harness intervention:/i.test(n))).toBe(true);
      // stopReason:"error" turns are excluded before this function ever
      // fires (see the test below), so the log text claiming this can be
      // an error turn was always stale.
      expect(h.notifies.join("\n")).not.toMatch(/errored with empty content/i);
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

    it("fires at most three times per session", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      await turn(h, assistantTurn({ text: "One." }));
      await turn(h, assistantTurn({ text: "Two." }));
      await turn(h, assistantTurn({ text: "Three." }));
      await turn(h, assistantTurn({ text: "Four." }));
      expect(h.sent).toHaveLength(3);
    });

    it("a real trial's shape (two early pauses, then a real final declaration) now gets a third fire where it previously would have been suppressed", async () => {
      // Two early pauses then a real final declaration -- the third fire the
      // old cap of 2 suppressed.
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(180);
      await newSession(h);
      await turn(h, assistantTurn({ text: "I think I'm done for now." })); // fire #1 (early pause)
      await turn(h, assistantTurn({ text: "Let me pause here too." })); // fire #2 (early pause)
      await turn(h, assistantTurn({ text: "That's everything." })); // the real final declaration
      expect(h.sent).toHaveLength(3);
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
      await turn(h, assistantTurn({ text: "Three." })); // fire #3 this session
      await turn(h, assistantTurn({ text: "Four." })); // would be #4 — blocked
      expect(h.sent).toHaveLength(3);
    });

    it("resets on session_start, allowing fresh fires for a new task", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      await turn(h, assistantTurn({ text: "One." }));
      await turn(h, assistantTurn({ text: "Two." }));
      await turn(h, assistantTurn({ text: "Three." }));
      expect(h.sent).toHaveLength(3); // session limit reached

      setDeadlineMinutesFromNow(30);
      await newSession(h); // session_start resets the latch
      await turn(h, assistantTurn({ text: "Four." }));
      expect(h.sent).toHaveLength(4);
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

    it("sends exactly the builder's text, including the baseline pointer", async () => {
      process.env.LITTLE_CODER_INITIAL_SNAPSHOT = "succeeded";
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      await turn(h, assistantTurn({ text: "I think that's everything." }));
      expect(h.sent).toHaveLength(1);
      // Exact equality, so the builder unit tests below are pinning the text
      // that is genuinely delivered rather than a parallel copy of it.
      expect(h.sent[0].text).toBe(buildTriggerAMessage(30, "succeeded"));
    });

    it("says nothing about the start-of-trial copy when the adapter staged none", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      await turn(h, assistantTurn({ text: "I think that's everything." }));
      // What a TB1.0 trial gets: same benchmark name, no container-side copy.
      expect(h.sent[0].text).toBe(buildTriggerAMessage(30, undefined));
      expect(h.sent[0].text).not.toContain(INITIAL_SNAPSHOT_APP_DIR);
    });

    it("strengthens the message when a write happened with no test-invocation since a real test seen earlier this session", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      // Primes testInvocationEverSeen so the categorical "since your last
      // test run" claim is warranted -- that claim is gated on having ever
      // seen a real test this session (see the "honest fallback" tests
      // below for the zero-ever-seen case).
      await turn(h, shellTurn(["pytest tests/"]));
      await turn(h, shellTurn(["echo done > /app/result.txt"])); // non-scratch write, no test since
      await turn(h, assistantTurn({ text: "I think that's everything." }));
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].text).toContain(
        "You edited a file since your last test run without re-running your tests",
      );
    });

    it("uses the honest fallback (not the categorical claim) when no test has ever run this session", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      // No test invocation anywhere this session — testInvocationEverSeen
      // stays false, so the strong "since your last test run" claim (which
      // presupposes a last test run that never happened) must not fire.
      await turn(h, shellTurn(["echo done > /app/result.txt"])); // non-scratch write, zero tests ever
      await turn(h, assistantTurn({ text: "I think that's everything." }));
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].text).not.toContain(
        "You edited a file since your last test run without re-running your tests",
      );
      expect(h.sent[0].text).toContain("You have not run any tests yet this session");
    });

    it("keeps testInvocationEverSeen across runs in the same session (session_start-scoped, not before_agent_start)", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      await turn(h, shellTurn(["pytest tests/"])); // a real test, run 1
      await startRun(h); // new run, SAME session — before_agent_start must not reset it
      await turn(h, shellTurn(["echo done > /app/result.txt"])); // write, no test this run
      await turn(h, assistantTurn({ text: "I think that's everything." }));
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].text).toContain(
        "You edited a file since your last test run without re-running your tests",
      );

      await newSession(h); // session_start DOES reset it
      await turn(h, shellTurn(["echo done > /app/result.txt"])); // write, zero tests this session
      await turn(h, assistantTurn({ text: "I think that's everything." }));
      expect(h.sent).toHaveLength(2);
      expect(h.sent[1].text).toContain("You have not run any tests yet this session");
    });

    it("does not add the retest clause when a test-invocation ran after the write", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      await turn(h, shellTurn(["echo done > /app/result.txt"])); // non-scratch write
      await turn(h, shellTurn(["pytest tests/"])); // test-invocation clears the flag
      await turn(h, assistantTurn({ text: "I think that's everything." }));
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].text).not.toContain("without re-running your tests");
    });

    it("clears the flag when the write and the test run land in the same turn", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      await turn(h, shellTurn(["echo done > /app/result.txt", "pytest tests/"]));
      await turn(h, assistantTurn({ text: "I think that's everything." }));
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].text).not.toContain("without re-running your tests");
    });

    it("does not clear the flag when a heredoc body merely mentions a test framework", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      // Primes testInvocationEverSeen so this test isolates the heredoc
      // invariant (dirty stays true) from the separate, session-scoped
      // categorical-vs-fallback clause choice covered elsewhere.
      await turn(h, shellTurn(["pytest tests/"]));
      // A heredoc write whose BODY text contains "pytest" — no test invocation
      // actually ran, so the flag must stay dirty.
      await turn(
        h,
        shellTurn(["cat > /app/solution.py <<'EOF'\nimport pytest\n\ndef test_x(): ...\nEOF"]),
      );
      await turn(h, assistantTurn({ text: "I think that's everything." }));
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].text).toContain(
        "You edited a file since your last test run without re-running your tests",
      );
    });

    it("still dirties when a chained write after the SAME command's own test invocation is a genuine edit, not test-output-shaped (iteration 3)", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      // The carve-out exempts only test-output-shaped writes; an in-place
      // edit after the same command's test must still dirty.
      await turn(h, shellTurn(["python -m pytest && sed -i 's/x/y/' /app/main.py"]));
      await turn(h, assistantTurn({ text: "I think that's everything." }));
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].text).toContain(
        "You edited a file since your last test run without re-running your tests",
      );
    });

    it("still dirties when a chained write after the SAME command's own test invocation is a copy onto the deliverable (iteration 3)", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      // Same root cause as the `sed -i` case above, exercised via `cp`
      // instead of an in-place edit: a copy onto a deliverable path is not
      // test-output-shaped either, so it must dirty even though it follows
      // the same command's own test invocation.
      await turn(h, shellTurn(["make test; cp /tmp/fixed.py /app/main.py"]));
      await turn(h, assistantTurn({ text: "I think that's everything." }));
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].text).toContain(
        "You edited a file since your last test run without re-running your tests",
      );
    });

    it("still dirties on a write in a DIFFERENT, later command even after the same turn's test invocation", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      // Guards against over-suppression: the same-command carve-out above
      // must not silently swallow a write that is its own separate command.
      await turn(h, shellTurn(["pytest tests/", "echo done > /app/DONE"]));
      await turn(h, assistantTurn({ text: "I think that's everything." }));
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].text).toContain(
        "You edited a file since your last test run without re-running your tests",
      );
    });

    it("does not re-dirty on a pipe to the test's own output (pytest 2>&1 | tee log)", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      // The pipe's second segment is a non-scratch write, and being last in
      // execution order, a strict last-wins rule wrongly re-dirtied the flag
      // even though this is the test's own output being logged, not a
      // deliverable edit.
      await turn(h, shellTurn(["pytest 2>&1 | tee /app/test.log"]));
      await turn(h, assistantTurn({ text: "I think that's everything." }));
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].text).not.toContain(
        "You edited a file since your last test run without re-running your tests",
      );
      expect(h.sent[0].text).not.toContain("You have not run any tests yet this session");
    });

    it("does not re-dirty on a test chained with a redirect of its own output (pytest && echo done > DONE)", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      await turn(h, shellTurn(["pytest && echo done > /app/DONE"]));
      await turn(h, assistantTurn({ text: "I think that's everything." }));
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].text).not.toContain(
        "You edited a file since your last test run without re-running your tests",
      );
      expect(h.sent[0].text).not.toContain("You have not run any tests yet this session");
    });

    it("a plain redirect that merely writes the word 'pytest' as file content is not a test invocation", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      // `echo pytest > file` must not be read as running pytest — it writes
      // the word "pytest" as file content. No real test has ever run this
      // session, so the flag stays dirty AND the message must use the
      // honest fallback, not the categorical claim.
      await turn(h, shellTurn(["echo pytest > /app/result.txt"]));
      await turn(h, assistantTurn({ text: "I think that's everything." }));
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].text).not.toContain(
        "You edited a file since your last test run without re-running your tests",
      );
      expect(h.sent[0].text).toContain("You have not run any tests yet this session");
    });

    it("still clears the flag for a real, unquoted, unchained test invocation", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      await turn(h, shellTurn(["echo done > /app/result.txt"])); // non-scratch write
      await turn(h, shellTurn(["python -m pytest"])); // the working case — must not regress
      await turn(h, assistantTurn({ text: "I think that's everything." }));
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].text).not.toContain("without re-running your tests");
    });

    it("judges two SEPARATE same-turn tool calls by real completion order, not by their position in the turn's tool-call array", async () => {
      // see the tool_result handler for why ordering comes from there.
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);

      await fire(h.pi, "turn_start", {}, h.ctx);
      // Real completion order: test result arrives first...
      await fire(h.pi, "tool_result", { toolName: "ShellSession", input: { command: "pytest tests/" } }, h.ctx);
      // ...then the write's result arrives second (finished LAST, after the
      // test already ran) -- even though the message below lists it FIRST.
      await fire(
        h.pi,
        "tool_result",
        { toolName: "ShellSession", input: { command: "echo done > /app/result.txt" } },
        h.ctx,
      );
      await fire(
        h.pi,
        "turn_end",
        {
          message: {
            content: [
              { type: "toolCall", name: "ShellSession", arguments: { command: "echo done > /app/result.txt" } },
              { type: "toolCall", name: "ShellSession", arguments: { command: "pytest tests/" } },
            ],
            stopReason: undefined,
          },
        },
        h.ctx,
      );

      await turn(h, assistantTurn({ text: "I think that's everything." }));
      expect(h.sent).toHaveLength(1);
      // Real completion order says the write came after the test -> dirty.
      // The array's own listed order would have said the opposite (test
      // last, so clean) if array position were still driving this.
      expect(h.sent[0].text).toContain(
        "You edited a file since your last test run without re-running your tests",
      );
    });
  });

  describe("looksLikeTestInvocation", () => {
    it("matches common test-runner invocations across languages", () => {
      const positives = [
        "pytest",
        "python -m pytest tests/",
        "python3 -m pytest -k foo",
        "python -m unittest discover",
        "go test ./...",
        "cargo test",
        "npm test",
        "npm run test",
        "yarn test",
        "make test",
        "make check",
        "rspec spec/",
        "rake test",
        "mvn test",
        "gradle test",
      ];
      for (const cmd of positives) expect(looksLikeTestInvocation(cmd)).toBe(true);
    });

    it("does not match plausible near-misses that merely contain the word 'test'", () => {
      const negatives = [
        "cat test_notes.txt",
        "ls test_data/",
        "echo 'contest results' > /app/out.txt",
        "vim test_plan.md",
        "grep -r test src/",
        // A quoted/embedded runner name in an unrelated command must not
        // match — "make test" here sits inside the commit message, not on
        // the command line.
        "git commit -m 'make test pass'",
        // A plain (non-heredoc) redirect that writes the word "pytest" as
        // file content, unquoted, is not a test invocation — the runner name
        // must be the command being run, not an argument to an unrelated
        // command like `echo`.
        "echo pytest > /app/result.txt",
      ];
      for (const cmd of negatives) expect(looksLikeTestInvocation(cmd)).toBe(false);
    });
  });

  describe("buildTriggerAMessage", () => {
    it("demands a falsifiable check by a different method", () => {
      const msg = buildTriggerAMessage(30, undefined);
      expect(msg).toContain("prove your answer WRONG");
      expect(msg).toContain("could actually produce that result");
      expect(msg).toContain("Recompute the result by a different method");
    });

    it("forbids verifying against the model's own artifacts", () => {
      // The overfull-hbox failure: a task file diffed against a backup the
      // model made after corrupting it, reported as "changed positions: 0".
      const msg = buildTriggerAMessage(30, undefined);
      expect(msg).toContain("another file you created this session");
      expect(msg).toContain("both can be wrong the same way");
    });

    it("keeps the cleanup-scoping and literal-reading clauses", () => {
      const msg = buildTriggerAMessage(30, undefined);
      expect(msg).toContain("remove only ones you created yourself");
      expect(msg).toContain("never anything that was already there");
      expect(msg).toContain("most literal reading of the task text");
      expect(msg).toContain("ShellSession");
    });

    it("mentions no snapshot path at all without a baseline", () => {
      const msg = buildTriggerAMessage(30, undefined);
      expect(msg).not.toContain(INITIAL_SNAPSHOT_APP_DIR);
      expect(msg).not.toContain("/tmp/.lc-initial");
    });

    it("points at the start-of-trial copy when one exists", () => {
      const msg = buildTriggerAMessage(30, "succeeded");
      expect(msg).toContain(`${INITIAL_SNAPSHOT_APP_DIR}/`);
      expect(msg).toContain(`${INITIAL_SNAPSHOT_APP_DIR}/somefile mirrors /app/somefile`);
      expect(msg).toContain("predates every change you made");
    });

    it("scopes the baseline diff to suspect inputs instead of directing a blanket one", () => {
      // The pointer must stay reactive, matching the adapter's own framing
      // of this copy: a standing "diff your inputs against it" costs a turn
      // on every trial, and some tasks' only input is a huge binary.
      const msg = buildTriggerAMessage(30, "succeeded");
      expect(msg).toContain("could have been changed this session");
      expect(msg).toContain("diff just those files");
      expect(msg).not.toContain("confirm those inputs are still intact");
    });

    it("adds the truncation caveat only for a partial copy", () => {
      const partial = buildTriggerAMessage(30, "partial");
      expect(partial).toContain("file-count cap");
      expect(partial).toContain("may still have existed at the start");
      expect(buildTriggerAMessage(30, "succeeded")).not.toContain("file-count cap");
    });

    it("warns every baseline, including a fully-succeeded copy, may omit large or deep files", () => {
      // Matches _initial_snapshot_advertisement's unconditional caveat: the
      // per-file size cap and -maxdepth apply regardless of outcome, so
      // "succeeded" alone doesn't mean the copy is exhaustive.
      expect(buildTriggerAMessage(30, "succeeded")).toContain(
        "may not contain very large (>10MB) or deeply nested files",
      );
      expect(buildTriggerAMessage(30, "partial")).toContain(
        "may not contain very large (>10MB) or deeply nested files",
      );
    });

    it("reports the remaining minutes it was given", () => {
      expect(buildTriggerAMessage(7, undefined)).toContain("roughly 7 minutes");
    });

    it("warns against restoring the baseline over a completed solution", () => {
      // Matches _initial_snapshot_advertisement's own restraint: pointing a
      // model at a "reference copy" without this warning risks it reading
      // that as license to overwrite its own finished work with the
      // pristine original.
      expect(buildTriggerAMessage(30, "succeeded")).toContain("never over your own");
      expect(buildTriggerAMessage(30, undefined)).not.toContain("never over your own");
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

  describe("Trigger C — dead run: errored or empty final message on agent_end", () => {
    it("does not fire when a mid-run retry recovers before agent_end (the whole point of snapshotting from turn_end)", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await newSession(h);
      // An errored turn, papered over by pi's own internal retry, followed by
      // a healthy turn before the run actually ends -- lastTurnMessage must
      // reflect the LAST turn_end, not the errored one.
      await turn(h, assistantTurn({ text: "oops", stopReason: "error" }));
      await turn(h, assistantTurn({ text: "All done.", toolCalls: 1, stopReason: "stop" }));
      await settle(h);
      expect(h.sent).toHaveLength(0);
    });

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

    it("sends exactly the recovery builder's text on the calm branch", async () => {
      // Exact equality, not toContain: the delivered text is a fixed prefix
      // concatenated with the builder's output, so toContain alone would
      // also pass if unrelated text were appended after the builder's --
      // pin the whole thing, matching the Trigger A sibling test above.
      process.env.LITTLE_CODER_INITIAL_SNAPSHOT = "succeeded";
      const h = makeHarness();
      setupExtension(h.pi as any);
      await newSession(h);
      await turn(h, assistantTurn({ stopReason: "error" }));
      await settle(h);
      const prefix =
        "Your previous turn ended with an error or an empty response. The task is NOT " +
        "complete. ";
      expect(h.sent[0].text).toBe(prefix + buildTriggerCRecoveryMessage("succeeded"));
    });

    it("says nothing about the start-of-trial copy on the calm branch when none was staged", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      await newSession(h);
      await turn(h, assistantTurn({ stopReason: "error" }));
      await settle(h);
      expect(h.sent[0].text).toContain(buildTriggerCRecoveryMessage(undefined));
      expect(h.sent[0].text).not.toContain("/tmp/.lc-initial");
    });

    it("leaves the near-deadline branch as the unmodified finalize message", async () => {
      // That message tells the model to stop verifying and save -- the
      // opposite regime from the recovery branch, so a staged baseline must
      // not pull any verification demand into it.
      process.env.LITTLE_CODER_INITIAL_SNAPSHOT = "succeeded";
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(5);
      await newSession(h);
      await turn(h, assistantTurn({ stopReason: "error" }));
      await settle(h);
      expect(h.sent[0].text).toContain(resolveFinalizeMessage("terminal_bench"));
      expect(h.sent[0].text).not.toContain("re-verify adversarially");
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

      // Exhaust Trigger A's session cap (3 fires).
      await turn(h, assistantTurn({ text: "One." })); // turn 1
      await turn(h, assistantTurn({ text: "Two." })); // turn 2
      await turn(h, assistantTurn({ text: "Two-point-five." })); // turn 2.5
      expect(h.sent).toHaveLength(3);

      // Trigger C fires on a settled run with an errored final message,
      // unaffected by Trigger A's now-exhausted cap.
      await turn(h, assistantTurn({ stopReason: "error" })); // turn 3
      await settle(h);
      expect(h.sent).toHaveLength(4);

      // A fresh run, still within the same session — Trigger C fires again.
      await startRun(h, 10);
      await turn(h, assistantTurn({ stopReason: "error" })); // turn 1 of the new run
      await settle(h);
      expect(h.sent).toHaveLength(5);

      // Trigger C's own session-scoped cap is now exhausted; a 3rd settled
      // error run does not fire.
      await startRun(h, 10);
      await turn(h, assistantTurn({ stopReason: "error" })); // turn 1 of yet another run
      await settle(h);
      expect(h.sent).toHaveLength(5);

      // Trigger A's cap is still exhausted, unaffected by Trigger C's fires.
      await turn(h, assistantTurn({ text: "Three." })); // turn 2 of this run
      expect(h.sent).toHaveLength(5);

      // Drive turns up to Trigger B's arm point (turn 6 for capForRun=10)
      // with no-op shell turns, then two non-compliant turns to fire B —
      // proving B's arming/latch was untouched by A's and C's exhausted caps.
      await turn(h, shellTurn(["ls -la"])); // turn 3
      await turn(h, shellTurn(["ls -la"])); // turn 4
      await turn(h, shellTurn(["ls -la"])); // turn 5
      await turn(h, shellTurn(["ls -la"])); // turn 6 — arms
      await turn(h, shellTurn(["ls -la"])); // turn 7 — 1st non-compliant turn
      await turn(h, shellTurn(["ls -la"])); // turn 8 — 2nd non-compliant turn; fires B
      expect(h.sent).toHaveLength(6);
    });
  });

  describe("Trigger D — budget-progress checkpoint", () => {
    it("never fires without a published trial-start instant", async () => {
      // A deadline alone cannot express a fraction: an adapter that
      // publishes only one end (TB before this change) leaves this trigger
      // off entirely, same as the ones that publish neither (GAIA, aider,
      // interactive pi).
      const h = makeHarness();
      setupExtension(h.pi as any);
      setDeadlineMinutesFromNow(30);
      await newSession(h);
      for (let i = 0; i < 4; i++) await turn(h, shellTurn(["ls -la"]));
      expect(h.sent).toHaveLength(0);
    });

    it("fires at the half-way milestone when nothing has been written outside /tmp", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setBudget(120, 0.5);
      await newSession(h);
      await turn(h, shellTurn(["ls -la"]));
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].options).toEqual({ deliverAs: "steer" });
      expect(h.sent[0].text).toBe(buildTriggerDMessage(0.5, 60, false));
      expect(h.notifies.some((n) => /harness intervention:/i.test(n))).toBe(true);
    });

    it("does not repeat a milestone it has already spent", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setBudget(120, 0.5);
      await newSession(h);
      await turn(h, shellTurn(["ls -la"]));
      expect(h.sent).toHaveLength(1);
      advanceMinutes(12); // 60% spent -- still short of the next milestone
      await turn(h, shellTurn(["ls -la"]));
      await turn(h, shellTurn(["ls -la"]));
      expect(h.sent).toHaveLength(1);
    });

    it("stays silent at half-time once a deliverable has been written", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setBudget(120, 0.4);
      await newSession(h);
      await turn(h, shellTurn(["echo 42 > /app/answer.txt"]));
      advanceMinutes(12); // 50%
      await turn(h, shellTurn(["ls -la"]));
      expect(h.sent).toHaveLength(0);
    });

    it("does not nag a run that spent its first 40% reading before writing anything", async () => {
      // The false positive this gate exists to avoid: a deliberate
      // read-then-write trajectory looks identical to a stalled one until
      // it writes, so the evidence -- not the reading -- is what decides.
      const h = makeHarness();
      setupExtension(h.pi as any);
      setBudget(120, 0.05);
      await newSession(h);
      await turn(h, shellTurn(["ls -la /app"]));
      await turn(h, shellTurn(["cat /app/instructions.md"]));
      advanceMinutes(24); // 25% spent, still only reading
      await turn(h, shellTurn(["grep -rn TODO /app", "echo notes > /tmp/notes.md"]));
      advanceMinutes(18); // 40% spent
      await turn(h, shellTurn(["cat > /app/solution.py"])); // the plan, written
      expect(h.sent).toHaveLength(0);
      advanceMinutes(12); // 50% -- the milestone, with evidence on disk
      await turn(h, shellTurn(["python3 /app/solution.py"]));
      advanceMinutes(12); // 60%
      await turn(h, shellTurn(["python3 /app/solution.py"]));
      expect(h.sent).toHaveLength(0);
    });

    it("fires the milder checkpoint at the later milestone when a deliverable exists", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setBudget(120, 0.4);
      await newSession(h);
      await turn(h, shellTurn(["echo 42 > /app/answer.txt"]));
      advanceMinutes(42); // 75%
      await turn(h, shellTurn(["ls -la"]));
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].text).toBe(buildTriggerDMessage(0.75, 30, true));
    });

    it("fires the strong checkpoint at the later milestone when nothing has been written", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setBudget(120, 0.75);
      await newSession(h);
      await turn(h, shellTurn(["ls -la"]));
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].text).toBe(buildTriggerDMessage(0.75, 30, false));
    });

    it("sends one message, not a backlog, when several milestones come due at once", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setBudget(120, 0.3);
      await newSession(h);
      await turn(h, shellTurn(["ls -la"]));
      expect(h.sent).toHaveLength(0);
      advanceMinutes(60); // 80% -- both milestones crossed inside one turn
      await turn(h, shellTurn(["ls -la"]));
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].text).toBe(buildTriggerDMessage(0.8, 24, false));
      await turn(h, shellTurn(["ls -la"]));
      expect(h.sent).toHaveLength(1);
    });

    it("stands down inside finalize-warn's wall-clock window, which owns the endgame", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setBudget(120, 11 / 12); // exactly WARN_REMAINING_MS left
      await newSession(h);
      await turn(h, shellTurn(["ls -la"]));
      expect(h.sent).toHaveLength(0);
    });

    it("still fires one minute outside that window", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setBudget(120, 109 / 120); // 11 minutes left
      await newSession(h);
      await turn(h, shellTurn(["ls -la"]));
      expect(h.sent).toHaveLength(1);
    });

    it("stands down at the turn cap without spending the milestone", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setBudget(120, 0.3);
      await fire(h.pi, "session_start", {}, h.ctx);
      await startRun(h, 3); // too small for finalize-warn's turn window to ever open
      await turn(h, shellTurn(["ls -la"])); // turn 1
      await turn(h, shellTurn(["ls -la"])); // turn 2
      advanceMinutes(60); // 80%
      await turn(h, shellTurn(["ls -la"])); // turn 3 == capForRun
      expect(h.sent).toHaveLength(0);

      // A later run in the same session has headroom again, and the
      // milestone was never marked, so it is still owed.
      await startRun(h, 10);
      await turn(h, shellTurn(["ls -la"]));
      expect(h.sent).toHaveLength(1);
    });

    it("latches nothing when the send throws, and retries on the next turn", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setBudget(120, 0.5);
      await newSession(h);

      h.state.sendThrows = true;
      await turn(h, shellTurn(["ls -la"]));
      expect(h.sent).toEqual([]);
      expect(h.notifies.some((n) => /harness intervention:/i.test(n))).toBe(false);

      h.state.sendThrows = false;
      await turn(h, shellTurn(["ls -la"]));
      expect(h.sent).toHaveLength(1);
    });

    it("keeps its milestones across a continuation and resets them for a new session", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setBudget(120, 0.5);
      await newSession(h);
      await turn(h, shellTurn(["ls -la"]));
      expect(h.sent).toHaveLength(1);

      await startRun(h); // a continuation: the trial clock did not restart
      await turn(h, shellTurn(["ls -la"]));
      expect(h.sent).toHaveLength(1);

      await newSession(h); // a new task
      await turn(h, shellTurn(["ls -la"]));
      expect(h.sent).toHaveLength(2);
    });

    it("counts a write from long before Trigger B could arm", async () => {
      // Regression pin for hoisting the per-turn write scan out of
      // maybeAdvanceTriggerB: it used to run only while Trigger B was armed,
      // which is always inside the endgame this trigger stands down for.
      const h = makeHarness();
      setupExtension(h.pi as any);
      setBudget(120, 0.3);
      await fire(h.pi, "session_start", {}, h.ctx);
      await startRun(h, 10); // Trigger B cannot arm before turn 6
      await turn(h, shellTurn(["echo 42 > /app/answer.txt"])); // turn 1
      advanceMinutes(30); // 55% -- half-way milestone, spent silently
      await turn(h, shellTurn(["ls -la"])); // turn 2
      expect(h.sent).toHaveLength(0);
      advanceMinutes(30); // 80%
      await turn(h, shellTurn(["ls -la"])); // turn 3
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].text).toBe(buildTriggerDMessage(0.8, 24, true));
    });

    it("is silent outside terminal_bench", async () => {
      process.env.LITTLE_CODER_BENCHMARK = "gaia";
      const h = makeHarness();
      setupExtension(h.pi as any);
      setBudget(120, 0.8);
      await newSession(h);
      await turn(h, shellTurn(["ls -la"]));
      expect(h.calls).toEqual([]);
    });

    it("does not demand a non-executable deliverable be run", async () => {
      // This trial's whole deliverable is a value in a text file, so
      // "run it" would be an impossible demand.
      const h = makeHarness();
      setupExtension(h.pi as any);
      setBudget(120, 0.3);
      await newSession(h);
      await turn(h, shellTurn(["echo 42 > /app/answer.txt"]));
      advanceMinutes(60); // 80%
      await turn(h, shellTurn(["cat /app/answer.txt"]));
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].text).toBe(buildTriggerDMessage(0.8, 24, true));
      expect(h.sent[0].text).toContain("otherwise compare the file or state you produced");
      expect(h.sent[0].text).not.toContain("run it to confirm it executes");
    });

    it("stays silent in a turn-cap-bound endgame after Trigger B has already fired", async () => {
      // `armed` cannot re-latch once triggerBFired is set, and
      // finalizeWarnWouldFire's turn half is edge-triggered, so every turn
      // of a turn-cap-bound warn window except the edge one looks calm to
      // both. Only the level-triggered window check suppresses here.
      const h = makeHarness();
      setupExtension(h.pi as any);
      setBudget(150, 0.4); // 90 minutes left: nowhere near the wall-clock window
      await fire(h.pi, "session_start", {}, h.ctx);

      // Run 1: drive Trigger B to fire, which permanently disarms `armed`.
      await startRun(h, 10);
      for (let i = 0; i < 5; i++) await turn(h, shellTurn(["ls -la"])); // turns 1-5
      await turn(h, shellTurn(["ls -la"])); // turn 6 — arms
      await turn(h, shellTurn(["echo 42 > /app/answer.txt"])); // turn 7 — compliant
      await turn(h, shellTurn(["ls -la"])); // turn 8
      await turn(h, shellTurn(["ls -la"])); // turn 9 — Trigger B fires
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].text).toMatch(/still have not written your answer/i);

      // Run 2: a continuation, still short of the later milestone.
      await startRun(h, 10);
      for (let i = 0; i < 5; i++) await turn(h, shellTurn(["ls -la"])); // turns 1-5
      expect(h.sent).toHaveLength(1);

      advanceMinutes(60); // 80% spent, 30 minutes left
      await turn(h, shellTurn(["ls -la"])); // turn 6 — the warn edge; calm to nobody
      await turn(h, shellTurn(["ls -la"])); // turn 7 — inside the window, past the edge
      await turn(h, shellTurn(["ls -la"])); // turn 8
      expect(h.sent).toHaveLength(1);
    });

    it("does not also fire Trigger A on the very turn it just nudged", async () => {
      // Trigger D fires at this turn's turn_start; if that same turn's own
      // response is toolless, Trigger A's turn_end would otherwise stack a
      // second, different steer right behind it -- pi delivers every queued
      // steer, it does not coalesce them.
      const h = makeHarness();
      setupExtension(h.pi as any);
      setBudget(120, 0.5); // 60 minutes left -- clears Trigger A's own gate
      await newSession(h);
      await turn(h, assistantTurn({ text: "Let me think about this." }));
      expect(h.sent).toHaveLength(1);
      expect(h.sent[0].text).toBe(buildTriggerDMessage(0.5, 60, false));
    });

    it("lets Trigger A fire normally on a later turn once D has already spoken", async () => {
      const h = makeHarness();
      setupExtension(h.pi as any);
      setBudget(120, 0.5);
      await newSession(h);
      await turn(h, assistantTurn({ text: "Let me think about this." })); // D fires
      expect(h.sent).toHaveLength(1);
      await turn(h, assistantTurn({ text: "Still thinking." })); // a later, separate turn
      expect(h.sent).toHaveLength(2);
      expect(h.sent[1].text).toBe(buildTriggerAMessage(60, undefined));
    });
  });

  describe("buildTriggerDMessage", () => {
    it("reports the share of the budget spent and the minutes left", () => {
      const msg = buildTriggerDMessage(0.5, 60, false);
      expect(msg).toContain("about 50% of this task's time budget");
      expect(msg).toContain("~60 minutes remain");
      expect(buildTriggerDMessage(0.77, 12, true)).toContain("about 77%");
    });

    it("names the missing deliverable only when there is no evidence of one", () => {
      expect(buildTriggerDMessage(0.5, 60, false)).toContain(
        "nothing has been written outside /tmp",
      );
      const mild = buildTriggerDMessage(0.75, 30, true);
      expect(mild).not.toContain("nothing has been written outside /tmp");
      expect(mild).toContain("Make sure your best current version is saved");
    });

    it("conditions its verification demand on the deliverable being executable", () => {
      // Many TB deliverables are a repaired file, a git state, or a value
      // at a path; an unconditional "run it" costs those trials the very
      // turns this nudge exists to save.
      for (const hasEvidence of [false, true]) {
        const msg = buildTriggerDMessage(0.75, 30, hasEvidence);
        expect(msg).toContain("if your deliverable is a program or script, run it");
        expect(msg).toContain("otherwise compare the file or state you produced");
        expect(msg).not.toContain("run it to confirm it executes");
        expect(msg).not.toContain("runnable version");
        expect(msg).not.toContain("runs end-to-end");
      }
    });
  });

  describe("buildTriggerCRecoveryMessage", () => {
    it("keeps the keep-working framing", () => {
      const msg = buildTriggerCRecoveryMessage(undefined);
      expect(msg).toContain("ample time");
      expect(msg).toContain("do not wrap up");
      expect(msg).toContain("verifying and testing as you normally would");
    });

    it("forbids verifying against the model's own artifacts either way", () => {
      for (const baseline of [undefined, "succeeded", "partial"] as const) {
        expect(buildTriggerCRecoveryMessage(baseline)).toContain(
          "never against another file you created this session",
        );
      }
    });

    it("offers the start-of-trial copy only on suspicion, and only when it exists", () => {
      const withCopy = buildTriggerCRecoveryMessage("succeeded");
      expect(withCopy).toContain("if you suspect a task-provided file was damaged");
      expect(withCopy).toContain(`${INITIAL_SNAPSHOT_APP_DIR}/`);
      expect(buildTriggerCRecoveryMessage(undefined)).not.toContain("/tmp/.lc-initial");
    });

    it("carries the same baseline caveats and restore restraint as Trigger A", () => {
      // A missing baseline file must not read as "never existed" here
      // either -- the same misreading Trigger A's own caveat exists to
      // prevent -- and the same restore-restraint risk applies: a model
      // "verifying... from where you left off" could misread this pointer
      // as license to overwrite its own progress with the pristine original.
      const partial = buildTriggerCRecoveryMessage("partial");
      expect(partial).toContain("may not contain very large (>10MB) or deeply nested files");
      expect(partial).toContain("file-count cap");
      expect(buildTriggerCRecoveryMessage("succeeded")).not.toContain("file-count cap");
      expect(buildTriggerCRecoveryMessage("succeeded")).toContain("never over your own");
      expect(buildTriggerCRecoveryMessage(undefined)).not.toContain("never over your own");
    });

    it("does not double-punctuate the sentence the baseline clause closes", () => {
      // Regression: the baseline clause and the fixed "Remember the task..."
      // continuation each supplied a sentence-ending period independently.
      expect(buildTriggerCRecoveryMessage("succeeded")).not.toMatch(/\.\.\s/);
      expect(buildTriggerCRecoveryMessage(undefined)).not.toMatch(/\.\.\s/);
    });
  });

  describe("Trigger E — byte-limit-aware finalize guard", () => {
    const REAL_PROMPT =
      "I have downloaded the gpt-2 weights stored as a TF .ckpt. Write me a " +
      "dependency-free C file that samples from the model with arg-max " +
      "sampling. Call your program /app/gpt2.c, I will compile with gcc -O3 " +
      "-lm. It should read the .ckpt and the .bpe file. Your c program must " +
      'be <5000 bytes. I will run it /app/a.out gpt2-124M.ckpt vocab.bpe ' +
      '"[input string here]" and you should continue the output under ' +
      "whatever GPT-2 would print for the next 20 tokens.";

    describe("limit parsing", () => {
      it("resolves the real gpt2-codegolf prompt's limit to 5000 bytes", () => {
        expect(parseByteLimit(REAL_PROMPT)).toBe(5000);
      });

      it("resolves kb/kilobytes to bytes and other phrasings", () => {
        expect(parseByteLimit("under 2KB")).toBe(2048);
        expect(parseByteLimit("must be at most 10000 bytes")).toBe(10000);
      });

      it("finds nothing in a prompt with no size language — false-negative bias", () => {
        expect(parseByteLimit("Write a function that reverses a string.")).toBeUndefined();
      });

      it("does not misread a floor ('no/not less than') as this trigger's ceiling", () => {
        expect(parseByteLimit("The file must be no less than 5000 bytes.")).toBeUndefined();
        expect(parseByteLimit("Output should be not less than 2KB.")).toBeUndefined();
        // The un-negated phrase still fires — only the floor-shaped negation
        // is excluded.
        expect(parseByteLimit("Output should be less than 2KB.")).toBe(2048);
      });

      it("does not misread a floor stated as 'must/should not BE less than' either — confirmed regression", () => {
        // Confirmed by execution: the old lookbehind only inspected the
        // single token immediately before "less" ("be", not "not"/"no"), so
        // both of these parsed as a 5000-byte CEILING despite stating a
        // floor.
        expect(parseByteLimit("The file must not be less than 5000 bytes.")).toBeUndefined();
        expect(parseByteLimit("The file should not be less than 5000 bytes.")).toBeUndefined();
      });

      it("does not silently pick the wrong constraint when the prompt states two distinct sizes", () => {
        // Confirmed by execution: first-match-with-no-ambiguity-check used to
        // return 64 here (an unrelated per-record constraint), not 5000 (the
        // deliverable's own limit).
        expect(
          parseByteLimit("each record must be 64 bytes; the program must be under 5000 bytes"),
        ).not.toBe(64);
      });

      it("does not mistake an output-size constraint for the deliverable's own limit", () => {
        // Confirmed by execution: used to return 100 (the output-size cap),
        // not the deliverable's own (unstated, here) byte limit.
        expect(
          parseByteLimit(
            "Write a program that prints at most 100 bytes of output. Call your program /app/gpt2.c.",
          ),
        ).toBeUndefined();
      });

      it("does not arm a bare 'must be N bytes' as a limit — an exact/fixed size states no bound", () => {
        // Confirmed by execution: a bare "must be" alongside no other bound
        // word used to parse as a false ceiling ({value: 16, inclusive:
        // false}), arming the guard against a file with no real size
        // constraint at all -- "must be" only introduces a size statement,
        // it doesn't say which direction it bounds.
        expect(parseByteLimit("Save your output to /app/header.bin. The header must be 16 bytes.")).toBeUndefined();
      });
    });

    describe("path parsing", () => {
      it("resolves the real gpt2-codegolf prompt's path to /app/gpt2.c", () => {
        expect(parseDeliverablePath(REAL_PROMPT)).toBe("/app/gpt2.c");
      });

      it("stays inert (never calls the proxy) when a limit exists but no path confidently resolves", async () => {
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        await newSessionWithPrompt(h, "Your output must be under 5000 bytes total.");
        await turn(h, shellTurn(["gcc -o /app/gpt2.c gpt2.c"]));
        expect(h.calls.filter((c) => c === "proxy-input")).toHaveLength(0);
      });

      it("does not silently latch an input path when the prompt also names a distinct output path", () => {
        // Confirmed by execution: first-match-with-no-disambiguation used to
        // resolve to /app/in (the "write ... to" trigger reaches the input
        // clause first), not /app/out (the actual deliverable) -- neither
        // path is more textually "the deliverable" than the other from the
        // regex's own point of view, so it must not guess.
        expect(
          parseDeliverablePath("Write your input to /app/in, save your output to /app/out."),
        ).not.toBe("/app/in");
        expect(
          parseDeliverablePath("Write your input to /app/in, save your output to /app/out."),
        ).toBeUndefined();
      });

      it("resolves a single quoted or backticked path instead of failing to match or corrupting the capture", () => {
        expect(parseDeliverablePath("Save it as `/app/out.c`.")).toBe("/app/out.c");
        expect(parseDeliverablePath('Call your program "/app/gpt2.c".')).toBe("/app/gpt2.c");
      });
    });

    describe("size-check triggering", () => {
      it("checks size exactly once when a command writes the deliverable path", async () => {
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        h.state.tbProxyResponses.push(wcResult(3000, "/app/gpt2.c"));
        await newSessionWithPrompt(h, REAL_PROMPT);
        await turn(h, shellTurn(["gcc -O3 -o /app/gpt2.c gpt2.c"]));
        expect(h.calls.filter((c) => c === "proxy-input")).toHaveLength(1);
      });

      it("does not check size when the command touches an unrelated file", async () => {
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        await newSessionWithPrompt(h, REAL_PROMPT);
        await turn(h, shellTurn(["cat /app/vocab.bpe > /tmp/scratch.txt"]));
        expect(h.calls.filter((c) => c === "proxy-input")).toHaveLength(0);
      });
    });

    describe("over-budget escalation", () => {
      async function driveOverBudget(h: ReturnType<typeof makeHarness>, checks: number) {
        await newSessionWithPrompt(h, REAL_PROMPT);
        for (let i = 0; i < checks; i++) {
          h.state.tbProxyResponses.push(wcResult(6927, "/app/gpt2.c"));
          await turn(h, shellTurn(["gcc -O3 -o /app/gpt2.c gpt2.c"]));
        }
      }

      it("does not escalate on the 1st or 2nd consecutive over-budget check", async () => {
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        setDeadlineMinutesFromNow(120);
        await driveOverBudget(h, 2);
        expect(h.calls.filter((c) => c === "send")).toHaveLength(0);
      });

      it("escalates on the 3rd consecutive over-budget check with the right byte-delta text", async () => {
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        setDeadlineMinutesFromNow(120);
        await driveOverBudget(h, 3);
        expect(h.sent).toHaveLength(1);
        expect(h.sent[0].text).toBe(buildTriggerEMessage(1927, 5000));
        expect(h.notifies.some((n) => n.startsWith("harness intervention:"))).toBe(true);
      });

      it("resets the count on a check that comes back under the limit", async () => {
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        setDeadlineMinutesFromNow(120);
        await newSessionWithPrompt(h, REAL_PROMPT);
        h.state.tbProxyResponses.push(wcResult(6927, "/app/gpt2.c"));
        await turn(h, shellTurn(["gcc -O3 -o /app/gpt2.c gpt2.c"]));
        h.state.tbProxyResponses.push(wcResult(4500, "/app/gpt2.c"));
        await turn(h, shellTurn(["gcc -O3 -o /app/gpt2.c gpt2.c"]));
        h.state.tbProxyResponses.push(wcResult(6927, "/app/gpt2.c"));
        await turn(h, shellTurn(["gcc -O3 -o /app/gpt2.c gpt2.c"]));
        h.state.tbProxyResponses.push(wcResult(6927, "/app/gpt2.c"));
        await turn(h, shellTurn(["gcc -O3 -o /app/gpt2.c gpt2.c"]));
        // Only 2 consecutive over-budget checks since the under-budget reset
        // (turns 3-4), not 3 — must not have escalated yet.
        expect(h.sent).toHaveLength(0);
      });

      it("treats a file of exactly the stated limit as over budget, not compliant (strict <N, not <=N)", async () => {
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        setDeadlineMinutesFromNow(120);
        // REAL_PROMPT's limit is 5000 bytes — a file of EXACTLY 5000 bytes
        // must be judged over budget (the real constraint is a strict "<5000
        // bytes" ceiling), so 3 consecutive at-exactly-limit checks must
        // still escalate.
        await driveOverBudget(h, 0); // just resolves the prompt's limit/path
        for (let i = 0; i < 3; i++) {
          h.state.tbProxyResponses.push(wcResult(5000, "/app/gpt2.c"));
          await turn(h, shellTurn(["gcc -O3 -o /app/gpt2.c gpt2.c"]));
        }
        expect(h.sent).toHaveLength(1);
        expect(h.sent[0].text).toBe(buildTriggerEMessage(0, 5000));
      });

      it("does NOT flag a file at exactly an inclusive limit ('at most N') as over budget", async () => {
        // "at most 5000 bytes" is an INCLUSIVE upper bound -- exactly 5000 is
        // compliant, unlike REAL_PROMPT's exclusive "<5000 bytes" (covered by
        // the strict-comparator test above, which must keep failing at
        // exactly the limit).
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        setDeadlineMinutesFromNow(120);
        await newSessionWithPrompt(
          h,
          "Your program must be at most 5000 bytes. Call your program /app/gpt2.c.",
        );
        for (let i = 0; i < 3; i++) {
          h.state.tbProxyResponses.push(wcResult(5000, "/app/gpt2.c"));
          await turn(h, shellTurn(["gcc -O3 -o /app/gpt2.c gpt2.c"]));
        }
        expect(h.sent).toHaveLength(0);
      });

      it("still flags a file one byte over an inclusive limit ('no more than N')", async () => {
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        setDeadlineMinutesFromNow(120);
        await newSessionWithPrompt(
          h,
          "Your program must be no more than 5000 bytes. Call your program /app/gpt2.c.",
        );
        for (let i = 0; i < 3; i++) {
          h.state.tbProxyResponses.push(wcResult(5001, "/app/gpt2.c"));
          await turn(h, shellTurn(["gcc -O3 -o /app/gpt2.c gpt2.c"]));
        }
        expect(h.sent).toHaveLength(1);
        expect(h.sent[0].text).toBe(buildTriggerEMessage(1, 5000));
      });
    });

    describe("near-deadline degrade", () => {
      it("switches to resolveFinalizeMessage's save-what-you-have framing instead of the escalation text", async () => {
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        // Inside Trigger A/B/D's own near-deadline window (finalize-warn's
        // WARN_REMAINING_MS), so `armed`/`finalizeWarnWouldFire` reads true.
        setDeadlineMinutesFromNow(5);
        await newSessionWithPrompt(h, REAL_PROMPT);
        for (let i = 0; i < 3; i++) {
          h.state.tbProxyResponses.push(wcResult(6927, "/app/gpt2.c"));
          await turn(h, shellTurn(["gcc -O3 -o /app/gpt2.c gpt2.c"]));
        }
        expect(h.sent).toHaveLength(1);
        expect(h.sent[0].text).toBe(resolveFinalizeMessage("terminal_bench"));
        expect(h.sent[0].text).not.toBe(buildTriggerEMessage(1927, 5000));
      });
    });

    describe("proxy failure modes — every one is silence", () => {
      it("does not escalate when the proxy call throws", async () => {
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        setDeadlineMinutesFromNow(120);
        h.ctx.ui.input = async () => {
          throw new Error("proxy unreachable");
        };
        await newSessionWithPrompt(h, REAL_PROMPT);
        for (let i = 0; i < 3; i++) {
          await turn(h, shellTurn(["gcc -O3 -o /app/gpt2.c gpt2.c"]));
        }
        expect(h.sent).toHaveLength(0);
      });

      it("does not escalate on a timed-out check", async () => {
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        setDeadlineMinutesFromNow(120);
        await newSessionWithPrompt(h, REAL_PROMPT);
        for (let i = 0; i < 3; i++) {
          h.state.tbProxyResponses.push("6927 /app/gpt2.c\n[exit=0 cwd=/app timed_out=true]");
          await turn(h, shellTurn(["gcc -O3 -o /app/gpt2.c gpt2.c"]));
        }
        expect(h.sent).toHaveLength(0);
      });

      it("does not escalate on a non-numeric response", async () => {
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        setDeadlineMinutesFromNow(120);
        await newSessionWithPrompt(h, REAL_PROMPT);
        for (let i = 0; i < 3; i++) {
          h.state.tbProxyResponses.push("wc: /app/gpt2.c: No such file\n[exit=1 cwd=/app timed_out=false]");
          await turn(h, shellTurn(["gcc -O3 -o /app/gpt2.c gpt2.c"]));
        }
        expect(h.sent).toHaveLength(0);
      });

      it("never checks size at all when not in TB mode", async () => {
        const h = makeHarness();
        setupExtension(h.pi as any);
        // LITTLE_CODER_TB_MODE deliberately left unset.
        await newSessionWithPrompt(h, REAL_PROMPT);
        await turn(h, shellTurn(["gcc -O3 -o /app/gpt2.c gpt2.c"]));
        expect(h.calls.filter((c) => c === "proxy-input")).toHaveLength(0);
      });
    });

    describe("continuation-prompt latching", () => {
      // Regression: a mid-run-compaction or error-retry continuation re-fires
      // before_agent_start with COMPACTION_CONTINUE_PROMPT/ERROR_RETRY_PROMPT
      // (rpc_client.py), not the task text. Re-parsing unconditionally on
      // every run used to silently wipe an already-resolved limit/path the
      // first time either fired — this trigger going dark for the rest of
      // the trial with no signal that it happened.
      const COMPACTION_CONTINUE_PROMPT =
        "Your session context was compacted to free space, which interrupted " +
        "what you were doing. The task is not complete — please continue from " +
        "where you left off. If the task is actually already complete and " +
        "verified, say so explicitly and stop.";

      it("keeps the limit/path resolved from the first run across a continuation run's own before_agent_start", async () => {
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        setDeadlineMinutesFromNow(120);
        await newSessionWithPrompt(h, REAL_PROMPT);
        // Simulate the harness re-firing before_agent_start for a
        // post-compaction continuation — same session, no session_start.
        await fire(
          h.pi,
          "before_agent_start",
          { systemPromptOptions: {}, prompt: COMPACTION_CONTINUE_PROMPT },
          h.ctx,
        );
        h.state.tbProxyResponses.push(wcResult(6927, "/app/gpt2.c"));
        await turn(h, shellTurn(["gcc -O3 -o /app/gpt2.c gpt2.c"]));
        // Still checks size — the limit/path survived the continuation's
        // own before_agent_start instead of being wiped to undefined.
        expect(h.calls.filter((c) => c === "proxy-input")).toHaveLength(1);
      });

      it("clears the latch at session_start so a later trial doesn't inherit a stale limit/path", async () => {
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        await newSessionWithPrompt(h, REAL_PROMPT);
        // A fresh session with no size language at all — the module-level
        // latch must not carry the previous session's resolved values over.
        await fire(h.pi, "session_start", {}, h.ctx);
        await fire(
          h.pi,
          "before_agent_start",
          { systemPromptOptions: {}, prompt: "Write a function that reverses a string." },
          h.ctx,
        );
        await turn(h, shellTurn(["gcc -O3 -o /app/gpt2.c gpt2.c"]));
        expect(h.calls.filter((c) => c === "proxy-input")).toHaveLength(0);
      });
    });

    describe("relative-path deliverable writes", () => {
      // Regression: the harness's own prompt prefix ("Default working
      // directory is /app", "cd <path> persists") teaches exactly the
      // relative command forms strict path equality never matched — the
      // real gpt2-codegolf trial's own `sed -i ... gpt2.c` pass on the file
      // it had just written was one of them.
      it("checks size when the write targets the deliverable via a relative path", async () => {
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        h.state.tbProxyResponses.push(wcResult(3000, "/app/gpt2.c"));
        await newSessionWithPrompt(h, REAL_PROMPT);
        await turn(h, shellTurn(["cd /app && sed -i 's/foo/bar/g' gpt2.c"]));
        expect(h.calls.filter((c) => c === "proxy-input")).toHaveLength(1);
      });

      it("does not check size for an unrelated file sharing the deliverable's basename at a different absolute path", async () => {
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        await newSessionWithPrompt(h, REAL_PROMPT);
        await turn(h, shellTurn(["cat /app/vocab.bpe > /tmp/gpt2.c"]));
        expect(h.calls.filter((c) => c === "proxy-input")).toHaveLength(0);
      });

      it("still does not check size for a genuinely different filename (e.g. the compiled binary)", async () => {
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        await newSessionWithPrompt(h, REAL_PROMPT);
        await turn(h, shellTurn(["gcc -O3 -lm gpt2.c -o a.out"]));
        expect(h.calls.filter((c) => c === "proxy-input")).toHaveLength(0);
      });
    });

    describe("interpreter-driven writes (previously undetected)", () => {
      // Confirmed gap: a write via `python3 -c "open(path,'w').write(...)"`
      // produced zero detected writes, so this check never ran for that
      // turn even though the deliverable had just been rewritten.
      it("checks size when the deliverable is written via a python3 -c inline script", async () => {
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        h.state.tbProxyResponses.push(wcResult(3000, "/app/gpt2.c"));
        await newSessionWithPrompt(h, REAL_PROMPT);
        await turn(
          h,
          shellTurn([`python3 -c "open('/app/gpt2.c','w').write('int main(){}')"`]),
        );
        expect(h.calls.filter((c) => c === "proxy-input")).toHaveLength(1);
      });
    });

    describe("path quoting — the deliverable path can carry shell metacharacters (task-prompt injection)", () => {
      // Confirmed gap: `checkDeliverableSize` interpolated the parsed
      // deliverable path directly into `wc -c ${path}` with no quoting, so a
      // task prompt (untrusted, task-author-controlled text) whose path
      // capture contained shell metacharacters could inject a second command
      // into the container shell that runs `wc -c`.
      const MARKER = `/tmp/tb-finalize-guard-injection-marker-${process.pid}`;
      // `${IFS}` spells a word break with no literal space; `touch` keeps the
      // live-shell run below offline.
      const MALICIOUS_PATH = `/app/gpt2.c;touch\${IFS}${MARKER};true`;

      afterEach(() => {
        try {
          unlinkSync(MARKER);
        } catch {
          // Nothing to clean up — the point of the test.
        }
      });

      it("quotes the deliverable path before it reaches tbProxyRun, proven by actually running the captured command in a real shell", async () => {
        expect(existsSync(MARKER)).toBe(false);

        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        h.state.tbProxyResponses.push(wcResult(3000, MALICIOUS_PATH));
        await newSessionWithPrompt(
          h,
          `Your program must be under 5000 bytes. Call your program ${MALICIOUS_PATH}`,
        );
        await turn(h, shellTurn([`gcc -O3 -o "${MALICIOUS_PATH}" gpt2.c`]));
        expect(h.calls.filter((c) => c === "proxy-input")).toHaveLength(1);

        const payload = JSON.parse(h.inputs[0].slice(TB_PROXY_PREFIX.length));
        const command: string = payload.command;
        expect(command.startsWith("wc -c ")).toBe(true);

        // Live-code proof: actually run the exact command tbProxyRun was
        // handed, in a real shell — an unquoted interpolation would run the
        // injected `touch` (and create MARKER); a quoted one can't.
        try {
          execSync(command, { shell: "/bin/bash", stdio: "pipe" });
        } catch {
          // `wc -c` on a made-up filename exits non-zero — expected, not a
          // signal of anything.
        }
        expect(existsSync(MARKER)).toBe(false);
      });
    });

    describe("agent_end / no-write-this-turn corrective check", () => {
      // Confirmed gap: the only check at agent_end was maybeFireTriggerC;
      // an over-budget deliverable that wasn't rewritten on the run's final
      // turn(s) (only compiled/run, or simply never touched again) reached
      // the run's end with zero corrective steer.
      it("produces a corrective nudge at agent_end even when no write happened this run", async () => {
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        setDeadlineMinutesFromNow(120);
        h.state.tbProxyResponses.push(wcResult(6927, "/app/gpt2.c"));
        await newSessionWithPrompt(h, REAL_PROMPT);
        // No turn_start/turn_end at all this run — no write evidence, and
        // the per-turn escalation's 3-consecutive-checks counter never even
        // starts.
        await settle(h);
        expect(h.sent).toHaveLength(1);
        expect(h.sent[0].text).toBe(buildTriggerEMessage(1927, 5000));
      });

      it("does not fire when the deliverable is at or under the limit at agent_end", async () => {
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        setDeadlineMinutesFromNow(120);
        h.state.tbProxyResponses.push(wcResult(3000, "/app/gpt2.c"));
        await newSessionWithPrompt(h, REAL_PROMPT);
        await settle(h);
        expect(h.sent).toHaveLength(0);
      });

      it("does not fire when no limit/path resolved at all", async () => {
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        await newSession(h);
        await settle(h);
        expect(h.calls.filter((c) => c === "proxy-input")).toHaveLength(0);
        expect(h.sent).toHaveLength(0);
      });

      it("stands down on an aborted run instead of stacking a steer on thinking-budget's own abort-recovery", async () => {
        // Confirmed gap: this check had neither of maybeFireTriggerC's two
        // guards, so an aborted run (thinking-budget abort-then-recover,
        // context-watchdog compaction resume) still got Trigger E's own
        // steer stacked on top of whatever recovery message the abort
        // already queued.
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        setDeadlineMinutesFromNow(120);
        h.state.tbProxyResponses.push(wcResult(6927, "/app/gpt2.c"));
        await newSessionWithPrompt(h, REAL_PROMPT);
        await turn(h, assistantTurn({ thinking: "let me think...", stopReason: "aborted" }));
        await settle(h);
        expect(h.sent).toHaveLength(0);
        expect(h.calls.filter((c) => c === "proxy-input")).toHaveLength(0);
      });

      it("stands down at the turn cap (no later turn for a corrective nudge to land on)", async () => {
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        setDeadlineMinutesFromNow(120);
        h.state.tbProxyResponses.push(wcResult(6927, "/app/gpt2.c"));
        await fire(h.pi, "session_start", {}, h.ctx);
        await fire(
          h.pi,
          "before_agent_start",
          { systemPromptOptions: { littleCoder: { maxTurns: 2 } }, prompt: REAL_PROMPT },
          h.ctx,
        );
        await turn(h, shellTurn(["ls -la"])); // turn 1
        await turn(h, shellTurn(["ls -la"])); // turn 2 == capForRun
        await settle(h);
        expect(h.sent).toHaveLength(0);
        expect(h.calls.filter((c) => c === "proxy-input")).toHaveLength(0);
      });

      it("does not also fire (a second, redundant nudge) when Trigger C already fired on this same agent_end", async () => {
        // Trigger C already told the model the run died/errored; stacking
        // Trigger E's own over-budget nudge on the same agent_end would be a
        // second contradictory steer for one event.
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        setDeadlineMinutesFromNow(120);
        h.state.tbProxyResponses.push(wcResult(6927, "/app/gpt2.c"));
        await newSessionWithPrompt(h, REAL_PROMPT);
        await turn(h, assistantTurn({ stopReason: "error" })); // qualifies Trigger C
        await settle(h);
        expect(h.sent).toHaveLength(1);
        expect(h.sent[0].text).toMatch(/previous turn ended with an error or an empty response/i);
        expect(h.calls.filter((c) => c === "proxy-input")).toHaveLength(0);
      });

      it("does not throw out of agent_end when ctx.ui.notify fails on the post-await harnessIntervention call", async () => {
        // Confirmed gap: only the pi.sendUserMessage call was try/caught;
        // harnessIntervention's own ctx.ui.notify right after it was not, so
        // a stale ctx after an await that crossed a session-replacing
        // abort/compaction boundary would throw unhandled out of agent_end.
        const h = makeHarness();
        setupExtension(h.pi as any);
        process.env.LITTLE_CODER_TB_MODE = "1";
        setDeadlineMinutesFromNow(120);
        h.state.tbProxyResponses.push(wcResult(6927, "/app/gpt2.c"));
        await newSessionWithPrompt(h, REAL_PROMPT);
        let notifyCalls = 0;
        h.ctx.ui.notify = () => {
          notifyCalls++;
          throw new Error("ctx.ui is stale after the await");
        };
        await expect(settle(h)).resolves.toBeUndefined();
        expect(h.sent).toHaveLength(1); // the steer itself still went out
        expect(notifyCalls).toBeGreaterThan(0);
      });
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
