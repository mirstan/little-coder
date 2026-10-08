import { compact, SessionManager } from "@earendil-works/pi-coding-agent";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";
import setup, { ENV_ENABLE, runReuse, type Capture } from "./index.ts";
import { buildReplayBody } from "./replay.ts";
import {
  assistantText,
  contextMessages,
  longTurn,
  MODEL,
  piCompaction,
  realPayload,
  SETTINGS,
  userMsg,
} from "./test-support.ts";

// ── Fake OpenAI-compatible server on an ephemeral port ─────────────────────

type Reply = (req: IncomingMessage, res: ServerResponse, body: string) => void;
let server: Server;
let baseUrl = "";
let reply: Reply;
const received: { body: string; headers: IncomingMessage["headers"] }[] = [];

const sse = (res: ServerResponse, chunks: unknown[], opts: { end?: boolean } = {}) => {
  res.writeHead(200, { "content-type": "text/event-stream" });
  for (const c of chunks) res.write(`data: ${JSON.stringify(c)}\n\n`);
  if (opts.end !== false) {
    res.write("data: [DONE]\n\n");
    res.end();
  }
};

const answer = (content: string, extra: { toolCalls?: boolean } = {}): Reply => (_req, res) =>
  sse(res, [
    { choices: [{ delta: { reasoning_content: "brief" } }] },
    { choices: [{ delta: { content } }] },
    ...(extra.toolCalls
      ? [{ choices: [{ delta: { tool_calls: [{ index: 0, id: "t", function: { name: "bash", arguments: "{}" } }] } }] }]
      : []),
    { choices: [{ delta: {}, finish_reason: extra.toolCalls ? "tool_calls" : "stop" }] },
    { choices: [], usage: { prompt_tokens: 120_000, completion_tokens: 900, prompt_tokens_details: { cached_tokens: 118_784 } } },
  ]);

beforeAll(async () => {
  server = createServer((req, res) => {
    let body = "";
    req.setEncoding("utf8");
    req.on("data", (d) => (body += d));
    req.on("end", () => {
      received.push({ body, headers: req.headers });
      reply(req, res, body);
    });
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}/v1`;
});
afterAll(() => new Promise<void>((r) => server.close(() => r())));

const savedEnv = { ...process.env };
beforeEach(() => {
  received.length = 0;
  process.env[ENV_ENABLE] = "1";
  process.env.LITTLE_CODER_TELEMETRY = "1";
});
afterEach(() => {
  for (const k of [ENV_ENABLE, "LITTLE_CODER_TELEMETRY"]) {
    if (savedEnv[k] === undefined) delete process.env[k];
    else process.env[k] = savedEnv[k];
  }
});

// ── Wiring helpers ─────────────────────────────────────────────────────────

type Handler = (event: any, ctx: any) => any;
function wire() {
  const handlers: Record<string, Handler[]> = {};
  const entries: { type: string; data: any }[] = [];
  const pi = {
    on(name: string, h: Handler) {
      (handlers[name] ??= []).push(h);
    },
    appendEntry(type: string, data: any) {
      entries.push({ type, data });
    },
  };
  setup(pi as any);
  const fire = async (name: string, event: any, ctx: any) => {
    let result: any;
    for (const h of handlers[name] ?? []) {
      const r = await h(event, ctx);
      if (r !== undefined) result = r;
    }
    return result;
  };
  return { handlers, entries, fire };
}

function makeCtx(over: Record<string, unknown> = {}) {
  return {
    model: { ...MODEL, baseUrl },
    modelRegistry: { getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "sk-test", headers: { "x-test": "1" } }) },
    sessionManager: { getSessionId: () => "sess-1" },
    ...over,
  };
}

interface Scenario {
  sm: SessionManager;
  prep: any;
  branch: any[];
  payload: any;
}

async function scenario(build: (sm: SessionManager) => void): Promise<Scenario> {
  const sm = SessionManager.inMemory("/tmp");
  build(sm);
  const { prepareCompaction } = await piCompaction();
  const branch = sm.getBranch();
  const prep = prepareCompaction(branch, SETTINGS);
  if (!prep) throw new Error("nothing to compact");
  return { sm, prep, branch, payload: await realPayload(contextMessages(sm)) };
}

const event = (s: Scenario, over: Record<string, unknown> = {}) => ({
  type: "session_before_compact",
  preparation: s.prep,
  branchEntries: s.branch,
  reason: "manual",
  willRetry: false,
  signal: new AbortController().signal,
  ...over,
});

const capture = (s: Scenario, over: Partial<Capture> = {}): Capture => ({
  payload: structuredClone(s.payload),
  provider: "omlx",
  modelId: MODEL.id,
  sessionId: "sess-1",
  ...over,
});

const deps = { fetch: globalThis.fetch, now: Date.now, timeoutMs: 5000 };

const HISTORY = "## Goal\nBuild the thing\n\n## Progress\n### Done\n- [x] read sources\n\n## Next Steps\n1. run make";
const PREFIX = "## Original Request\nFix the build\n\n## Early Progress\n- ran make\n\n## Context for Suffix\n- linker errors";
const both = `<history-summary>\n${HISTORY}\n</history-summary>\n\n<turn-prefix-summary>\n${PREFIX}\n</turn-prefix-summary>`;

// pi's own compact() on the same preparation, with an LLM that returns the
// same section texts, is the reference for every field but usage.
async function piReference(s: Scenario) {
  const streamFn = async (_m: any, ctx: any) => {
    const prompt = ctx.messages[0].content[0].text as string;
    const text = prompt.includes("This is the PREFIX of a turn") ? PREFIX : HISTORY;
    return {
      result: async () => ({
        role: "assistant",
        content: [{ type: "text", text }],
        stopReason: "stop",
        usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
      }),
    };
  };
  return compact(s.prep, MODEL, "k", undefined, undefined, undefined, undefined, streamFn as any);
}

// ── Scenarios pi really produces ───────────────────────────────────────────

const nonSplit = () =>
  scenario((sm) => {
    for (let i = 0; i < 8; i++) {
      sm.appendMessage(userMsg(`request ${i}: ` + `context ${i} `.repeat(4000)));
      sm.appendMessage(assistantText(`done with request ${i}`));
    }
  });
const splitWithHistory = () =>
  scenario((sm) => {
    longTurn(sm, "Set up the toolchain in /app.", 12, "a");
    longTurn(sm, "Now fix the build in /app.", 24, "b");
  });
const splitNoHistory = () => scenario((sm) => longTurn(sm, "Fix the build in /app.", 30, "c"));

describe("scenario shapes (sanity: these are the cases the suite claims to cover)", () => {
  it("non-split, split with history, split without history", async () => {
    const a = await nonSplit();
    expect(a.prep.isSplitTurn).toBe(false);
    const b = await splitWithHistory();
    expect(b.prep.isSplitTurn).toBe(true);
    expect(b.prep.messagesToSummarize.length).toBeGreaterThan(0);
    const c = await splitNoHistory();
    expect(c.prep.isSplitTurn).toBe(true);
    expect(c.prep.messagesToSummarize.length).toBe(0);
  });
});

describe("reuse path", () => {
  it("is inert unless opted in", () => {
    delete process.env[ENV_ENABLE];
    expect(Object.keys(wire().handlers)).toEqual([]);
  });

  it("sends exactly the captured payload plus one instruction, and matches pi's result on a non-split compaction", async () => {
    const s = await nonSplit();
    reply = answer(HISTORY);
    const w = wire();
    const ctx = makeCtx();
    await w.fire("before_provider_request", { type: "before_provider_request", payload: s.payload }, ctx);
    const result = await w.fire("session_before_compact", event(s), ctx);

    expect(received).toHaveLength(1);
    const sent = JSON.parse(received[0].body);
    const instruction = sent.messages.at(-1).content;
    const expected = buildReplayBody(s.payload, instruction, { maxTokens: sent.max_completion_tokens ?? sent.max_tokens, thinkingBudget: 2048 });
    // Byte-identical request body, not merely deep-equal.
    expect(received[0].body).toBe(JSON.stringify(expected));
    expect(received[0].headers.authorization).toBe("Bearer sk-test");
    expect(received[0].headers["x-test"]).toBe("1");

    const ref = await piReference(s);
    expect(result.compaction.summary).toBe(ref.summary);
    expect(result.compaction.firstKeptEntryId).toBe(ref.firstKeptEntryId);
    expect(result.compaction.tokensBefore).toBe(ref.tokensBefore);
    expect(result.compaction.details).toEqual(ref.details);
    expect(result.compaction.usage).toMatchObject({ input: 1216, cacheRead: 118_784, output: 900 });

    const t = w.entries.find((e) => e.data.kind === "compaction_reuse")!.data;
    expect(t).toMatchObject({ path: "reuse", prompt_tokens: 120_000, cache_read: 118_784, summary_tokens: 900, reason: "manual" });
    expect(t.ttft_s).toBeGreaterThanOrEqual(0);
  });

  it("matches pi's merged split-turn summary, with history", async () => {
    const s = await splitWithHistory();
    reply = answer(both);
    const out = await runReuse(event(s), makeCtx(), capture(s), 2048, deps);
    expect(out.ok).toBe(true);
    const ref = await piReference(s);
    expect((out as any).compaction.summary).toBe(ref.summary);
    expect((out as any).compaction.details).toEqual(ref.details);
    expect((out as any).compaction.firstKeptEntryId).toBe(ref.firstKeptEntryId);
    const instruction = JSON.parse(received[0].body).messages.at(-1).content as string;
    expect(instruction).toContain("<history-summary>");
    expect(instruction).toContain("<turn-prefix-summary>");
    expect(instruction).toContain("Now fix the build in /app.");
  });

  it("matches pi's 'No prior history.' split when nothing precedes the turn", async () => {
    const s = await splitNoHistory();
    reply = answer(`<turn-prefix-summary>\n${PREFIX}\n</turn-prefix-summary>`);
    const out = await runReuse(event(s), makeCtx(), capture(s), 2048, deps);
    expect(out.ok).toBe(true);
    const ref = await piReference(s);
    expect((out as any).compaction.summary).toBe(ref.summary);
    expect((out as any).compaction.summary.startsWith("No prior history.")).toBe(true);
  });

  it("uses pi's update prompt when a previous compaction summary opens the payload", async () => {
    const s0 = await splitWithHistory();
    const keep = s0.prep.firstKeptEntryId;
    s0.sm.appendCompaction("## Goal\nOLD SUMMARY GOAL\n\n## Next Steps\n1. continue", keep, 999, { readFiles: ["/old"], modifiedFiles: [] });
    longTurn(s0.sm, "Your session context was compacted; continue.", 30, "d");
    const { prepareCompaction } = await piCompaction();
    const branch = s0.sm.getBranch();
    const s: Scenario = { sm: s0.sm, branch, prep: prepareCompaction(branch, SETTINGS), payload: await realPayload(contextMessages(s0.sm)) };
    expect(s.prep.previousSummary).toContain("OLD SUMMARY GOAL");
    reply = answer(both);
    const out = await runReuse(event(s), makeCtx(), capture(s), 2048, deps);
    expect(out.ok).toBe(true);
    const instruction = JSON.parse(received[0].body).messages.at(-1).content as string;
    expect(instruction).toContain("PRESERVE all existing information from the previous summary");
    const ref = await piReference(s);
    expect((out as any).compaction.summary).toBe(ref.summary);
    expect((out as any).compaction.details).toEqual(ref.details);

    // The payload from before that compaction does not hold its summary: stale.
    const old = await runReuse(event(s), makeCtx(), { ...capture(s), payload: s0.payload }, 2048, deps);
    expect(old).toMatchObject({ ok: false, fallback: "stale" });
  });

  it("keeps a valid summary even if the model also started a tool call", async () => {
    const s = await nonSplit();
    reply = answer(HISTORY, { toolCalls: true });
    const out = await runReuse(event(s), makeCtx(), capture(s), 2048, deps);
    expect(out.ok).toBe(true);
    expect((out as any).telemetry.tool_calls).toBe(1);
  });

  it("forgets the capture on session_compact and model_select", async () => {
    const s = await nonSplit();
    for (const ev of ["session_compact", "model_select"]) {
      reply = answer(HISTORY);
      const w = wire();
      const ctx = makeCtx();
      await w.fire("before_provider_request", { payload: s.payload }, ctx);
      await w.fire(ev, {}, ctx);
      expect(await w.fire("session_before_compact", event(s), ctx)).toBeUndefined();
      expect(w.entries.at(-1)!.data).toMatchObject({ path: "native", fallback: "no_capture" });
    }
  });

  it("does not alter the payload pi sends", async () => {
    const s = await nonSplit();
    const w = wire();
    const payload = structuredClone(s.payload);
    expect(await w.fire("before_provider_request", { payload }, makeCtx())).toBeUndefined();
    expect(payload).toEqual(s.payload);
  });
});

describe("fallbacks hand the compaction to pi (return nothing)", () => {
  it.each([
    ["overflow", { reason: "overflow", willRetry: true }],
    ["overflow", { reason: "threshold", willRetry: true }],
  ])("%s", async (fallback, over) => {
    const s = await nonSplit();
    expect(await runReuse(event(s, over), makeCtx(), capture(s), 2048, deps)).toMatchObject({ ok: false, fallback });
    expect(received).toHaveLength(0);
  });

  it("api, no_capture, model_mismatch, session_mismatch, auth", async () => {
    const s = await nonSplit();
    const cases: [string, any, Capture | null][] = [
      ["api", makeCtx({ model: { ...MODEL, api: "anthropic-messages" } }), capture(s)],
      ["api", makeCtx({ model: undefined }), capture(s)],
      ["no_capture", makeCtx(), null],
      ["model_mismatch", makeCtx(), capture(s, { modelId: "other" })],
      ["model_mismatch", makeCtx(), capture(s, { provider: "rapidmlx" })],
      ["session_mismatch", makeCtx(), capture(s, { sessionId: "sess-0" })],
      ["auth", makeCtx({ modelRegistry: { getApiKeyAndHeaders: async () => ({ ok: false, error: "no key" }) } }), capture(s)],
    ];
    for (const [fallback, ctx, cap] of cases) {
      expect(await runReuse(event(s), ctx, cap, 2048, deps)).toMatchObject({ ok: false, fallback });
    }
    expect(received).toHaveLength(0);
  });

  it("anchor_missing when the capture predates the history pi summarizes", async () => {
    const s = await splitWithHistory();
    const early = await realPayload(contextMessages(s.sm).slice(0, 5));
    expect(await runReuse(event(s), makeCtx(), { ...capture(s), payload: early }, 2048, deps)).toMatchObject({
      ok: false,
      fallback: "anchor_missing",
    });
  });

  it("window when payload + instruction + answer cannot fit", async () => {
    const s = await nonSplit();
    const prep = { ...s.prep, tokensBefore: MODEL.contextWindow - 3000 };
    expect(await runReuse(event(s, { preparation: prep }), makeCtx(), capture(s), 2048, deps)).toMatchObject({
      ok: false,
      fallback: "window",
    });
  });

  it("clamps the output budget to the room left in the window", async () => {
    const s = await nonSplit();
    reply = answer(HISTORY);
    const prep = { ...s.prep, tokensBefore: MODEL.contextWindow - 12_000 };
    const out = await runReuse(event(s, { preparation: prep }), makeCtx(), capture(s), 2048, deps);
    expect(out.ok).toBe(true);
    const sent = JSON.parse(received[0].body);
    expect(sent.max_completion_tokens ?? sent.max_tokens).toBeLessThan(12_000);
  });

  it("http_error on a non-2xx answer or an unreachable server", async () => {
    const s = await nonSplit();
    reply = (_q, res) => {
      res.writeHead(500);
      res.end("boom");
    };
    expect(await runReuse(event(s), makeCtx(), capture(s), 2048, deps)).toMatchObject({ ok: false, fallback: "http_error" });
    const dead = makeCtx({ model: { ...MODEL, baseUrl: "http://127.0.0.1:9/v1" } });
    expect(await runReuse(event(s), dead, capture(s), 2048, deps)).toMatchObject({ ok: false, fallback: "http_error" });
  });

  it("aborted when pi aborts the compaction mid-stream", async () => {
    const s = await nonSplit();
    const ac = new AbortController();
    reply = (_q, res) => {
      sse(res, [{ choices: [{ delta: { reasoning_content: "thinking" } }] }], { end: false });
      setTimeout(() => ac.abort(), 20);
      setTimeout(() => res.end(), 2000);
    };
    expect(await runReuse(event(s, { signal: ac.signal }), makeCtx(), capture(s), 2048, deps)).toMatchObject({
      ok: false,
      fallback: "aborted",
    });
  });

  it("timeout when the server never answers in time", async () => {
    const s = await nonSplit();
    reply = (_q, res) => setTimeout(() => res.end(), 2000);
    expect(await runReuse(event(s), makeCtx(), capture(s), 2048, { ...deps, timeoutMs: 50 })).toMatchObject({
      ok: false,
      fallback: "timeout",
    });
  });

  it("tool_call when the model only calls a tool; garbage on an unusable answer", async () => {
    const s = await nonSplit();
    reply = answer("", { toolCalls: true });
    expect(await runReuse(event(s), makeCtx(), capture(s), 2048, deps)).toMatchObject({ ok: false, fallback: "tool_call" });
    reply = answer("Sure, I will keep working on make.");
    expect(await runReuse(event(s), makeCtx(), capture(s), 2048, deps)).toMatchObject({ ok: false, fallback: "garbage" });
    const split = await splitWithHistory();
    reply = answer(`<history-summary>\n${HISTORY}\n</history-summary>`);
    expect(await runReuse(event(split), makeCtx(), capture(split), 2048, deps)).toMatchObject({
      ok: false,
      fallback: "garbage",
    });
  });

  it("the wired hook returns undefined and records the fallback", async () => {
    const s = await nonSplit();
    const w = wire();
    expect(await w.fire("session_before_compact", event(s, { reason: "overflow", willRetry: true }), makeCtx())).toBeUndefined();
    expect(w.entries).toEqual([
      { type: "lc-telemetry", data: expect.objectContaining({ kind: "compaction_reuse", path: "native", fallback: "overflow" }) },
    ]);
  });
});
