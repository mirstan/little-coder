import { describe, it, expect, beforeEach, afterEach } from "vitest";
import {
  DEFAULT_CMD_KEEP,
  DEFAULT_DEMOTE_PENDING_BYTES,
  DEFAULT_KEEP_RESULT_HEAD,
  DEFAULT_KEEP_RESULT_TAIL,
  DEFAULT_MIN_PAIR_BYTES,
  DEFAULT_RETAIN_RAW,
  DEFAULT_STALE_DISTANCE,
  ENV_CMD_KEEP,
  ENV_DEMOTE_BATCH,
  ENV_DEMOTE_PENDING_BYTES,
  ENV_KEEP_RESULT_HEAD,
  ENV_KEEP_RESULT_TAIL,
  ENV_MIN_PAIR_BYTES,
  ENV_RETAIN_RAW,
  ENV_STALE_DISTANCE,
  demoteMessages,
  demoteMessagesWithStats,
  type RetentionArchive,
  type RetentionOptions,
} from "./retention.ts";
import setupShellRetention from "./index.ts";

const FOOTER = "[exit=0 cwd=/app timed_out=false backend=harbor-env]";

const OPTS: RetentionOptions = {
  retainRaw: DEFAULT_RETAIN_RAW,
  minPairBytes: DEFAULT_MIN_PAIR_BYTES,
  staleDistance: DEFAULT_STALE_DISTANCE,
  keepResultHeadBytes: DEFAULT_KEEP_RESULT_HEAD,
  keepResultTailBytes: DEFAULT_KEEP_RESULT_TAIL,
  cmdKeepBytes: DEFAULT_CMD_KEEP,
  demoteBatch: 1,
  demotePendingBytes: DEFAULT_DEMOTE_PENDING_BYTES,
};

function memArchive(): RetentionArchive {
  const entries = new Map<string, string>();
  return {
    save(id, text) {
      entries.set(id, text);
      return true;
    },
    size(id) {
      const t = entries.get(id);
      return t === undefined ? undefined : Buffer.byteLength(t, "utf-8");
    },
    readRange(id, start, length) {
      const t = entries.get(id);
      return t === undefined ? undefined : Buffer.from(t, "utf-8").subarray(start, start + length);
    },
  };
}

function body(bytes: number, tag: string): string {
  const line = `${tag} ${"x".repeat(60)}\n`;
  return line.repeat(Math.ceil(bytes / line.length)).slice(0, bytes);
}

function call(id: string, command: string, blockExtra: Record<string, unknown> = {}, msgExtra: Record<string, unknown> = {}, leading: any[] = []) {
  return {
    role: "assistant",
    ...msgExtra,
    content: [...leading, { type: "toolCall", id, name: "ShellSession", arguments: { command }, ...blockExtra }],
  };
}

function result(id: string, text: string) {
  return {
    role: "toolResult",
    toolCallId: id,
    toolName: "ShellSession",
    content: [{ type: "text", text: `${text}\n${FOOTER}` }],
    isError: false,
    timestamp: 1700000000000,
  };
}

function sixPairs(): any[] {
  const msgs: any[] = [{ role: "user", content: "go" }];
  for (let i = 0; i < 6; i++) msgs.push(call(`p${i}`, `echo ${i}`), result(`p${i}`, body(4096, `p${i}`)));
  return msgs;
}

const HEREDOC = `cat > /tmp/s.c <<'EOF'\n${body(5000, "src")}\nEOF\ngcc /tmp/s.c`;

describe("demoteMessagesWithStats", () => {
  it("counts the projection: pairs, due, prefix, demoted and bytes saved", () => {
    const out = demoteMessagesWithStats(sixPairs(), memArchive(), OPTS);
    expect(out.demotedCount).toBe(2);
    expect(out.stats).toMatchObject({
      pairs: 6, large: 6, due: 2, prefix: 2, demoted: 2, flushed: false,
      signed: 0, skippedNoShrink: 0, skippedArchive: 0,
    });
    expect(out.stats.bytesBefore).toBeGreaterThan(2 * 4096);
    expect(out.stats.bytesAfter).toBeLessThan(out.stats.bytesBefore / 2);
  });

  it("leaves demoteMessages' own result shape and content unchanged", () => {
    const msgs = sixPairs();
    const withStats = demoteMessagesWithStats(msgs, memArchive(), OPTS);
    expect(demoteMessages(msgs, memArchive(), OPTS)).toEqual({
      messages: withStats.messages,
      demotedCount: withStats.demotedCount,
    });
  });

  it("shows a due pair that a signature kept raw: prefix 1, demoted 0, signed 1", () => {
    const msgs = [{ role: "user", content: "go" }, call("sig", HEREDOC, { thoughtSignature: "CtYBsig" }), result("sig", "compiled")];
    const out = demoteMessagesWithStats(msgs, memArchive(), { ...OPTS, retainRaw: 0 });
    expect(out.stats).toMatchObject({
      large: 1, due: 1, prefix: 1, demoted: 0, signed: 1, skippedNoShrink: 1, bytesBefore: 0, bytesAfter: 0,
    });
  });

  it("does not count openai-completions reasoning_content as a signature", () => {
    const msgs = [
      { role: "user", content: "go" },
      call("oc", HEREDOC, {}, { api: "openai-completions" }, [
        { type: "thinking", thinking: "plan", thinkingSignature: "reasoning_content" },
      ]),
      result("oc", "compiled"),
    ];
    const out = demoteMessagesWithStats(msgs, memArchive(), { ...OPTS, retainRaw: 0 });
    expect(out.stats).toMatchObject({ prefix: 1, demoted: 1, signed: 0, skippedNoShrink: 0 });
  });

  it("counts a refused archive save", () => {
    const refusing: RetentionArchive = { save: () => false, size: () => undefined, readRange: () => undefined };
    const out = demoteMessagesWithStats(sixPairs(), refusing, OPTS);
    expect(out.stats).toMatchObject({ prefix: 2, demoted: 0, skippedArchive: 2 });
  });
});

// ── wired through the real hooks ────────────────────────────────────────────

const PINNED_ENV: Record<string, string | undefined> = {
  [ENV_STALE_DISTANCE]: String(DEFAULT_STALE_DISTANCE),
  [ENV_RETAIN_RAW]: String(DEFAULT_RETAIN_RAW),
  [ENV_MIN_PAIR_BYTES]: String(DEFAULT_MIN_PAIR_BYTES),
  [ENV_KEEP_RESULT_HEAD]: String(DEFAULT_KEEP_RESULT_HEAD),
  [ENV_KEEP_RESULT_TAIL]: String(DEFAULT_KEEP_RESULT_TAIL),
  [ENV_CMD_KEEP]: String(DEFAULT_CMD_KEEP),
  [ENV_DEMOTE_BATCH]: "1",
  [ENV_DEMOTE_PENDING_BYTES]: String(DEFAULT_DEMOTE_PENDING_BYTES),
  LITTLE_CODER_SHELL_DEMOTE_MIN_SAVE_RATIO: undefined,
  LITTLE_CODER_SHELL_DEMOTE_OPEN_AT_PERCENT: undefined,
  LITTLE_CODER_SHELL_DEMOTE_FORCE_PENDING_BYTES: undefined,
  LITTLE_CODER_COMPACT_AT_PERCENT: undefined,
  LITTLE_CODER_SHELL_RETENTION_BUDGET_BYTES: String(256 * 1024 * 1024),
  LITTLE_CODER_NO_SHELL_RETENTION: undefined,
  LITTLE_CODER_TELEMETRY: "1",
};

/** One big pair, then 60 user turns: stale under the per-pair rule. */
function staleSeedMessages(): any[] {
  const bigBody = ("x".repeat(50) + "\n").repeat(200);
  return [
    { role: "user", content: "go" },
    { role: "assistant", content: [{ type: "toolCall", id: "tc-1", name: "ShellSession", arguments: { command: "run big thing" } }] },
    { role: "toolResult", toolCallId: "tc-1", toolName: "ShellSession", content: [{ type: "text", text: bigBody }] },
    ...Array.from({ length: 60 }, (_, i) => ({ role: "user", content: `turn ${i}` })),
  ];
}

type Handler = (event: any, ctx: any) => any;

function wire() {
  const handlers: Record<string, Handler[]> = {};
  const entries: Array<{ customType: string; data: any }> = [];
  const pi = {
    on(name: string, h: Handler) {
      (handlers[name] ??= []).push(h);
    },
    registerTool() {},
    appendEntry(customType: string, data: any) {
      entries.push({ customType, data });
    },
  };
  setupShellRetention(pi as any);
  return { handlers, entries };
}

function makeCtx() {
  return { cwd: "/tmp", ui: { notify: (_m: string) => {} } };
}

async function project(handlers: Record<string, Handler[]>, messages: any[]) {
  let out: any;
  for (const h of handlers.context ?? []) {
    const r = await h({ messages }, makeCtx());
    if (r) out = r;
  }
  return out;
}

describe("shell-retention telemetry (wired)", () => {
  let savedEnv: Record<string, string | undefined>;
  let wired: ReturnType<typeof wire> | undefined;

  beforeEach(() => {
    savedEnv = {};
    for (const [name, value] of Object.entries(PINNED_ENV)) {
      savedEnv[name] = process.env[name];
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    }
  });

  afterEach(async () => {
    for (const h of wired?.handlers.session_shutdown ?? []) await h({}, makeCtx());
    wired = undefined;
    for (const [name, value] of Object.entries(savedEnv)) {
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    }
  });

  it("emits one shell_retention snapshot per request, and none under the kill switch", async () => {
    wired = wire();
    const out = await project(wired.handlers, staleSeedMessages());
    expect(wired.entries.map((e) => [e.customType, e.data.kind])).toEqual([["lc-telemetry", "shell_retention"]]);
    expect(wired.entries[0].data).toMatchObject({ v: 1, pairs: 1, large: 1, due: 1, prefix: 1, demoted: 1, signed: 0 });
    expect(wired.entries[0].data.bytesAfter).toBeLessThan(wired.entries[0].data.bytesBefore);
    expect(out?.messages[2].content[0].text).toMatch(/^\[shell result demoted/);

    // pi hands the hook pristine history every request, so the same history
    // yields the same snapshot again: a count of what is demoted, not of news.
    await project(wired.handlers, staleSeedMessages());
    expect(wired.entries).toHaveLength(2);
    expect(wired.entries[1].data.demoted).toBe(1);

    delete process.env.LITTLE_CODER_TELEMETRY;
    const killed = await project(wired.handlers, staleSeedMessages());
    expect(wired.entries).toHaveLength(2);
    expect(killed?.messages[2].content[0].text).toMatch(/^\[shell result demoted/);
  });

  it("emits a snapshot even when nothing is due", async () => {
    wired = wire();
    await project(wired.handlers, [{ role: "user", content: "hi" }]);
    expect(wired.entries.map((e) => e.data)).toEqual([
      {
        v: 1, kind: "shell_retention", pairs: 0, large: 0, due: 0, prefix: 0, demoted: 0, flushed: false,
        signed: 0, skippedNoShrink: 0, skippedArchive: 0, bytesBefore: 0, bytesAfter: 0,
        gate: "off", gateReason: null, skippedCost: 0, sticky: 0,
        estSaveTokens: 0, estReprefillTokens: 0, estContextTokens: 0,
      },
    ]);
  });

  it("records the cost gate's verdict and estimates from pi's context usage", async () => {
    process.env[ENV_DEMOTE_BATCH] = "4";
    process.env.LITTLE_CODER_SHELL_DEMOTE_OPEN_AT_PERCENT = "75"; // the context clause is off by default
    wired = wire();
    // Eight 4KB pairs then a long, shell-free tail: the oldest four are a due
    // batch whose break would re-prefill the whole tail.
    const msgs: any[] = [{ role: "user", content: "go" }];
    for (let i = 0; i < 8; i++) msgs.push(call(`g${i}`, `echo ${i}`), result(`g${i}`, body(4096, `g${i}`)));
    for (let i = 0; i < 10; i++) msgs.push({ role: "assistant", content: [{ type: "thinking", thinking: "t".repeat(40_000) }] });
    const ctxAt = (tokens: number) => ({
      ...makeCtx(),
      getContextUsage: () => ({ tokens, contextWindow: 262144, percent: (100 * tokens) / 262144 }),
    });

    for (const h of wired.handlers.context ?? []) await h({ messages: msgs }, ctxAt(150_000));
    expect(wired.entries.at(-1)?.data).toMatchObject({
      kind: "shell_retention", due: 4, prefix: 0, demoted: 0, gate: "deferred", gateReason: null,
      skippedCost: 4, estContextTokens: 150_000,
    });
    expect(wired.entries.at(-1)?.data.estReprefillTokens).toBeGreaterThan(100_000);

    let out: any;
    for (const h of wired.handlers.context ?? []) out = await h({ messages: msgs }, ctxAt(200_000));
    expect(wired.entries.at(-1)?.data).toMatchObject({ demoted: 4, gate: "open", gateReason: "context", skippedCost: 0 });
    expect(out?.messages[2].content[0].text).toMatch(/^\[shell result demoted/);
  });

  it("reads the window from ctx.model when usage is unavailable, and turns the gate off when ctx throws", async () => {
    wired = wire();
    const msgs = staleSeedMessages();
    for (const h of wired.handlers.context ?? []) {
      await h({ messages: msgs }, { ...makeCtx(), getContextUsage: () => undefined, model: { contextWindow: 262144 } });
    }
    expect(wired.entries.at(-1)?.data).toMatchObject({ gate: "open", gateReason: "cold", demoted: 1 });

    const stale = {
      ...makeCtx(),
      getContextUsage: () => {
        throw new Error("stale ctx");
      },
      get model(): unknown {
        throw new Error("stale ctx");
      },
    };
    for (const h of wired.handlers.context ?? []) await h({ messages: msgs }, stale);
    expect(wired.entries.at(-1)?.data).toMatchObject({ gate: "off", demoted: 1 });
  });

  it("emits echo_block when it blocks a call quoting a live placeholder", async () => {
    wired = wire();
    const out = await project(wired.handlers, staleSeedMessages());
    const m = /ShellRecall id=(sr-[0-9a-f]{16})/.exec(out?.messages[2].content[0].text ?? "");
    expect(m).not.toBeNull();
    const id = m![1];
    let blocked: any;
    for (const h of wired.handlers.tool_call ?? []) {
      const r = await h(
        { toolName: "ShellSession", input: { command: `printf '%s' '[... 1.6KB of command text demoted — ShellRecall id=${id} ...]' > f` } },
        makeCtx(),
      );
      if (r?.block) blocked = r;
    }
    expect(blocked?.block).toBe(true);
    expect(wired.entries.filter((e) => e.data.kind === "echo_block").map((e) => e.data)).toEqual([
      { v: 1, kind: "echo_block", source: "shell_retention", tool: "ShellSession", ids: 1 },
    ]);
  });
});
