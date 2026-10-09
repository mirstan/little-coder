import { describe, it, expect, afterEach } from "vitest";
import {
  archiveId,
  blockStart,
  demoteMessagesWithStats,
  demotionGate,
  demotionVeto,
  prefillSeconds,
  resolveGateOptions,
  DEFAULT_CMD_KEEP,
  DEFAULT_DEMOTE_BATCH,
  DEFAULT_DEMOTE_CEILING_SECONDS,
  DEFAULT_DEMOTE_MIN_TOKENS_PER_SECOND,
  DEFAULT_DEMOTE_NEAR_COMPACT_TOKENS,
  DEFAULT_PREFILL_LINEAR_SECONDS,
  DEFAULT_PREFILL_QUADRATIC_SECONDS,
  DEFAULT_PREFIX_BLOCK_TOKENS,
  DEFAULT_DEMOTE_FORCE_PENDING_BYTES,
  DEFAULT_DEMOTE_MIN_SAVE_RATIO,
  DEFAULT_DEMOTE_OPEN_AT_PERCENT,
  DEFAULT_DEMOTE_PENDING_BYTES,
  DEFAULT_KEEP_RESULT_HEAD,
  DEFAULT_KEEP_RESULT_TAIL,
  DEFAULT_MIN_PAIR_BYTES,
  DEFAULT_RETAIN_RAW,
  DEFAULT_STALE_DISTANCE,
  MAX_GATE_WINDOW,
  MIN_GATE_WINDOW,
  ENV_DEMOTE_CEILING_SECONDS,
  ENV_DEMOTE_FORCE_PENDING_BYTES,
  ENV_DEMOTE_MIN_TOKENS_PER_SECOND,
  ENV_DEMOTE_NEAR_COMPACT_TOKENS,
  ENV_PREFILL_LINEAR_SECONDS,
  ENV_PREFILL_QUADRATIC_SECONDS,
  ENV_PREFIX_BLOCK_TOKENS,
  ENV_DEMOTE_MIN_SAVE_RATIO,
  ENV_DEMOTE_OPEN_AT_PERCENT,
  RESULT_DEMOTED_PREFIX,
  type GateOptions,
  type RetentionArchive,
  type RetentionOptions,
} from "./retention.ts";

const FOOTER = "[exit=0 cwd=/app timed_out=false backend=harbor-env]";

const OPTS: RetentionOptions = {
  retainRaw: DEFAULT_RETAIN_RAW,
  minPairBytes: DEFAULT_MIN_PAIR_BYTES,
  staleDistance: DEFAULT_STALE_DISTANCE,
  keepResultHeadBytes: DEFAULT_KEEP_RESULT_HEAD,
  keepResultTailBytes: DEFAULT_KEEP_RESULT_TAIL,
  cmdKeepBytes: DEFAULT_CMD_KEEP,
  demoteBatch: DEFAULT_DEMOTE_BATCH,
  demotePendingBytes: DEFAULT_DEMOTE_PENDING_BYTES,
};

const GATE: GateOptions = {
  minSaveRatio: DEFAULT_DEMOTE_MIN_SAVE_RATIO,
  openAtPercent: DEFAULT_DEMOTE_OPEN_AT_PERCENT,
  forcePendingBytes: DEFAULT_DEMOTE_FORCE_PENDING_BYTES,
  ceilingSeconds: DEFAULT_DEMOTE_CEILING_SECONDS,
  minTokensPerSecond: DEFAULT_DEMOTE_MIN_TOKENS_PER_SECOND,
  nearCompactTokens: DEFAULT_DEMOTE_NEAR_COMPACT_TOKENS,
  compactAtPercent: 80,
  prefillLinearSeconds: DEFAULT_PREFILL_LINEAR_SECONDS,
  prefillQuadraticSeconds: DEFAULT_PREFILL_QUADRATIC_SECONDS,
  prefixBlockTokens: DEFAULT_PREFIX_BLOCK_TOKENS,
};
/** #83's gate: the clauses without the re-prefill ceiling, for tests of the clauses themselves. */
const NO_CEILING: GateOptions = { ...GATE, ceilingSeconds: 0 };
/** The context clause is off by default; these tests turn it on where they exercise it. */
const WITH_CONTEXT: GateOptions = { ...NO_CEILING, openAtPercent: 75 };

const WINDOW = 262144;

function memArchive(): RetentionArchive & { entries: Map<string, string> } {
  const entries = new Map<string, string>();
  return {
    entries,
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

function call(id: string, command: string) {
  return { role: "assistant", content: [{ type: "toolCall", id, name: "ShellSession", arguments: { command } }] };
}

function result(id: string, text: string) {
  return {
    role: "toolResult",
    toolCallId: id,
    toolName: "ShellSession",
    content: [{ type: "text", text: `${text}\n${FOOTER}` }],
    isError: false,
  };
}

/** `n` big pairs, then `tailChars` of plain conversation: the turn-122 shape when the tail is large. */
function history(n: number, pairBytes: number, tailChars: number, tag = "p"): any[] {
  const msgs: any[] = [{ role: "user", content: "go" }];
  for (let i = 0; i < n; i++) msgs.push(call(`${tag}${i}`, `echo ${i}`), result(`${tag}${i}`, body(pairBytes, `${tag}${i}`)));
  // Thinking-heavy turns, as with preserve_thinking: they grow the context but hold no shell output.
  const chunk = 40_000; // few messages, so no pair passes staleDistance
  for (let c = 0; c < tailChars; c += chunk) {
    msgs.push({ role: "assistant", content: [{ type: "thinking", thinking: "t".repeat(chunk) }, { type: "text", text: "ok" }] });
    msgs.push({ role: "user", content: "continue" });
  }
  return msgs;
}

function demotedIds(messages: any[]): string[] {
  return messages
    .filter((m) => m.role === "toolResult" && m.content[0].text.startsWith(RESULT_DEMOTED_PREFIX))
    .map((m) => m.toolCallId);
}

describe("demotionGate", () => {
  const base = { estSaveTokens: 9289, estReprefillTokens: 138377, contextTokens: 184059, contextWindow: WINDOW, pendingBytes: 35418, estReprefillSeconds: 0 };

  it("defers the turn-122 jump: small saving, deep break, context under the open point", () => {
    expect(demotionGate(base, GATE)).toBeNull();
  });

  it("opens on the save ratio", () => {
    expect(demotionGate({ ...base, estSaveTokens: DEFAULT_DEMOTE_MIN_SAVE_RATIO * base.estReprefillTokens }, GATE)).toBe("ratio");
    expect(demotionGate({ ...base, estSaveTokens: DEFAULT_DEMOTE_MIN_SAVE_RATIO * base.estReprefillTokens - 1 }, GATE)).toBeNull();
  });

  it("opens once the context reaches openAtPercent of the window", () => {
    const at = Math.ceil(0.75 * WINDOW);
    expect(demotionGate({ ...base, contextTokens: at }, WITH_CONTEXT)).toBe("context");
    expect(demotionGate({ ...base, contextTokens: at - 1 }, WITH_CONTEXT)).toBeNull();
    // Off by default: near compaction a deep break only precedes the compaction's own re-prefill.
    expect(demotionGate({ ...base, contextTokens: WINDOW }, GATE)).toBeNull();
  });

  it("opens on pending raw bytes", () => {
    expect(demotionGate({ ...base, pendingBytes: DEFAULT_DEMOTE_FORCE_PENDING_BYTES }, GATE)).toBe("bytes");
    expect(demotionGate({ ...base, pendingBytes: DEFAULT_DEMOTE_FORCE_PENDING_BYTES - 1 }, GATE)).toBeNull();
  });

  it("treats a non-positive ratio or byte budget as disabled, and openAtPercent >= 100 as never by context", () => {
    const off: GateOptions = { ...GATE, minSaveRatio: 0, openAtPercent: 100, forcePendingBytes: 0 };
    expect(demotionGate({ ...base, estSaveTokens: 1e9, contextTokens: WINDOW, pendingBytes: 1e9 }, off)).toBeNull();
  });

  it("opens when the break re-prefills nothing", () => {
    expect(demotionGate({ ...base, estSaveTokens: 1, estReprefillTokens: 0 }, GATE)).toBe("ratio");
  });
});

describe("resolveGateOptions", () => {
  const names = [
    ENV_DEMOTE_MIN_SAVE_RATIO,
    ENV_DEMOTE_OPEN_AT_PERCENT,
    ENV_DEMOTE_FORCE_PENDING_BYTES,
    ENV_DEMOTE_CEILING_SECONDS,
    ENV_DEMOTE_MIN_TOKENS_PER_SECOND,
    ENV_DEMOTE_NEAR_COMPACT_TOKENS,
    ENV_PREFILL_LINEAR_SECONDS,
    ENV_PREFILL_QUADRATIC_SECONDS,
    ENV_PREFIX_BLOCK_TOKENS,
  ];
  const defaults = {
    minSaveRatio: 0.12,
    openAtPercent: 100,
    forcePendingBytes: 262144,
    ceilingSeconds: 60,
    minTokensPerSecond: 38,
    nearCompactTokens: 40000,
    compactAtPercent: 80,
    prefillLinearSeconds: 1.05e-3,
    prefillQuadraticSeconds: 3.64e-8,
    prefixBlockTokens: 4096,
  };
  const saved = Object.fromEntries(names.map((n) => [n, process.env[n]]));
  afterEach(() => {
    for (const n of names) {
      if (saved[n] === undefined) delete process.env[n];
      else process.env[n] = saved[n];
    }
  });

  it("reads the LITTLE_CODER_SHELL_DEMOTE_* knobs with conservative defaults", () => {
    expect(names).toEqual([
      "LITTLE_CODER_SHELL_DEMOTE_MIN_SAVE_RATIO",
      "LITTLE_CODER_SHELL_DEMOTE_OPEN_AT_PERCENT",
      "LITTLE_CODER_SHELL_DEMOTE_FORCE_PENDING_BYTES",
      "LITTLE_CODER_SHELL_DEMOTE_CEILING_SECONDS",
      "LITTLE_CODER_SHELL_DEMOTE_MIN_TOKENS_PER_SECOND",
      "LITTLE_CODER_SHELL_DEMOTE_NEAR_COMPACT_TOKENS",
      "LITTLE_CODER_SHELL_PREFILL_LINEAR_SECONDS",
      "LITTLE_CODER_SHELL_PREFILL_QUADRATIC_SECONDS",
      "LITTLE_CODER_SHELL_PREFIX_BLOCK_TOKENS",
    ]);
    for (const n of names) delete process.env[n];
    const watchdog = process.env.LITTLE_CODER_COMPACT_AT_PERCENT;
    delete process.env.LITTLE_CODER_COMPACT_AT_PERCENT;
    const noWatchdog = process.env.LITTLE_CODER_NO_COMPACT_WATCHDOG;
    delete process.env.LITTLE_CODER_NO_COMPACT_WATCHDOG;
    try {
      expect(resolveGateOptions()).toEqual(defaults);
      process.env[ENV_DEMOTE_MIN_SAVE_RATIO] = "0.5";
      process.env[ENV_DEMOTE_OPEN_AT_PERCENT] = "0";
      process.env[ENV_DEMOTE_FORCE_PENDING_BYTES] = "-1";
      process.env[ENV_DEMOTE_CEILING_SECONDS] = "90";
      process.env[ENV_DEMOTE_MIN_TOKENS_PER_SECOND] = "0";
      process.env[ENV_DEMOTE_NEAR_COMPACT_TOKENS] = "30000";
      process.env[ENV_PREFILL_LINEAR_SECONDS] = "1.1e-3";
      process.env[ENV_PREFILL_QUADRATIC_SECONDS] = "4e-8";
      process.env[ENV_PREFIX_BLOCK_TOKENS] = "1";
      process.env.LITTLE_CODER_COMPACT_AT_PERCENT = "70";
      expect(resolveGateOptions()).toEqual({
        minSaveRatio: 0.5,
        openAtPercent: 0,
        forcePendingBytes: -1,
        ceilingSeconds: 90,
        minTokensPerSecond: 0,
        nearCompactTokens: 30000,
        compactAtPercent: 70,
        prefillLinearSeconds: 1.1e-3,
        prefillQuadraticSeconds: 4e-8,
        prefixBlockTokens: 1,
      });
      // With the watchdog off, the RPC harness's trigger is the one to measure from.
      process.env.LITTLE_CODER_NO_COMPACT_WATCHDOG = "1";
      expect(resolveGateOptions().compactAtPercent).toBeNull();
      process.env.LITTLE_CODER_NO_COMPACT_WATCHDOG = "0";
      process.env.LITTLE_CODER_COMPACT_AT_PERCENT = "0";
      expect(resolveGateOptions().compactAtPercent).toBeNull();
    } finally {
      if (watchdog === undefined) delete process.env.LITTLE_CODER_COMPACT_AT_PERCENT;
      else process.env.LITTLE_CODER_COMPACT_AT_PERCENT = watchdog;
      if (noWatchdog === undefined) delete process.env.LITTLE_CODER_NO_COMPACT_WATCHDOG;
      else process.env.LITTLE_CODER_NO_COMPACT_WATCHDOG = noWatchdog;
    }
  });
});

describe("demoteMessagesWithStats with the cost gate", () => {
  const B = DEFAULT_DEMOTE_BATCH;
  const R = DEFAULT_RETAIN_RAW;
  // B + R big pairs: the oldest B are a due batch. 4KB each saves ~3.3KB, while
  // the 400K chars of thinking after them is the re-prefill the break would cost.
  const deep = () => history(B + R, 4096, 400_000);

  it("defers a due batch whose saving is small next to the re-prefill it forces", () => {
    const msgs = deep();
    const out = demoteMessagesWithStats(msgs, memArchive(), OPTS, {
      options: GATE,
      context: { contextTokens: 120_000, contextWindow: WINDOW },
    });
    expect(out.demotedCount).toBe(0);
    expect(out.messages).toEqual(msgs);
    expect(out.stats).toMatchObject({ due: B, prefix: 0, demoted: 0, gate: "deferred", gateReason: null, skippedCost: B });
    expect(out.stats.estReprefillTokens).toBeGreaterThan(100_000);
    expect(out.stats.estSaveTokens).toBeGreaterThan(0);
    expect(out.stats.estSaveTokens).toBeLessThan(DEFAULT_DEMOTE_MIN_SAVE_RATIO * out.stats.estReprefillTokens);
    expect(out.stats.estContextTokens).toBe(120_000);
  });

  it("lets the same batch through near the compaction point", () => {
    const out = demoteMessagesWithStats(deep(), memArchive(), OPTS, {
      options: WITH_CONTEXT,
      context: { contextTokens: 0.8 * WINDOW, contextWindow: WINDOW },
    });
    expect(out.demotedCount).toBe(B);
    expect(out.stats).toMatchObject({ prefix: B, demoted: B, gate: "open", gateReason: "context", skippedCost: 0 });
  });

  it("lets a batch through when its saving is large next to a shallow break", () => {
    const shallow = history(B + R, 4096, 0);
    const out = demoteMessagesWithStats(shallow, memArchive(), OPTS, {
      options: GATE,
      context: { contextTokens: 20_000, contextWindow: WINDOW },
    });
    expect(out.demotedCount).toBe(B);
    expect(out.stats).toMatchObject({ gate: "open", gateReason: "ratio" });
  });

  it("keeps an approved batch demoted on the next request even when the gate would now defer it", () => {
    const archive = memArchive();
    const msgs = deep();
    const first = demoteMessagesWithStats(msgs, archive, OPTS, {
      options: WITH_CONTEXT,
      context: { contextTokens: 0.8 * WINDOW, contextWindow: WINDOW },
    });
    expect(first.demotedCount).toBe(B);

    // Next request: the default gate (no context clause) would now defer this
    // jump, and the history grew. Without the latch the
    // pairs would go back to raw and break the prefix a second time.
    const next = [...msgs, { role: "user", content: "more" }];
    const second = demoteMessagesWithStats(next, archive, OPTS, {
      options: GATE,
      context: { contextTokens: 0.7 * WINDOW, contextWindow: WINDOW },
    });
    expect(second.demotedCount).toBe(B);
    expect(demotedIds(second.messages)).toEqual(demotedIds(first.messages));
    expect(second.messages.slice(0, msgs.length)).toEqual(first.messages);
    expect(second.stats).toMatchObject({ prefix: B, gate: "none", skippedCost: 0 });
  });

  it("gates only the new part of the prefix once earlier pairs are latched", () => {
    const archive = memArchive();
    const msgs = deep();
    demoteMessagesWithStats(msgs, archive, OPTS, {
      options: WITH_CONTEXT,
      context: { contextTokens: 0.8 * WINDOW, contextWindow: WINDOW },
    });
    // B more big pairs make a second batch due; the default gate defers it,
    // and the first batch stays exactly as it was.
    const grown = [...msgs, ...history(B, 4096, 0, "q").slice(1)];
    const out = demoteMessagesWithStats(grown, archive, OPTS, {
      options: GATE,
      context: { contextTokens: 0.7 * WINDOW, contextWindow: WINDOW },
    });
    expect(out.stats).toMatchObject({ due: 2 * B, prefix: B, demoted: B, gate: "deferred", skippedCost: B });
    expect(demotedIds(out.messages)).toEqual(Array.from({ length: B }, (_, i) => `p${i}`));
  });

  it("is off, and matches today's projection exactly, without a context window or with openAtPercent <= 0", () => {
    const msgs = deep();
    const today = demoteMessagesWithStats(msgs, memArchive(), OPTS);
    expect(today.demotedCount).toBe(B);
    expect(today.stats).toMatchObject({ gate: "off", skippedCost: 0 });

    const noWindow = demoteMessagesWithStats(msgs, memArchive(), OPTS, {
      options: GATE,
      context: { contextTokens: 120_000, contextWindow: null },
    });
    expect(noWindow.messages).toEqual(today.messages);
    expect(noWindow.stats.gate).toBe("off");

    const disabled = demoteMessagesWithStats(msgs, memArchive(), OPTS, {
      options: { ...GATE, openAtPercent: 0 },
      context: { contextTokens: 120_000, contextWindow: WINDOW },
    });
    expect(disabled.messages).toEqual(today.messages);
    expect(disabled.stats.gate).toBe("off");
  });

  it("opens on a cold cache: pi has no usage between a compaction and the next response", () => {
    const out = demoteMessagesWithStats(deep(), memArchive(), OPTS, {
      options: GATE,
      context: { contextTokens: null, contextWindow: WINDOW },
    });
    expect(out.demotedCount).toBe(B);
    expect(out.stats).toMatchObject({ gate: "open", gateReason: "cold" });
    // ~430K chars at 4 chars/token, from the messages themselves.
    expect(out.stats.estContextTokens).toBeGreaterThan(100_000);
  });

  it("treats the first request after a model switch as cold: the new server holds no prefix", () => {
    const msgs = deep();
    for (const m of msgs) if (m.role === "assistant") Object.assign(m, { provider: "omlx", model: "old-model" });
    const ctx = { contextTokens: 120_000, contextWindow: WINDOW };
    const same = demoteMessagesWithStats(msgs, memArchive(), OPTS, {
      options: GATE,
      context: { ...ctx, model: { provider: "omlx", id: "old-model" } },
    });
    expect(same.stats.gate).toBe("deferred");
    const switched = demoteMessagesWithStats(msgs, memArchive(), OPTS, {
      options: GATE,
      context: { ...ctx, model: { provider: "omlx", id: "new-model" } },
    });
    expect(switched.stats).toMatchObject({ gate: "open", gateReason: "cold" });
  });

  it("stays off above MAX_GATE_WINDOW, past the windows its defaults were fitted on", () => {
    const out = demoteMessagesWithStats(deep(), memArchive(), OPTS, {
      options: GATE,
      context: { contextTokens: 120_000, contextWindow: MAX_GATE_WINDOW + 1 },
    });
    expect(out.stats.gate).toBe("off");
    expect(out.demotedCount).toBe(B);
  });

  it("opens below a lowered context-watchdog threshold, never at or past it", () => {
    const names = ["LITTLE_CODER_COMPACT_AT_PERCENT", ENV_DEMOTE_OPEN_AT_PERCENT];
    const prev = names.map((n) => process.env[n]);
    try {
      process.env.LITTLE_CODER_COMPACT_AT_PERCENT = "70";
      delete process.env[ENV_DEMOTE_OPEN_AT_PERCENT];
      expect(resolveGateOptions().openAtPercent).toBe(100); // the clause is off; nothing to lower
      process.env[ENV_DEMOTE_OPEN_AT_PERCENT] = "75";
      expect(resolveGateOptions().openAtPercent).toBe(65);
      process.env.LITTLE_CODER_COMPACT_AT_PERCENT = "90";
      expect(resolveGateOptions().openAtPercent).toBe(75);
      process.env.LITTLE_CODER_COMPACT_AT_PERCENT = "0";
      expect(resolveGateOptions().openAtPercent).toBe(75);
      // Unset means the watchdog's default 80, so an opted-in 78 opens at 75.
      delete process.env.LITTLE_CODER_COMPACT_AT_PERCENT;
      process.env[ENV_DEMOTE_OPEN_AT_PERCENT] = "78";
      expect(resolveGateOptions().openAtPercent).toBe(75);
    } finally {
      names.forEach((n, i) => {
        if (prev[i] === undefined) delete process.env[n];
        else process.env[n] = prev[i];
      });
    }
  });

  it("stays off below MIN_GATE_WINDOW, where its defaults were never fitted", () => {
    const msgs = deep();
    const out = demoteMessagesWithStats(msgs, memArchive(), OPTS, {
      options: GATE,
      context: { contextTokens: 20_000, contextWindow: MIN_GATE_WINDOW - 1 },
    });
    expect(out.stats.gate).toBe("off");
    expect(out.demotedCount).toBe(B);
  });

  it("does not count history the server has not cached yet as re-prefill", () => {
    // The same due batch with the long tail arriving as new tool output after the
    // last response: the server prefills it on this request either way.
    const msgs: any[] = history(B + R, 4096, 0);
    msgs.push(call("late", "cat big"), result("late", "y".repeat(400_000)));
    const atEnd = demoteMessagesWithStats(msgs.slice(0, -1).concat([{ role: "user", content: "z".repeat(400_000) }]), memArchive(), OPTS, {
      options: GATE,
      context: { contextTokens: 50_000, contextWindow: WINDOW },
    });
    expect(atEnd.stats.estReprefillTokens).toBeLessThan(20_000);
    expect(atEnd.stats).toMatchObject({ gate: "open", gateReason: "ratio" });
  });

  it("never lets the latch reach past today's candidate (compaction, a changed knob, a reused id)", () => {
    const archive = memArchive();
    // Every pair archived by an earlier projection, but only the oldest B are due now.
    const msgs = history(B + R, 4096, 400_000);
    for (let i = 0; i < B + R; i++) archive.save(archiveId(`p${i}`), "x");
    const out = demoteMessagesWithStats(msgs, archive, OPTS, {
      options: GATE,
      context: { contextTokens: 120_000, contextWindow: WINDOW },
    });
    expect(out.stats).toMatchObject({ sticky: B, prefix: B, gate: "none" });
    expect(demotedIds(out.messages)).toEqual(Array.from({ length: B }, (_, i) => `p${i}`));
  });

  it("defers a pending-bytes flush like any other jump", () => {
    // Under the force budget but over the 64 KiB flush: the flush makes the jump
    // due, the gate still judges it.
    const msgs: any[] = [{ role: "user", content: "go" }];
    msgs.push(call("big", `cat > /tmp/a <<'EOF'\n${body(70 * 1024, "src")}\nEOF`), result("big", "ok"));
    for (let i = 0; i < 30; i++) msgs.push({ role: "user", content: `u${i}` }, { role: "assistant", content: [{ type: "thinking", thinking: "t".repeat(40_000) }] });
    const out = demoteMessagesWithStats(msgs, memArchive(), OPTS, {
      options: GATE,
      context: { contextTokens: 150_000, contextWindow: WINDOW },
    });
    expect(out.stats).toMatchObject({ due: 1, flushed: true, gate: "deferred", skippedCost: 1, prefix: 0 });
    // The break would start inside the call message, at its toolCall block:
    // the first block of a prompt the context reading puts at 150K tokens.
    expect(out.stats.estReprefillTokens).toBe(150_000);
  });

  it("counts bashExecution and summary messages in the re-prefill, as pi's estimator does", () => {
    const msgs: any[] = [{ role: "user", content: "go" }];
    for (let i = 0; i < B + R; i++) msgs.push(call(`p${i}`, `echo ${i}`), result(`p${i}`, body(4096, `p${i}`)));
    msgs.push(
      { role: "bashExecution", command: "cat log", output: "o".repeat(200_000), exitCode: 0 },
      { role: "compactionSummary", summary: "s".repeat(200_000), tokensBefore: 1 },
      { role: "assistant", content: [{ type: "text", text: "ok" }] },
    );
    const out = demoteMessagesWithStats(msgs, memArchive(), OPTS, {
      options: GATE,
      context: { contextTokens: 120_000, contextWindow: WINDOW },
    });
    expect(out.stats.estReprefillTokens).toBeGreaterThan(100_000);
    expect(out.stats.gate).toBe("deferred");
  });

  it("ignores pairs that rewrite nothing when placing the break", () => {
    // A signed heredoc whose result is tiny cannot shrink; the break is the next pair's result.
    const msgs: any[] = [{ role: "user", content: "go" }];
    msgs.push(
      { role: "assistant", content: [{ type: "toolCall", id: "s0", name: "ShellSession", arguments: { command: body(5000, "h") }, thoughtSignature: "sig" }] },
      result("s0", "ok"),
    );
    msgs.push({ role: "assistant", content: [{ type: "thinking", thinking: "t".repeat(200_000) }] });
    for (let i = 1; i < B + R; i++) msgs.push(call(`p${i}`, `echo ${i}`), result(`p${i}`, body(4096, `p${i}`)));
    const out = demoteMessagesWithStats(msgs, memArchive(), OPTS, {
      options: GATE,
      context: { contextTokens: 60_000, contextWindow: WINDOW },
    });
    expect(out.stats.estReprefillTokens).toBeLessThan(20_000);
    expect(out.stats).toMatchObject({ gate: "open", skippedNoShrink: 1, signed: 1 });
  });

  it("archives nothing for a deferred pair, so the latch cannot claim it", () => {
    const archive = memArchive();
    demoteMessagesWithStats(deep(), archive, OPTS, {
      options: GATE,
      context: { contextTokens: 120_000, contextWindow: WINDOW },
    });
    expect(archive.entries.has(archiveId("p0"))).toBe(false);
  });
});

describe("re-prefill ceiling", () => {
  it("prices prefill as a(p-c) + b(p²-c²)/2 seconds", () => {
    expect(prefillSeconds(0, 200_000, GATE)).toBeCloseTo(1.05e-3 * 200_000 + (3.64e-8 * 200_000 ** 2) / 2, 6);
    // ~120 tok/s at 200K against ~950 at 0: depth, not length, dominates.
    expect(prefillSeconds(190_000, 200_000, GATE)).toBeGreaterThan(6 * prefillSeconds(0, 10_000, GATE));
    expect(prefillSeconds(5, 5, GATE)).toBe(0);
    expect(prefillSeconds(6, 5, GATE)).toBe(0);
  });

  it("re-prefills from the start of the 4096-token block holding the divergence", () => {
    // make-doom-for-mips turn 88: diverged at token 12,250, omlx reused 8,192.
    expect(blockStart(12_250, 4096)).toBe(8192);
    expect(blockStart(8192, 4096)).toBe(8192);
    expect(blockStart(-3, 4096)).toBe(0);
    expect(blockStart(12_250.5, 1)).toBe(12_250.5);
    expect(blockStart(12_250, 0)).toBe(12_250);
  });

  // Live 2026-10-08 (job 2026-10-08__09-29-28): each turn's shell_retention
  // estimates, re-anchored as the gate now places them (end of prompt = the
  // context reading, break block-aligned).
  function live(S: number, R: number, ctx: number) {
    const from = blockStart(ctx - R, 4096);
    return {
      estSaveTokens: S,
      estReprefillTokens: ctx - from,
      contextTokens: ctx,
      contextWindow: WINDOW,
      pendingBytes: 0,
      estReprefillSeconds: prefillSeconds(from, ctx, GATE),
    };
  }
  const deferred: Array<[string, number, number, number, string]> = [
    ["gcode-to-text 97 (493 s live)", 17371, 113469, 163953, "ceiling"],
    ["gcode-to-text 119 (477 s)", 15505, 95155, 204965, "near"],
    ["gcode-to-text 122 (321 s)", 22132, 70054, 209578, "near"],
    ["gcode-to-text 123 (328 s)", 12196, 51080, 203876, "near"],
    ["gcode-to-text 124 (348 s)", 9389, 45103, 210522, "near"],
    ["make-doom-for-mips 88 (324 s)", 11044, 88796, 124103, "ceiling"],
  ];
  for (const [name, S, R, ctx, why] of deferred) {
    it(`defers ${name}, which the save ratio opened`, () => {
      const input = live(S, R, ctx);
      expect(demotionGate(input, GATE)).toBe("ratio");
      expect(input.estReprefillSeconds).toBeGreaterThan(300);
      expect(demotionVeto(input, GATE)).toBe(why);
    });
  }

  const opened: Array<[string, number, number, number]> = [
    ["make-doom-for-mips 10 (14 s live)", 4290, 9697, 20006],
    ["make-doom-for-mips 17 (21 s)", 4269, 11453, 22280],
    // Over the ceiling (69 s), but 62 tokens saved per second paid.
    ["make-doom-for-mips 43 (75 s)", 4306, 30503, 47521],
    ["build-cython-ext 23 (45 s)", 2909, 20151, 31474],
    ["fix-ocaml-gc 26 (58 s)", 3426, 25478, 38179],
  ];
  for (const [name, S, R, ctx] of opened) {
    it(`still opens ${name}`, () => {
      const input = live(S, R, ctx);
      expect(demotionGate(input, GATE)).toBe("ratio");
      expect(demotionVeto(input, GATE)).toBeNull();
    });
  }

  it("lets an over-ceiling jump through when it saves enough per second, away from compaction", () => {
    // The #83 replay's raman-fitting turn 30: 470 s at 137K for 19.8K tokens kept the trial under compaction.
    const input = { ...live(19_781, 124_788, 137_076), estReprefillSeconds: 470 };
    expect(demotionVeto(input, GATE)).toBeNull();
    expect(demotionVeto({ ...input, estSaveTokens: 38 * 470 - 1 }, GATE)).toBe("ceiling");
    // The near clause is what holds gcode-to-text 122 (43 tokens per second).
    expect(demotionVeto(live(22132, 70054, 209578), { ...GATE, nearCompactTokens: 0 })).toBeNull();
  });

  it("measures 'near' from the compaction percent of the window", () => {
    const at = 0.8 * WINDOW - DEFAULT_DEMOTE_NEAR_COMPACT_TOKENS;
    const over = { ...live(1, 50_000, at), estReprefillSeconds: 61 };
    expect(demotionVeto(over, GATE)).toBe("near");
    expect(demotionVeto({ ...over, contextTokens: at - 1 }, GATE)).toBe("ceiling");
    expect(demotionVeto({ ...over, contextTokens: at - 1, estSaveTokens: 1e9 }, GATE)).toBeNull();
    // Watchdog off: the RPC harness compacts at min(220K, 84% of the window).
    const piAt = 220_000 - DEFAULT_DEMOTE_NEAR_COMPACT_TOKENS;
    expect(demotionVeto({ ...over, contextWindow: 200_000, contextTokens: 168_000 - DEFAULT_DEMOTE_NEAR_COMPACT_TOKENS, estSaveTokens: 1e9 }, { ...GATE, compactAtPercent: null })).toBe("near");
    expect(demotionVeto({ ...over, contextTokens: piAt - 1, estSaveTokens: 1e9 }, { ...GATE, compactAtPercent: null })).toBeNull();
    expect(demotionVeto({ ...over, contextTokens: piAt, estSaveTokens: 1e9 }, { ...GATE, compactAtPercent: null })).toBe("near");
  });

  it("treats non-positive knobs as off: no ceiling, no per-second override, no near clause", () => {
    const t119 = live(15505, 95155, 204965);
    expect(demotionVeto(t119, { ...GATE, ceilingSeconds: 0 })).toBeNull();
    expect(demotionVeto(live(4306, 30503, 47521), { ...GATE, minTokensPerSecond: 0 })).toBe("ceiling");
    expect(demotionVeto({ ...t119, estSaveTokens: 1e9 }, { ...GATE, nearCompactTokens: -1 })).toBeNull();
    expect(demotionVeto({ ...t119, estReprefillSeconds: DEFAULT_DEMOTE_CEILING_SECONDS }, GATE)).toBeNull();
  });
});

describe("demoteMessagesWithStats with the re-prefill ceiling", () => {
  const B = DEFAULT_DEMOTE_BATCH;
  const R = DEFAULT_RETAIN_RAW;
  // A due batch under ~107K tokens of later history: deep, at ~200K context.
  const deep = () => history(B + R, 4096, 400_000);

  it("defers a jump the ratio opens when its re-prefill is long and compaction is near", () => {
    const msgs = deep();
    const opts = { ...GATE, minSaveRatio: 0.01 };
    const out = demoteMessagesWithStats(msgs, memArchive(), OPTS, {
      options: opts,
      context: { contextTokens: 200_000, contextWindow: WINDOW },
    });
    expect(out.demotedCount).toBe(0);
    expect(out.messages).toEqual(msgs);
    expect(out.stats).toMatchObject({ gate: "deferred", gateReason: "near", skippedCost: B, prefix: 0, ceilingSeconds: 60 });
    expect(out.stats.estReprefillSeconds).toBeGreaterThan(300);
    // The same jump on #83's gate.
    const before = demoteMessagesWithStats(msgs, memArchive(), OPTS, {
      options: { ...opts, ceilingSeconds: 0 },
      context: { contextTokens: 200_000, contextWindow: WINDOW },
    });
    expect(before.stats).toMatchObject({ gate: "open", gateReason: "ratio", ceilingSeconds: 0 });
  });

  it("places the re-prefill at the start of the divergence's block, counted back from the context reading", () => {
    const out = demoteMessagesWithStats(deep(), memArchive(), OPTS, {
      options: GATE,
      context: { contextTokens: 200_000, contextWindow: WINDOW },
    });
    // The prompt ends in "continue" (8 chars) after the last response.
    const cachedEnd = 200_000 - 8 / 4;
    expect((cachedEnd - out.stats.estReprefillTokens) % 4096).toBe(0);
    expect(out.stats.estReprefillSeconds).toBeCloseTo(prefillSeconds(cachedEnd - out.stats.estReprefillTokens, cachedEnd, GATE), 0);
    const exact = demoteMessagesWithStats(deep(), memArchive(), OPTS, {
      options: { ...GATE, prefixBlockTokens: 1 },
      context: { contextTokens: 200_000, contextWindow: WINDOW },
    });
    expect(exact.stats.estReprefillTokens).toBeLessThan(out.stats.estReprefillTokens);
    expect(out.stats.estReprefillTokens - exact.stats.estReprefillTokens).toBeLessThan(4096);
  });

  it("holds a forced pending-bytes flush too", () => {
    const msgs = deep();
    const out = demoteMessagesWithStats(msgs, memArchive(), OPTS, {
      options: { ...GATE, forcePendingBytes: 1 },
      context: { contextTokens: 200_000, contextWindow: WINDOW },
    });
    expect(out.stats).toMatchObject({ gate: "deferred", gateReason: "near", demoted: 0 });
  });

  it("never holds a cold-cache jump: its break is free", () => {
    const out = demoteMessagesWithStats(deep(), memArchive(), OPTS, {
      options: { ...GATE, minSaveRatio: 0.01 },
      context: { contextTokens: null, contextWindow: WINDOW },
    });
    expect(out.stats).toMatchObject({ gate: "open", gateReason: "cold", demoted: B });
  });

  it("opens a cheap jump at low context", () => {
    const out = demoteMessagesWithStats(history(B + R, 4096, 0), memArchive(), OPTS, {
      options: GATE,
      context: { contextTokens: 20_000, contextWindow: WINDOW },
    });
    expect(out.stats).toMatchObject({ gate: "open", gateReason: "ratio", demoted: B });
    expect(out.stats.estReprefillSeconds).toBeLessThan(DEFAULT_DEMOTE_CEILING_SECONDS);
  });

  it("holds a new due batch near compaction and keeps the latched one exactly as it was", () => {
    const archive = memArchive();
    const msgs = deep();
    const first = demoteMessagesWithStats(msgs, archive, OPTS, {
      options: GATE,
      context: { contextTokens: null, contextWindow: WINDOW },
    });
    expect(first.demotedCount).toBe(B);
    const grown = [...msgs, ...history(B, 4096, 200_000, "q").slice(1)];
    const out = demoteMessagesWithStats(grown, archive, OPTS, {
      options: { ...GATE, minSaveRatio: 0.01 },
      context: { contextTokens: 200_000, contextWindow: WINDOW },
    });
    expect(out.stats).toMatchObject({ due: 2 * B, sticky: B, prefix: B, demoted: B, gate: "deferred", gateReason: "near", skippedCost: B });
    expect(out.messages.slice(0, msgs.length)).toEqual(first.messages);
  });

  it("places the break the same whether earlier pairs are latched or already demoted in storage", () => {
    // pi's context reading counts the latched pairs demoted; pristine chars do
    // not. Counting back from the end keeps them out of the estimate.
    const archive = memArchive();
    const msgs = deep();
    const first = demoteMessagesWithStats(msgs, archive, OPTS, {
      options: GATE,
      context: { contextTokens: null, contextWindow: WINDOW },
    });
    const tail = history(B, 4096, 200_000, "q").slice(1);
    const ctx = { contextTokens: 200_000, contextWindow: WINDOW };
    const latched = demoteMessagesWithStats([...msgs, ...tail], archive, OPTS, { options: GATE, context: ctx });
    const stored = demoteMessagesWithStats([...first.messages, ...tail], memArchive(), OPTS, { options: GATE, context: ctx });
    expect(latched.stats.sticky).toBe(B);
    expect(stored.stats.sticky).toBe(B);
    expect(latched.stats.estReprefillTokens).toBeGreaterThan(0);
    expect(latched.stats.estReprefillTokens).toBe(stored.stats.estReprefillTokens);
    expect(latched.stats.estReprefillFromToken).toBe(stored.stats.estReprefillFromToken);
  });

  it("lets a pending-bytes flush through under the ceiling", () => {
    const out = demoteMessagesWithStats(history(B + R, 4096, 0), memArchive(), OPTS, {
      options: { ...GATE, minSaveRatio: 0, forcePendingBytes: 1 },
      context: { contextTokens: 20_000, contextWindow: WINDOW },
    });
    expect(out.stats).toMatchObject({ gate: "open", gateReason: "bytes", demoted: B });
  });

  it("keeps a latched prefix demoted when the ceiling would now hold the jump", () => {
    const archive = memArchive();
    const msgs = deep();
    const first = demoteMessagesWithStats(msgs, archive, OPTS, {
      options: { ...GATE, minSaveRatio: 0.01 },
      context: { contextTokens: null, contextWindow: WINDOW },
    });
    expect(first.demotedCount).toBe(B);
    const next = [...msgs, { role: "user", content: "more" }];
    const second = demoteMessagesWithStats(next, archive, OPTS, {
      options: { ...GATE, minSaveRatio: 0.01 },
      context: { contextTokens: 200_000, contextWindow: WINDOW },
    });
    expect(second.demotedCount).toBe(B);
    expect(second.messages.slice(0, msgs.length)).toEqual(first.messages);
  });
});
