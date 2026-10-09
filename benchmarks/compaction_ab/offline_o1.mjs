// Compaction A/B, Stage 1 offline gate O1 (plan §4.3). Uses the GPU via omlx.
//
// For each compaction point in a saved pi session, it rebuilds the request pi
// would have sent last before compacting. It does this the way #84's evidence
// did (pi-ai's real payload builder, with fetch stubbed so nothing is sent),
// then runs each arm's summary path against omlx:
//   warm  the captured request at max_tokens 1, which puts the prefix in cache
//         the way a live trial would have it
//   A1    #84's runReuse (planReuse + send + parse), k = 2
//   A3    pi-prefix-cache-compaction's buildSummaryBody + streamMessages (the
//         vendored code; it runs only after the security sign-off), k = 2
//   A0    pi's own compact() at thinking level "high", k = 1
// Output: one JSON row per request with wall, ttft, text and reasoning
// lengths, finish reason or error, and usage. The omlx log slices (prompt,
// cached, TTFT) are matched afterwards by time.
//
// Usage: node benchmarks/compaction_ab/offline_o1.mjs <session.jsonl> <out.jsonl> [pointIndex ...]
// Point indexes count the omlx-era compaction entries, starting at 0.
import { appendFileSync, readFileSync } from "node:fs";
import { request as httpRequest } from "node:http";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const ROOT = new URL("../../", import.meta.url).pathname;
const [sessionPath, outPath, ...pointArgs] = process.argv.slice(2);
const imp = (p) => import(pathToFileURL(join(ROOT, p)).href);
const pi = await imp("node_modules/@earendil-works/pi-coding-agent/dist/index.js");
const { prepareCompaction, compact } = await imp("node_modules/@earendil-works/pi-coding-agent/dist/core/compaction/compaction.js");
const { stream } = await imp(
  "node_modules/@earendil-works/pi-coding-agent/node_modules/@earendil-works/pi-ai/dist/api/openai-completions.js",
);
const reuse = await imp(".pi/extensions/cache-reuse-compaction/index.ts");
const a3 = await imp("vendor/compaction-ab/pi-prefix-cache-compaction@0.3.0/src/core.ts");

const models = JSON.parse(readFileSync(join(ROOT, "models.json"), "utf-8"));
const prov = models.providers.omlx;
const m = prov.models.find((x) => x.id === "tiel-coder-oq6e-fp16");
const model = { ...m, api: prov.api, provider: "omlx", baseUrl: prov.baseUrl };
const URL_CC = `${model.baseUrl.replace(/\/+$/, "")}/chat/completions`;
const systemPrompt = readFileSync(join(ROOT, "AGENTS.md"), "utf-8");
const tools = ["read", "write", "edit", "bash", "glob", "grep"].map((name) => ({
  name,
  description: `The ${name} tool.`,
  parameters: { type: "object", properties: { path: { type: "string" }, command: { type: "string" } } },
}));

async function payloadFor(messages) {
  let captured;
  const s = stream(
    model,
    { systemPrompt, messages: pi.convertToLlm(messages), tools },
    {
      apiKey: "IGNORED",
      maxTokens: model.maxTokens,
      reasoningEffort: "high",
      onPayload: (p) => {
        captured = p;
        return undefined;
      },
      fetch: async () => {
        throw new Error("offline");
      },
      maxRetries: 0,
    },
  );
  await s.result();
  return { ...captured, temperature: 0.2 };
}

const out = (row) => {
  appendFileSync(outPath, JSON.stringify({ t: Date.now() / 1000, ...row }) + "\n");
  console.log(JSON.stringify(row).slice(0, 400));
};

/** Our own node:http SSE sender (no undici 300 s idle limit), used for the warm-up. */
function send(body) {
  return new Promise((resolve, reject) => {
    const t0 = Date.now();
    let ttft = null;
    let buf = "";
    const req = httpRequest(URL_CC, { method: "POST", headers: { "content-type": "application/json", authorization: "Bearer IGNORED" } }, (res) => {
      res.setEncoding("utf8");
      res.on("data", (c) => {
        if (ttft === null && /"(content|reasoning_content|reasoning)":"[^"]/.test(c)) ttft = (Date.now() - t0) / 1000;
        buf += c;
      });
      res.on("end", () => resolve({ status: res.statusCode, ttft, wall: (Date.now() - t0) / 1000, tail: buf.slice(-400) }));
    });
    req.setTimeout(0);
    req.on("error", reject);
    req.end(JSON.stringify(body));
  });
}

const all = readFileSync(sessionPath, "utf-8").trim().split("\n").map((l) => JSON.parse(l));
const entries = all.slice(1);
const omlxStart = entries.findIndex((e) => e.type === "model_change" && e.provider === "omlx");
const points = entries.map((e, i) => (e.type === "compaction" && i > omlxStart ? i : -1)).filter((i) => i >= 0);
const chosen = pointArgs.length ? pointArgs.map(Number).map((k) => points[k]) : points;

for (const ci of chosen) {
  const branch = entries.slice(0, ci);
  const ctxAll = pi.buildSessionContext(branch).messages;
  let lastAssistant = ctxAll.length - 1;
  while (lastAssistant >= 0 && ctxAll[lastAssistant].role !== "assistant") lastAssistant--;
  const captured = await payloadFor(ctxAll.slice(0, lastAssistant));
  const prep = prepareCompaction(branch, { enabled: true, reserveTokens: 16384, keepRecentTokens: 20000 });
  const point = { point: ci, tokensBefore: prep.tokensBefore, split: prep.isSplitTurn, n_messages: captured.messages.length };
  out({ ...point, arm: "info" });

  // Warm the prefix as a live trial would have it.
  const warm = await send({ ...captured, max_tokens: 1, stream: true });
  out({ ...point, arm: "warm", ...warm, tail: undefined });

  const event = { reason: "manual", willRetry: false, preparation: prep, branchEntries: branch, signal: new AbortController().signal };
  const ctx = {
    model,
    sessionManager: { getSessionId: () => "s" },
    modelRegistry: { getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "IGNORED" }) },
  };
  for (let k = 0; k < 2; k++) {
    const t0 = Date.now();
    const o = await reuse.runReuse(event, ctx, [{ payload: captured, provider: "omlx", modelId: model.id, sessionId: "s", completed: true }], 2048, {
      fetch: globalThis.fetch,
      now: Date.now,
      ttftTimeoutMs: 240_000,
      stallTimeoutMs: 60_000,
      maxOutputTokens: 600 * 20,
      genDeadlineMs: 2 * 600 * 1000,
    });
    out({
      ...point,
      arm: "A1",
      k,
      wall: (Date.now() - t0) / 1000,
      ok: o.ok,
      telemetry: o.telemetry,
      summary_chars: o.ok ? o.compaction.summary.length : null,
      summary: o.ok ? o.compaction.summary : null,
    });
  }

  const budget = a3.summaryTokenBudget(model.contextWindow, prep.tokensBefore, a3.mergeConfig({ warmup: false }));
  for (let k = 0; k < 2; k++) {
    const t0 = Date.now();
    const collector = a3.createCollector("openai-completions");
    try {
      const r = await a3.streamMessages(
        URL_CC,
        a3.requestHeaders("openai-completions", undefined, { apiKey: "IGNORED" }),
        JSON.stringify(a3.buildSummaryBody(captured, budget, "openai-completions")),
        collector,
        new AbortController().signal,
      );
      out({ ...point, arm: "A3", k, budget, wall: (Date.now() - t0) / 1000, ok: true, finish: r.stopReason, usage: r.usage, thinking_chars: collector.thinkingChars, summary_chars: r.text.length, summary: r.text });
    } catch (err) {
      out({ ...point, arm: "A3", k, budget, wall: (Date.now() - t0) / 1000, ok: false, error: String(err?.message ?? err), thinking_chars: collector.thinkingChars, text_chars: collector.text.length, finish: collector.finishReason });
    }
  }

  {
    const t0 = Date.now();
    try {
      const r = await compact(prep, model, "IGNORED", undefined, undefined, new AbortController().signal, "high");
      out({ ...point, arm: "A0", wall: (Date.now() - t0) / 1000, ok: true, usage: r.usage, summary_chars: r.summary.length, summary: r.summary });
    } catch (err) {
      out({ ...point, arm: "A0", wall: (Date.now() - t0) / 1000, ok: false, error: String(err?.message ?? err) });
    }
  }
}
