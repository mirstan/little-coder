// Multi-depth replay, step 4: run every arm's compaction at every cut point.
// Uses the GPU when pointed at omlx; for a dry run, point it at fake_openai.py.
//
// Per point, in ascending depth (points nest: each request is a prefix of
// the next point's request):
//   1. warm   request-N at max_tokens 1, so the prefix is in cache exactly as in a
//             live trial at that turn. Its cost is reported, but no arm is charged.
//   2. A2     blackhole in-process (vendored dist, pinned config). No model call.
//   3. A1, A3 the two reuse arms, k samples each, in seeded random order per
//             point (mulberry32(seed + turn)). Both read the shared warm prefix,
//             as they would live.
//             A1: #84 runReuse with the last two captures [N completed, N+1 aborted]
//             A3: the vendored buildSummaryBody(request N+1) + streamMessages
//   4. A0     pi compact() at the session's thinking level, LAST at each point.
//             Its system prompt gets a per-request nonce line, so its serialized
//             history can never reuse an earlier point's native prompt from cache
//             (native is always cold live). This is the only change to a native
//             request: one short line before the system prompt.
// Each row records wall, ttft, prompt, cached and completion tokens, finish
// reason, and the summary. omlx log lines (Chat completion, Prefix cache
// restore) for the request's window are attached when --omlx-log is set.
//
// Usage: node mdr_run.mjs --points DIR --out FILE [--turns 261,349,...] [--k 2]
//          [--base-url http://127.0.0.1:8000/v1] [--omlx-log /opt/homebrew/var/log/omlx.log|none]
//          [--seed 20261009] [--arms A2,A1,A3,A0] [--agent-dir DIR]
import { randomUUID } from "node:crypto";
import { appendFileSync, closeSync, existsSync, mkdirSync, openSync, readFileSync, readSync, statSync, writeFileSync } from "node:fs";
import { request as httpRequest } from "node:http";
import { homedir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, v, i, a) => (v.startsWith("--") ? [...acc, [v.slice(2), a[i + 1]]] : acc), []),
);
const ROOT = new URL("../../", import.meta.url).pathname;
const POINTS = args.points;
const OUT = args.out;
const K = Number(args.k ?? 2);
const SEED = Number(args.seed ?? 20261009);
const ARMS = (args.arms ?? "A2,A1,A3,A0").split(",");
const OMLX_LOG = args["omlx-log"] ?? "/opt/homebrew/var/log/omlx.log";
const imp = (p) => import(pathToFileURL(join(ROOT, p)).href);

// Blackhole reads its config from the agent dir; the pinned copy is placed there (never ~/.pi).
const agentDir = args["agent-dir"] ?? join(ROOT, ".cache", "mdr-agent");
mkdirSync(join(agentDir, "pi-blackhole"), { recursive: true });
writeFileSync(
  join(agentDir, "pi-blackhole", "pi-blackhole-config.json"),
  readFileSync(join(ROOT, "vendor/compaction-ab/config/pi-blackhole-config.json")),
);
process.env.PI_CODING_AGENT_DIR = agentDir;
process.env.GIT_OPTIONAL_LOCKS = "0";
for (const k of Object.keys(process.env)) if (k.startsWith("PI_BLACKHOLE_")) delete process.env[k];

const { prepareCompaction, compact } = await imp("node_modules/@earendil-works/pi-coding-agent/dist/core/compaction/compaction.js");
const { streamSimple } = await imp("node_modules/@earendil-works/pi-coding-agent/node_modules/@earendil-works/pi-ai/dist/compat.js");
const reuse = await imp(".pi/extensions/cache-reuse-compaction/index.ts");
const a3 = await imp("vendor/compaction-ab/pi-prefix-cache-compaction@0.3.0/src/core.ts");

const realModels = JSON.parse(readFileSync(join(homedir(), ".config/little-coder/models.json"), "utf-8"));
const prov = realModels.providers.omlx;
const mdef = prov.models.find((x) => x.id === "tiel-coder-oq6e-fp16");
const BASE = (args["base-url"] ?? prov.baseUrl).replace(/\/+$/, "");
const model = { ...mdef, api: prov.api, provider: "omlx", baseUrl: BASE };
const URL_CC = `${BASE}/chat/completions`;

function mulberry32(a) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const logOffset = () => (OMLX_LOG !== "none" && existsSync(OMLX_LOG) ? statSync(OMLX_LOG).size : null);
function logSlice(from) {
  if (from === null) return null;
  const size = statSync(OMLX_LOG).size;
  const fd = openSync(OMLX_LOG, "r");
  const buf = Buffer.alloc(size - from);
  readSync(fd, buf, 0, buf.length, from);
  closeSync(fd);
  return buf
    .toString("utf-8")
    .split("\n")
    .filter((l) => /Chat completion:|Prefix cache restore|re-prefills|HTTP 507|ERROR/.test(l))
    .map((l) => l.replace(/^.*? - omlx\.\w+ - \w+ - /, ""));
}

const out = (row) => {
  appendFileSync(OUT, JSON.stringify({ t: Date.now() / 1000, ...row }) + "\n");
  const { summary, ...brief } = row;
  console.log(JSON.stringify(brief).slice(0, 420));
};

/** node:http SSE request (no undici idle limit). Returns ttft, wall, text, reasoning length, finish, usage. */
function sendSSE(body, signal) {
  return new Promise((resolve, reject) => {
    const t0 = Date.now();
    let ttft = null;
    let buf = "";
    const acc = { text: "", reasoning: 0, finish: null, usage: null, toolCalls: 0 };
    const req = httpRequest(URL_CC, { method: "POST", headers: { "content-type": "application/json", authorization: "Bearer IGNORED" } }, (res) => {
      res.setEncoding("utf8");
      res.on("data", (chunk) => {
        buf += chunk;
        let i;
        while ((i = buf.indexOf("\n\n")) >= 0) {
          const frame = buf.slice(0, i);
          buf = buf.slice(i + 2);
          for (const line of frame.split("\n")) {
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim();
            if (!data || data === "[DONE]") continue;
            let ev;
            try {
              ev = JSON.parse(data);
            } catch {
              continue;
            }
            if (ev.usage) acc.usage = ev.usage;
            const ch = ev.choices?.[0];
            if (!ch) continue;
            const d = ch.delta ?? {};
            if (ttft === null && (d.content || d.reasoning_content || d.reasoning)) ttft = (Date.now() - t0) / 1000;
            if (typeof d.content === "string") acc.text += d.content;
            if (typeof d.reasoning_content === "string") acc.reasoning += d.reasoning_content.length;
            if (Array.isArray(d.tool_calls)) acc.toolCalls += d.tool_calls.length;
            if (ch.finish_reason) acc.finish = ch.finish_reason;
          }
        }
      });
      res.on("end", () => resolve({ status: res.statusCode, ttft, wall: (Date.now() - t0) / 1000, ...acc }));
      res.on("error", reject);
    });
    req.setTimeout(0);
    req.on("error", reject);
    signal?.addEventListener("abort", () => req.destroy(new Error("aborted")), { once: true });
    req.end(JSON.stringify(body));
  });
}

/**
 * Blackhole loaded the way a live A2 trial loads it: pi's own extension loader
 * (jiti plus pi's module aliases) on the real .pi/extensions/zz-arm-blackhole shim,
 * with LC_COMPACTION_ARM=blackhole. Returns the loaded extension's handler map.
 */
async function loadBlackhole() {
  const { loadExtensions } = await imp("node_modules/@earendil-works/pi-coding-agent/dist/core/extensions/loader.js");
  const prev = process.env.LC_COMPACTION_ARM;
  process.env.LC_COMPACTION_ARM = "blackhole";
  const shim = join(ROOT, ".pi/extensions/zz-arm-blackhole/index.ts");
  const { extensions, errors } = await loadExtensions([shim], ROOT);
  if (prev === undefined) delete process.env.LC_COMPACTION_ARM;
  else process.env.LC_COMPACTION_ARM = prev;
  if (errors.length) throw new Error(`blackhole load failed: ${JSON.stringify(errors)}`);
  const handlers = extensions[0]?.handlers;
  if (!handlers?.get("session_before_compact")?.length) throw new Error("blackhole registered no session_before_compact handler");
  return handlers;
}

const points = (args.turns ? args.turns.split(",") : JSON.parse(readFileSync(join(POINTS, "build-report.json"))).points.map((p) => p.turn))
  .map(Number)
  .sort((a, b) => a - b);
const meta = Object.fromEntries(JSON.parse(readFileSync(join(POINTS, "build-report.json"))).points.map((p) => [p.turn, p]));
const bh = ARMS.includes("A2") ? await loadBlackhole() : null;
mkdirSync(join(POINTS, "summaries"), { recursive: true });

for (const turn of points) {
  const dir = join(POINTS, `turn-${turn}`);
  const reqN = JSON.parse(readFileSync(join(dir, "request-N.json")));
  const reqN1 = existsSync(join(dir, "request-N1.json")) ? JSON.parse(readFileSync(join(dir, "request-N1.json"))) : null;
  const { header, entries: branch } = JSON.parse(readFileSync(join(dir, "branch.json")));
  const prep = prepareCompaction(branch, { enabled: true, reserveTokens: 16384, keepRecentTokens: 20000 });
  const level = meta[turn]?.thinking_level ?? "high";
  const base = { turn, tokensBefore: prep.tokensBefore, split: prep.isSplitTurn, thinking_level: level };
  const saveSummary = (arm, k, text) => text && writeFileSync(join(POINTS, "summaries", `turn-${turn}.${arm}.k${k}.md`), text);

  // 1. warm
  {
    const off = logOffset();
    const r = await sendSSE({ ...reqN, stream: true, stream_options: { include_usage: true }, max_tokens: 1, max_completion_tokens: undefined });
    out({ ...base, arm: "warm", wall: r.wall, ttft: r.ttft, usage: r.usage, finish: r.finish, omlx: logSlice(off) });
  }

  // 2. A2 blackhole (no model call)
  if (bh) {
    const t0 = Date.now();
    try {
      let result;
      const ctx = {
        cwd: header.cwd && existsSync(header.cwd) ? header.cwd : ROOT,
        model,
        hasUI: false,
        ui: { notify: () => undefined },
        sessionManager: {
          getSessionId: () => header.id,
          getEntries: () => branch,
          getBranch: () => branch,
          getLeafId: () => branch.at(-1)?.id,
          getSessionFile: () => undefined,
        },
      };
      for (const h of bh.get("session_start") ?? []) await h({ type: "session_start" }, ctx);
      for (const h of bh.get("session_before_compact") ?? []) {
        const r = await h(
          { type: "session_before_compact", preparation: prep, branchEntries: branch, reason: "manual", willRetry: false, signal: new AbortController().signal },
          ctx,
        );
        if (r !== undefined) result = r;
      }
      const c = result?.compaction;
      saveSummary("A2", 0, c?.summary);
      out({ ...base, arm: "A2", k: 0, wall: (Date.now() - t0) / 1000, ok: !!c, cancelled: !!result?.cancel, summary_chars: c?.summary?.length ?? null, firstKeptEntryId: c?.firstKeptEntryId, same_cut_as_pi: c?.firstKeptEntryId === prep.firstKeptEntryId, compactor: c?.details?.compactor ?? null, summary: c?.summary });
    } catch (err) {
      out({ ...base, arm: "A2", k: 0, wall: (Date.now() - t0) / 1000, ok: false, error: String(err?.stack ?? err).slice(0, 600) });
    }
  }

  // 3. A1 and A3, seeded random order
  const rnd = mulberry32(SEED + turn);
  const reuseArms = ARMS.filter((a) => a === "A1" || a === "A3");
  if (reuseArms.length === 2 && rnd() < 0.5) reuseArms.reverse();
  for (const arm of reuseArms) {
    for (let k = 0; k < K; k++) {
      const off = logOffset();
      const t0 = Date.now();
      if (arm === "A1") {
        const captures = [
          { payload: reqN, provider: "omlx", modelId: model.id, sessionId: "s", completed: true },
          ...(reqN1 ? [{ payload: reqN1, provider: "omlx", modelId: model.id, sessionId: "s", completed: false }] : []),
        ];
        const o = await reuse.runReuse(
          { reason: "manual", willRetry: false, preparation: prep, branchEntries: branch, signal: new AbortController().signal },
          { model, sessionManager: { getSessionId: () => "s" }, modelRegistry: { getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "IGNORED" }) } },
          captures,
          2048,
          { fetch: globalThis.fetch, now: Date.now, ttftTimeoutMs: 240_000, stallTimeoutMs: 60_000, maxOutputTokens: 600 * 20, genDeadlineMs: 2 * 600 * 1000 },
        );
        saveSummary("A1", k, o.ok ? o.compaction.summary : null);
        out({ ...base, arm, k, wall: (Date.now() - t0) / 1000, ok: o.ok, fallback: o.ok ? null : o.fallback, telemetry: o.telemetry, summary_chars: o.ok ? o.compaction.summary.length : null, omlx: logSlice(off), summary: o.ok ? o.compaction.summary : null });
      } else {
        const budget = a3.summaryTokenBudget(model.contextWindow, prep.tokensBefore, a3.mergeConfig({ warmup: false }));
        const collector = a3.createCollector("openai-completions");
        try {
          const r = await a3.streamMessages(URL_CC, a3.requestHeaders("openai-completions", undefined, { apiKey: "IGNORED" }), JSON.stringify(a3.buildSummaryBody(reqN1 ?? reqN, budget, "openai-completions")), collector, new AbortController().signal);
          const text = r.text + a3.fileListSuffix(prep.fileOps);
          saveSummary("A3", k, text);
          out({ ...base, arm, k, budget, wall: (Date.now() - t0) / 1000, ok: true, finish: r.stopReason, usage: r.usage, thinking_chars: collector.thinkingChars, summary_chars: text.length, omlx: logSlice(off), summary: text });
        } catch (err) {
          out({ ...base, arm, k, budget, wall: (Date.now() - t0) / 1000, ok: false, error: String(err?.message ?? err), thinking_chars: collector.thinkingChars, omlx: logSlice(off) });
        }
      }
    }
  }

  // 4. A0 native, last, cold via a per-request nonce line
  if (ARMS.includes("A0")) {
    const off = logOffset();
    const t0 = Date.now();
    const nonceStream = (m, context, options) =>
      streamSimple(m, { ...context, systemPrompt: `[mdr ${randomUUID()}]\n${context.systemPrompt ?? ""}` }, options);
    try {
      const r = await compact(prep, model, "IGNORED", undefined, undefined, new AbortController().signal, level, nonceStream);
      saveSummary("A0", 0, r.summary);
      out({ ...base, arm: "A0", k: 0, wall: (Date.now() - t0) / 1000, ok: true, usage: r.usage, summary_chars: r.summary.length, omlx: logSlice(off), summary: r.summary });
    } catch (err) {
      out({ ...base, arm: "A0", k: 0, wall: (Date.now() - t0) / 1000, ok: false, error: String(err?.message ?? err), omlx: logSlice(off) });
    }
  }
}
console.log("done");
