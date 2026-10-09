// Compaction A/B, O0 (no GPU, no network): for each compaction point in a saved pi
// session, build the request bodies whose shared prefix with the captured last
// request decides the prefix-cache hit. #84's evidence/render_prefix.py renders
// and tokenizes them.
//   A1 replay  #84 planReuse body
//   A3 summary pi-prefix-cache-compaction buildSummaryBody (vendored, pure)
//   A3 warm-up pi-ai's payload for {systemPrompt:"", tools:[]} with buildWarmupBody
//              (the W1 check), compared with the next real turn
//   A0         pi's native summary request (system + one user message)
// Usage: node benchmarks/compaction_ab/o0_pairs.mjs <session.jsonl> <pairs.json>
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const ROOT = new URL("../../", import.meta.url).pathname;
const [sessionPath, outPath] = process.argv.slice(2);
const imp = (p) => import(pathToFileURL(join(ROOT, p)).href);
const pi = await imp("node_modules/@earendil-works/pi-coding-agent/dist/index.js");
const { prepareCompaction } = await imp("node_modules/@earendil-works/pi-coding-agent/dist/core/compaction/compaction.js");
const { stream } = await imp(
  "node_modules/@earendil-works/pi-coding-agent/node_modules/@earendil-works/pi-ai/dist/api/openai-completions.js",
);
const reuse = await imp(".pi/extensions/cache-reuse-compaction/index.ts");
const a3 = await imp("vendor/compaction-ab/pi-prefix-cache-compaction@0.3.0/src/core.ts");

const models = JSON.parse(readFileSync(join(ROOT, "models.json"), "utf-8"));
const prov = models.providers.omlx;
const m = prov.models.find((x) => x.id === "tiel-coder-oq6e-fp16");
const model = { ...m, api: prov.api, provider: "omlx", baseUrl: prov.baseUrl };
const systemPrompt = readFileSync(join(ROOT, "AGENTS.md"), "utf-8");
const tools = ["read", "write", "edit", "bash", "glob", "grep"].map((name) => ({
  name,
  description: `The ${name} tool.`,
  parameters: { type: "object", properties: { path: { type: "string" }, command: { type: "string" } } },
}));

async function payloadFor(messages, sys = systemPrompt, tl = tools) {
  let captured;
  const s = stream(model, { systemPrompt: sys, messages: pi.convertToLlm(messages), tools: tl }, {
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
  });
  await s.result();
  return { ...captured, temperature: 0.2 };
}

const all = readFileSync(sessionPath, "utf-8").trim().split("\n").map((l) => JSON.parse(l));
const entries = all.slice(1);
const omlxStart = entries.findIndex((e) => e.type === "model_change" && e.provider === "omlx");
const pairs = [];
const report = [];
for (let ci = 0; ci < entries.length; ci++) {
  if (entries[ci].type !== "compaction" || ci < omlxStart) continue;
  const branch = entries.slice(0, ci);
  const ctxAll = pi.buildSessionContext(branch).messages;
  let lastAssistant = ctxAll.length - 1;
  while (lastAssistant >= 0 && ctxAll[lastAssistant].role !== "assistant") lastAssistant--;
  const captured = await payloadFor(ctxAll.slice(0, lastAssistant));
  const prep = prepareCompaction(branch, { enabled: true, reserveTokens: 16384, keepRecentTokens: 20000 });
  const plan = reuse.planReuse(
    { reason: "manual", willRetry: false, preparation: prep, branchEntries: branch },
    { model, sessionManager: { getSessionId: () => "s" } },
    { payload: captured, provider: "omlx", modelId: model.id, sessionId: "s", completed: true },
    2048,
  );
  const budget = a3.summaryTokenBudget(model.contextWindow, prep.tokensBefore, a3.mergeConfig({ warmup: false }));
  const a3body = a3.buildSummaryBody(captured, budget, "openai-completions");
  // W1: what A3's warm-up would send after a compaction, against the next real turn.
  const after = pi.buildSessionContext(entries.slice(0, ci + 1)).messages;
  const nextReal = await payloadFor(after);
  const warmBuilt = await payloadFor(after, "", []);
  const warm = a3.buildWarmupBody(captured, warmBuilt);
  report.push({ point: ci, tokensBefore: prep.tokensBefore, split: prep.isSplitTurn, a1: "fallback" in plan ? plan.fallback : "reuse", a3_budget: budget });
  if (!("fallback" in plan)) pairs.push({ name: `@${ci} A1 captured->replay`, captured, replay: plan.body });
  pairs.push({ name: `@${ci} A3 captured->summary`, captured, replay: a3body });
  pairs.push({ name: `@${ci} A3 warm-up vs next real turn (W1)`, captured: nextReal, replay: warm });
}
writeFileSync(outPath, JSON.stringify(pairs));
console.log(JSON.stringify(report, null, 1));
