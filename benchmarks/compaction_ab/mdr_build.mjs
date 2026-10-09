// Multi-depth replay, step 2: rebuild the trial's exact requests at the cut
// points, and each arm's compaction request bodies. No network.
//
// For a cut at turn N (turns.jsonl numbering = the N-th assistant message):
//   branch       session entries up to, not including, assistant message N+1.
//                This is the state when the harness compaction fires at
//                turn_end of N.
//   request N    the last *completed* request. Context before assistant N,
//                with the shell-retention projection as of N, converted by
//                pi-ai, inside the captured template (system + tools + params).
//   request N+1  the request the harness aborts. It is the one
//                pi-prefix-cache-compaction (A3) captures last (plan §2.7 R4).
//   a1.json      #84 planReuse body from request N, the newest completed capture
//   a3.json      A3 buildSummaryBody from request N+1
//   prep.json    pi prepareCompaction(branch), which A0 and A2 consume
// The shell-retention projection is replayed with its own code. It is fed a
// fake archive that latches what earlier turns demoted, and the gate context
// those turns recorded (turns.jsonl retention.estContextTokens). The result is
// checked against the recorded stats. Request message counts are checked
// against the live observer's request rows, and system/tools sha1 against its
// fingerprints.
//
// Usage: node mdr_build.mjs <session.jsonl> <turns.jsonl> <ab-observer.jsonl> <template.json> <outdir> N [N ...]
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const ROOT = new URL("../../", import.meta.url).pathname;
const [sessionPath, turnsPath, obsPath, templatePath, outDir, ...cuts] = process.argv.slice(2);
const imp = (p) => import(pathToFileURL(join(ROOT, p)).href);
const pi = await imp("node_modules/@earendil-works/pi-coding-agent/dist/index.js");
const { prepareCompaction } = await imp("node_modules/@earendil-works/pi-coding-agent/dist/core/compaction/compaction.js");
const { stream } = await imp(
  "node_modules/@earendil-works/pi-coding-agent/node_modules/@earendil-works/pi-ai/dist/api/openai-completions.js",
);
const retention = await imp(".pi/extensions/shell-retention/retention.ts");
const reuse = await imp(".pi/extensions/cache-reuse-compaction/index.ts");
const a3 = await imp("vendor/compaction-ab/pi-prefix-cache-compaction@0.3.0/src/core.ts");

const sha16 = (v) => createHash("sha1").update(JSON.stringify(v ?? null)).digest("hex").slice(0, 16);
const jsonl = (p) =>
  readFileSync(p, "utf-8")
    .trim()
    .split("\n")
    .map((l) => JSON.parse(l));

const realModels = JSON.parse(readFileSync(join(homedir(), ".config/little-coder/models.json"), "utf-8"));
const prov = realModels.providers.omlx;
const mdef = prov.models.find((x) => x.id === "tiel-coder-oq6e-fp16");
const model = { ...mdef, compat: { ...mdef.compat, chatTemplateKwargs: undefined }, api: prov.api, provider: "omlx", baseUrl: prov.baseUrl };
const template = JSON.parse(readFileSync(templatePath, "utf-8"));
const systemText = template.messages[0].content;

const all = jsonl(sessionPath);
const entries = all.slice(1);
const turns = jsonl(turnsPath);
const requests = jsonl(obsPath).filter((o) => o.kind === "request");
const assistantIdx = entries.map((e, i) => (e.type === "message" && e.message.role === "assistant" ? i : -1)).filter((i) => i >= 0);

// Fake shell-retention archive: remembers what was "saved" so later projections see those pairs as latched.
const saved = new Set();
const archive = { save: (id) => (saved.add(id), true), size: (id) => (saved.has(id) ? 1 : undefined), readRange: () => undefined };
const gateFor = (t) => ({
  options: retention.resolveGateOptions(),
  context: { contextTokens: t?.retention?.estContextTokens ?? null, contextWindow: model.contextWindow, model: { provider: "omlx", id: model.id } },
});

/** Thinking level in force before assistant message n (the model config maps it to enable_thinking). */
function thinkingLevelForTurn(n) {
  let level = null;
  for (const e of entries.slice(0, assistantIdx[n - 1])) if (e.type === "thinking_level_change") level = e.thinkingLevel;
  return level;
}

/** Agent messages pi would send for turn n (1-based), projected; plus the projection stats. */
function contextForTurn(n) {
  const upto = entries.slice(0, assistantIdx[n - 1]);
  const msgs = pi.buildSessionContext(upto).messages;
  const r = retention.demoteMessagesWithStats(msgs, archive, retention.resolveOptions(), gateFor(turns[n - 1]));
  return { messages: r.messages, stats: r.stats };
}

async function payloadFor(agentMessages, thinkingLevel) {
  let captured;
  const s = stream(model, { systemPrompt: systemText, messages: pi.convertToLlm(agentMessages), tools: [] }, {
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
  // Everything but the conversation comes from the live template (system, tools, params).
  // enable_thinking follows the session's thinking level ({$var: thinking.enabled} in models.json):
  // a thinking-budget latch to "off" turns it off for every later request.
  const kwargs = { ...template.chat_template_kwargs, enable_thinking: thinkingLevel !== "off" };
  return { ...template, chat_template_kwargs: kwargs, messages: [template.messages[0], ...captured.messages.slice(1)] };
}

// Replay every latching demotion before the earliest cut, in turn order.
const latchTurns = turns.filter((t) => (t.demoted_new ?? 0) > 0).map((t) => t.turn);
const replay = [];
for (const n of latchTurns) {
  const { stats } = contextForTurn(n);
  replay.push({ turn: n, demoted: stats.demoted, recorded: turns[n - 1].retention?.demoted, gate: stats.gate });
}

mkdirSync(outDir, { recursive: true });
const summary = { latch_replay: replay, points: [] };
for (const nStr of cuts) {
  const n = Number(nStr);
  const t = turns[n - 1];
  const { messages: msgsN, stats: statsN } = contextForTurn(n);
  const reqN = await payloadFor(msgsN, thinkingLevelForTurn(n));
  const hasNext = n < assistantIdx.length;
  const { messages: msgsN1, stats: statsN1 } = hasNext ? contextForTurn(n + 1) : { messages: null, stats: null };
  const reqN1 = hasNext ? await payloadFor(msgsN1, thinkingLevelForTurn(n + 1)) : null;
  const branch = entries.slice(0, hasNext ? assistantIdx[n] : entries.length);
  const prep = prepareCompaction(branch, { enabled: true, reserveTokens: 16384, keepRecentTokens: 20000 });
  const ctx = { model, sessionManager: { getSessionId: () => "s" } };
  if (!prep) {
    // Too early to compact (probe turns): keep only the rebuilt requests.
    const dir = join(outDir, `turn-${n}`);
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, "request-N.json"), JSON.stringify(reqN));
    if (reqN1) writeFileSync(join(dir, "request-N1.json"), JSON.stringify(reqN1));
    summary.points.push({ turn: n, recorded_prompt_tokens: t.prompt_tokens, n_messages: reqN.messages.length, observer_n_messages: requests[n - 1]?.n_messages, prep: null });
    continue;
  }
  const plan = reuse.planReuse(
    { reason: "manual", willRetry: false, preparation: prep, branchEntries: branch },
    ctx,
    { payload: reqN, provider: "omlx", modelId: model.id, sessionId: "s", completed: true },
    2048,
    600 * 20,
  );
  const budget = a3.summaryTokenBudget(model.contextWindow, prep.tokensBefore, a3.mergeConfig({ warmup: false }));
  const a3body = a3.buildSummaryBody(reqN1 ?? reqN, budget, "openai-completions");
  const dir = join(outDir, `turn-${n}`);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, "request-N.json"), JSON.stringify(reqN));
  if (reqN1) writeFileSync(join(dir, "request-N1.json"), JSON.stringify(reqN1));
  if (!("fallback" in plan)) writeFileSync(join(dir, "a1.json"), JSON.stringify(plan.body));
  writeFileSync(join(dir, "a3.json"), JSON.stringify(a3body));
  writeFileSync(join(dir, "branch.json"), JSON.stringify({ header: all[0], entries: branch }));
  const statKeys = ["demoted", "sticky", "gate", "prefix", "due"];
  const pick = (s) => (s ? Object.fromEntries(statKeys.map((k) => [k, s[k]])) : null);
  const point = {
    turn: n,
    recorded_prompt_tokens: t.prompt_tokens,
    recorded_prompt_tokens_next: hasNext ? turns[n].prompt_tokens : null,
    n_messages: reqN.messages.length,
    observer_n_messages: requests[n - 1]?.n_messages,
    n_messages_next: reqN1?.messages.length ?? null,
    observer_n_messages_next: hasNext ? requests[n]?.n_messages : null,
    system_sha1_ok: sha16(reqN.messages[0]) === requests[n - 1]?.system_sha1,
    tools_sha1_ok: sha16(reqN.tools) === requests[n - 1]?.tools_sha1,
    retention: pick(statsN),
    retention_recorded: pick(t.retention),
    retention_next: pick(statsN1),
    retention_recorded_next: hasNext ? pick(turns[n].retention) : null,
    tokensBefore: prep.tokensBefore,
    split: prep.isSplitTurn,
    a1: "fallback" in plan ? `fallback:${plan.fallback}` : "reuse",
    a1_max_tokens: "fallback" in plan ? null : plan.maxTokens,
    a3_budget: budget,
    thinking_level: thinkingLevelForTurn(n),
    a3_source: reqN1 ? "request N+1 (aborted, as live)" : "request N",
  };
  summary.points.push(point);
  console.log(JSON.stringify(point));
}
writeFileSync(join(outDir, "build-report.json"), JSON.stringify(summary, null, 1));
console.log(JSON.stringify({ latch_replay: replay }));
