// Multi-depth replay, step 5: quality inputs (plan §4.4 Q1 and Q2).
//
// For every point and arm (the k=0 summary from mdr_run's output):
//   post-<arm>.json   The request the agent would send right after that arm's
//                     compaction: branch + compaction entry (the arm's summary
//                     and firstKeptEntryId) + COMPACTION_CONTINUE_PROMPT, through
//                     pi's buildSessionContext and the shell-retention projection.
//                     The gate is cold (contextTokens null, as on the first request
//                     after any compaction) and the earlier demotions stay latched.
//                     Then the live template (system, tools, params, enable_thinking
//                     per thinking level).
//   --probe           Q2 on the GPU (or the fake server). For each post payload:
//                     one max_tokens 1 send (the cold post-compaction prefill: the
//                     cost metric), then k samples with thinking_budget 32768,
//                     max_tokens 40000. Records text, reasoning size, tool calls
//                     and finish reason.
//   --packets         Judge packets for Claude (no GPU), per point:
//                       transcript.md   the pre-compaction history, each tool
//                                       result cut to 4,000 chars (head and tail);
//                       summaries.md    the arms' summaries, blinded S1..Sn (seeded
//                                       shuffle), pi's file-list tags stripped;
//                       q2.md           the agent's real next 5 actions, then each
//                                       arm's sampled next actions, blinded;
//                       key.json        the label-to-arm map (keep it from the judge).
// Usage: node mdr_quality.mjs --points DIR --run RUN.jsonl --session S.jsonl --turns T.jsonl
//          --template template.json [--probe OUT.jsonl --base-url URL --k 3] [--packets DIR] [--seed N]
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { request as httpRequest } from "node:http";
import { homedir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, v, i, a) => (v.startsWith("--") ? [...acc, [v.slice(2), a[i + 1]]] : acc), []),
);
const ROOT = new URL("../../", import.meta.url).pathname;
const imp = (p) => import(pathToFileURL(join(ROOT, p)).href);
const pi = await imp("node_modules/@earendil-works/pi-coding-agent/dist/index.js");
const { stream } = await imp("node_modules/@earendil-works/pi-coding-agent/node_modules/@earendil-works/pi-ai/dist/api/openai-completions.js");
const retention = await imp(".pi/extensions/shell-retention/retention.ts");
const { prepareCompaction } = await imp("node_modules/@earendil-works/pi-coding-agent/dist/core/compaction/compaction.js");

const COMPACTION_CONTINUE_PROMPT =
  "Your session context was compacted to free space, which interrupted what you were doing. The task is not complete — please continue from where you left off. If the task is actually already complete and verified, say so explicitly and stop.";
const jsonl = (p) => readFileSync(p, "utf-8").trim().split("\n").map((l) => JSON.parse(l));
const POINTS = args.points;
const SEED = Number(args.seed ?? 20261009);
const K = Number(args.k ?? 3);
const template = JSON.parse(readFileSync(args.template, "utf-8"));
const turns = jsonl(args.turns);
const sessionEntries = jsonl(args.session).slice(1);
const runRows = jsonl(args.run).filter((r) => ["A0", "A1", "A2", "A3"].includes(r.arm) && r.ok && (r.k ?? 0) === 0);

const realModels = JSON.parse(readFileSync(join(homedir(), ".config/little-coder/models.json"), "utf-8"));
const prov = realModels.providers.omlx;
const mdef = prov.models.find((x) => x.id === "tiel-coder-oq6e-fp16");
const model = { ...mdef, compat: { ...mdef.compat, chatTemplateKwargs: undefined }, api: prov.api, provider: "omlx", baseUrl: prov.baseUrl };

function mulberry32(a) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Same latch replay as mdr_build: earlier demotions stay demoted after compaction.
const saved = new Set();
const archive = { save: (id) => (saved.add(id), true), size: (id) => (saved.has(id) ? 1 : undefined), readRange: () => undefined };
const assistantIdx = sessionEntries.map((e, i) => (e.type === "message" && e.message.role === "assistant" ? i : -1)).filter((i) => i >= 0);
for (const t of turns.filter((t) => (t.demoted_new ?? 0) > 0)) {
  const msgs = pi.buildSessionContext(sessionEntries.slice(0, assistantIdx[t.turn - 1])).messages;
  retention.demoteMessagesWithStats(msgs, archive, retention.resolveOptions(), {
    options: retention.resolveGateOptions(),
    context: { contextTokens: t.retention?.estContextTokens ?? null, contextWindow: model.contextWindow, model: { provider: "omlx", id: model.id } },
  });
}

async function payloadFor(agentMessages, thinkingLevel) {
  let captured;
  const s = stream(model, { systemPrompt: template.messages[0].content, messages: pi.convertToLlm(agentMessages), tools: [] }, {
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
  const kwargs = { ...template.chat_template_kwargs, enable_thinking: thinkingLevel !== "off" };
  return { ...template, chat_template_kwargs: kwargs, messages: [template.messages[0], ...captured.messages.slice(1)] };
}

function postPayloadEntries(branch, summary, firstKeptEntryId, tokensBefore, arm) {
  const last = branch.at(-1);
  const comp = { type: "compaction", id: `mdr-${arm}`, parentId: last.id, timestamp: new Date().toISOString(), summary, firstKeptEntryId, tokensBefore, details: {}, fromHook: arm !== "A0" };
  const user = { type: "message", id: `mdr-${arm}-u`, parentId: comp.id, timestamp: new Date().toISOString(), message: { role: "user", content: [{ type: "text", text: COMPACTION_CONTINUE_PROMPT }], timestamp: Date.now() } };
  return [...branch, comp, user];
}

const byPoint = new Map();
for (const r of runRows) byPoint.set(r.turn, [...(byPoint.get(r.turn) ?? []), r]);

// Build the post-compaction payloads.
for (const [turn, rows] of byPoint) {
  const { entries: branch } = JSON.parse(readFileSync(join(POINTS, `turn-${turn}`, "branch.json")));
  const prep = prepareCompaction(branch, { enabled: true, reserveTokens: 16384, keepRecentTokens: 20000 });
  for (const r of rows) {
    const firstKept = r.arm === "A2" && r.firstKeptEntryId ? r.firstKeptEntryId : prep.firstKeptEntryId;
    const entries = postPayloadEntries(branch, r.summary, firstKept, prep.tokensBefore, r.arm);
    const msgs = pi.buildSessionContext(entries).messages;
    const proj = retention.demoteMessagesWithStats(msgs, archive, retention.resolveOptions(), {
      options: retention.resolveGateOptions(),
      context: { contextTokens: null, contextWindow: model.contextWindow, model: { provider: "omlx", id: model.id } },
    });
    const body = await payloadFor(proj.messages, r.thinking_level);
    writeFileSync(join(POINTS, `turn-${turn}`, `post-${r.arm}.json`), JSON.stringify(body));
    console.log(JSON.stringify({ turn, arm: r.arm, post_messages: body.messages.length, post_bytes: JSON.stringify(body).length, demoted: proj.stats.demoted, gate: proj.stats.gate }));
  }
}

function sendSSE(url, body) {
  return new Promise((resolve, reject) => {
    const t0 = Date.now();
    let ttft = null;
    let buf = "";
    const acc = { text: "", reasoning: 0, finish: null, usage: null, toolCalls: [] };
    const req = httpRequest(url, { method: "POST", headers: { "content-type": "application/json", authorization: "Bearer IGNORED" } }, (res) => {
      res.setEncoding("utf8");
      res.on("data", (c) => {
        buf += c;
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
            if (ttft === null && (d.content || d.reasoning_content || d.tool_calls)) ttft = (Date.now() - t0) / 1000;
            if (typeof d.content === "string") acc.text += d.content;
            if (typeof d.reasoning_content === "string") acc.reasoning += d.reasoning_content.length;
            for (const tc of d.tool_calls ?? []) {
              const slot = (acc.toolCalls[tc.index ?? 0] ??= { name: "", arguments: "" });
              if (tc.function?.name) slot.name += tc.function.name;
              if (tc.function?.arguments) slot.arguments += tc.function.arguments;
            }
            if (ch.finish_reason) acc.finish = ch.finish_reason;
          }
        }
      });
      res.on("end", () => resolve({ status: res.statusCode, ttft, wall: (Date.now() - t0) / 1000, ...acc }));
      res.on("error", reject);
    });
    req.setTimeout(0);
    req.on("error", reject);
    req.end(JSON.stringify(body));
  });
}

if (args.probe) {
  const url = `${(args["base-url"] ?? prov.baseUrl).replace(/\/+$/, "")}/chat/completions`;
  for (const [turn, rows] of [...byPoint].sort((a, b) => a[0] - b[0])) {
    const order = rows.map((r) => r.arm).sort();
    const rnd = mulberry32(SEED + turn * 7);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    for (const arm of order) {
      const body = JSON.parse(readFileSync(join(POINTS, `turn-${turn}`, `post-${arm}.json`)));
      const { max_completion_tokens: _drop, ...rest } = body;
      const cold = await sendSSE(url, { ...rest, max_tokens: 1, stream: true, stream_options: { include_usage: true } });
      appendFileSync(args.probe, JSON.stringify({ turn, arm, kind: "post_prefill", ...cold, text: undefined }) + "\n");
      for (let k = 0; k < K; k++) {
        const r = await sendSSE(url, { ...rest, max_tokens: 40000, thinking_budget: 32768, stream: true, stream_options: { include_usage: true } });
        appendFileSync(args.probe, JSON.stringify({ turn, arm, kind: "q2", k, ...r }) + "\n");
        console.log(JSON.stringify({ turn, arm, k, wall: r.wall, finish: r.finish, tools: r.toolCalls.map((t) => t.name) }));
      }
    }
  }
}

if (args.packets) {
  const probes = args.probe ? jsonl(args.probe).filter((p) => p.kind === "q2") : [];
  const cut = (s, n = 4000) => (s.length <= n ? s : `${s.slice(0, n / 2)}\n… [${s.length - n} chars cut] …\n${s.slice(-n / 2)}`);
  const textOf = (m) =>
    typeof m.content === "string" ? m.content : (m.content ?? []).map((c) => (c.type === "text" ? c.text : c.type === "toolCall" ? `→ ${c.name}(${JSON.stringify(c.arguments)})` : "")).join("\n");
  for (const [turn, rows] of byPoint) {
    const dir = join(args.packets, `turn-${turn}`);
    mkdirSync(dir, { recursive: true });
    const { entries: branch } = JSON.parse(readFileSync(join(POINTS, `turn-${turn}`, "branch.json")));
    const msgs = pi.buildSessionContext(branch).messages;
    const transcript = msgs.map((m, i) => `### [${i}] ${m.role}${m.toolName ? ` (${m.toolName})` : ""}\n${cut(textOf(m))}`).join("\n\n");
    writeFileSync(join(dir, "transcript.md"), `# Pre-compaction transcript, turn ${turn}\n\n${transcript}\n`);
    const rnd = mulberry32(SEED + turn);
    const shuffled = [...rows].sort(() => rnd() - 0.5);
    const key = {};
    const strip = (s) => s.replace(/<read-files>[\s\S]*?<\/read-files>/g, "").replace(/<modified-files>[\s\S]*?<\/modified-files>/g, "").trim();
    writeFileSync(
      join(dir, "summaries.md"),
      shuffled.map((r, i) => ((key[`S${i + 1}`] = r.arm), `## S${i + 1}\n\n${strip(r.summary)}\n`)).join("\n"),
    );
    writeFileSync(join(dir, "key.json"), JSON.stringify(key, null, 1));
    const after = sessionEntries.slice(sessionEntries.findIndex((e) => e.id === branch.at(-1).id) + 1).filter((e) => e.type === "message" && e.message.role === "assistant").slice(0, 5);
    const real = after.map((e, i) => `${i + 1}. ${textOf(e.message).slice(0, 600)}`).join("\n");
    const q2 = shuffled
      .map((r, i) => {
        const samples = probes.filter((p) => p.turn === turn && p.arm === r.arm);
        return `## S${i + 1}\n` + samples.map((p) => `- sample ${p.k}: ${p.toolCalls.map((t) => `${t.name}(${t.arguments.slice(0, 500)})`).join("; ") || "(no tool call) " + p.text.slice(0, 500)} [finish=${p.finish}]`).join("\n");
      })
      .join("\n\n");
    writeFileSync(join(dir, "q2.md"), `# Next-action probe, turn ${turn}\n\n## What the agent actually did next (reference)\n${real}\n\n${q2}\n`);
  }
  console.log(`packets written to ${args.packets}`);
}
