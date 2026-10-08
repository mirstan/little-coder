import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { convertToLlm, sessionEntryToContextMessages } from "@earendil-works/pi-coding-agent";
import { envNumber } from "../_shared/env-number.ts";
import { emitTelemetry } from "../_shared/telemetry.ts";
import { computeFileLists, formatFileOperations, mergeSplitTurn, PI_NO_PRIOR_HISTORY } from "./pi-compat.ts";
import {
  buildInstruction,
  buildReplayBody,
  locateAnchor,
  parseSummaryOutput,
  type Payload,
} from "./replay.ts";
import { readChatCompletionStream, type StreamUsage } from "./sse.ts";

// Compaction that reuses the server's prefix cache (opt-in).
//
// pi summarizes with a separate request: its own system prompt and the whole
// history serialized into one user message. That shares no token prefix with
// the conversation, so an exact-prefix KV cache (omlx: whole 4096-token
// blocks) re-prefills everything: 820–955 s to first token at ~190–200k
// tokens on the TB2.1 runs. This extension keeps the last chat-completions
// payload exactly as it went out (after every `context` hook and every
// earlier `before_provider_request` handler), and on `session_before_compact`
// replays it with one extra user message asking for pi's summary. The server
// then prefills only the uncached tail and the instruction. The result is
// returned in pi's own CompactionResult shape (pi's firstKeptEntryId,
// tokensBefore, split-turn merge and file lists). One difference pi imposes:
// it stores a returned compaction with fromHook: true and does not carry its
// file lists into the next compaction, so carriedFileOps() does that for the
// next reuse compaction (a native fallback after one still loses them).
//
// What makes the prefix identical (replay.ts buildReplayBody):
// - the captured body is cloned, one user message is appended, and only
//   max_tokens/max_completion_tokens and omlx's request-level thinking_budget
//   change; omlx applies both outside the chat template;
// - tool_choice is never sent (omlx drops `tools` from the template on
//   "none"), nor reasoning_effort (omlx merges it into the template kwargs),
//   and the instruction carries no `<|think_*|>` marker (the template scans
//   every user message for them and rewrites the first system block).
//
// Load order: extensions load in directory-name order (bin/little-coder.mjs)
// and pi adopts only RETURNED payloads, so this capture sees what
// benchmark-profiles returned. A later extension that rewrites the payload
// would be invisible here; load-order.test.ts pins the current set.
//
// Every doubt falls back to pi's native compaction by returning nothing:
// overflow recovery, a missing/stale/foreign capture, an anchor the payload
// does not hold, no room in the window, a network/HTTP error, abort, timeout,
// a tool call, a truncated answer or an unusable summary.
//
//   LITTLE_CODER_CACHE_REUSE_COMPACTION=1                      opt in (default off)
//   LITTLE_CODER_CACHE_REUSE_COMPACTION_THINKING_BUDGET=2048   omlx thinking cap; <=0 omits it
//   LITTLE_CODER_CACHE_REUSE_COMPACTION_TTFT_TIMEOUT_S=240     no first token by then: a cache miss, fall back
//   LITTLE_CODER_CACHE_REUSE_COMPACTION_TIMEOUT_S=360          whole-request timeout
//
// The two timeouts aim to keep a slow replay plus pi's own fallback inside
// the harness's 1800 s wait for a compaction (rpc_client PI_IDLE_WAIT_CAP_SEC).
// That assumes pi's own request takes about what it measured at 190-200k
// (820-955 s to first token, ~1000 s in all); it is not enforced, and a
// fallback at a larger context can still overrun it.
// A hit's TTFT is a normal turn's (5-50 s at 190-220k); a miss would prefill
// the whole payload, kept tail and untruncated tool output included, which
// costs more than pi's own request, so it is abandoned early.
//
// Capture choice: the harness asks for a compaction at turn_end while pi is
// already sending the next request, which compact() then aborts. That request
// may never have been prefilled, and may carry a fresh shell-retention
// demotion deep in history. The last two payloads are kept and the newest one
// whose response completed is preferred; the turn after it lies in the kept
// tail, which the anchor check confirms.

export const ENV_ENABLE = "LITTLE_CODER_CACHE_REUSE_COMPACTION";
export const ENV_THINKING_BUDGET = "LITTLE_CODER_CACHE_REUSE_COMPACTION_THINKING_BUDGET";
export const ENV_TIMEOUT_S = "LITTLE_CODER_CACHE_REUSE_COMPACTION_TIMEOUT_S";
export const ENV_TTFT_TIMEOUT_S = "LITTLE_CODER_CACHE_REUSE_COMPACTION_TTFT_TIMEOUT_S";
const DEFAULT_THINKING_BUDGET = 2048;
const DEFAULT_TIMEOUT_S = 360;
const DEFAULT_TTFT_TIMEOUT_S = 240;
// Room the window must still have for the answer after the thinking budget.
const MIN_ANSWER_TOKENS = 2048;
const WINDOW_MARGIN_TOKENS = 1024;

export interface Capture {
  payload: Payload;
  provider: string;
  modelId: string;
  sessionId: string;
  /** An assistant response to this payload finished (not aborted, not an error). */
  completed?: boolean;
}

export type Fallback =
  | "overflow"
  | "api"
  | "no_capture"
  | "model_mismatch"
  | "session_mismatch"
  | "stale"
  | "anchor_missing"
  | "anchor_ambiguous"
  | "window"
  | "auth"
  | "http_error"
  | "aborted"
  | "timeout"
  | "ttft_timeout"
  | "tool_call"
  | "truncated"
  | "garbage";

export interface ReusePlan {
  body: Payload;
  want: { history: boolean; prefix: boolean };
  maxTokens: number;
}

function sessionId(ctx: any): string | undefined {
  try {
    return ctx?.sessionManager?.getSessionId?.();
  } catch {
    return undefined;
  }
}

function textOf(m: any): string {
  if (typeof m?.content === "string") return m.content;
  if (!Array.isArray(m?.content)) return "";
  return m.content.map((p: any) => (typeof p?.text === "string" ? p.text : "")).join("\n");
}

const norm = (s: string) => s.replace(/\s+/g, " ").trim();

/**
 * Everything short of the network: is this compaction one the capture can
 * serve, and what exactly to send. Pure apart from reading pi's helpers.
 */
export function planReuse(
  event: any,
  ctx: any,
  capture: Capture | null,
  thinkingBudget: number,
): ReusePlan | { fallback: Fallback } {
  if (event.reason === "overflow" || event.willRetry) return { fallback: "overflow" };
  const model = ctx.model;
  if (!model || model.api !== "openai-completions") return { fallback: "api" };
  if (!capture) return { fallback: "no_capture" };
  if (capture.provider !== model.provider || capture.modelId !== model.id) return { fallback: "model_mismatch" };
  if (capture.sessionId !== sessionId(ctx)) return { fallback: "session_mismatch" };

  const prep = event.preparation;
  const messages = capture.payload.messages;
  if (prep.previousSummary) {
    const needle = norm(prep.previousSummary).slice(0, 200);
    if (!messages.some((m: any) => m?.role === "user" && norm(textOf(m)).includes(needle))) return { fallback: "stale" };
  }

  // pi's findCutPoint walks the cut back over adjacent entries that put
  // nothing in context (lc-telemetry custom entries precede every request in
  // a benchmark run), so firstKeptEntryId often names one of those. The first
  // message the model will still see is the next context-visible entry.
  const branch: any[] = event.branchEntries ?? [];
  let cutMsg: any;
  for (let i = branch.findIndex((e) => e?.id === prep.firstKeptEntryId); i >= 0 && i < branch.length; i++) {
    if (branch[i]?.type === "compaction") break;
    const visible = sessionEntryToContextMessages(branch[i]);
    if (visible.length > 0) {
      cutMsg = convertToLlm(visible)[0];
      break;
    }
  }
  if (!cutMsg) return { fallback: "anchor_missing" };
  const split = !!prep.isSplitTurn && prep.turnPrefixMessages.length > 0;
  const summarized = convertToLlm(prep.messagesToSummarize);
  const cut = locateAnchor(messages, {
    message: cutMsg as any,
    before: [...summarized, ...(split ? convertToLlm(prep.turnPrefixMessages) : [])] as any,
  });
  if ("error" in cut) return { fallback: cut.error };
  let turnStart: { index: number; excerpt: string } | null = null;
  if (split) {
    const startMsg = convertToLlm([prep.turnPrefixMessages[0]])[0];
    const found = startMsg ? locateAnchor(messages, { message: startMsg as any, before: summarized as any }) : null;
    if (found && "error" in found) return { fallback: found.error };
    if (!found || found.index >= cut.index) return { fallback: "anchor_missing" };
    turnStart = found;
  }

  const want = { history: !split || prep.messagesToSummarize.length > 0, prefix: split };
  const instruction = buildInstruction({
    cut,
    turnStart,
    wantHistory: want.history,
    wantPrefix: want.prefix,
    hasPrevious: !!prep.previousSummary,
    customInstructions: event.customInstructions,
  });

  // pi's own budgets (compaction.js generateSummaryWithUsage / generateTurnPrefixSummary).
  const reserve = prep.settings?.reserveTokens ?? 16384;
  const cap = (n: number) => Math.min(Math.floor(n), model.maxTokens > 0 ? model.maxTokens : Number.POSITIVE_INFINITY);
  const thinking = Math.max(0, Math.floor(thinkingBudget));
  const desired = (want.history ? cap(0.8 * reserve) : 0) + (want.prefix ? cap(0.5 * reserve) : 0) + thinking;
  // tokensBefore is pi's estimate of the whole current context, which the
  // payload is a prefix of; chars/3 over-counts the instruction on purpose.
  const room =
    (model.contextWindow ?? 0) - (prep.tokensBefore ?? 0) - Math.ceil(instruction.length / 3) - WINDOW_MARGIN_TOKENS;
  if (room < thinking + MIN_ANSWER_TOKENS) return { fallback: "window" };
  const maxTokens = Math.min(desired, room);

  return { body: buildReplayBody(capture.payload, instruction, { maxTokens, thinkingBudget: thinking }), want, maxTokens };
}

/**
 * pi's fileOps plus the file lists of the latest compaction when an extension
 * made it. pi stores a returned compaction with fromHook: true, and its own
 * extractFileOperations only carries details forward from compactions it made
 * itself (compaction.js extractFileOperations), so without this every reuse
 * compaction would reset the file record. A native fallback after a reuse
 * compaction still loses them: that is pi's code path.
 */
export function carriedFileOps(fileOps: any, branchEntries: any[]): any {
  const out = { read: new Set<string>(fileOps.read), written: new Set<string>(fileOps.written), edited: new Set<string>(fileOps.edited) };
  let last: any;
  for (const e of branchEntries) if (e?.type === "compaction") last = e;
  if (last?.fromHook && last.details) {
    for (const f of Array.isArray(last.details.readFiles) ? last.details.readFiles : []) if (typeof f === "string") out.read.add(f);
    for (const f of Array.isArray(last.details.modifiedFiles) ? last.details.modifiedFiles : []) if (typeof f === "string") out.edited.add(f);
  }
  return out;
}

function toUsage(u: StreamUsage | null, model: any) {
  const cacheRead = u?.cachedTokens ?? 0;
  const input = Math.max(0, (u?.promptTokens ?? 0) - cacheRead);
  const output = u?.completionTokens ?? 0;
  const rate = model?.cost ?? {};
  const cost = {
    input: (input * (rate.input ?? 0)) / 1e6,
    output: (output * (rate.output ?? 0)) / 1e6,
    cacheRead: (cacheRead * (rate.cacheRead ?? 0)) / 1e6,
    cacheWrite: 0,
    total: 0,
  };
  cost.total = cost.input + cost.output + cost.cacheRead;
  return { input, output, cacheRead, cacheWrite: 0, totalTokens: input + output + cacheRead, cost };
}

export interface RunDeps {
  fetch: typeof fetch;
  now: () => number;
  timeoutMs: number;
  ttftTimeoutMs: number;
}

export type ReuseOutcome =
  | { ok: true; compaction: any; telemetry: Record<string, unknown> }
  | { ok: false; fallback: Fallback; telemetry: Record<string, unknown> };

/** Plan, send, parse. Never throws: every failure is a fallback. */
export async function runReuse(
  event: any,
  ctx: any,
  captures: Capture | Capture[] | null,
  thinkingBudget: number,
  deps: RunDeps,
): Promise<ReuseOutcome> {
  const tsStart = deps.now();
  const base: Record<string, unknown> = {
    source: "compaction",
    reason: event.reason,
    split: !!event.preparation?.isSplitTurn,
    tokens_before: event.preparation?.tokensBefore ?? null,
    ts_start: tsStart / 1000,
  };
  const fail = (fallback: Fallback, extra: Record<string, unknown> = {}): ReuseOutcome => {
    const tsEnd = deps.now();
    return {
      ok: false,
      fallback,
      telemetry: { ...base, path: "native", fallback, ...extra, ts_end: tsEnd / 1000, duration_s: (tsEnd - tsStart) / 1000 },
    };
  };

  // Completed responses first, newest first; then whatever was in flight.
  const list = (Array.isArray(captures) ? captures : captures ? [captures] : [])
    .map((c, i) => ({ c, i }))
    .sort((a, b) => Number(!!b.c.completed) - Number(!!a.c.completed) || b.i - a.i)
    .map((x) => x.c);
  let plan: ReusePlan | { fallback: Fallback } = { fallback: "no_capture" };
  let used: Capture | undefined;
  for (const c of list.length > 0 ? list : [null]) {
    let p: ReusePlan | { fallback: Fallback };
    try {
      p = planReuse(event, ctx, c, thinkingBudget);
    } catch {
      p = { fallback: "anchor_missing" };
    }
    if (!("fallback" in p)) {
      plan = p;
      used = c ?? undefined;
      break;
    }
    if (c === list[0] || c === null) plan = p;
  }
  if ("fallback" in plan) return fail(plan.fallback);
  base.capture = used?.completed ? "completed" : "in_flight";

  const model = ctx.model;
  let auth: any;
  try {
    auth = await ctx.modelRegistry.getApiKeyAndHeaders(model);
  } catch {
    auth = { ok: false };
  }
  if (!auth?.ok) return fail("auth");
  const headers: Record<string, string> = {
    "content-type": "application/json",
    accept: "text/event-stream",
    ...(model.headers ?? {}),
    ...(auth.apiKey ? { authorization: `Bearer ${auth.apiKey}` } : {}),
    ...(auth.headers ?? {}),
  };
  const url = `${String(model.baseUrl).replace(/\/+$/, "")}/chat/completions`;
  const ttft = new AbortController();
  let ttftExpired = false;
  const ttftTimer = setTimeout(() => {
    ttftExpired = true;
    ttft.abort();
  }, deps.ttftTimeoutMs);
  const signals = [AbortSignal.timeout(deps.timeoutMs), ttft.signal];
  if (event.signal) signals.push(event.signal);
  const signal = AbortSignal.any(signals);

  let result;
  try {
    const res = await deps.fetch(url, { method: "POST", headers, body: JSON.stringify(plan.body), signal });
    if (!res.ok || !res.body) {
      try {
        await res.body?.cancel();
      } catch {
        // nothing to release
      }
      return fail("http_error", { status: res.status });
    }
    result = await readChatCompletionStream(res.body as ReadableStream<Uint8Array>, deps.now, () => clearTimeout(ttftTimer));
  } catch (err) {
    if (event.signal?.aborted) return fail("aborted");
    if (ttftExpired) return fail("ttft_timeout");
    if (signal.aborted) return fail("timeout");
    return fail("http_error", { error: String((err as Error)?.message ?? err).slice(0, 200) });
  } finally {
    clearTimeout(ttftTimer);
  }

  const tsEnd = deps.now();
  const metrics = {
    prompt_tokens: result.usage?.promptTokens ?? null,
    cache_read: result.usage?.cachedTokens ?? null,
    summary_tokens: result.usage?.completionTokens ?? null,
    ttft_s: result.firstTokenAt === null ? null : (result.firstTokenAt - tsStart) / 1000,
    finish_reason: result.finishReason,
    max_tokens: plan.maxTokens,
  };
  // Only an answer the server closed normally: "length" was cut off by the
  // budget, and no finish reason at all means the stream ended early.
  // A tool call means the model went back to work instead of summarizing;
  // whatever text came before it is not trusted as a finished summary.
  if (result.toolCalls > 0 || result.finishReason === "tool_calls") {
    return fail("tool_call", { ...metrics, tool_calls: result.toolCalls });
  }
  if (result.finishReason !== "stop") {
    return fail("truncated", { ...metrics, tool_calls: result.toolCalls });
  }
  const parsed = parseSummaryOutput(result.content, plan.want);
  if ("error" in parsed) {
    return fail("garbage", { ...metrics, tool_calls: result.toolCalls });
  }

  const prep = event.preparation;
  let summary = plan.want.prefix
    ? mergeSplitTurn(parsed.history ?? PI_NO_PRIOR_HISTORY, parsed.prefix as string)
    : (parsed.history as string);
  const { readFiles, modifiedFiles } = computeFileLists(carriedFileOps(prep.fileOps, event.branchEntries ?? []));
  summary += formatFileOperations(readFiles, modifiedFiles);
  return {
    ok: true,
    compaction: {
      summary,
      firstKeptEntryId: prep.firstKeptEntryId,
      tokensBefore: prep.tokensBefore,
      usage: toUsage(result.usage, model),
      details: { readFiles, modifiedFiles },
    },
    telemetry: {
      ...base,
      path: "reuse",
      ...metrics,
      tool_calls: result.toolCalls,
      ts_end: tsEnd / 1000,
      duration_s: (tsEnd - tsStart) / 1000,
    },
  };
}

export default function (pi: ExtensionAPI) {
  if (process.env[ENV_ENABLE] !== "1") return;

  // The last two payloads sent (see "Capture choice" above).
  let captures: Capture[] = [];
  const forget = async () => {
    captures = [];
  };

  // Registered for its side effect only: returning undefined keeps whatever
  // payload the earlier handlers settled on.
  pi.on("before_provider_request", async (event, ctx) => {
    const payload = (event as any).payload;
    const model = (ctx as any).model;
    if (!payload || typeof payload !== "object" || !Array.isArray(payload.messages) || !model) return;
    try {
      const next: Capture = {
        payload: structuredClone(payload),
        provider: String(model.provider),
        modelId: String(model.id),
        sessionId: String(sessionId(ctx)),
      };
      captures = [...captures, next].slice(-2);
    } catch {
      // An uncloneable payload is not replayable; keep what we had.
    }
  });
  pi.on("message_end", async (event) => {
    const m: any = (event as any).message;
    if (m?.role !== "assistant" || m.stopReason === "aborted" || m.stopReason === "error") return;
    const last = captures[captures.length - 1];
    if (last) last.completed = true;
  });
  pi.on("session_compact", forget);
  pi.on("session_start", forget);
  pi.on("session_shutdown", forget);
  pi.on("session_tree", forget);
  pi.on("session_before_fork", forget);
  pi.on("model_select", forget);

  pi.on("session_before_compact", async (event, ctx) => {
    const outcome = await runReuse(event, ctx, captures, envNumber(ENV_THINKING_BUDGET, DEFAULT_THINKING_BUDGET), {
      fetch: globalThis.fetch,
      now: Date.now,
      timeoutMs: Math.max(1, envNumber(ENV_TIMEOUT_S, DEFAULT_TIMEOUT_S)) * 1000,
      ttftTimeoutMs: Math.max(1, envNumber(ENV_TTFT_TIMEOUT_S, DEFAULT_TTFT_TIMEOUT_S)) * 1000,
    });
    emitTelemetry(pi, "compaction_reuse", outcome.telemetry);
    if (outcome.ok) return { compaction: outcome.compaction };
    return undefined;
  });
}
