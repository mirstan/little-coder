// Compaction A/B observer: arm-neutral recording for the compaction experiment
// (scratchpad plan compaction-ab/plan.md §4.2, §5.4). Opt-in twice over: it is
// inert unless LITTLE_CODER_AB_OBSERVER=1 AND LITTLE_CODER_PI_SESSION_DIR names
// the directory pi persists the session in (rpc_client's session_dir).
//
// It never returns a value from any hook, so it changes no payload and supplies
// no compaction; every handler swallows its own errors. The directory name
// sorts after every other extension (zzz-), so its before_provider_request sees
// the final payload, the same bytes cache-reuse-compaction and zz-arm-* capture.
//
// Writes, all under LITTLE_CODER_PI_SESSION_DIR:
//   ab-observer.jsonl                     one JSON row per event (below)
//   compaction-<n>-payload-last.json      the newest request before compaction n
//   compaction-<n>-payload-completed.json the newest request whose response completed
//   compaction-<n>-post.json              the first request after compaction n
// Rows: arm_env (session_start: arm env vars, command names, tool names),
// request (per provider request: sha1 of messages[0] and of tools, message
// count), before_compact, compact (fromExtension, details.compactor, summary
// size). The request fingerprints are the drift check that replaced
// pi-prefix-stabilizer in the plan.

import { createHash } from "node:crypto";
import { appendFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export const ENV_OBSERVER = "LITTLE_CODER_AB_OBSERVER";
export const ENV_SESSION_DIR = "LITTLE_CODER_PI_SESSION_DIR";
/** Env recorded in every arm_env row; the driver checks them per trial. */
export const RECORDED_ENV = [
  "LC_COMPACTION_ARM",
  "LITTLE_CODER_CACHE_REUSE_COMPACTION",
  "LITTLE_CODER_PI_SESSION_DIR_IN_LOGS",
  "LITTLE_CODER_TB_MODE",
] as const;

type Capture = { payload: unknown; completed: boolean; ts: number };

export function sha1(value: unknown): string {
  return createHash("sha1")
    .update(JSON.stringify(value ?? null))
    .digest("hex")
    .slice(0, 16);
}

export function recordedEnv(env: NodeJS.ProcessEnv = process.env): Record<string, string | null> {
  const out: Record<string, string | null> = {};
  for (const key of RECORDED_ENV) out[key] = env[key] ?? null;
  out.PI_BLACKHOLE_vars = Object.keys(env)
    .filter((k) => k.startsWith("PI_BLACKHOLE_"))
    .sort()
    .join(",") || null;
  return out;
}

export function createObserver(dir: string, now: () => number = Date.now) {
  let captures: Capture[] = [];
  let compactions = 0;
  let pendingPost: number | null = null;

  const row = (kind: string, data: Record<string, unknown>) => {
    try {
      appendFileSync(join(dir, "ab-observer.jsonl"), JSON.stringify({ kind, ts: now() / 1000, ...data }) + "\n");
    } catch {
      // Recording must never break the agent.
    }
  };
  const dump = (name: string, payload: unknown) => {
    try {
      writeFileSync(join(dir, name), JSON.stringify(payload));
      return true;
    } catch {
      return false;
    }
  };

  return {
    onSessionStart(commands: string[], tools: string[]) {
      row("arm_env", { env: recordedEnv(), commands, tools });
    },
    onRequest(payload: any) {
      if (!payload || typeof payload !== "object" || !Array.isArray(payload.messages)) return;
      row("request", {
        n_messages: payload.messages.length,
        system_sha1: sha1(payload.messages[0]),
        tools_sha1: sha1(payload.tools),
      });
      if (pendingPost !== null) {
        dump(`compaction-${pendingPost}-post.json`, payload);
        pendingPost = null;
      }
      try {
        captures = [...captures, { payload: structuredClone(payload), completed: false, ts: now() / 1000 }].slice(-2);
      } catch {
        // Uncloneable: keep what we had.
      }
    },
    onMessageEnd(message: any) {
      if (message?.role !== "assistant" || message.stopReason === "aborted" || message.stopReason === "error") return;
      const last = captures[captures.length - 1];
      if (last) last.completed = true;
    },
    onBeforeCompact(event: any) {
      compactions += 1;
      const n = compactions;
      const last = captures[captures.length - 1];
      const completed = [...captures].reverse().find((c) => c.completed);
      const wroteLast = last ? dump(`compaction-${n}-payload-last.json`, last.payload) : false;
      const wroteCompleted = completed ? dump(`compaction-${n}-payload-completed.json`, completed.payload) : false;
      row("before_compact", {
        n,
        reason: event?.reason ?? null,
        will_retry: event?.willRetry ?? null,
        tokens_before: event?.preparation?.tokensBefore ?? null,
        first_kept_entry_id: event?.preparation?.firstKeptEntryId ?? null,
        last_completed: last?.completed ?? null,
        dumped_last: wroteLast,
        dumped_completed: wroteCompleted,
      });
    },
    onCompact(event: any) {
      const entry = event?.compactionEntry;
      row("compact", {
        n: compactions,
        from_extension: event?.fromExtension ?? null,
        compactor: entry?.details?.compactor ?? null,
        summary_chars: typeof entry?.summary === "string" ? entry.summary.length : null,
        first_kept_entry_id: entry?.firstKeptEntryId ?? null,
        tokens_before: entry?.tokensBefore ?? null,
      });
      pendingPost = compactions;
      captures = [];
    },
  };
}

export default function (pi: ExtensionAPI) {
  const dir = process.env[ENV_SESSION_DIR];
  if (process.env[ENV_OBSERVER] !== "1" || !dir) return;
  const obs = createObserver(dir);
  const safe =
    <A extends unknown[]>(fn: (...args: A) => void) =>
    async (...args: A): Promise<undefined> => {
      try {
        fn(...args);
      } catch {
        // never break a hook
      }
      return undefined;
    };

  pi.on(
    "session_start",
    safe(() => {
      const commands = ((pi as any).getCommands?.() ?? []).map((c: any) => String(c?.name ?? c?.invocationName ?? ""));
      const tools = ((pi as any).getAllTools?.() ?? []).map((t: any) => String(t?.name ?? ""));
      obs.onSessionStart(commands, tools);
    }),
  );
  pi.on("before_provider_request", safe((event: any) => obs.onRequest(event?.payload)));
  pi.on("message_end", safe((event: any) => obs.onMessageEnd(event?.message)));
  pi.on("session_before_compact", safe((event: any) => obs.onBeforeCompact(event)));
  pi.on("session_compact", safe((event: any) => obs.onCompact(event)));
}
