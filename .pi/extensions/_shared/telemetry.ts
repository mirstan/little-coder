// Structured counters from little-coder's extensions to the benchmark harness.
//
// Channel: pi.appendEntry(customType, data). pi stores a plain "custom" session
// entry and synchronously emits the session event {type: "entry_appended",
// entry} (pi-coding-agent dist/core/agent-session.js:1864-1870); rpc mode writes
// every session event to stdout (dist/modes/rpc/rpc-mode.js:264), where
// benchmarks/rpc_client.py hands it to on_event and benchmarks/turn_ledger.py
// folds it into the trial's turns.jsonl.
//
// Safe from a `context` or `tool_call` hook: appendCustomEntry touches only the
// session tree (dist/core/session-manager.js:754-759, 820-831), a plain custom
// entry projects to no LLM message (session-manager.js:162-188) and is never a
// compaction cut point (dist/core/compaction/compaction.js:267-278). The TUI
// shows a custom entry only for a customType with a registered renderer
// (dist/modes/interactive/interactive-mode.js:2601-2605); none is registered.
//
// Opt-in through LITTLE_CODER_TELEMETRY=1, which benchmarks/rpc_client.py sets
// for every benchmark pi it spawns (those run --no-session, so nothing reaches
// disk). An interactive session with persistence would otherwise gain one
// session-file entry per LLM request.
//
// Telemetry must never break a hook, so every failure (including the "stale
// ctx" throw pi raises after a session replacement, loader.js:136-140) is
// swallowed and reported as `false`. Callers that abort must emit BEFORE
// ctx.abort(), never after.
//
// No index.ts here: the launcher only auto-loads <subdir>/index.ts, so this
// file is a library, like intervention.ts.

export const TELEMETRY_CUSTOM_TYPE = "lc-telemetry";
export const TELEMETRY_VERSION = 1;
export const ENV_TELEMETRY = "LITTLE_CODER_TELEMETRY";

export type TelemetryKind =
  | "shell_retention"
  | "length_stub"
  | "echo_block"
  | "guard_abort"
  | "guard_stand_down";

export function telemetryEnabled(): boolean {
  return process.env[ENV_TELEMETRY] === "1";
}

export function emitTelemetry(pi: unknown, kind: TelemetryKind, data: Record<string, unknown>): boolean {
  try {
    if (!telemetryEnabled()) return false;
    const append = (pi as { appendEntry?: unknown } | null | undefined)?.appendEntry;
    if (typeof append !== "function") return false;
    append.call(pi, TELEMETRY_CUSTOM_TYPE, { ...data, v: TELEMETRY_VERSION, kind });
    return true;
  } catch {
    return false;
  }
}
