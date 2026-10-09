// Projection-layer demotion of stale, oversized shell (call, result) pairs,
// generalizing browser-extract-retention's rank-based staleness to the shell
// tools. Session history is never touched — see index.ts for the hook.
//
// Two deviations from that precedent, both forced by shell output's shape:
// the demotion unit is the PAIR, because a heredoc'd source paste can be 10KB
// of tool-call text whose result is `compiled`; and nothing demotes until an
// archive copy exists, because shell output has neither an out-of-band
// evidence store nor a re-fetchable origin.

import { createHash } from "node:crypto";
import { envNumber } from "../_shared/env-number.ts";
import { byteLen, formatSize } from "../shell-session/helpers.ts";

// bash/Bash/ShellSession matches truncated-view's ANNOTATED_TOOLS. ShellLog is
// included where truncated-view had to exclude it: demotion needs no command
// parsing, only sizes. ShellRecall is here so recalled dumps age out too.
export const DEMOTABLE_TOOLS = new Set([
  "bash",
  "Bash",
  "ShellSession",
  "ShellLog",
  "ShellRecall",
]);

export const RESULT_DEMOTED_PREFIX = "[shell result demoted";
export const CMD_DEMOTED_INFIX = "of command text demoted — ShellRecall id=";

export const ENV_RETAIN_RAW = "LITTLE_CODER_SHELL_RETAIN_RAW";
export const ENV_MIN_PAIR_BYTES = "LITTLE_CODER_SHELL_DEMOTE_MIN_BYTES";
export const ENV_STALE_DISTANCE = "LITTLE_CODER_SHELL_STALE_DISTANCE";
export const ENV_KEEP_RESULT_HEAD = "LITTLE_CODER_SHELL_KEEP_RESULT_HEAD_BYTES";
export const ENV_KEEP_RESULT_TAIL = "LITTLE_CODER_SHELL_KEEP_RESULT_TAIL_BYTES";
export const ENV_CMD_KEEP = "LITTLE_CODER_SHELL_CMD_KEEP_BYTES";
export const ENV_RECALL_DEFAULT = "LITTLE_CODER_SHELL_RECALL_DEFAULT_BYTES";
export const ENV_RECALL_MAX = "LITTLE_CODER_SHELL_RECALL_MAX_BYTES";
export const ENV_DEMOTE_BATCH = "LITTLE_CODER_SHELL_DEMOTE_BATCH";
export const ENV_DEMOTE_PENDING_BYTES = "LITTLE_CODER_SHELL_DEMOTE_PENDING_BYTES";

export const DEFAULT_RETAIN_RAW = 4;
export const DEFAULT_MIN_PAIR_BYTES = 3072;
export const DEFAULT_STALE_DISTANCE = 50;
export const DEFAULT_KEEP_RESULT_HEAD = 512;
export const DEFAULT_KEEP_RESULT_TAIL = 256;
export const DEFAULT_CMD_KEEP = 1024;
export const DEFAULT_RECALL_BYTES = 8192;
// Equal to MAX_BODY_HEAD_BYTES + MAX_BODY_TAIL_BYTES, so a recall can never
// reinject more than the tool result was allowed to carry in the first place.
export const DEFAULT_RECALL_MAX = 49152;
// Pairs demoted per jump of the demoted prefix. Each jump rewrites history
// mid-prompt, and a prefix-caching server re-prefills everything after the
// first rewritten pair, so a jump of B pairs removes about 1 - 1/B of the
// breaks that per-pair demotion causes. The price is up to B - 1 older pairs
// left raw that per-pair demotion would have shrunk. Each is at least
// minPairBytes; a ShellSession result is capped at ~48KB and pi's built-in
// bash output at 50KB, but a heredoc'd command has no cap, so those raw bytes
// are bounded by DEFAULT_DEMOTE_PENDING_BYTES rather than by their count.
// 4 (= retainRaw) gets 75% of the saving for at most 3 extra pairs; for a
// small context window, set 2, or 1 to restore per-pair demotion.
export const DEFAULT_DEMOTE_BATCH = 4;
// Raw bytes allowed in pairs that per-pair demotion would already have shrunk
// but whose batch is not yet due. Past it, every due pair demotes at once, so
// those bytes never exceed this budget (<= 0 disables the flush), unless the
// cost gate defers the flush, which forcePendingBytes then bounds, unless the
// re-prefill ceiling vetoes that too (then only compaction does). The trigger
// is bytes, not age: a break late in history re-prefills the most of it, and
// an age-triggered flush measured worse than per-pair demotion in sparse
// sessions. A lone stale pair under the budget therefore stays raw for good;
// it is cached, so it costs no prefill, only context. For a small context
// window, lower this, or set the batch to 2 or 1.
export const DEFAULT_DEMOTE_PENDING_BYTES = 65536;

// ── Cost gate ───────────────────────────────────────────────────────────────
//
// Every growth of the demoted prefix makes a prefix-caching server re-prefill
// everything after the first rewritten message (R tokens) to save S tokens on
// later turns. Measured on omlx (Harbor TB2.1 replay, 473 demotion breaks):
// re-prefill costs a(p-c) + b(p²-c²)/2 seconds with a = 1.05e-3 and
// b = 3.7e-8, so a break at 184K tokens that saved 9K cost 665 s, while 9K
// fewer tokens speed each later turn up by ~2 s. What savings buy is keeping
// a trial under compaction, which costs a full re-prefill plus the summary
// (946 s for a 187K-token compaction request, live 2026-10-08). A due jump
// therefore goes ahead only when one of these holds, and the re-prefill
// ceiling below does not veto it; otherwise its pairs stay raw (and cached)
// and are judged again on the next request. R starts at the 4096-token block
// holding the break, counted back from pi's context reading.
//  - S >= minSaveRatio * R: the break is shallow next to what it sheds.
//  - context >= openAtPercent% of the window. Off by default (100): in the
//    replay, demoting near compaction never avoided one, it only paid a deep
//    break before the compaction re-prefilled everything anyway. When set, it
//    is kept 5 points under context-watchdog's LITTLE_CODER_COMPACT_AT_PERCENT.
//  - raw bytes pending in the jump >= forcePendingBytes: a burst of huge
//    outputs. Never fired in the replay; a backstop only.
// Replay (57 trial segments, compaction at 220K priced in, accounting stopped
// at the first one): today's policy costs 12.2 h (6.4 h re-prefill, 16
// compactions). Never demoting costs 9.4 h with 20 compactions. The ratio
// clause alone costs 8.3-8.8 h for ratios 0.1-0.15 with no extra compaction;
// from 0.2 up compactions climb. A context clause at 75% added back 3 h.
export const ENV_DEMOTE_MIN_SAVE_RATIO = "LITTLE_CODER_SHELL_DEMOTE_MIN_SAVE_RATIO";
export const ENV_DEMOTE_OPEN_AT_PERCENT = "LITTLE_CODER_SHELL_DEMOTE_OPEN_AT_PERCENT";
export const ENV_DEMOTE_FORCE_PENDING_BYTES = "LITTLE_CODER_SHELL_DEMOTE_FORCE_PENDING_BYTES";
/** <= 0 disables the clause. The middle of the replay's flat 0.1-0.15 optimum. */
export const DEFAULT_DEMOTE_MIN_SAVE_RATIO = 0.12;
/** <= 0 turns the whole gate off (today's behaviour); >= 100, the default, never opens on context. */
export const DEFAULT_DEMOTE_OPEN_AT_PERCENT = 100;
/** <= 0 disables the clause. Four times the batch flush budget. */
export const DEFAULT_DEMOTE_FORCE_PENDING_BYTES = 262144;
/**
 * The gate runs only on windows in [MIN_GATE_WINDOW, MAX_GATE_WINDOW]: its
 * defaults were fitted on 262,144-token windows, and a small window reaches
 * compaction in few turns, where per-turn savings weigh more.
 */
export const MIN_GATE_WINDOW = 131072;
export const MAX_GATE_WINDOW = 262144;
/** The open point stays this many points under context-watchdog's LITTLE_CODER_COMPACT_AT_PERCENT. */
const WATCHDOG_MARGIN_PERCENT = 5;
/** context-watchdog's DEFAULT_PERCENT, mirrored rather than imported: its index.ts is a pi extension entry point. */
const WATCHDOG_DEFAULT_PERCENT = 80;
/** pi's estimateTokens convention (compaction.js), so estimates line up with getContextUsage's trailing part. */
export const CHARS_PER_TOKEN = 4;

// ── Re-prefill ceiling ──────────────────────────────────────────────────────
//
// The ratio clause prices a break in tokens, but a token re-prefilled at 200K
// costs ~6x one at 10K (~8x one at depth 0): omlx prefill time is a(p-c) + b(p²-c²)/2 seconds for
// tokens c..p, refitted on 162 requests with >= 16K uncached tokens (Sept 20 -
// Oct 8; median error 0%, p10 -9%, p90 +2%), about 950 tok/s at depth 0, 213
// at 100K and 120 at 200K. omlx reuses whole 4096-token blocks only, so a break
// at token d re-prefills from floor(d / 4096) * 4096.
// On 2026-10-08 the ratio clause opened 320-500 s breaks at 124-210K
// (gcode-to-text turns 97, 119, 122-124; make-doom-for-mips turn 88), and
// both trials compacted anyway. A jump whose estimated re-prefill exceeds
// ceilingSeconds is therefore deferred unless it saves at least
// minTokensPerSecond tokens per second it costs, and never goes ahead within
// nearCompactTokens of the compaction threshold, where the compaction is about
// to re-prefill everything anyway. A flat ceiling is not used: in the #83
// replay it held back the 200-475 s jumps at 85-141K that kept four trials
// under compaction (8.67 h and 20 compactions, against 7.69 h and 16). This
// rule: 7.02 h, 16 compactions; flat for ceilings 45-120 s and for
// nearCompactTokens 30-60K, and minTokensPerSecond 45 costs a compaction.
// The veto binds every clause but a cold cache, whose break is free, so an
// opted-in context clause (openAtPercent) no longer pays an over-ceiling
// break that close to compaction.
export const ENV_DEMOTE_CEILING_SECONDS = "LITTLE_CODER_SHELL_DEMOTE_CEILING_SECONDS";
export const ENV_DEMOTE_MIN_TOKENS_PER_SECOND = "LITTLE_CODER_SHELL_DEMOTE_MIN_TOKENS_PER_SECOND";
export const ENV_DEMOTE_NEAR_COMPACT_TOKENS = "LITTLE_CODER_SHELL_DEMOTE_NEAR_COMPACT_TOKENS";
export const ENV_PREFILL_LINEAR_SECONDS = "LITTLE_CODER_SHELL_PREFILL_LINEAR_SECONDS";
export const ENV_PREFILL_QUADRATIC_SECONDS = "LITTLE_CODER_SHELL_PREFILL_QUADRATIC_SECONDS";
export const ENV_PREFIX_BLOCK_TOKENS = "LITTLE_CODER_SHELL_PREFIX_BLOCK_TOKENS";
/** <= 0 disables the veto (#83's gate, but for R's block alignment: PREFIX_BLOCK_TOKENS=1 restores that). */
export const DEFAULT_DEMOTE_CEILING_SECONDS = 60;
/** <= 0: every jump over the ceiling is deferred. Under the 42 tok/s of the slowest jump that avoided a compaction. */
export const DEFAULT_DEMOTE_MIN_TOKENS_PER_SECOND = 38;
/**
 * <= 0 disables the near-compaction clause. Measured from context-watchdog's
 * threshold (80% = 209.7K of a 262K window); the Harbor harness compacts at
 * 220K, about 50K past the clause. In the replay (harness at 220K) margins of
 * 30-60K score the same, 80K costs a compaction; under +-20% noise on the
 * estimates, minTokensPerSecond 30-40 and margins 40-60K all keep 16.
 */
export const DEFAULT_DEMOTE_NEAR_COMPACT_TOKENS = 40000;
/** pi's default compaction reserveTokens (compaction.js): it compacts past window - reserve. */
const PI_RESERVE_TOKENS = 16384;
/** Seconds per prefilled token at depth 0. */
export const DEFAULT_PREFILL_LINEAR_SECONDS = 1.05e-3;
/** Extra seconds per prefilled token per token of depth. */
export const DEFAULT_PREFILL_QUADRATIC_SECONDS = 3.64e-8;
/** <= 1: reuse up to the exact divergence token. */
export const DEFAULT_PREFIX_BLOCK_TOKENS = 4096;

export interface GateOptions {
  minSaveRatio: number;
  openAtPercent: number;
  forcePendingBytes: number;
  ceilingSeconds: number;
  minTokensPerSecond: number;
  nearCompactTokens: number;
  /** context-watchdog's trigger, percent of the window; null when it is off and pi's own compaction (window - reserve) applies. */
  compactAtPercent: number | null;
  prefillLinearSeconds: number;
  prefillQuadraticSeconds: number;
  prefixBlockTokens: number;
}

/** What the hook knows about the prompt pi is about to send. */
export interface GateContext {
  /** pi's getContextUsage().tokens: the last request's usage plus an estimate of what followed; null when unknown. */
  contextTokens: number | null;
  contextWindow: number | null;
  /** The model this request goes to; a history last answered by another one has no warm prefix there. */
  model?: { provider?: string; id?: string } | null;
}

export interface GateInput {
  estSaveTokens: number;
  estReprefillTokens: number;
  contextTokens: number;
  contextWindow: number;
  pendingBytes: number;
  /** Seconds the server would spend re-prefilling estReprefillTokens (prefillSeconds). */
  estReprefillSeconds: number;
}

export type GateReason = "ratio" | "context" | "bytes" | "cold";
/** Why a jump some clause opened was deferred anyway. */
export type VetoReason = "ceiling" | "near";

export function resolveGateOptions(): GateOptions {
  return {
    minSaveRatio: envNumber(ENV_DEMOTE_MIN_SAVE_RATIO, DEFAULT_DEMOTE_MIN_SAVE_RATIO),
    openAtPercent: openAtPercent(),
    forcePendingBytes: envNumber(ENV_DEMOTE_FORCE_PENDING_BYTES, DEFAULT_DEMOTE_FORCE_PENDING_BYTES),
    ceilingSeconds: envNumber(ENV_DEMOTE_CEILING_SECONDS, DEFAULT_DEMOTE_CEILING_SECONDS),
    minTokensPerSecond: envNumber(ENV_DEMOTE_MIN_TOKENS_PER_SECOND, DEFAULT_DEMOTE_MIN_TOKENS_PER_SECOND),
    nearCompactTokens: envNumber(ENV_DEMOTE_NEAR_COMPACT_TOKENS, DEFAULT_DEMOTE_NEAR_COMPACT_TOKENS),
    compactAtPercent: watchdogPercent(),
    prefillLinearSeconds: envNumber(ENV_PREFILL_LINEAR_SECONDS, DEFAULT_PREFILL_LINEAR_SECONDS),
    prefillQuadraticSeconds: envNumber(ENV_PREFILL_QUADRATIC_SECONDS, DEFAULT_PREFILL_QUADRATIC_SECONDS),
    prefixBlockTokens: envNumber(ENV_PREFIX_BLOCK_TOKENS, DEFAULT_PREFIX_BLOCK_TOKENS),
  };
}

/** context-watchdog's trigger percent, or null when it is off or out of range. */
function watchdogPercent(): number | null {
  if (process.env.LITTLE_CODER_NO_COMPACT_WATCHDOG === "1") return null;
  // Same resolution as context-watchdog's thresholdPercent(): unset means its default 80.
  const watchdog = envNumber("LITTLE_CODER_COMPACT_AT_PERCENT", WATCHDOG_DEFAULT_PERCENT);
  return watchdog <= 0 || watchdog >= 100 ? null : watchdog;
}

// An operator who lowers context-watchdog's threshold (default 80) below an
// enabled open point would otherwise have it compact before the gate opened.
function openAtPercent(): number {
  const open = envNumber(ENV_DEMOTE_OPEN_AT_PERCENT, DEFAULT_DEMOTE_OPEN_AT_PERCENT);
  if (open <= 0 || open >= 100) return open;
  const watchdog = watchdogPercent();
  return watchdog === null ? open : Math.min(open, watchdog - WATCHDOG_MARGIN_PERCENT);
}

/** omlx's prefill time for tokens [from, to): a(to-from) + b(to²-from²)/2 seconds; 0 when to <= from. */
export function prefillSeconds(from: number, to: number, o: GateOptions): number {
  if (!(to > from)) return 0;
  return o.prefillLinearSeconds * (to - from) + (o.prefillQuadraticSeconds * (to * to - from * from)) / 2;
}

/** The first token a prefix cache of `block`-token blocks re-prefills after diverging at `token`. */
export function blockStart(token: number, block: number): number {
  const t = Math.max(0, token);
  return Number.isFinite(block) && block > 1 ? Math.floor(t / block) * block : t;
}

/** Why a jump some clause opened must still wait, or null to let it go ahead. A cold cache never reaches here. */
export function demotionVeto(input: GateInput, o: GateOptions): VetoReason | null {
  if (!(o.ceilingSeconds > 0) || input.estReprefillSeconds <= o.ceilingSeconds) return null;
  const compactAt =
    o.compactAtPercent === null
      ? input.contextWindow - PI_RESERVE_TOKENS
      : (Math.min(o.compactAtPercent, 100) / 100) * input.contextWindow;
  if (o.nearCompactTokens > 0 && input.contextTokens >= compactAt - o.nearCompactTokens) return "near";
  if (o.minTokensPerSecond > 0 && input.estSaveTokens >= o.minTokensPerSecond * input.estReprefillSeconds) return null;
  return "ceiling";
}

/** Why a due jump may go ahead, or null to defer it. */
export function demotionGate(input: GateInput, o: GateOptions): GateReason | null {
  if (o.minSaveRatio > 0 && input.estSaveTokens >= o.minSaveRatio * input.estReprefillTokens) return "ratio";
  if (o.openAtPercent < 100 && input.contextTokens >= (o.openAtPercent / 100) * input.contextWindow) return "context";
  if (o.forcePendingBytes > 0 && input.pendingBytes >= o.forcePendingBytes) return "bytes";
  return null;
}

/**
 * Characters pi would count for a message: estimateTokens' fields per role
 * (compaction.js), images at its 4800-char flat rate.
 */
function messageChars(m: any, uptoBlock = Infinity): number {
  const str = (v: unknown) => (typeof v === "string" ? v.length : 0);
  if (m?.role === "bashExecution") return str(m.command) + str(m.output);
  if (m?.role === "branchSummary" || m?.role === "compactionSummary") return str(m.summary);
  if (typeof m?.content === "string") return m.content.length;
  if (!Array.isArray(m?.content)) return 0;
  let n = 0;
  m.content.forEach((b: any, i: number) => {
    if (i >= uptoBlock) return;
    if (b?.type === "text" && typeof b.text === "string") n += b.text.length;
    else if (b?.type === "thinking" && typeof b.thinking === "string") n += b.thinking.length;
    else if (b?.type === "toolCall") n += String(b.name ?? "").length + JSON.stringify(b.arguments ?? {}).length;
    else if (b?.type === "image") n += 4800;
  });
  return n;
}

export interface RetentionOptions {
  retainRaw: number;
  minPairBytes: number;
  staleDistance: number;
  keepResultHeadBytes: number;
  keepResultTailBytes: number;
  cmdKeepBytes: number;
  demoteBatch: number;
  demotePendingBytes: number;
}

export interface RecallOptions {
  defaultBytes: number;
  maxBytes: number;
}

export interface RetentionArchive {
  /** False when the copy could not be stored, which cancels the demotion. */
  save(id: string, text: string): boolean;
  /** Total archived byte size for `id`, or undefined when `id` is unknown. */
  size(id: string): number | undefined;
  /**
   * Up to `length` bytes starting at byte `start` — never the whole archive,
   * so paging a small slice out of a large one can't itself re-inflate the
   * memory usage this extension exists to reduce.
   */
  readRange(id: string, start: number, length: number): Buffer | undefined;
}

export function resolveOptions(): RetentionOptions {
  return {
    retainRaw: envNumber(ENV_RETAIN_RAW, DEFAULT_RETAIN_RAW),
    minPairBytes: envNumber(ENV_MIN_PAIR_BYTES, DEFAULT_MIN_PAIR_BYTES),
    staleDistance: envNumber(ENV_STALE_DISTANCE, DEFAULT_STALE_DISTANCE),
    keepResultHeadBytes: envNumber(ENV_KEEP_RESULT_HEAD, DEFAULT_KEEP_RESULT_HEAD),
    keepResultTailBytes: envNumber(ENV_KEEP_RESULT_TAIL, DEFAULT_KEEP_RESULT_TAIL),
    cmdKeepBytes: envNumber(ENV_CMD_KEEP, DEFAULT_CMD_KEEP),
    demoteBatch: envNumber(ENV_DEMOTE_BATCH, DEFAULT_DEMOTE_BATCH),
    demotePendingBytes: envNumber(ENV_DEMOTE_PENDING_BYTES, DEFAULT_DEMOTE_PENDING_BYTES),
  };
}

export function resolveRecallOptions(): RecallOptions {
  return {
    defaultBytes: envNumber(ENV_RECALL_DEFAULT, DEFAULT_RECALL_BYTES),
    maxBytes: envNumber(ENV_RECALL_MAX, DEFAULT_RECALL_MAX),
  };
}

/** Archive id for a pair. Deterministic so every re-projection maps to one file. */
export function archiveId(toolCallId: string): string {
  return `sr-${createHash("sha256").update(toolCallId).digest("hex").slice(0, 16)}`;
}

function contentText(m: any): string {
  if (typeof m?.content === "string") return m.content;
  if (Array.isArray(m?.content)) {
    return m.content
      .filter((c: any) => c?.type === "text")
      .map((c: any) => c.text ?? "")
      .join("\n");
  }
  return "";
}

/**
 * Keep the first `headBytes` and last `tailBytes` of `s`, preferring line
 * boundaries but never requiring one — the technique of shell-session's
 * capBytesHeadTail, with the joining marker left to the caller so demotion
 * markers stay outside that module's byte-identical marker contract.
 *
 * `dropped === 0` means the whole string fit and `tail` is empty.
 */
export function splitHeadTail(
  s: string,
  headBytes: number,
  tailBytes: number,
): { head: string; tail: string; dropped: number } {
  const buf = Buffer.from(s, "utf-8");
  if (buf.length <= headBytes + tailBytes) return { head: s, tail: "", dropped: 0 };

  let headEnd = Math.max(0, headBytes);
  // A negative byteOffset would search backward from the END of the buffer.
  // The `> 0` guard rejects a boundary newline that would leave head empty —
  // a long first line must still contribute something, not vanish entirely.
  const headNl = headEnd > 0 ? buf.lastIndexOf(0x0a, headEnd - 1) : -1;
  if (headNl > 0) {
    headEnd = headNl;
  } else {
    while (headEnd > 0 && (buf[headEnd] & 0xc0) === 0x80) headEnd--;
  }

  let tailStart = buf.length - Math.max(0, tailBytes);
  const tailNl = buf.indexOf(0x0a, tailStart);
  // The `+ 1 < buf.length` guard rejects a boundary newline that would leave
  // tail empty — the real failure signal often sits on the output's last
  // line, so silently dropping it here defeats the whole point of the tail.
  if (tailNl >= 0 && tailNl + 1 < buf.length) {
    tailStart = tailNl + 1;
  } else {
    while (tailStart < buf.length && (buf[tailStart] & 0xc0) === 0x80) tailStart++;
  }

  if (tailStart <= headEnd) return { head: s, tail: "", dropped: 0 };
  return {
    head: buf.subarray(0, headEnd).toString("utf-8"),
    tail: buf.subarray(tailStart).toString("utf-8"),
    dropped: tailStart - headEnd,
  };
}

// A real footer (the `[exit=…]` line, or GAIA's plain last line) is always
// short. Without this cap, a giant single-line result — no newline at all,
// or a huge final line like a recalled minified file — gets its whole body
// mistaken for "the footer" and copied verbatim, defeating demotion entirely.
const MAX_FOOTER_BYTES = 512;

/**
 * Split off the trailing `[exit=… cwd=… timed_out=…]` footer, or — for GAIA's
 * built-in bash, which has no such footer — the last literal line. Returns
 * an empty footer when the last line is too large to plausibly be one, so
 * the whole text is treated as body instead.
 */
export function splitFooter(text: string): { body: string; footer: string } {
  const lines = text.split("\n");
  let f = lines.length - 1;
  // Trailing blank lines would otherwise be kept in place of the footer.
  while (f > 0 && lines[f].trim() === "") f--;
  if (byteLen(lines[f]) > MAX_FOOTER_BYTES) return { body: text, footer: "" };
  return { body: lines.slice(0, f).join("\n"), footer: lines.slice(f).join("\n") };
}

/**
 * Replacement text for a demoted result, or null when it would not shrink the
 * message.
 *
 * The tail slice is not decoration: the harness footer reports the exit status
 * of the model's own wrapper, so a program that segfaulted under
 * `prog; echo "exit=$?"` still footers as `exit=0` and only the body's last
 * lines carry the real failure. Head-only would render that as a success.
 */
export function demoteResultText(
  resultText: string,
  toolName: string,
  id: string,
  opts: RetentionOptions,
): string | null {
  const { body, footer } = splitFooter(resultText);
  const { head, tail, dropped } = splitHeadTail(
    body,
    opts.keepResultHeadBytes,
    opts.keepResultTailBytes,
  );
  if (dropped <= 0) return null;

  const lines = [
    `${RESULT_DEMOTED_PREFIX} — ${formatSize(byteLen(resultText))} originally; ` +
      `ShellRecall id=${id} pages back the full command+output]`,
  ];
  if (toolName === "ShellLog") {
    lines.push("[ShellLog re-pages the live job buffer, which may have dropped its oldest lines]");
  }
  lines.push(head, `  [... ${formatSize(dropped)} demoted ...]`, tail);
  if (footer) lines.push(footer);

  const next = lines.join("\n");
  return byteLen(next) < byteLen(resultText) ? next : null;
}

/** Replacement command text, or null when it would not shrink the call. */
export function demoteCommandText(
  command: string,
  id: string,
  opts: RetentionOptions,
): string | null {
  if (byteLen(command) <= opts.cmdKeepBytes) return null;
  const half = Math.floor(opts.cmdKeepBytes / 2);
  // Often the first line names what was done, the last runs/builds it.
  const { head, tail, dropped } = splitHeadTail(command, half, opts.cmdKeepBytes - half);
  if (dropped <= 0) return null;

  const next = `${head}\n[... ${formatSize(dropped)} ${CMD_DEMOTED_INFIX}${id} ...]\n${tail}`;
  return byteLen(next) < byteLen(command) ? next : null;
}

export function archiveText(command: string, resultText: string): string {
  return command ? `$ ${command}\n\n${resultText}\n` : `${resultText}\n`;
}

interface Pair {
  resultIdx: number;
  callIdx: number;
  blockIdx: number;
  toolCallId: string;
  toolName: string;
  command: string;
  resultText: string;
  signed: boolean;
}

function collectPairs(messages: any[]): Pair[] {
  const calls = new Map<string, { msgIdx: number; blockIdx: number; block: any }>();
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    if (m?.role !== "assistant" || !Array.isArray(m.content)) continue;
    m.content.forEach((block: any, blockIdx: number) => {
      if (block?.type !== "toolCall") return;
      const id = block.id;
      if (typeof id !== "string" || calls.has(id)) return;
      calls.set(id, { msgIdx: i, blockIdx, block });
    });
  }

  const pairs: Pair[] = [];
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    if (m?.role !== "toolResult") continue;
    const toolName = String(m.toolName ?? "");
    if (!DEMOTABLE_TOOLS.has(toolName)) continue;
    const toolCallId = typeof m.toolCallId === "string" ? m.toolCallId : "";
    // Without it there is no stable archive id, so recall could not be honored.
    if (!toolCallId) continue;

    const call = calls.get(toolCallId);
    pairs.push({
      resultIdx: i,
      callIdx: call?.msgIdx ?? -1,
      blockIdx: call?.blockIdx ?? -1,
      toolCallId,
      toolName,
      command: typeof call?.block?.arguments?.command === "string"
        ? call.block.arguments.command
        : "",
      resultText: contentText(m),
      // A signature can land on a sibling text/thinking block rather than the
      // toolCall block itself (google-shared.js: "can appear on ANY part
      // type... does NOT necessarily correspond to the functionCall") —
      // the whole message counts as signed, not just its toolCall block.
      signed: call ? isSignedMessage(messages[call.msgIdx]) : false,
    });
  }
  return pairs;
}

// pi-ai's openai-completions provider sets thinkingSignature to the name of
// the delta field the reasoning streamed in (its `reasoningFields` list), so it
// can replay the thinking under that key. Nothing is signed, so the toolCall
// args are free to rewrite. The exemption needs both that api and one of those
// names: another provider's thinkingSignature, or a message without `api`,
// keeps the protection. retention.test.ts pins this list against the installed
// provider, since `reasoningFields` is internal to pi-ai.
export const REASONING_FIELD_NAMES = new Set(["reasoning_content", "reasoning", "reasoning_text"]);

function isSignedMessage(m: any): boolean {
  if (!Array.isArray(m?.content)) return false;
  const fieldNameMarkers = m.api === "openai-completions";
  return m.content.some(
    (b: any) =>
      b?.thoughtSignature !== undefined ||
      b?.textSignature !== undefined ||
      (b?.thinkingSignature !== undefined &&
        !(fieldNameMarkers && REASONING_FIELD_NAMES.has(b.thinkingSignature))),
  );
}

// Anchored to the exact marker shape and capture the id — a loose match
// (even one requiring the right shape) can't tell this pair's own marker
// apart from a complete, validly-shaped one copied from a different pair
// (e.g. quoted as an example in a heredoc) — this repo is self-hosted, so
// that shape can appear as plain command text with no demotion involved.
const CMD_DEMOTED_MARKER_RE = /^\[\.\.\. \S+ of command text demoted — ShellRecall id=(sr-[0-9a-f]{16}) \.\.\.\]$/m;
// Anchored to the complete first line, not just "starts with the prefix and
// mentions an id somewhere" — real output that happens to open with the
// prefix text could otherwise coincidentally satisfy a looser check.
const RESULT_DEMOTED_MARKER_RE =
  /^\[shell result demoted — \S+ originally; ShellRecall id=(sr-[0-9a-f]{16}) pages back the full command\+output\]/;

function alreadyDemoted(p: Pair): boolean {
  const ownId = archiveId(p.toolCallId);
  const resultMatch = RESULT_DEMOTED_MARKER_RE.exec(p.resultText);
  if (resultMatch && resultMatch[1] === ownId) return true;
  const cmdMatch = CMD_DEMOTED_MARKER_RE.exec(p.command);
  return cmdMatch !== null && cmdMatch[1] === ownId;
}

// ── Marker-echo tripwire (index.ts's tool_call hook) ────────────────────────
//
// A model can copy a demoted placeholder verbatim into a NEW tool call (e.g.
// reconstructing a file from "what I remember writing"), baking the
// placeholder text into real output. That's a different job from
// alreadyDemoted above (this extension's own command/result text vs.
// arbitrary tool-call input), so it gets its own, more tolerant pattern.
//
// Per-leaf pre-gate: marker-free strings (the vast majority) never reach the
// regex below.
const MARKER_ECHO_GATE = "ShellRecall id=sr-";

// Anchored on the id phrase alone, not the "demoted — …" prose around it
// (both the command-infix and result-prefix marker lines carry it). The
// id must be byte-exact anyway for the liveness check downstream to match
// it, while the dash just before it is exactly the byte a model
// paraphrasing its own context from memory is free to mangle (em dash vs
// `--`/`-`/`...`). Matching more of the surrounding prose would only make
// this stricter than MARKER_ECHO_GATE, opening a gap where the pre-gate
// admits a call this regex then fails to catch.
//
// The lookahead is that byte-exactness enforced. Without it a longer hex run
// still yields its first 16 characters, and if those name a live entry the
// guard blocks a call that never carried that id. Both marker shapes put a
// space right after the id, so the boundary costs no genuine match.
const MARKER_ECHO_ID_RE = /ShellRecall id=(sr-[0-9a-f]{16})(?![0-9a-fA-F])/g;

/**
 * All sr-… ids that appear, anywhere among `input`'s string leaves, in
 * demotion-marker shape. Doesn't check liveness — the caller cross-checks
 * against the archive, so a marker-shaped id naming nothing there (a fixture's,
 * or one left in a file by an earlier session) doesn't trip the guard.
 */
export function findMarkerEchoIds(input: unknown): string[] {
  const ids = new Set<string>();
  const stack: unknown[] = [input];
  while (stack.length > 0) {
    const v = stack.pop();
    if (typeof v === "string") {
      if (!v.includes(MARKER_ECHO_GATE)) continue;
      MARKER_ECHO_ID_RE.lastIndex = 0;
      let m: RegExpExecArray | null;
      while ((m = MARKER_ECHO_ID_RE.exec(v)) !== null) ids.add(m[1]);
    } else if (Array.isArray(v)) {
      stack.push(...v);
    } else if (v !== null && typeof v === "object") {
      stack.push(...Object.values(v as Record<string, unknown>));
    }
  }
  return [...ids];
}

/**
 * How many leading slots of the ordered demotable-pair list to demote, given
 * `target`, the count the per-pair rule (rank >= retainRaw or distance >
 * staleDistance) selects. Rounding down to a multiple of `batch` makes the
 * demoted prefix grow in jumps of `batch` pairs, so the rendered history stays
 * byte-identical between jumps. A `batch` that is not a finite number >= 1
 * means per-pair; a fractional one is floored.
 */
export function demotedPrefixLength(target: number, batch: number): number {
  const b = Number.isFinite(batch) && batch >= 1 ? Math.floor(batch) : 1;
  return b * Math.floor(target / b);
}

/** Counts from one projection; see demoteMessagesWithStats. */
export interface DemotionStats {
  /** Demotable (call, result) pairs in history. */
  pairs: number;
  /** Pairs at or over minPairBytes, plus any already carrying their own marker. */
  large: number;
  /** Pairs the per-pair rule selects (rank >= retainRaw or distance > staleDistance). */
  due: number;
  /** Pairs the batch/flush rule lets demote on this request. */
  prefix: number;
  /** Pairs replaced in this projection; equals demotedCount. */
  demoted: number;
  /** True when the pending-bytes flush overrode the batch boundary (the cost gate may still defer it). */
  flushed: boolean;
  /** Pairs in the prefix whose call message counts as signed, so their command is never rewritten. */
  signed: number;
  /** Pairs in the prefix where neither the result nor the command could shrink. */
  skippedNoShrink: number;
  /** Pairs in the prefix whose archive save was refused. */
  skippedArchive: number;
  /** Command + result bytes of the demoted pairs, before and after. */
  bytesBefore: number;
  bytesAfter: number;
  /**
   * Cost gate verdict: "off" (no gate input, unknown window, or disabled),
   * "none" (nothing due past the latched prefix), "open" or "deferred".
   */
  gate: "off" | "none" | "open" | "deferred";
  /** The clause that opened the jump, or on "deferred" the veto that held it (null: no clause opened). */
  gateReason: GateReason | VetoReason | null;
  /** Due pairs the gate kept raw on this request. */
  skippedCost: number;
  /** Pairs latched demoted by earlier requests (see demoteMessagesWithStats). */
  sticky: number;
  /**
   * Estimates for the jump the gate judged; 0 when it judged none. Save is chars / CHARS_PER_TOKEN;
   * re-prefill runs from the break's 4096-token block to the end of the cached prefix.
   */
  estSaveTokens: number;
  estReprefillTokens: number;
  estContextTokens: number;
  /** prefillSeconds over the estReprefillTokens before the end of the cached prefix; 0 when it judged none. */
  estReprefillSeconds: number;
  /** The configured ceiling (GateOptions.ceilingSeconds) when the gate ran, else 0. */
  ceilingSeconds: number;
  /** First token the judged jump would re-prefill (block-aligned); 0 when it judged none. */
  estReprefillFromToken: number;
}

export interface GateArgs {
  options: GateOptions;
  context: GateContext;
}

/**
 * Replace stale, oversized shell pairs with archived placeholders.
 *
 * Staleness is displacement by newer qualifying pairs plus a distance floor,
 * both recomputable from the messages alone — pi hands every LLM call a fresh
 * clone of pristine history, so no cross-call state may be relied on.
 *
 * Both triggers are monotone in age, so the pairs they select are always a
 * prefix of the size-qualifying pairs in history order. Only whole batches of
 * that prefix are demoted (demotedPrefixLength): every growth of the demoted
 * set rewrites history mid-prompt and forces the server to re-prefill from the
 * first rewritten pair on, so growing in jumps keeps the prompt byte-identical,
 * and its prefix cache warm, for every turn between jumps. The one exception
 * is byte pressure: when the due pairs past the last whole batch (not yet
 * demoted) hold more than demotePendingBytes raw, the prefix takes all of
 * them, so raw-but-due bytes never exceed that budget (unless the cost gate
 * below defers the flush; then forcePendingBytes bounds them, unless the
 * re-prefill ceiling vetoes that too). Bytes, not age, because
 * a late break re-prefills the most history. Until the batch boundary passes
 * them, those pending pairs only gain members as history is appended, so once
 * a flush fires every later call flushes again or the boundary has already
 * moved past them. A due pair under the budget waits for its batch, however
 * long that takes.
 *
 * A pair that cannot shrink, or whose archive save is refused, keeps its slot
 * and stays raw; the boundary is never pulled past it, so which pairs demote
 * depends on counts alone. A pair already carrying its own marker keeps its
 * slot too, so this function's output fed back in selects the same prefix and
 * demotes nothing further.
 *
 * The prefix only ever grows while history is appended to. A non-append edit
 * such as compaction recomputes it from scratch and may return pairs to raw,
 * but that edit has already invalidated the cached prefix anyway.
 *
 * With `gate`, the jump from the latched prefix to the one above must also
 * pass demotionGate and escape demotionVeto, or it is deferred and its pairs stay raw. The gate's
 * inputs are not monotone in history, so this mode reads one piece of
 * cross-call state: the archive, whose entries mark pairs an earlier request
 * demoted. Without `gate`, or with no context window, it is unchanged.
 *
 * `stats` describes this one projection. pi's context hook receives stored,
 * undemoted history on every request (pi-agent-core agent-loop.js:178-185:
 * transformContext's result goes only to convertToLlm), so `demoted` is how
 * many pairs this request's prompt carries demoted, not how many are new, and
 * `done` is true only for stored text that already carries its own marker.
 * benchmarks/turn_ledger.py diffs consecutive snapshots for "new".
 */
export function demoteMessagesWithStats(
  messages: any[],
  archive: RetentionArchive,
  opts: RetentionOptions = resolveOptions(),
  gate?: GateArgs,
): { messages: any[]; demotedCount: number; stats: DemotionStats } {
  const result = [...messages];
  const pairs = collectPairs(messages);
  const slots = pairs
    .map((p) => ({ p, done: alreadyDemoted(p) }))
    .filter(({ p, done }) => done || byteLen(p.command) + byteLen(p.resultText) >= opts.minPairBytes);

  let target = 0;
  for (let k = 0; k < slots.length; k++) {
    const rank = slots.length - 1 - k;
    const distance = messages.length - 1 - slots[k].p.resultIdx;
    if (rank >= opts.retainRaw || distance > opts.staleDistance) target = k + 1;
  }

  // Batch 1 gives batched === target, so nothing is pending and the flush
  // cannot change per-pair output.
  const batched = demotedPrefixLength(target, opts.demoteBatch);
  let pending = 0;
  for (let k = batched; k < target; k++) {
    const { p, done } = slots[k];
    if (!done) pending += byteLen(p.command) + byteLen(p.resultText);
  }
  const budget = opts.demotePendingBytes;
  const flush = Number.isFinite(budget) && budget > 0 && pending > budget;
  const candidate = flush ? target : batched;

  // Replacement texts, computed once: the gate's saving estimate and the
  // rewrite below must agree on what each pair turns into.
  const plans = new Map<number, { id: string; nextResult: string | null; nextCommand: string | null }>();
  const planFor = (k: number) => {
    let plan = plans.get(k);
    if (!plan) {
      const { p } = slots[k];
      const id = archiveId(p.toolCallId);
      plan = {
        id,
        nextResult: demoteResultText(p.resultText, p.toolName, id, opts),
        // Google-style providers replay a thoughtSignature bound to the original
        // args, so a rewritten command would be replayed against a stale signature.
        nextCommand: p.signed || p.callIdx < 0 ? null : demoteCommandText(p.command, id, opts),
      };
      plans.set(k, plan);
    }
    return plan;
  };

  const stats: DemotionStats = {
    pairs: pairs.length,
    large: slots.length,
    due: target,
    prefix: candidate,
    demoted: 0,
    flushed: flush,
    signed: 0,
    skippedNoShrink: 0,
    skippedArchive: 0,
    bytesBefore: 0,
    bytesAfter: 0,
    gate: "off",
    gateReason: null,
    skippedCost: 0,
    sticky: 0,
    estSaveTokens: 0,
    estReprefillTokens: 0,
    estContextTokens: 0,
    estReprefillSeconds: 0,
    ceilingSeconds: 0,
    estReprefillFromToken: 0,
  };

  let prefix = candidate;
  const window = gate?.context.contextWindow;
  if (
    gate && gate.options.openAtPercent > 0 &&
    typeof window === "number" && window >= MIN_GATE_WINDOW && window <= MAX_GATE_WINDOW
  ) {
    // The latch. Each request sees pristine history, and the gate's inputs
    // are not monotone in it (the context reading drops once a jump lands),
    // so a jump approved on one request could be denied on the next and its
    // pairs return to raw: a second prefix break. A pair demoted by an
    // earlier request has an archive entry (save precedes every rewrite, and
    // nothing else saves), so the prefix never shrinks below the newest one.
    // Clamped to the candidate, which only grows while history is appended
    // to: after a compaction, a changed knob or a reused toolCallId the latch
    // cannot demote anything today's rule would not.
    let sticky = 0;
    for (let k = 0; k < candidate; k++) {
      if (slots[k].done || archive.size(archiveId(slots[k].p.toolCallId)) !== undefined) sticky = k + 1;
    }
    stats.sticky = sticky;

    let totalChars = 0;
    let cachedEnd = 0;
    const before: number[] = [];
    messages.forEach((m, i) => {
      before.push(totalChars);
      totalChars += messageChars(m);
      // The server cached the prompt through the last response it produced;
      // what follows it is prefilled on this request whatever we do.
      if (m?.role === "assistant") cachedEnd = totalChars;
    });
    const savedChars = (from: number, to: number) => {
      let n = 0;
      for (let k = from; k < to; k++) {
        const { p, done } = slots[k];
        if (done) continue;
        const { nextResult, nextCommand } = planFor(k);
        if (nextResult !== null) n += p.resultText.length - nextResult.length;
        if (nextCommand !== null) n += p.command.length - nextCommand.length;
      }
      return n;
    };
    // Cold when pi has no usage reading (between a compaction and the next
    // response), or when the last response came from another model: either
    // way the server this request goes to holds no prefix to break.
    const lastAssistant = [...messages].reverse().find((m) => m?.role === "assistant");
    const want = gate.context.model;
    const switched = !!want && !!lastAssistant &&
      ((typeof lastAssistant.provider === "string" && typeof want.provider === "string" &&
        lastAssistant.provider !== want.provider) ||
       (typeof lastAssistant.model === "string" && typeof want.id === "string" && lastAssistant.model !== want.id));
    const cold = gate.context.contextTokens === null || switched;
    const contextTokens = gate.context.contextTokens ??
      Math.ceil((totalChars - savedChars(0, sticky)) / CHARS_PER_TOKEN);
    stats.estContextTokens = contextTokens;
    stats.ceilingSeconds = gate.options.ceilingSeconds;

    if (candidate <= sticky) {
      stats.gate = "none";
    } else {
      // The break starts at the first message the jump rewrites: the call's
      // toolCall block when its command shrinks, else the result.
      let breakAt = totalChars;
      let pendingBytes = 0;
      for (let k = sticky; k < candidate; k++) {
        const { p, done } = slots[k];
        if (done) continue;
        pendingBytes += byteLen(p.command) + byteLen(p.resultText);
        const { nextResult, nextCommand } = planFor(k);
        if (nextCommand !== null) {
          breakAt = Math.min(breakAt, before[p.callIdx] + messageChars(messages[p.callIdx], p.blockIdx));
        } else if (nextResult !== null) {
          breakAt = Math.min(breakAt, before[p.resultIdx]);
        }
      }
      // Positions in the server's tokens, counted back from the end of the
      // prompt: contextTokens is measured there, and the system prompt and
      // tool definitions ahead of the first message are not in the chars.
      const tokenAt = (chars: number) => contextTokens - Math.max(0, totalChars - chars) / CHARS_PER_TOKEN;
      const cachedEndToken = Math.max(0, tokenAt(cachedEnd));
      // A break past the cached prefix rewrites only what is prefilled anyway.
      const from = breakAt >= cachedEnd ? cachedEndToken : blockStart(tokenAt(breakAt), gate.options.prefixBlockTokens);
      const input: GateInput = {
        estSaveTokens: Math.ceil(savedChars(sticky, candidate) / CHARS_PER_TOKEN),
        estReprefillTokens: Math.ceil(Math.max(0, cachedEndToken - from)),
        contextTokens,
        contextWindow: window,
        pendingBytes,
        estReprefillSeconds: prefillSeconds(from, cachedEndToken, gate.options),
      };
      stats.estSaveTokens = input.estSaveTokens;
      stats.estReprefillTokens = input.estReprefillTokens;
      stats.estReprefillSeconds = Math.round(input.estReprefillSeconds * 10) / 10;
      stats.estReprefillFromToken = Math.floor(from);
      // A break on a cold cache is free.
      const opened = cold ? "cold" : demotionGate(input, gate.options);
      const veto = opened && opened !== "cold" ? demotionVeto(input, gate.options) : null;
      const reason = veto ? null : opened;
      stats.gateReason = veto ?? opened;
      if (reason) {
        stats.gate = "open";
        prefix = candidate;
      } else {
        stats.gate = "deferred";
        stats.skippedCost = candidate - sticky;
        prefix = sticky;
      }
    }
    stats.prefix = prefix;
  }

  let demotedCount = 0;
  for (let k = 0; k < prefix; k++) {
    const { p, done } = slots[k];
    if (done) continue;
    if (p.signed) stats.signed++;

    const { id, nextResult, nextCommand } = planFor(k);
    if (nextResult === null && nextCommand === null) {
      stats.skippedNoShrink++;
      continue;
    }
    if (!archive.save(id, archiveText(p.command, p.resultText))) {
      stats.skippedArchive++;
      continue;
    }

    if (nextResult !== null) {
      result[p.resultIdx] = {
        ...result[p.resultIdx],
        content: [{ type: "text" as const, text: nextResult }],
      };
    }
    if (nextCommand !== null) {
      const callMsg = result[p.callIdx];
      const content = callMsg.content.map((block: any, i: number) =>
        i === p.blockIdx
          ? { ...block, arguments: { ...block.arguments, command: nextCommand } }
          : block,
      );
      result[p.callIdx] = { ...callMsg, content };
    }
    stats.bytesBefore += byteLen(p.command) + byteLen(p.resultText);
    stats.bytesAfter += byteLen(nextCommand ?? p.command) + byteLen(nextResult ?? p.resultText);
    demotedCount++;
  }

  stats.demoted = demotedCount;
  return { messages: result, demotedCount, stats };
}

/** demoteMessagesWithStats without the stats: the shape the existing tests use. */
export function demoteMessages(
  messages: any[],
  archive: RetentionArchive,
  opts: RetentionOptions = resolveOptions(),
): { messages: any[]; demotedCount: number } {
  const { messages: out, demotedCount } = demoteMessagesWithStats(messages, archive, opts);
  return { messages: out, demotedCount };
}

// UTF-8 continuation bytes span at most 3 extra bytes on either side of a
// requested boundary — padding the disk read by this much gives the
// alignment walk real neighboring bytes to inspect without reading further.
const UTF8_BOUNDARY_PAD = 3;

/** One page of an archived pair, in ShellLog's paging-header style. */
export function recallSlice(
  id: string,
  archive: RetentionArchive,
  offset: number | undefined,
  bytes: number | undefined,
  opts: RecallOptions,
): { text: string; isError: boolean } {
  if (!id) return { text: "Error: id is required", isError: true };
  const total = archive.size(id);
  if (total === undefined) {
    return { text: `Error: no archived shell observation with id '${id}'`, isError: true };
  }

  const want = Math.max(
    1,
    Math.min(Number.isFinite(bytes as number) ? Number(bytes) : opts.defaultBytes, opts.maxBytes),
  );
  const wantStart = Math.max(0, Math.min(Number.isFinite(offset as number) ? Number(offset) : 0, total));
  const readStart = Math.max(0, wantStart - UTF8_BOUNDARY_PAD);
  const readEnd = Math.min(total, wantStart + want + UTF8_BOUNDARY_PAD);
  const buf = archive.readRange(id, readStart, readEnd - readStart);
  if (buf === undefined) {
    return { text: `Error: failed to read archived shell observation '${id}'`, isError: true };
  }

  let start = wantStart - readStart;
  while (start < buf.length && (buf[start] & 0xc0) === 0x80) start++;
  let end = Math.min(start + want, buf.length);
  while (end < buf.length && (buf[end] & 0xc0) === 0x80) end++;

  const absEnd = readStart + end;
  const header = `[${id} bytes ${readStart + start}–${absEnd} of ${total}]`;
  const slice = buf.subarray(start, end).toString("utf-8");
  const more = absEnd < total
    ? `\n[has_more=true — call ShellRecall with offset=${absEnd} for the next page]`
    : "";
  return { text: `${header}\n${slice}${more}`, isError: false };
}
