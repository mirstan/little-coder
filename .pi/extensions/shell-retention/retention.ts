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

export interface RetentionOptions {
  retainRaw: number;
  minPairBytes: number;
  staleDistance: number;
  keepResultHeadBytes: number;
  keepResultTailBytes: number;
  cmdKeepBytes: number;
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

function isSignedMessage(m: any): boolean {
  if (!Array.isArray(m?.content)) return false;
  return m.content.some(
    (b: any) =>
      b?.thoughtSignature !== undefined ||
      b?.textSignature !== undefined ||
      b?.thinkingSignature !== undefined,
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

/**
 * Replace stale, oversized shell pairs with archived placeholders.
 *
 * Staleness is displacement by newer qualifying pairs plus a distance floor,
 * both recomputable from the messages alone — pi hands every LLM call a fresh
 * clone of pristine history, so no cross-call state may be relied on.
 */
export function demoteMessages(
  messages: any[],
  archive: RetentionArchive,
  opts: RetentionOptions = resolveOptions(),
): { messages: any[]; demotedCount: number } {
  const result = [...messages];
  const pairs = collectPairs(messages);
  const qualifying = pairs.filter(
    (p) =>
      !alreadyDemoted(p) &&
      byteLen(p.command) + byteLen(p.resultText) >= opts.minPairBytes,
  );

  let demotedCount = 0;
  for (let k = 0; k < qualifying.length; k++) {
    const p = qualifying[k];
    const rank = qualifying.length - 1 - k;
    const distance = messages.length - 1 - p.resultIdx;
    if (rank < opts.retainRaw && distance <= opts.staleDistance) continue;

    const id = archiveId(p.toolCallId);
    const nextResult = demoteResultText(p.resultText, p.toolName, id, opts);
    // Google-style providers replay a thoughtSignature bound to the original
    // args, so a rewritten command would be replayed against a stale signature.
    const nextCommand = p.signed || p.callIdx < 0
      ? null
      : demoteCommandText(p.command, id, opts);
    if (nextResult === null && nextCommand === null) continue;
    if (!archive.save(id, archiveText(p.command, p.resultText))) continue;

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
    demotedCount++;
  }

  return { messages: result, demotedCount };
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
