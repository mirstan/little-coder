// Detection and note-building for a shell command that truncated its OWN
// output with a trailing `head`/`tail`.
//
// The harness's 200-line/48KB cap marks its cuts, so a model can see them. A
// `| head -50` the model wrote itself leaves no mark at all: the result looks
// complete, and a model that concludes "X is absent" from it is wrong with no
// way to notice. This module spots that shape and appends one line saying the
// view may be partial.
//
// Everything here is a pure function of (command, result text) — no session
// state, no clock, no filesystem — so a given result always annotates the same
// way and every rule is testable without a shell.
//
// Honesty rule, the property this module lives or dies by: the note asserts
// only what is checked here — the command's final segment, and the size of
// the output being annotated. It never claims anything about what the
// truncator stage itself returned, because a multi-statement command composes
// output from stages this module does not analyze.

import { basename } from "node:path";
import { envNumber } from "../_shared/env-number.ts";
import { splitCommandChain, splitWords } from "../_shared/shell-write.ts";
import { MAX_RAW_HEAD_BYTES, MAX_RAW_TAIL_BYTES } from "../shell-session/helpers.ts";

export interface TrailingTruncator {
  tool: "head" | "tail";
  unit: "lines" | "bytes";
  limit: number;
}

export const MIN_LINES_ENV = "LITTLE_CODER_PARTIAL_VIEW_MIN_LINES";
export const MIN_BYTES_ENV = "LITTLE_CODER_PARTIAL_VIEW_MIN_BYTES";

// Below these, the model asked for a peek (`head -5`, `head -c 80`) and
// already knows it got one.
export const DEFAULT_MIN_LINES = 10;
export const DEFAULT_MIN_BYTES = 1024;

const DEFAULT_LINES = 10; // coreutils' count when no flag gives one

// Flags that carry no count, so skipping one cannot make the parsed limit
// disagree with what the shell will do.
const COUNTLESS_FLAGS = new Set([
  "-q",
  "-v",
  "-z",
  "--quiet",
  "--silent",
  "--verbose",
  "--zero-terminated",
]);

function parseCount(token: string | undefined): number | null {
  return token !== undefined && /^\d+$/.test(token) ? Number(token) : null;
}

/**
 * The `head`/`tail` limit a command ends in, or null.
 *
 * Only the final segment counts. Mid-pipeline (`… | head -50 | grep x`) the
 * limit says nothing about the output that comes back, and mid-chain
 * (`head -5 a && make`) the output can't be attributed to it. Both are
 * accepted false negatives.
 */
export function detectTrailingTruncator(command: string): TrailingTruncator | null {
  const segments = splitCommandChain(command);
  const last = segments[segments.length - 1];
  if (!last) return null;
  const words = splitWords(last);
  const tool = basename(words[0] ?? "");
  if (tool !== "head" && tool !== "tail") return null;

  let unit: "lines" | "bytes" = "lines";
  let limit: number | null = null;

  for (let i = 1; i < words.length; i++) {
    const word = words[i];
    if (word === "-" || !word.startsWith("-")) continue; // file operand
    if (COUNTLESS_FLAGS.has(word)) continue;

    let wordUnit: "lines" | "bytes";
    let raw: string | undefined;
    if (word === "-n" || word === "--lines") {
      wordUnit = "lines";
      raw = words[++i];
    } else if (word === "-c" || word === "--bytes") {
      wordUnit = "bytes";
      raw = words[++i];
    } else if (word.startsWith("--lines=")) {
      wordUnit = "lines";
      raw = word.slice("--lines=".length);
    } else if (word.startsWith("--bytes=")) {
      wordUnit = "bytes";
      raw = word.slice("--bytes=".length);
    } else if (word.startsWith("-n")) {
      wordUnit = "lines";
      raw = word.slice(2);
    } else if (word.startsWith("-c")) {
      wordUnit = "bytes";
      raw = word.slice(2);
    } else if (/^-\d+$/.test(word)) {
      wordUnit = "lines";
      raw = word.slice(1);
    } else {
      // An unread flag may still be the one setting the count (`--lines 50`,
      // `-qn50`, `-f`); falling back to 10 would put a false limit in the note.
      return null;
    }

    // `+5` (print from, not through), `-5` (GNU all-but-last), `1K`, `"50"`.
    const count = parseCount(raw);
    if (count === null) return null;
    unit = wordUnit;
    limit = count; // last count flag wins, as in coreutils
  }

  return { tool, unit, limit: limit ?? DEFAULT_LINES };
}

// Markers and footer are emitted byte-identically by shell-session/helpers.ts
// and its two Python twins, so they can be re-parsed here.
const FOOTER_RE = /^\[exit=-?\d+ cwd=.* timed_out=(?:true|false).*\]$/;
const RAW_BYTES_RE = /\braw_bytes=(\d+)\b/;
const DUPLICATE_MARKER_RE = /^\s*\[\.\.\. (\d+) duplicate line\(s\) collapsed \.\.\.\]$/;
const LINES_MARKER_RE = /^\s*\[\.\.\. (\d+) lines truncated \.\.\.\]$/;
const BYTES_MARKER_RE = /^\s*\[\.\.\. [\d.]+(?:B|KB|MB) truncated \.\.\.\]$/;
const OVERFLOW_NOTE_RE = /^(?:Full output|Partial output)\b/;

export interface ReconstructedCount {
  count: number;
  unknownCut: boolean;
}

/**
 * Lines the command produced, as a LOWER BOUND: each harness marker is
 * credited with the lines it replaced, but a marker can itself be swallowed by
 * a later stage's cut, and a byte cut has no line count to credit at all
 * (`unknownCut`). An under-count only suppresses a note, and every result it
 * can happen on already carries a harness marker of its own.
 *
 * pi's built-in bash tool truncates with a format of its own
 * (`[Showing lines X-Y of Z. …]`), deliberately not parsed: that footer
 * already tells the model the total, so the missed note costs nothing.
 */
export function reconstructLineCount(body: string): ReconstructedCount {
  const lines = body.split("\n");
  if (lines[lines.length - 1] === "") lines.pop();

  let count = 0;
  let unknownCut = false;
  for (const line of lines) {
    const duplicates = DUPLICATE_MARKER_RE.exec(line);
    if (duplicates) {
      count += Number(duplicates[1]);
      continue;
    }
    const truncated = LINES_MARKER_RE.exec(line);
    if (truncated) {
      count += Number(truncated[1]);
      continue;
    }
    if (BYTES_MARKER_RE.test(line)) {
      unknownCut = true;
      continue;
    }
    if (OVERFLOW_NOTE_RE.test(line)) continue;
    count += 1;
  }
  return { count, unknownCut };
}

/** Split a formatted result into its body and the `[exit=… ]` footer, if any. */
export function splitFooter(text: string): { body: string; footer: string | null } {
  const lines = text.split("\n");
  const last = lines[lines.length - 1] ?? "";
  if (!FOOTER_RE.test(last)) return { body: text, footer: null };
  return { body: lines.slice(0, -1).join("\n"), footer: last };
}

export function partialViewNote(truncator: TrailingTruncator, unknownCut: boolean): string {
  const { tool, unit, limit } = truncator;
  const spelling = unit === "bytes" ? `${tool} -c ${limit}` : `${tool} -${limit}`;
  const observed =
    unit === "bytes"
      ? `the output reaches ${limit} bytes`
      : unknownCut
        ? "the output was truncated before its line count could be established"
        : `the output reaches ${limit} lines`;
  const consequence =
    tool === "head"
      ? unit === "bytes"
        ? "the source may continue past what is shown"
        : "the source may continue past the last line shown"
      : unit === "bytes"
        ? "the source may have earlier content before what is shown"
        : "the source may have earlier lines before the first line shown";
  const advice =
    unit === "lines"
      ? " Don't conclude something is absent from this view — re-run with a larger " +
        "limit or a targeted filter (grep/awk) if it matters."
      : "";
  return `[partial view: this command ends in '${spelling}' and ${observed}, so ${consequence}.${advice}]`;
}

/**
 * `text` with the partial-view note added, or null to leave it unchanged.
 *
 * The note goes above the footer so the footer stays the last line, which
 * `_extract_exit_code`'s last-match rule and the log previewers rely on.
 */
export function annotate(command: string, text: string): string | null {
  const truncator = detectTrailingTruncator(command);
  if (!truncator) return null;
  const floor =
    truncator.unit === "lines"
      ? envNumber(MIN_LINES_ENV, DEFAULT_MIN_LINES)
      : envNumber(MIN_BYTES_ENV, DEFAULT_MIN_BYTES);
  if (truncator.limit < floor) return null;

  const { body, footer } = splitFooter(text);
  const footerRawBytes = footer ? Number(RAW_BYTES_RE.exec(footer)?.[1] ?? NaN) : NaN;

  let unknownCut = false;
  if (truncator.unit === "bytes") {
    const bytes = Number.isFinite(footerRawBytes)
      ? footerRawBytes
      : Buffer.byteLength(body, "utf-8");
    if (bytes < truncator.limit) return null;
  } else {
    const counted = reconstructLineCount(body);
    // The pre-dedup cap fires exactly when the raw output exceeds its budget,
    // and its marker is the one a later line cut can drop.
    unknownCut =
      counted.unknownCut ||
      (Number.isFinite(footerRawBytes) &&
        footerRawBytes > MAX_RAW_HEAD_BYTES + MAX_RAW_TAIL_BYTES);
    if (counted.count < truncator.limit && !unknownCut) return null;
  }

  const note = partialViewNote(truncator, unknownCut);
  return footer ? `${body}\n${note}\n${footer}` : `${body.trimEnd()}\n${note}`;
}
