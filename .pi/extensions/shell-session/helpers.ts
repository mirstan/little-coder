// Shared output-formatting helpers for ShellSession. Mirrors
// local/tools/shell_session.py's _strip_ansi / _dedup_lines / _truncate_lines
// so output reaches the model in the same format across backends.

import { randomBytes } from "node:crypto";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const ANSI_RE = /\x1b\[[0-?]*[ -/]*[@-~]/g;
export const MAX_LINES = 200;
export const DEFAULT_TIMEOUT = 30;

// Byte caps, which the line cap alone cannot enforce: one 1MB line is one
// "line", so a `grep` hit on a single-line JSON file used to reach the model
// whole and blow the context window. Head/tail split mirrors the 2:1 ratio
// truncateLines already uses.
export const MAX_BODY_HEAD_BYTES = 32 * 1024;
export const MAX_BODY_TAIL_BYTES = 16 * 1024;
// Pre-dedup gate: bounds the cost of split/dedupLines, which otherwise walk
// the whole output before any truncation runs.
export const MAX_RAW_HEAD_BYTES = 256 * 1024;
export const MAX_RAW_TAIL_BYTES = 128 * 1024;

// Stops a pathological loop from filling the disk with overflow files. Counts
// bytes written for the life of the pi process, i.e. one trial.
export const OVERFLOW_BUDGET_BYTES = 512 * 1024 * 1024;
let overflowBytesWritten = 0;

export function stripAnsi(s: string): string {
  return s.replace(ANSI_RE, "");
}

export function byteLen(s: string): number {
  return Buffer.byteLength(s, "utf-8");
}

/** Human-readable byte count, matching pi's own truncation markers. */
export function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}

/**
 * Keep the first `headBytes` and last `tailBytes` of `s`, joined by a marker.
 *
 * Deliberately local rather than pi's exported `truncateHead`: that one
 * returns *empty* content when the first line alone exceeds the byte budget,
 * which is exactly the giant-single-line case this exists to survive. Cuts
 * prefer a line boundary but must never require one.
 */
export function capBytesHeadTail(
  s: string,
  headBytes: number,
  tailBytes: number,
): { text: string; dropped: number } {
  const buf = Buffer.from(s, "utf-8");
  if (buf.length <= headBytes + tailBytes) return { text: s, dropped: 0 };

  // Last newline in the head window, so the kept head is as long as the
  // budget allows; the first newline would legally cut at byte 10 of 32K.
  let headEnd = headBytes;
  const headNl = buf.lastIndexOf(0x0a, headEnd - 1);
  if (headNl >= 0) {
    headEnd = headNl;
  } else {
    // Back off continuation bytes, keeping the shorter valid prefix.
    while (headEnd > 0 && (buf[headEnd] & 0xc0) === 0x80) headEnd--;
  }

  let tailStart = buf.length - tailBytes;
  const tailNl = buf.indexOf(0x0a, tailStart);
  if (tailNl >= 0) {
    tailStart = tailNl + 1;
  } else {
    // Mirror image of the head side: skip *forward* off a continuation byte.
    while (tailStart < buf.length && (buf[tailStart] & 0xc0) === 0x80) tailStart++;
  }

  const dropped = tailStart - headEnd;
  const head = buf.subarray(0, headEnd).toString("utf-8");
  const tail = buf.subarray(tailStart).toString("utf-8");
  return { text: `${head}\n  [... ${formatSize(dropped)} truncated ...]\n${tail}`, dropped };
}

export function dedupLines(lines: string[]): string[] {
  const out: string[] = [];
  let last: string | null = null;
  let dup = 0;
  for (const ln of lines) {
    if (ln === last) {
      dup++;
      continue;
    }
    if (dup > 0) out.push(`  [... ${dup} duplicate line(s) collapsed ...]`);
    dup = 0;
    out.push(ln);
    last = ln;
  }
  if (dup > 0) out.push(`  [... ${dup} duplicate line(s) collapsed ...]`);
  return out;
}

export function truncateLines(lines: string[], cap = MAX_LINES): { lines: string[]; truncated: boolean } {
  if (lines.length <= cap) return { lines, truncated: false };
  const head = Math.floor(cap / 2);
  const tail = Math.floor(cap / 4);
  const skipped = lines.length - head - tail;
  return {
    lines: [...lines.slice(0, head), `  [... ${skipped} lines truncated ...]`, ...lines.slice(-tail)],
    truncated: true,
  };
}

// Overflow content is arbitrary command output — env dumps, credential files,
// private source — so it must not land loose in a shared world-readable /tmp.
// mkdtempSync gives a 0700 directory with an unguessable name (matching
// deep-research/index.ts), which also denies an attacker the pre-created
// symlink that a predictable path in /tmp invites. One per process, made on
// first use so a session that never overflows leaves nothing behind.
let overflowDir: string | null = null;
let overflowDirUnavailable = false;

function ensureOverflowDir(): string | null {
  if (overflowDir || overflowDirUnavailable) return overflowDir;
  try {
    overflowDir = mkdtempSync(join(tmpdir(), "lc-shell-"));
  } catch {
    overflowDirUnavailable = true; // tmp isn't writable; stop retrying
  }
  return overflowDir;
}

/**
 * Save the captured output so the model can still read what was cut. Only the
 * local subprocess backend may call this: under the tmux/harbor proxies the
 * command runs in a container that cannot see this host path.
 *
 * `captureTruncated` means the caller's own buffer already lost the tail, so
 * the file is a prefix and must not be advertised as the whole output.
 *
 * Returns the model-facing line, or null when nothing was written.
 */
function writeOverflowFile(cleaned: string, captureTruncated: boolean): string | null {
  const label = captureTruncated
    ? "Partial output (first ~10MB only; command exceeded the capture buffer)"
    : "Full output";
  const bytes = byteLen(cleaned);
  if (overflowBytesWritten + bytes > OVERFLOW_BUDGET_BYTES) {
    return `${label}: not saved (per-session overflow-file budget exhausted)`;
  }
  const dir = ensureOverflowDir();
  if (!dir) return null;
  const path = join(dir, `${process.pid}-${Date.now()}-${randomBytes(4).toString("hex")}.log`);
  try {
    writeFileSync(path, cleaned, { encoding: "utf-8", mode: 0o600, flag: "wx" });
  } catch {
    // "wx" refuses an existing path rather than overwriting it, so a name
    // collision surfaces here as EEXIST. Dropping the note is deliberate:
    // there is no logger in this module, and the caller already renders a
    // missing note as "no overflow file" rather than as an error.
    return null;
  }
  overflowBytesWritten += bytes;
  return `${label}: ${path}`;
}

// execSubprocess's real SIGTERM path: Node actually killed the child, so the
// process is confirmed gone and cleanup logic confirmed did not run.
const KILLED_TIMED_OUT_WARNING =
  "WARNING: this command hit its timeout and was killed. Any file it was " +
  "mid-way through writing may now be HALF-WRITTEN, and any cleanup/restore logic at " +
  "the end of a script did NOT run -- re-verify (cat/wc/diff) any file it touched " +
  "before trusting it. If it simply needed more time, re-run with a larger timeout.";

// execTmuxProxy's no-response fallback: nothing was killed or interrupted --
// the command is likely still executing in the tmux pane, and no output was
// recovered at all (the "response" that would have carried it never came
// back). Deliberately does NOT suggest a re-run: re-running a command that
// may still be executing would start a duplicate copy with duplicate side
// effects.
const UNKNOWN_TIMED_OUT_WARNING =
  "WARNING: this command hit its timeout with no response from the session -- nothing " +
  "was killed or interrupted, and it is likely STILL RUNNING. No output was recovered " +
  "for this call, and any file it is writing may be incomplete. Do NOT re-run it -- " +
  "check whether it is still running first (e.g. capture the pane, or ps in the " +
  "session) to avoid starting a duplicate copy.";

export function formatOutput(
  raw: string,
  code: number,
  cwd: string,
  timedOut: boolean,
  backendNote: string,
  opts: { overflowFile?: boolean; captureTruncated?: boolean; timedOutKind?: "killed" | "unknown" } = {},
): string {
  const cleaned = stripAnsi(raw).replace(/\r/g, "");
  const rawBytes = byteLen(cleaned);
  const pre = capBytesHeadTail(cleaned, MAX_RAW_HEAD_BYTES, MAX_RAW_TAIL_BYTES);
  const dedupped = dedupLines(pre.text.split("\n"));
  const { lines, truncated } = truncateLines(dedupped);
  const post = capBytesHeadTail(lines.join("\n"), MAX_BODY_HEAD_BYTES, MAX_BODY_TAIL_BYTES);
  let body = post.text;
  const byteCapped = pre.dropped > 0 || post.dropped > 0;

  if (byteCapped && opts.overflowFile) {
    const note = writeOverflowFile(cleaned, opts.captureTruncated === true);
    if (note) body = body ? `${body}\n${note}` : note;
  }

  if (timedOut) {
    const warning = opts.timedOutKind === "unknown" ? UNKNOWN_TIMED_OUT_WARNING : KILLED_TIMED_OUT_WARNING;
    body = body ? `${body}\n${warning}` : warning;
  }

  const footerBits = [`exit=${code}`, `cwd=${cwd}`, `timed_out=${timedOut ? "true" : "false"}`];
  if (truncated || byteCapped) {
    footerBits.push("output_truncated=true");
    // Only alongside output_truncated: untruncated output is its own raw size,
    // and the existing footer shape stays byte-identical for normal results.
    footerBits.push(`raw_bytes=${rawBytes}`);
  }
  // rawBytes counts what the caller managed to capture, which for an ENOBUFS
  // result is a ~10MB prefix of what the command actually produced.
  if (opts.captureTruncated) footerBits.push("raw_bytes_exact=false");
  if (backendNote) footerBits.push(backendNote);
  const footer = `[${footerBits.join(" ")}]`;
  return body ? `${body}\n${footer}` : footer;
}
