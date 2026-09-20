// Pure logic for the post-write syntax check (see index.ts for the wiring).
//
// Everything here is a function of (command, paths, checker output) — no
// shell, no proxy, no clock — so every rule is testable without a container.

import { basename } from "node:path";
import { detectDeliverableWrites, splitCommandChain, splitWords } from "../_shared/shell-write.ts";
import { resolveShellTarget } from "../checkpoint/index.ts";
import { splitFooter } from "../truncated-view/truncation.ts";

export const SYNTAX_CHECK_ENV = "LITTLE_CODER_SYNTAX_CHECK";

/** Read at call time, not module load, so a mid-session unset takes effect. */
export function syntaxCheckEnabled(): boolean {
  return process.env[SYNTAX_CHECK_ENV] !== "0";
}

/** Echoed by a check whose checker isn't installed — silence, not a verdict. */
export const SKIP_SENTINEL = "__LC_SKIP__";

export const CHECK_TIMEOUT_SEC = 10;
export const MAX_FILES_PER_COMMAND = 3;
export const MAX_DIAG_LINES = 12;
export const MAX_DIAG_CHARS = 1000;

const TRUNCATION_MARK = "[… truncated]";

export interface Checker {
  /** Quoted back to the model in the diagnostic header. */
  label: string;
  /** Keys the per-session "this checker is missing" cache. */
  key: string;
  /** Shell snippet run with `$f` already bound to the file. */
  body: string;
}

// Only checks a container running this language already has the binary for
// (if the model wrote .pl it is about to run perl). Syntax-only throughout:
// no imports resolved, no project flags — a not-yet-installed dependency or a
// missing -D must never surface as "your file is wrong".
const CHECKERS: Record<string, Checker> = {};

function register(extensions: string[], checker: Checker): void {
  for (const ext of extensions) CHECKERS[ext] = checker;
}

register([".py"], {
  label: "python3 ast.parse",
  key: "python3",
  body:
    `if command -v python3 >/dev/null 2>&1; then ` +
    `python3 -c 'import ast,sys; ast.parse(open(sys.argv[1],"rb").read(), sys.argv[1])' "$f"; ` +
    `else echo ${SKIP_SENTINEL}; fi`,
});

register([".pl", ".pm", ".t"], {
  label: "perl -c",
  key: "perl",
  // `-c` runs BEGIN/use blocks. Accepted: the file is model-authored, runs
  // in-container, and the model was about to execute it anyway.
  body: `if command -v perl >/dev/null 2>&1; then perl -c "$f"; else echo ${SKIP_SENTINEL}; fi`,
});

register([".c", ".h"], {
  label: "gcc/cc -fsyntax-only",
  key: "cc",
  body:
    `if __cc=$(command -v gcc || command -v cc); then ` +
    `"$__cc" -fsyntax-only -I. -I"$(dirname "$f")" "$f"; ` +
    `else echo ${SKIP_SENTINEL}; fi`,
});

register([".sh", ".bash"], {
  label: "bash/sh -n",
  key: "bash",
  body:
    `if command -v bash >/dev/null 2>&1; then bash -n "$f"; ` +
    `elif command -v sh >/dev/null 2>&1; then sh -n "$f"; ` +
    `else echo ${SKIP_SENTINEL}; fi`,
});

export function checkerFor(path: string): Checker | null {
  const name = basename(path).toLowerCase();
  const dot = name.lastIndexOf(".");
  if (dot <= 0) return null;
  return CHECKERS[name.slice(dot)] ?? null;
}

function shellQuote(s: string): string {
  return `'${s.replace(/'/g, `'\\''`)}'`;
}

/**
 * The full one-shot check. Fail-open at both ends: a write that never landed
 * and a checker that is not installed both end in silence, never a diagnostic.
 */
export function buildCheckCommand(path: string, checker: Checker): string {
  return `f=${shellQuote(path)}; [ -f "$f" ] || exit 0; ${checker.body}`;
}

// Interpreters and compilers whose presence in the same command means the
// model is about to see the compile error anyway.
const RUNNERS = new Set([
  "perl", "python", "python3", "bash", "sh", "node", "gcc", "cc", "cobc", "prove", "make",
]);

function isRunner(word: string): boolean {
  const name = basename(word);
  // The set cannot enumerate every pinned version (`python3.11`).
  return RUNNERS.has(name) || /^python\d[\d.]*$/.test(name);
}

// Matches _shared/shell-write.ts's own (unexported) unquote exactly: a
// quoted word loses its quotes, an unquoted one loses backslash escapes
// instead -- never both, since a real shell never applies both to one word.
// Kept in sync by hand rather than importing, since that helper isn't
// exported; drift here would only ever cost a redundant round trip (a missed
// skip), not a wrong diagnostic, so it's a low-severity thing to keep in sync.
function unquote(word: string): string {
  if (word.length >= 2 && (word[0] === '"' || word[0] === "'") && word[word.length - 1] === word[0]) {
    return word.slice(1, -1);
  }
  return word.replace(/\\(.)/g, "$1");
}

/**
 * Does `segments` (the command chain from the write onward) already execute
 * `rawPath`?
 *
 * The model commonly writes `cat > f.pl <<EOF … EOF` and runs `perl f.pl` in
 * the same call, where the run itself reports the syntax error — a second
 * check would only duplicate it and pay another round trip. Matching is on
 * the path as SPELLED in the command, not the resolved absolute path, which
 * never appears in the segment. The basename-only fallback is scoped to a
 * bare filename (no directory component) — `rawPath` naming a directory
 * (`/app/f.pl`) requires an exact operand match, so a run of an unrelated
 * `/tmp/f.pl` sharing the same basename can't false-match it.
 */
export function commandRunsFile(segments: string[], rawPath: string): boolean {
  const bareBase = rawPath.includes("/") ? null : rawPath;
  for (const segment of segments) {
    const words = splitWords(segment);
    if (words.length === 0 || !isRunner(words[0])) continue;
    for (const word of words.slice(1)) {
      const operand = unquote(word);
      if (operand === rawPath) return true;
      if (bareBase !== null && basename(operand) === bareBase) return true;
    }
  }
  return false;
}

export interface CheckTarget {
  /** As spelled in the command — what the skip heuristic matches on. */
  raw: string;
  /** Resolved against the shell's cwd — what the checker is handed. */
  path: string;
  checker: Checker;
}

/**
 * Files a command wrote that are worth checking, in command order.
 *
 * Walks the chain one segment at a time so the skip heuristic only sees
 * segments AFTER a given write -- `perl f.pl; cat > f.pl <<EOF …` ran the
 * PRE-write file, so that run must not suppress the check on what just
 * replaced it. `splitCommandChain` strips heredoc bodies itself, so a
 * `perl` line inside the written file's own text can't be mistaken for a
 * run in either this loop or `commandRunsFile`.
 */
export function checkTargets(cmd: string, cwd: string): CheckTarget[] {
  const targets: CheckTarget[] = [];
  const seen = new Set<string>();
  const segments = splitCommandChain(cmd);
  outer: for (let i = 0; i < segments.length; i++) {
    for (const write of detectDeliverableWrites(segments[i])) {
      const checker = checkerFor(write.path);
      if (!checker) continue;
      if (commandRunsFile(segments.slice(i + 1), write.path)) continue;
      const path = resolveShellTarget(write.path, cwd);
      if (seen.has(path)) continue;
      seen.add(path);
      targets.push({ raw: write.path, path, checker });
      if (targets.length >= MAX_FILES_PER_COMMAND) break outer;
    }
  }
  return targets;
}

const FOOTER_EXIT_RE = /^\[exit=(-?\d+) /;
// Greedy up to ` timed_out=`, because a cwd may legitimately contain spaces.
const FOOTER_CWD_RE = / cwd=(.*) timed_out=/;
const FOOTER_TIMED_OUT_RE = / timed_out=(true|false)/;

export function footerExit(footer: string | null): number | null {
  const m = footer && FOOTER_EXIT_RE.exec(footer);
  return m ? Number(m[1]) : null;
}

export function footerCwd(footer: string | null): string | null {
  const m = footer && FOOTER_CWD_RE.exec(footer);
  const cwd = m?.[1]?.trim();
  return cwd ? cwd : null;
}

export function footerTimedOut(footer: string | null): boolean {
  const m = footer && FOOTER_TIMED_OUT_RE.exec(footer);
  return m?.[1] === "true";
}

export type CheckResult =
  | { kind: "verdict"; exit: number; output: string }
  /** Checker absent — cached, so the next file of that language skips the trip. */
  | { kind: "skip" }
  /** Fail open: no footer, a timeout, nothing usable to say. */
  | { kind: "none" };

// Every checker's `else` branch is a bare `echo SKIP_SENTINEL` with nothing
// else on that shell invocation, so a genuine skip is the WHOLE trimmed
// output, not merely a substring of it -- a real syntax error whose message
// happens to quote this exact token (a file testing marker strings, say;
// this repo is self-hosted) must not be mistaken for a missing checker.
export function classifyCheckOutput(exit: number, output: string): CheckResult {
  if (output.trim() === SKIP_SENTINEL) return { kind: "skip" };
  return { kind: "verdict", exit, output };
}

/**
 * Read a proxied check's own result, which the adapter wraps in the same
 * `[exit=… ]` footer as any model-issued command — so the exit code comes from
 * that footer, never from raw stdout.
 */
export function readCheckResult(text: string): CheckResult {
  const { body, footer } = splitFooter(text);
  if (footer === null || footerTimedOut(footer)) return { kind: "none" };
  const exit = footerExit(footer);
  if (exit === null) return { kind: "none" };
  return classifyCheckOutput(exit, body);
}

/** Advisory by design — the extension never blocks, edits, or re-runs. */
export function formatDiagnostic(a: {
  label: string;
  path: string;
  exit: number;
  output: string;
}): string {
  const header =
    `[syntax-check] ${a.label} ${a.path} FAILED (exit ${a.exit}) — fix before running:`;
  let lines = a.output.replace(/\s+$/, "").split("\n").filter((l) => l.trim() !== "");
  let truncated = false;

  const room = MAX_DIAG_LINES - 1;
  if (lines.length > room) {
    lines = lines.slice(0, room - 1); // leaves a line for the mark
    truncated = true;
  }
  let block = [header, ...lines].join("\n");

  const charRoom = MAX_DIAG_CHARS - TRUNCATION_MARK.length - 1;
  if (block.length > MAX_DIAG_CHARS || (truncated && block.length > charRoom)) {
    block = block.slice(0, charRoom).replace(/\s+$/, "");
    truncated = true;
  }
  return truncated ? `${block}\n${TRUNCATION_MARK}` : block;
}

/**
 * `text` with `block` added above its `[exit=… ]` footer.
 *
 * Not a trailing append: `truncated-view` also hooks tool_result on the same
 * results and locates the footer as the last line, so a diagnostic appended
 * after it would break whichever of the two runs second.
 */
export function insertAboveFooter(text: string, block: string): string {
  const { body, footer } = splitFooter(text);
  if (footer !== null) return `${body}\n${block}\n${footer}`;
  return body.endsWith("\n") ? `${body}${block}` : `${body}\n${block}`;
}
