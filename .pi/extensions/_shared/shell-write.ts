// Shell command analysis for the write/permission guards (issue #70).
//
// Background: little-coder refuses `write` on an existing file so small models
// are pushed to `Edit` instead of rewriting whole files, and it whitelists bash
// commands. Both guards matched on tool NAME — write-guard on `write`,
// permission-gate on `bash` — so once `write` got refused, the model could
// reach the same bytes through a shell:
//
//     cat > backend/main.py << 'ENDOFFILE'
//     … whole file …
//     ENDOFFILE
//
// rvanswieten saw exactly that 5x in one session. And `ShellSession` hit
// neither guard at all. The bash whitelist wouldn't have caught it either:
// `isSafeBash` was `startsWith` on the raw string, and "cat" is whitelisted —
// with no awareness of the `>` sitting right after it.
//
// Matching on the heredoc delimiter would be trivially routed around with a
// different delimiter, so this module looks at what actually writes: shell
// redirection. Everything here is pure and string-only so it can be unit
// tested without a shell.
//
// This dir has no `index.ts` on purpose — the launcher's extension discovery
// requires one, so `_shared` is skipped and stays a plain library.

/** An operator/keyword through which a command writes to a path. */
/**
 * Every tool that hands a string to a shell, and can therefore reach the
 * filesystem without going near the `write` tool.
 *
 * This list is shared by permission-gate and write-guard deliberately. Issue
 * #70 happened because each guard matched its own hardcoded set: the gate knew
 * about `bash` but not `ShellSession`, so a model refused a whole-file write
 * simply re-ran it through the other tool and landed in an unguarded execSync.
 * Two copies of a security-relevant list will drift again, and the drift is
 * invisible until someone exploits it — so there is one copy, here, and adding
 * a new shell tool means adding it once.
 *
 * `ShellSessionCwd` / `ShellSessionReset` take no command (fixed `pwd`, or a
 * no-op) and stay ungated. `ShellSend` writes to an already-running job's
 * stdin rather than starting a command, so it is gated by whatever approved
 * that job in the first place.
 */
export const SHELL_TOOLS: ReadonlySet<string> = new Set([
  "bash",
  "Bash",
  "ShellSession",
  "ShellStart",
]);

export type WriteKind =
  | "redirect"
  | "append"
  | "tee"
  | "dd"
  | "copy"
  | "move"
  | "inplace"
  | "compile";

export interface ShellWrite {
  /** The (possibly relative) path the command writes to. */
  path: string;
  kind: WriteKind;
  /**
   * For `copy`/`move` only: the source operands, in command order.
   *
   * `cp a.md docs/` writes `docs/a.md`, not `docs` — `path` alone cannot say
   * which file gets clobbered, because that depends on whether `path` is a
   * directory on disk, which this pure string module deliberately does not
   * look at. Consumers that care (checkpoint, deciding what to snapshot) stat
   * `path` themselves and recombine it with these basenames. Purely additive:
   * every existing consumer reads only `path`/`kind` and is unaffected.
   */
  sources?: string[];
}

// Operators that chain one command into the next. Splitting on these lets the
// whitelist judge every segment rather than only the first one, so
// `ls && rm -rf /` can't ride in on `ls`.
const CHAIN_OPERATORS = ["&&", "||", ";", "|", "\n"];

/**
 * Walk `cmd` once, tracking quote state, and hand each character to `visit`.
 *
 * Quote tracking is what keeps every consumer here from firing on text that
 * merely looks like shell syntax — `grep "a > b" file` writes nothing. A
 * backslash escape outside single quotes hides the next character too.
 */
function scan(
  cmd: string,
  visit: (ch: string, index: number, quoted: boolean) => void,
): void {
  let quote: '"' | "'" | null = null;
  for (let i = 0; i < cmd.length; i++) {
    const ch = cmd[i];
    if (quote === null && ch === "\\") {
      i++; // skip the escaped character entirely
      continue;
    }
    if (quote === "'" && ch === "\\") {
      // Backslash is literal inside single quotes — no escaping happens.
      visit(ch, i, true);
      continue;
    }
    if (quote === null && (ch === '"' || ch === "'")) {
      quote = ch;
      continue;
    }
    if (quote !== null && ch === quote) {
      quote = null;
      continue;
    }
    visit(ch, i, quote !== null);
  }
}

// `<< DELIM`, `<<-DELIM`, `<<'DELIM'`, `<<"DELIM"`. `<<<` is a here-string
// (single-line data, no body to strip) and must not match, hence the guard on
// a third `<` at the call site.
const HEREDOC_START = /<<-?[ \t]*(?:'([^']*)'|"([^"]*)"|([A-Za-z_][A-Za-z0-9_]*))/;

/**
 * Remove heredoc bodies, leaving the command line that opened them.
 *
 * A heredoc body is data, not shell syntax, and analyzing it produces noise in
 * both directions: a `>` in the payload (`if a > b:`) looks like a redirect,
 * and an apostrophe (`don't`) leaves the quote scanner stuck mid-string for
 * everything after it. The redirect we actually care about — the `>` in
 * `cat > main.py << 'EOF'` — always sits on the opening line, so dropping the
 * body loses nothing and removes every one of those false readings.
 */
export function stripHeredocBodies(cmd: string): string {
  let out = cmd;
  let searchFrom = 0;
  // One pass per heredoc; a command can open several.
  for (let guard = 0; guard < 32; guard++) {
    const rest = out.slice(searchFrom);
    const m = HEREDOC_START.exec(rest);
    if (!m || m.index === undefined) break;
    const at = searchFrom + m.index;
    if (out[at + 2] === "<") {
      // `<<<` here-string — nothing to strip, keep scanning past it.
      searchFrom = at + 3;
      continue;
    }
    const delim = m[1] ?? m[2] ?? m[3] ?? "";
    const bodyStart = out.indexOf("\n", at + m[0].length);
    if (bodyStart === -1 || !delim) {
      // Body never started (single-line command) — nothing to remove.
      searchFrom = at + m[0].length;
      continue;
    }
    const lines = out.slice(bodyStart + 1).split("\n");
    let consumed = 0;
    let closed = false;
    for (const line of lines) {
      consumed += line.length + 1;
      // `<<-` allows the terminator to be indented with tabs.
      if (line.trim() === delim) {
        closed = true;
        break;
      }
    }
    const bodyEnd = closed
      ? Math.min(bodyStart + consumed, out.length)
      : out.length; // unterminated heredoc: the rest of the string is body
    out = out.slice(0, at) + out.slice(bodyEnd);
    searchFrom = at;
  }
  return out;
}

/**
 * Split a command line into the individual commands it runs, on unquoted
 * `&&`, `||`, `;`, `|` and newlines. Heredoc bodies are stripped first so
 * their contents are never mistaken for commands.
 */
export function splitCommandChain(raw: string): string[] {
  const cmd = stripHeredocBodies(raw);
  const cuts: Array<{ at: number; len: number }> = [];
  scan(cmd, (_ch, i, quoted) => {
    if (quoted) return;
    for (const op of CHAIN_OPERATORS) {
      if (cmd.startsWith(op, i)) {
        // Don't cut inside an already-recorded operator (`&&` must not also
        // match as two separate cuts).
        const last = cuts[cuts.length - 1];
        if (last && i < last.at + last.len) return;
        cuts.push({ at: i, len: op.length });
        return;
      }
    }
  });

  const segments: string[] = [];
  let start = 0;
  for (const cut of cuts) {
    segments.push(cmd.slice(start, cut.at));
    start = cut.at + cut.len;
  }
  segments.push(cmd.slice(start));
  return segments.map((s) => s.trim()).filter((s) => s.length > 0);
}

// `2>`, `>&2`, `&>`, `1>&2` … — file-descriptor plumbing, not a file write.
// We only care about a redirect whose target is a path.
function isFdTarget(target: string): boolean {
  return target.startsWith("&");
}

// Redirecting to one of the kernel's special character devices destroys no
// file — `2>/dev/null` discards output, `>/dev/stdout` re-plumbs a stream — so
// the write guards must not treat these as file writes (issue #87). `/dev/null`
// is the overwhelmingly common one; small models emit it constantly.
// `/dev/fd/N` is fd plumbing by another spelling. Matched case-sensitively:
// these are the real POSIX device paths, not arbitrary files. Windows `nul` is
// deliberately NOT exempted here — write-guard blocks it as a reserved device
// name (issue #60) so a POSIX run can't author a landmine file named `nul`.
const NON_DESTRUCTIVE_TARGETS = new Set([
  "/dev/null",
  "/dev/stdout",
  "/dev/stderr",
  "/dev/stdin",
  "/dev/tty",
  "/dev/zero",
  "/dev/full",
  "/dev/random",
  "/dev/urandom",
]);

/** True for a redirect target that can't destroy a real file. */
export function isNonDestructiveTarget(path: string): boolean {
  return NON_DESTRUCTIVE_TARGETS.has(path) || /^\/dev\/fd\/\d+$/.test(path);
}

/**
 * Every path `cmd` writes to via shell redirection or a write-by-design tool.
 *
 * Covers `>`, `>>`, `tee` (with `-a`), and `dd of=`. Ignores fd duplication
 * (`2>&1`, `>&2`), process substitution (`>(…)`), here-strings/heredocs
 * (`<<`, `<<<` — those read), and anything inside quotes.
 *
 * Returns [] for a command that only reads, which is the common case, so
 * callers can treat a non-empty result as "this touches the filesystem".
 */
export function detectWriteTargets(raw: string): ShellWrite[] {
  const cmd = stripHeredocBodies(raw);
  const writes: ShellWrite[] = [];
  const redirects: Array<{ at: number; kind: WriteKind }> = [];

  scan(cmd, (ch, i, quoted) => {
    if (quoted || ch !== ">") return;
    // `>>` — record once, on the first angle bracket.
    if (cmd[i - 1] === ">") return;
    // `<>` opens read-write; treat as a write.
    const append = cmd[i + 1] === ">";
    redirects.push({ at: i + (append ? 2 : 1), kind: append ? "append" : "redirect" });
  });

  for (const { at, kind } of redirects) {
    const rest = cmd.slice(at);
    // Process substitution `>(cmd)` is not a file target.
    if (rest.trimStart().startsWith("(")) continue;
    // fd duplication / close (`2>&1`, `>&2`, `>&-`): no file involved. Checked
    // on the raw text because `&` is a word break, so `firstWord("&1")` is the
    // descriptor number alone and no longer looks like fd syntax.
    if (rest.startsWith("&")) continue;
    const target = firstWord(rest);
    if (!target || isFdTarget(target)) continue;
    writes.push({ path: unquote(target), kind });
  }

  for (const segment of splitCommandChain(cmd)) {
    const words = splitWords(segment);
    if (words.length === 0) continue;

    // `tee [-a] FILE…` writes every non-flag argument.
    if (words[0] === "tee") {
      for (const w of words.slice(1)) {
        if (w.startsWith("-")) continue;
        writes.push({ path: unquote(w), kind: "tee" });
      }
    }

    // `dd if=… of=PATH`
    if (words[0] === "dd") {
      for (const w of words.slice(1)) {
        if (w.startsWith("of=")) writes.push({ path: unquote(w.slice(3)), kind: "dd" });
      }
    }
  }

  // De-duplicate by path, keeping the first kind seen. Non-destructive device
  // targets (/dev/null, /dev/stderr, …) are dropped — redirecting to them
  // writes no file the guard needs to protect (issue #87).
  const seen = new Set<string>();
  return writes.filter((w) => {
    if (!w.path || seen.has(w.path) || isNonDestructiveTarget(w.path)) return false;
    seen.add(w.path);
    return true;
  });
}

/** True when the command writes to the filesystem through the shell. */
export function hasWriteRedirection(cmd: string): boolean {
  return detectWriteTargets(cmd).length > 0;
}

// ---------------------------------------------------------------------------
// detectDeliverableWrites — a broader superset, for non-gating consumers
// ---------------------------------------------------------------------------
// `detectWriteTargets` deliberately only covers redirection (`>`, `>>`,
// `tee`, `dd of=`) because it also feeds write-guard and permission-gate, and
// those two are security/permission gates: `cp`/`mv`/`sed -i`/a compiler's
// `-o` are all deliberately treated as SAFE, whitelisted, non-write commands
// there (see permission-gate's BUILTIN_SAFE_PREFIXES — "cp "/"mv " are
// explicitly routine filesystem scaffolding). Folding those into
// `detectWriteTargets` itself would make both guards start refusing commands
// they intentionally let through today (confirmed: permission-gate's own
// test asserts `isSafeBash("cp a b") === true`).
//
// This function's consumers have a different question to answer than
// write-guard/permission-gate's "is this command safe to run": tb-finalize-
// guard asks "did the model do something that plausibly produced its
// deliverable" (evidence-of-work), and checkpoint asks "might this command
// destroy a file I haven't snapshotted yet" (pre-write backup) — both are
// non-gating, best-effort consumers where over-detection is acceptable, unlike
// the two write-permission gates above. For those purposes `cp`/`mv`/
// `install`/`sed -i`/a compiler's `-o` are all evidence of a write, so this
// function layers detection for those on top of `detectWriteTargets`.
// Short flags of `cp`/`mv`/`install` whose value is a SEPARATE word, so the
// value must be consumed rather than read as an operand: install's `-m` mode,
// `-o` owner and `-g` group, and the `-S` backup suffix. Skipping only the
// flag itself made `install -m 644 out.bin /app/bin/` see `644` as a source
// file. It never affected the target (still the last operand), which is why
// this went unnoticed while `sources` did not exist.
const VALUE_FLAGS = new Set(["-m", "-o", "-g", "-S"]);

function lastOperandOrTargetFlag(
  words: string[],
): { target: string | undefined; sources: string[] } {
  let tDir: string | undefined;
  const operands: string[] = [];
  for (let i = 1; i < words.length; i++) {
    const w = words[i];
    if (w === "-t") {
      tDir = words[i + 1];
      i++;
      continue;
    }
    if (VALUE_FLAGS.has(w)) {
      i++; // consume the flag's value word
      continue;
    }
    if (w.startsWith("--target-directory=")) {
      tDir = w.slice("--target-directory=".length);
      continue;
    }
    if (w.startsWith("-")) continue; // flag — skip
    operands.push(w);
  }
  // With an explicit `-t DIR` every operand is a source; otherwise the last
  // operand is the destination and everything before it is a source.
  return tDir
    ? { target: tDir, sources: operands }
    : { target: operands[operands.length - 1], sources: operands.slice(0, -1) };
}

// `-i`, `-i.bak` (GNU, suffix glued on), `--in-place`, `--in-place=.bak`.
// BSD sed's mandatory suffix arg (`-i ''`) is NOT specially handled — this is
// a heuristic feeding a nudge, not a security gate, so over-detecting a
// trailing `''`/suffix token as an extra "target" is an acceptable false
// positive (better than missing a real in-place edit).
function hasSedInPlaceFlag(words: string[]): boolean {
  return words.some(
    (w) => w === "-i" || w.startsWith("-i") || w === "--in-place" || w.startsWith("--in-place="),
  );
}

/**
 * `detectWriteTargets` plus command-shape coverage that only matters for
 * non-gating consumers (tb-finalize-guard's evidence-of-work check, and
 * checkpoint's pre-write backup), never for permission-gating: `cp`/`mv`/
 * `install` (last non-flag operand, or the `-t DIR` argument), `sed -i`/
 * `--in-place` (every non-flag operand after the script), and a compiler's
 * `-o` output flag (`gcc -o`, `cc -o`, `ld -o`). See the block comment above
 * for why this is a separate function rather than a change to
 * `detectWriteTargets` itself.
 */
export function detectDeliverableWrites(raw: string): ShellWrite[] {
  const writes = [...detectWriteTargets(raw)];
  const cmd = stripHeredocBodies(raw);

  for (const segment of splitCommandChain(cmd)) {
    const words = splitWords(segment);
    if (words.length === 0) continue;
    const name = words[0];

    if (name === "cp" || name === "mv" || name === "install") {
      const { target, sources } = lastOperandOrTargetFlag(words);
      if (target) {
        writes.push({
          path: unquote(target),
          kind: name === "mv" ? "move" : "copy",
          sources: sources.map(unquote),
        });
      }
      continue;
    }

    if (name === "sed" && hasSedInPlaceFlag(words)) {
      const nonFlagOperands = words.slice(1).filter((w) => !w.startsWith("-"));
      for (const t of nonFlagOperands.slice(1)) {
        writes.push({ path: unquote(t), kind: "inplace" });
      }
      continue;
    }

    if (name === "gcc" || name === "cc" || name === "ld") {
      for (let i = 1; i < words.length; i++) {
        if (words[i] === "-o" && words[i + 1]) {
          writes.push({ path: unquote(words[i + 1]), kind: "compile" });
          i++;
        }
      }
    }
  }

  const seen = new Set<string>();
  return writes.filter((w) => {
    if (!w.path || seen.has(w.path) || isNonDestructiveTarget(w.path)) return false;
    seen.add(w.path);
    return true;
  });
}

// Roots that are obviously scratch space rather than a deliverable location.
// Deliberately short: over-fitting this list (e.g. every plausible temp dir a
// model might invent) risks quietly treating a *real* deliverable path as
// scratch. `/tmp` and `/var/tmp` are the two POSIX-standard temp roots;
// `/private/tmp` is macOS's resolved form of `/tmp` (relevant if a command's
// path already went through realpath-style resolution upstream — most raw
// model-issued commands will say `/tmp` directly).
const SCRATCH_ROOTS = ["/tmp", "/var/tmp", "/private/tmp"];

/**
 * Lexically collapse `.` and `..` segments in an absolute path, the way a
 * shell's path resolution would, without touching the filesystem. `path`
 * must already start with `/`. An empty stack absorbs a leading `..` (the
 * parent of `/` is `/`), which is exactly the POSIX semantics we want here.
 */
function normalizeAbsPath(path: string): string {
  const parts = path.split("/");
  const stack: string[] = [];
  for (const part of parts) {
    if (part === "" || part === ".") continue;
    if (part === "..") {
      stack.pop();
      continue;
    }
    stack.push(part);
  }
  return "/" + stack.join("/");
}

/**
 * True when `path` is rooted under an obviously-scratch directory (`/tmp`
 * and friends) rather than a plausible deliverable location.
 *
 * Used by tb-finalize-guard's post-finalize-warn compliance check: a
 * write-shaped command that only ever touches `/tmp` must not count as
 * evidence the model saved its actual answer. Intentionally conservative:
 * only an *absolute* path under one of `SCRATCH_ROOTS` counts as scratch; a
 * relative path or any other absolute path is treated as a plausible
 * deliverable target, since the harness cannot know the task's real
 * deliverable path (parsing it out of instruction.md was considered and
 * rejected as too fragile to rely on).
 *
 * The path is normalized (lexically) before the prefix check, so
 * `/tmp/../app/answer.txt` (which really resolves to `/app/answer.txt`) does
 * not misclassify as scratch, and `/app/../tmp/x` (which really resolves to
 * `/tmp/x`) does not misclassify as a deliverable. Normalization is
 * lexical-only and never resolves symlinks — which is exactly why
 * `/private/tmp` is listed in SCRATCH_ROOTS separately from `/tmp` rather
 * than relying on normalization to unify them (macOS's `/tmp` symlink can't
 * be resolved here). This only feeds a nudge heuristic in tb-finalize-guard,
 * not any write-permission gate, so a misclassification in either direction
 * is low-stakes.
 *
 * One contrived edge case, left unfixed on purpose: `/tmp/../dev/null`
 * normalizes to `/dev/null` (not scratch), but callers elsewhere in this
 * file that inspect the raw (non-normalized) command text may have already
 * treated it as a non-destructive write target before normalization ever
 * runs — so it can end up counted as a compliant deliverable write even
 * though it is actually `/dev/null`. Documented so it isn't a silent
 * surprise later, not something worth adding path-normalization elsewhere
 * in this file to close.
 */
export function isScratchPath(path: string): boolean {
  if (!path.startsWith("/")) return false;
  const normalized = normalizeAbsPath(path);
  return SCRATCH_ROOTS.some((root) => normalized === root || normalized.startsWith(root + "/"));
}

function unquote(word: string): string {
  if (word.length >= 2 && (word[0] === '"' || word[0] === "'") && word[word.length - 1] === word[0]) {
    return word.slice(1, -1);
  }
  return word.replace(/\\(.)/g, "$1");
}

/** The first whitespace-delimited word of `s`, honoring quotes. */
function firstWord(s: string): string {
  return splitWords(s)[0] ?? "";
}

// Unquoted characters that end a word even with no whitespace around them:
// redirects (`cat>f` is two words) and the control operators that separate one
// command from the next. Without the operators, `2>/dev/null;` parsed as a
// redirect to the file `/dev/null;`, which is not the device the #87 exemption
// knows about, so a perfectly ordinary chained `find ... 2>/dev/null; find ...`
// was refused as an unsafe file write (issue #107). The same held for `&&`,
// `||`, `|` and a closing subshell paren.
const WORD_BREAKS = new Set([">", "<", ";", "&", "|", "(", ")"]);

/** Split on unquoted whitespace and control operators, keeping quoted runs together. */
function splitWords(s: string): string[] {
  const words: string[] = [];
  let cur = "";
  let quote: '"' | "'" | null = null;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (quote === null && ch === "\\") {
      cur += ch + (s[i + 1] ?? "");
      i++;
      continue;
    }
    if (quote === null && (ch === '"' || ch === "'")) {
      quote = ch;
      cur += ch;
      continue;
    }
    if (quote !== null && ch === quote) {
      quote = null;
      cur += ch;
      continue;
    }
    if (quote === null && /\s/.test(ch)) {
      if (cur) words.push(cur);
      cur = "";
      continue;
    }
    if (quote === null && WORD_BREAKS.has(ch)) {
      if (cur) words.push(cur);
      cur = "";
      continue;
    }
    cur += ch;
  }
  if (cur) words.push(cur);
  return words;
}
