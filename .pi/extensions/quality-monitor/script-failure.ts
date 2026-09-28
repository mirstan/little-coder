// Trial-wide counter for a pattern `near_duplicate_loop`'s content-clustering
// definitionally cannot catch: a script gets rewritten, run, and fails --
// differently each time -- with real gaps (~10-30 turns) between events, so
// most pairs never share the fuzzy tracker's 8-turn window at all
// (overfull-hbox: historically stated here as "8 such events, only 2 caught
// by near_duplicate_loop"). Iteration 2's review (Codex + an independent
// Claude pass) reported that under iteration 1's final-segment-only +
// command-text-only-dedup logic, that count was actually 6 false positives
// (perl -c syntax checks, a command-not-found, a SIGTERM) and 0 real piped
// script runs detected at all -- the final-segment restriction excluded
// every real run piped to a filter (the trial's dominant shape:
// `perl solve.pl ... | grep -vE "..."`), and the command-text dedup
// swallowed the genuine write-rerun repeats. This header intentionally does
// NOT restate a new count for that historical "8" figure: replaying this
// trial's actual ShellSession exit codes found the trial's real piped
// perl/python runs come back exit=0 or the already-excluded exit=-1
// (timeout) far more often than a distinguishable non-timeout failure, so a
// specific new number here would not be a reproduced fact. What IS verified
// (script-failure.test.ts's "piped real-run attribution" tests, using this
// trial's own command text verbatim) is that the new logic now recognizes
// that shape at all, which iteration 1's logic did not for a single one of
// the trial's real runs. Recognition only: readResult() gates on the
// chain's reported exit, which a trailing filter masks, so a piped script
// failure is still not counted until the exit-code fix lands.
//
// Monotonic, not a streak or rolling window: the claim is "how many times has
// this happened at all this trial", true regardless of what ran in between,
// and needs no window-size tuning the way a rolling span would.

import { scan, splitCommandChain, stripHeredocBodies } from "../_shared/shell-write.ts";
import { splitFooter } from "../truncated-view/truncation.ts";
import { footerExit } from "../syntax-check/helpers.ts";
import { readResult, signatureOf } from "./failure-signature.ts";

const INTERPRETER_RE = /\b(?:perl|python3?|ruby|node|bash|sh)\b/i;
const SCRIPT_EXT_RE = /\.(?:pl|pm|t|py|rb|js|sh)\b/i;

function commandOf(input: unknown): string | undefined {
  const command = (input as { command?: unknown } | null | undefined)?.command;
  return typeof command === "string" ? command : undefined;
}

/** One `;`/`|`/newline-delimited top-level segment of a command chain. */
interface ChainSegment {
  text: string;
  /**
   * The operator that immediately preceded this segment -- `null` for the
   * chain's first segment. Distinguishes a segment reached by an
   * unconditional separator (`;`/newline: an independent, unrelated command)
   * from one reached by a pipe (`|`: this segment's INPUT is the previous
   * segment's output) -- `attributionSegment` below only walks back across
   * the latter.
   */
  precedingOp: ";" | "|" | "\n" | null;
}

/**
 * Split a shell command chain into its top-level segments, at `;`, `|`, and
 * newlines only -- separators where every segment unconditionally runs and
 * the LAST one's exit is what the whole command reports (plain bash, no
 * `pipefail`). Deliberately does NOT split on `&&`/`||`: those short-circuit,
 * so the exit code can belong to an EARLIER segment (the one that stopped
 * the chain) rather than the last one written -- splitting on them risks the
 * opposite mistake this function exists to prevent, excluding a script whose
 * own failure is exactly what aborted a trailing `&& next_step`.
 *
 * Reuses shell-write.ts's `stripHeredocBodies` and quote/escape-aware
 * `scan`: an unbalanced quote inside a heredoc body otherwise leaves the
 * quote tracker open and swallows every separator after it. Uses these
 * rather than `splitCommandChain` itself -- that function's cut set
 * (`&&`, `||`, `;`, `|`, `&`, newline) is the WRONG set here for the reason
 * above, so this mirrors its internals for the narrower boundary set this
 * module needs instead of composing/re-merging its output.
 */
function commandSegments(raw: string): ChainSegment[] {
  const cmd = stripHeredocBodies(raw);
  const cuts: Array<{ at: number; len: number; op: ";" | "|" | "\n" }> = [];
  // Exclusive upper bound of the most recently seen `&&`/`||` pair. `scan`
  // visits one character at a time, so without this the SECOND character of
  // `||` would be re-examined on its own and mistaken for the unconditional
  // single-pipe separator below.
  let shortCircuitEnd = -1;
  scan(cmd, (ch, i, quote) => {
    if (quote) return;
    if (i < shortCircuitEnd) return;
    if ((ch === "&" && cmd[i + 1] === "&") || (ch === "|" && cmd[i + 1] === "|")) {
      shortCircuitEnd = i + 2;
      return;
    }
    if (ch === ";" || ch === "|" || ch === "\n") {
      const last = cuts[cuts.length - 1];
      if (last && i < last.at + last.len) return;
      cuts.push({ at: i, len: 1, op: ch });
    }
  });

  const result: ChainSegment[] = [];
  let start = 0;
  let precedingOp: ";" | "|" | "\n" | null = null;
  for (const cut of cuts) {
    result.push({ text: cmd.slice(start, cut.at), precedingOp });
    precedingOp = cut.op;
    start = cut.at + cut.len;
  }
  result.push({ text: cmd.slice(start), precedingOp });
  return result;
}

// A small table of commands whose whole job is to reshape/filter another
// command's OUTPUT, never to be the thing that "ran" -- same shape as
// INTERPRETER_RE/SCRIPT_EXT_RE above, not a general argv parser. Used only to
// walk a pipe chain back to the segment actually worth attributing a script
// run to (see `attributionSegment`). `sed` is included only with `-n`
// (suppress-output mode, i.e. used as a filter/extractor); bare `sed`
// (typically `-i`, in-place editing) is not a pure filter and is
// deliberately excluded.
const PURE_FILTER_COMMANDS = new Set(["grep", "head", "tail", "wc", "cut"]);

function firstWordOf(s: string): string {
  const m = /^\S+/.exec(s);
  return m ? m[0] : "";
}

function isPureFilterSegment(segment: ChainSegment): boolean {
  const first = firstWordOf(segment.text.trimStart());
  if (PURE_FILTER_COMMANDS.has(first)) return true;
  return first === "sed" && /(^|\s)-n(\s|$)/.test(segment.text);
}

/**
 * Walk back across trailing PIPED pure-output filters (grep, head, tail,
 * sed -n, wc, cut) to the segment worth attributing a script run to. Shape
 * matching only -- the chain's reported exit is still the trailing filter's
 * own. The real-trial shape this module exists to catch (`perl solve.pl
 * 2>&1 | grep -vE "..."`, `python3 solve.py ... | head -40`) is a script
 * piped to exactly one of these.
 *
 * Only walks across a `|` boundary. A `;`/newline-separated filter is a
 * wholly INDEPENDENT command with its own unrelated exit code and no
 * data-flow relationship to what ran before it -- reattributing across THAT
 * boundary would misattribute an unrelated command's failure to the script,
 * which is `python3 solve.py; grep expected missing.txt`'s own case (the
 * chain's real exit is grep's, evaluated on its own terms, and correctly
 * does not match).
 */
function attributionSegment(segments: ChainSegment[]): ChainSegment {
  let idx = segments.length - 1;
  while (idx > 0 && segments[idx].precedingOp === "|" && isPureFilterSegment(segments[idx])) {
    idx--;
  }
  return segments[idx];
}

// Commands that transparently re-run their remaining arguments as a command
// -- real trial data wraps almost every long-running script call in one of
// these (`timeout 90 python3 solve.py`), so without stripping them the
// first-word invocation check below would reject every one of them the same
// way it correctly rejects `ls python3 missing.py`. Small and deliberately
// conservative, same rationale as PURE_FILTER_COMMANDS above.
const COMMAND_WRAPPERS = new Set(["timeout", "nice", "nohup", "setsid", "env", "sudo"]);

/**
 * True when `atom` -- one `&&`/`||`/`&`-joined sub-command, already isolated
 * by `splitCommandChain` -- actually LOOKS like an interpreter invoking a
 * script, not merely a string that happens to contain both an interpreter
 * name and a script extension somewhere in its text. Requires the
 * interpreter token to be the atom's own first word (after stripping a
 * transparent wrapper prefix and its flags/duration argument): this is what
 * rejects `ls python3 missing.py` (the interpreter word is `ls`'s own
 * argument, not the executable) and a literal `grep "python3 solve.py"
 * file.txt` (the executable is `grep`), which the old plain
 * substring-anywhere match could not tell apart from a real invocation.
 */
function looksLikeInvocation(atom: string): boolean {
  const words = atom.trim().split(/\s+/);
  let i = 0;
  while (i < words.length && COMMAND_WRAPPERS.has(words[i])) {
    i++;
    while (
      i < words.length &&
      (words[i].startsWith("-") ||
        /^[\d.]+[smhd]?$/.test(words[i]) ||
        /^[A-Za-z_][A-Za-z0-9_]*=/.test(words[i]))
    ) {
      i++;
    }
  }
  const first = words[i] ?? "";
  return INTERPRETER_RE.test(first) && SCRIPT_EXT_RE.test(atom);
}

/**
 * A shell command chain whose attributed segment (see `attributionSegment`)
 * both names a common interpreter and targets a common script extension, in
 * actual command-invocation position (see `looksLikeInvocation`).
 * Deliberately a small local table: this needs interpreter+extension shape,
 * not syntax-check's path→checker map.
 *
 * A `&&`/`||`-joined sub-command within the attributed segment is not split
 * further at the outer level (see `commandSegments`) but IS considered here,
 * atom by atom, via `splitCommandChain` -- so it stays eligible wherever it
 * sits within its enclosing segment (`cd /app && perl final.pl` matches on
 * its second atom).
 */
export function looksLikeScriptRun(input: unknown): boolean {
  const command = commandOf(input);
  if (command === undefined) return false;
  const segments = commandSegments(command);
  const candidate = attributionSegment(segments);
  return splitCommandChain(candidate.text).some(looksLikeInvocation);
}

export interface ScriptFailureDetection {
  count: number;
  /** The last message for this trial -- and the only one, if threshold 1 never got delivered. */
  escalated: boolean;
}

const THRESHOLD_1 = 3;
const THRESHOLD_2 = 6;

// Tail length fed to failure-signature.ts's `signatureOf` for the dedup key
// below -- matches failsigOptionsFromEnv's own default tailLines.
// This tracker doesn't expose an env override of its own; if that's ever
// needed it should read the same LITTLE_CODER_FAILSIG_TAIL_LINES knob rather
// than growing a second one.
const DEDUP_SIG_TAIL_LINES = 40;

/**
 * Counts script write-run-fail events across the whole trial, independent of
 * `FuzzyLoopTracker`'s content clustering. Steer-only, like the other two
 * quality-monitor watchdogs: a script failing repeatedly cannot prove the
 * NEXT rewrite is wrong, only that the current approach isn't converging.
 *
 * Three things are excluded from the count so it isn't counting the same
 * event twice or a non-event at all: `exit === -1` (ShellSession's own
 * timeout/proxy-no-response/sentinel code, not a script failure), a truly
 * identical immediate re-run of the immediately preceding COUNTED command
 * (same command text AND same failure signature -- see `record`'s own
 * comment), and a chain whose matched interpreter+extension token isn't in
 * the chain's own attributed segment (see `looksLikeScriptRun`).
 */
export class ScriptFailureTracker {
  private count = 0;
  private lastCounted: { command: string; sigHash: string } | null = null;
  /** Highest threshold whose message has actually been delivered. */
  private notifiedThreshold = 0;

  /**
   * Fold one tool result into the trial-wide count. Does NOT itself decide
   * whether a message is due -- see `due()`, which is computed fresh from
   * `count`/`notifiedThreshold` independent of whatever call caused a change,
   * and which the caller (index.ts's `turn_end`) now checks once per
   * ok-verdict turn regardless of whether THIS turn produced a new qualifying
   * result -- so `record`'s return value here is a convenience for
   * the many tests that call it directly, not the tracker's only trigger.
   */
  record(obs: { input: unknown; text: string; isError: boolean }): ScriptFailureDetection | null {
    if (!looksLikeScriptRun(obs.input)) return null;

    const { footer } = splitFooter(obs.text);
    if (footerExit(footer) === -1) return null;

    // minTokens 0: this tracker only cares whether the command failed, not
    // whether the result said enough to identify which failure it was --
    // that's readResult's `hasContent` floor, built for a different watchdog.
    const { failed } = readResult(obs.text, obs.isError, 0);
    if (!failed) return null;

    const command = commandOf(obs.input) ?? null;
    // Dedup by (command text, failure signature), not command text alone:
    // the canonical loop this whole tracker exists to catch is
    // write->run->fail, REWRITE->run->fail -- the command text is usually
    // byte-identical each retry (`perl solve.pl`), only the FILE changed in
    // between, so a command-text-only key suppressed every one of those
    // reruns as a "verbatim repeat" and could never reach threshold 3.
    // Reusing failure-signature.ts's own signature (a hash of the normalized
    // output tail) means a rerun that now fails DIFFERENTLY -- the common
    // case after a real edit -- still counts, and only a truly identical
    // immediate re-run (same command, same failure, nothing changed) stays
    // suppressed as the same attempt seen twice.
    const sigHash = signatureOf("scriptfailure", obs.text, DEDUP_SIG_TAIL_LINES).hash;
    if (
      command !== null &&
      this.lastCounted !== null &&
      command === this.lastCounted.command &&
      sigHash === this.lastCounted.sigHash
    ) {
      return null;
    }

    this.count++;
    this.lastCounted = command !== null ? { command, sigHash } : null;
    return this.due();
  }

  /**
   * Consume the message slot for whichever threshold `count` currently
   * satisfies. Called by the caller only once a detection has actually been
   * delivered (see `due()`) -- never from inside `record()` itself.
   */
  markNotified(): void {
    if (this.count >= THRESHOLD_2) this.notifiedThreshold = THRESHOLD_2;
    else if (this.count >= THRESHOLD_1) this.notifiedThreshold = THRESHOLD_1;
  }

  /**
   * Report whichever threshold `count` currently satisfies and that
   * `markNotified()` has not yet consumed -- computed fresh on every call,
   * never mutating. A crossing whose message never actually gets sent (the
   * turn it happened on is later classified non-ok, so the caller never
   * reaches `sendUserMessage`) must
   * not be lost: only `markNotified()` advances `notifiedThreshold`, so an
   * unmarked crossing keeps being reported on every subsequent call until it
   * is -- mirroring `FailureSignatureTracker`'s / `FuzzyLoopTracker`'s own
   * due()-computed-fresh + explicit markNotified()-only-when-sent split.
   *
   * Public: `record()` still calls this itself so every existing
   * caller of `record()` keeps getting a detection back the instant a
   * threshold crosses, but `due()` no longer has only that one trigger.
   * `record()` is only reached from inside index.ts's per-result loop, which
   * runs only for turns that produced a tool result at all -- a threshold
   * that crossed on an earlier non-ok turn has nothing to re-check it on a
   * LATER ok-verdict turn whose own results include no new qualifying script
   * result (e.g. the model's next attempt succeeds, or the turn has no
   * ShellSession call at all). index.ts now also calls `due()` directly,
   * once per ok-verdict `turn_end`, independent of that turn's own results.
   */
  due(): ScriptFailureDetection | null {
    if (this.count >= THRESHOLD_2 && this.notifiedThreshold < THRESHOLD_2) {
      return { count: this.count, escalated: true };
    }
    if (this.count >= THRESHOLD_1 && this.notifiedThreshold < THRESHOLD_1) {
      return { count: this.count, escalated: false };
    }
    return null;
  }
}
