import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { harnessIntervention } from "../_shared/intervention.ts";
import { resolveTurnCap } from "../_shared/turn-cap.ts";
import { resolveDeadlineEpochMs } from "../_shared/deadline.ts";
import { SHELL_TOOLS, detectDeliverableWrites, isScratchPath } from "../_shared/shell-write.ts";
import {
  finalizeWarnTurnWindowOpen,
  finalizeWarnWouldFire,
} from "../_shared/finalize-warn-trigger.ts";
import {
  PROGRESS_MILESTONES,
  budgetFractionUsed,
  resolveBudgetStartEpochMs,
} from "../_shared/budget-progress.ts";
import { resolveFinalizeMessage } from "../_shared/finalize-message.ts";
import {
  INITIAL_SNAPSHOT_APP_DIR,
  initialSnapshotOutcome,
  type InitialSnapshotOutcome,
} from "../_shared/snapshot-paths.ts";
import { inTbMode, tbProxyRun, tbSessionId, type ProxyUiCtx } from "../_shared/tb-proxy.ts";
import { CHECK_TIMEOUT_SEC, footerExit, footerTimedOut } from "../syntax-check/helpers.ts";
import { splitFooter } from "../truncated-view/truncation.ts";

// tb-finalize-guard: a merged guard for Terminal-Bench with four independent
// trigger conditions, scoped to LITTLE_CODER_BENCHMARK === "terminal_bench"
// only. GAIA has its own separate gaia-finalize-guard, and the two must
// never fire on the same benchmark's sessions.
//
// This is a sibling of finalize-warn, not an addition to it: abort policy
// (turn-cap, thinking-budget), warn policy (finalize-warn), and guard policy
// (this extension) are kept in separate extensions so each can be tuned or
// disabled independently — matching finalize-warn's own header convention.
//
// ---------------------------------------------------------------------------
// Trigger A — early voluntary quit
// ---------------------------------------------------------------------------
// Fires when a turn ends with text but no tool calls while a large share of
// the wall-clock budget still remains (at least EARLY_QUIT_MIN_REMAINING_MS)
// and there is still turn-cap headroom to deliver a nudge before the cap
// would abort the run.
//
// A turn whose stopReason is "aborted" or "error" is deliberately excluded
// from this shape check: that's a provider/transport failure, not a model
// decision, and steering can't fix it. Treating a repeating provider error as
// a "quit" would also be actively harmful here, since MAX_TRIGGER_A_FIRES is
// session-scoped — burning both fires on turns the model never controlled
// would leave the guard disarmed for a real early-quit later in the same
// session. The trade-off used to be a real limitation: a quit that manifests
// as a silent stopReason:"error" turn, rather than a toolless one with
// visible text, was not steerable by this trigger — turn_end fires on every
// turn including mid-run retries, so Trigger A can't tell "this turn errored
// but the run recovered" from "this turn errored and the run is dying" and
// must stay conservative. That gap is what Trigger C (below) exists to
// close: it looks at the run's actual last message via agent_settled instead
// of at every individual turn_end, so it only reacts to a turn that stayed
// erroring/empty. The turn_end instrumentation below remains useful
// independently of that, since it's the finer-grained per-turn record.
//
// The nudge demands adversarial re-verification rather than any re-check.
// Observed failures passed re-checks that could not have failed: one re-ran
// the same analysis that produced the wrong answer and re-confirmed it,
// another diffed a task file against a backup made after the file was
// already corrupted. So the text states a protocol — name the falsifying
// result first, use a different method, never compare against another
// artifact from this same session — and names the one reference that
// predates the model's own changes, the adapter's start-of-trial copy.
//
// That pointer is gated on the adapter-classified outcome in
// _shared/snapshot-paths.ts, because TB1.0 runs under this same benchmark
// name with no such copy, and it is scoped to inputs the model already has
// reason to suspect, matching the adapter's own reactive-only framing of the
// copy. A standing "diff against it" directive would spend turns on every
// trial, including tasks whose only input file is a multi-hundred-MB binary.
//
// ---------------------------------------------------------------------------
// Trigger B — post-finalize-warn non-compliance
// ---------------------------------------------------------------------------
// finalize-warn keeps no exported latch describing whether its "save now"
// nudge has already fired this run, so this guard independently re-derives
// finalize-warn's own trigger condition (turn-count OR wall-clock, computed
// the same way at turn_start) rather than reading finalize-warn's private
// state, via the shared `finalizeWarnWouldFire` in
// _shared/finalize-warn-trigger.ts.
//
// finalize-warn's message is delivered as deliverAs:"steer", which normally
// lands on the model's next turn, not the turn during which the trigger
// fired. So "armed" here also tracks the turn number at which arming
// happened, and compliance is judged starting from the turn AFTER that one.
//
// Known, accepted gap: if another extension's steer is queued the same turn
// as finalize-warn's, delivery slips a turn (see finalize-warn/index.ts's
// own comment) and this window starts judging compliance one turn before
// the model has actually seen the message -- Trigger B can then fire after
// only one turn of real exposure instead of two. Left as-is rather than
// widening the window: the consequence is an extra "write it now" nudge
// firing a turn earlier than ideal in an already-rare case, not a missed
// catch, and widening would loosen the tuned 2-turn window for every run
// that doesn't hit the slip.
//
// Compliance is judged via _shared/shell-write.ts's `detectDeliverableWrites`
// — a tb-finalize-guard-only superset of `detectWriteTargets` that also
// recognizes `cp`/`mv`/`install`, `sed -i`, and a compiler's `-o` flag as
// evidence of a write, not just shell redirection (`detectWriteTargets`
// itself stays redirect-only because write-guard and permission-gate also
// consume it, and both deliberately treat `cp`/`mv`/`sed -i` as safe,
// non-write commands — see that function's own comment). Extended with
// `isScratchPath` so a write that only ever lands in /tmp does not count as
// having saved the real deliverable.
//
// A turn's evidence-of-work also includes `ShellSend` (writing to an
// already-running interactive job's stdin) even though it is deliberately
// excluded from `SHELL_TOOLS` for permission-gating purposes — a model
// driving an editor/REPL through `ShellSend` is plainly working. This guard
// scans for it locally rather than adding it to the shared, security-relevant
// `SHELL_TOOLS` set.
//
// ---------------------------------------------------------------------------
// Trigger C — dead run: errored or empty final message, any time
// ---------------------------------------------------------------------------
// Trajectory analysis of failed trials found real deaths that neither
// Trigger A nor Trigger B can catch: the run's last assistant message came
// back with stopReason:"error" or with no text and no tool calls, far from
// any deadline or turn-cap boundary (22-63% of budget used in the observed
// cases) — nowhere near where Trigger B arms, and excluded from Trigger A's
// shape check by design (see Trigger A's comment above). Trigger A and B
// only ever look near a deadline/turn-cap boundary because that's when a
// slow-but-working run legitimately needs to be told to wrap up; Trigger C
// is different in kind — an errored or empty last message is never a sign of
// healthy progress, so it fires regardless of remaining budget.
//
// This evaluates at `agent_end` rather than `turn_end` deliberately:
// `turn_end` fires on every turn, including ones a mid-run retry papers
// over, so judging "the run is dead" from a single turn_end would misfire
// on transient errors the harness already recovered from. `agent_end`
// reports the run's actual last message once the agent loop segment has
// ended, so this only reacts to a turn that stayed erroring/empty.
//
// A prior version of this fired at `agent_settled` instead (pi's event for
// "after an agent run has fully settled and no automatic retry, compaction,
// or queued continuation will run"), on the theory that a transient error
// pi's own internal retry recovers from would never reach this trigger at
// all. That's true in isolation, but `sendUserMessage` called at settle time
// starts a genuinely FRESH run (the session is no longer streaming), and the
// benchmark harness (`benchmarks/rpc_client.py`) treats `agent_settled` as
// unconditionally terminal — it stops draining events and the adapter closes
// the pi process within a few seconds, killing that fresh run before the
// model ever sees the nudge. `agent_end` is the hook pi actually supports for
// this: while the session is still streaming, `sendUserMessage` queues into
// the SAME run's steering queue, and `_handlePostAgentRun`'s
// `hasQueuedMessages()` check turns that into a real continuation the
// harness already waits for. The accepted cost of firing at `agent_end`
// instead of `agent_settled`: since the extension-facing `agent_end` event
// doesn't carry `willRetry`, this can occasionally queue a redundant steer
// on a turn pi's own internal retry was about to recover on its own — bounded
// by `MAX_TRIGGER_C_FIRES` and harmless (the steer just rides along).
//
// `AgentEndEvent` also carries no reliably-shaped `messages` array to read
// the last assistant message from in every case, so it's still snapshotted
// from `turn_end` into the run-scoped `lastTurnMessage` below and read back
// here — this mechanism didn't need to change when the firing hook reverted.
//
// An "aborted" last message is excluded before the shape check even runs —
// see `maybeFireTriggerC`'s own comment on why (in short: an abort is a
// harness decision, not a dead run, and a run that aborted with only
// thinking tokens streamed looks empty to `contentShape`, which is
// otherwise invisible to thinking blocks). The nudge itself is two-toned:
// it uses `resolveFinalizeMessage`'s urgency framing only when the run is
// genuinely near its deadline or turn-cap per Trigger B's `armed` latch (or
// `finalizeWarnWouldFire` directly, for a deadline crossed since the last
// turn_start); otherwise it uses calmer recovery framing so it doesn't
// contradict Trigger A's "you have plenty of time" message in the same
// session.
//
// ---------------------------------------------------------------------------
// Trigger D — budget-progress checkpoint
// ---------------------------------------------------------------------------
// Otherwise the first time-pressure signal a trial ever gets is
// finalize-warn's, 10 minutes or 5 turns from the end. One observed trial
// spent ~70% of its budget on read-only exploration and wrote its first
// non-scratch file with a sliver of the clock left; the only real code it
// produced came after that warning. Nothing had told it at half-time that a
// deliverable should exist by then.
//
// So this fires once per session at each of PROGRESS_MILESTONES — fractions
// of the [trial start, deadline] interval, which is why it needs the start
// instant _shared/budget-progress.ts resolves and not just the deadline
// every other trigger here reads. An adapter that publishes neither end
// (GAIA, aider_polyglot, interactive pi) disables this trigger outright.
//
// The earliest milestone is gated on evidence — any non-scratch write seen
// this session, judged by the same detectDeliverableWrites/isScratchPath
// pair Trigger B measures compliance with — because interrupting a run that
// is demonstrably writing costs it a turn for nothing. Later milestones
// fire either way, with milder wording when that evidence exists. The
// asymmetry is deliberate: the evidence heuristic answers "does a
// deliverable-shaped artifact exist", not "does a working one", so letting
// it decide WHETHER a trial gets a mid-run signal at all would put a
// heuristic between a model and its only early warning. Deciding only the
// wording, a wrong answer costs a sentence.
//
// Whether the deliverable works is demanded of the model, not detected:
// many TB deliverables are not programs (a repaired .tex file, a git state,
// a value written to a path), so the text asks for a real check of whichever
// shape applies rather than for the file to be run. The deliverable's actual
// path stays unknowable here for the reason isScratchPath's own comment
// records — recovering it from the task text was considered and rejected.
//
// Endgame turns belong to finalize-warn and Trigger B, so this stands down
// while their window is open. That check can use neither Trigger B's
// `armed` latch alone, which stops re-latching once Trigger B has fired,
// nor `finalizeWarnWouldFire` alone, whose turn half is edge-triggered and
// so reads false on every turn of the window but one — hence the
// level-triggered finalizeWarnTurnWindowOpen alongside both.
//
// ---------------------------------------------------------------------------
// Trigger E — byte-limit-aware finalize guard
// ---------------------------------------------------------------------------
// Some TB tasks state a hard numeric size limit on the deliverable (a code-
// golf constraint). Terminal-Bench mode has no Write/Edit tool at all --
// every file write happens via an opaque shell command (a heredoc, `sed -i`,
// a compiler's `-o` flag), so a write's effect on the target file's size is
// never visible in the tool call's own arguments the way it would be if a
// Write tool passed full file content directly. Nothing else in this file
// tracks *how big* the deliverable is, only *whether* something was written
// and *when* in the budget -- a shrinking-then-growing-again file, or one
// that simply never gets under the limit, currently passes unnoticed.
//
// The limit and the deliverable's path are each parsed at most once, from
// the task prompt, at `before_agent_start` -- mirroring capForRun/
// deadlineForRun's own once-per-run resolution. Both regexes are
// false-negative-biased by design: if either fails to find a single
// confident match, this trigger stays permanently inert for the run rather
// than guessing. A guard armed with the wrong path or the wrong limit is
// worse than one that never fires -- it would tell the model to golf a file
// that isn't the deliverable, or accept a size that isn't the real cap.
//
// Getting the real byte count reuses syntax-check's own solution to the
// adjacent problem of reaching a file that lives in the TB container: the
// `__LC_TB_SHELL__` proxy channel (`_shared/tb-proxy.ts`), which pi's host
// process can use to run a command inside the container and read back its
// output. A `wc -c` over that channel is a real count, not an inference from
// command text. This only runs on a turn whose commands actually touched the
// parsed deliverable path -- a turn doing unrelated work pays nothing.
//
// The over-budget count is session- rather than run-scoped: it tracks a
// property of the whole trial's deliverable, so a mid-run compaction
// continuation must not reset progress already made toward the limit.
//
// There is no explicit "finalize" tool call in TB mode to block -- a trial
// simply runs until the model stops, the deadline hits, or turn-cap aborts
// it, and grading happens externally afterward. Every trigger in this file
// works the same way this one must: a `pi.sendUserMessage` steer nudge, not
// a hard refusal. The escalation is two-toned, like Trigger C: a calm
// "keep golfing" nudge ordinarily, or -- once inside the same near-deadline
// window Triggers A/B/D already key off -- the same save-what-you-have
// framing `resolveFinalizeMessage` gives Trigger C's own near-deadline
// branch, so a model that can no longer realistically close the gap isn't
// told to keep pushing on a constraint it may be out of turns to meet.
//
// ---------------------------------------------------------------------------
// Shared instrumentation
// ---------------------------------------------------------------------------
// On every terminal_bench turn_end, log the turn's stopReason and a coarse
// content-shape summary via ctx.ui.notify at "info"-but-diagnostic framing —
// NOT a harnessIntervention call, deliberately, so this doesn't inflate the
// intervention-count metric with pure diagnostics. This makes a silent
// stopReason:"error" turn — a shape Trigger A can't steer on and Trigger C
// only reacts to once it's the run's final message, see above — diagnosable
// from the run log instead of invisible.

// WARN_REMAINING_MS lives in _shared/finalize-warn-trigger.ts now — see that
// module's header for why (this constant used to be hand-copied here and in
// finalize-warn/index.ts with nothing enforcing they stayed in lockstep).
const EARLY_QUIT_MIN_REMAINING_MS = 20 * 60 * 1000; // double finalize-warn's WARN_REMAINING_MS
const MAX_TRIGGER_A_FIRES = 2; // per session

const NO_WRITE_TURNS_BEFORE_NUDGE = 2; // consecutive non-compliant turns

const MAX_TRIGGER_C_FIRES = 2; // per session

const SIZE_CHECK_TIMEOUT_SEC = CHECK_TIMEOUT_SEC; // same budget syntax-check gives its own container checks
const OVER_BUDGET_CHECKS_BEFORE_NUDGE = 3; // consecutive over-budget checks -- avoids spamming mid-golf

// ---- Trigger A state (session-scoped fire count; run-scoped turn/cap bookkeeping) ----
let triggerAFireCount = 0;
let turnsThisRun = 0;
let capForRun = 0;
let deadlineForRun = 0;

// ---- Trigger B state ----
// `armed`/`armedAtTurn`/`consecutiveNoWriteTurns` are run-scoped, like
// finalize-warn's own warnedThisRun, because they're derived from the
// run-scoped turnsThisRun/capForRun/deadlineForRun. `triggerBFired` is the
// actual one-shot latch and is session-scoped, like gaia-finalize-guard's.
let armed = false;
let armedAtTurn = 0;
let consecutiveNoWriteTurns = 0;
let triggerBFired = false;

// ---- Trigger C state ----
// Fire count is session-scoped, like Trigger A's. The message snapshot is
// run-scoped: AgentSettledEvent carries no messages (unlike agent_end), so
// Trigger C reads the last turn_end's assistant message, captured below.
let triggerCFireCount = 0;
let lastTurnMessage: any = undefined;

// ---- Trigger D state ----
// Both latches are session-scoped, like triggerBFired: the interval they
// are measured against is the whole trial's, which spans pi's internal run
// boundaries, so a continuation must neither resurrect a milestone already
// spent nor forget a write from an earlier run. `startForRun` is run-scoped
// only to be resolved next to deadlineForRun; the env var behind it does
// not change within a session.
let milestonesFired = new Set<number>();
let deliverableWriteEverSeen = false;
let startForRun = 0;
// Run-scoped: which turn (if any) Trigger D fired at, so Trigger A's own
// turn_end for that SAME turn can skip -- both are separately queued
// steering messages (pi delivers each one, never coalesces them), so
// firing both stacks two different nudges on the turn D already spoke to.
let triggerDFiredAtTurn = 0;

// ---- Trigger E state ----
// byteLimitForRun/deliverablePathForRun are SESSION-scoped, latched on first
// successful parse, not re-derived on every run like capForRun/deadlineForRun.
// The task prompt is only the actual task text on the trial's first
// before_agent_start -- a mid-run-compaction or error-retry continuation
// re-fires before_agent_start with COMPACTION_CONTINUE_PROMPT/
// ERROR_RETRY_PROMPT instead (rpc_client.py), neither of which mentions a
// size or path, so re-parsing on every run would silently wipe an
// already-resolved limit the moment either fires. Undefined means "no
// confident single match (yet)" and leaves this trigger inert until one
// resolves, never guessed at.
let byteLimitForRun: number | undefined;
let deliverablePathForRun: string | undefined;
// Session-scoped: the over-budget count and last known size are properties
// of the whole trial's deliverable, not one run segment, so a mid-run
// compaction continuation must not reset progress already made toward the
// limit.
let lastKnownSize: number | undefined;
let consecutiveOverBudgetChecks = 0;

function isTerminalBench(): boolean {
  return process.env.LITTLE_CODER_BENCHMARK === "terminal_bench";
}

function contentShape(message: any): { text: string; toolCallCount: number; toolCalls: any[] } {
  const content = Array.isArray(message?.content) ? message.content : [];
  const text = content
    .filter((c: any) => c?.type === "text")
    .map((c: any) => c.text ?? "")
    .join("\n");
  const toolCalls = content.filter((c: any) => c?.type === "toolCall");
  return { text, toolCallCount: toolCalls.length, toolCalls };
}

// Local addition to SHELL_TOOLS, for this guard's evidence-of-work purposes
// only. `SHELL_TOOLS` itself stays untouched — it's security-relevant and
// shared by write-guard/permission-gate, which deliberately leave ShellSend
// out (it writes to an already-running job's stdin rather than starting a
// command, so it's gated by whatever approved that job in the first place —
// see _shared/shell-write.ts's comment on SHELL_TOOLS). That's an
// authorization judgment, not a claim that ShellSend can't produce writes:
// a model driving an interactive editor/REPL through it is plainly working.
const EVIDENCE_ONLY_SHELL_TOOLS: ReadonlySet<string> = new Set(["ShellSend"]);

/**
 * Every command-shaped string found in this turn's tool calls: the `command`
 * argument for anything in SHELL_TOOLS, plus (for Trigger B's evidence-of-work
 * purposes only) ShellSend's `text` argument — confirmed against
 * bg-shell/index.ts's ShellSend tool definition, which takes `text`, not
 * `command`.
 */
function shellCommandsIn(toolCalls: any[]): string[] {
  const commands: string[] = [];
  for (const c of toolCalls) {
    if (typeof c?.name !== "string") continue;
    const args = c.arguments ?? c.input ?? {};
    if (SHELL_TOOLS.has(c.name)) {
      if (typeof args?.command === "string") commands.push(args.command);
    } else if (EVIDENCE_ONLY_SHELL_TOOLS.has(c.name)) {
      if (typeof args?.text === "string") commands.push(args.text);
    }
  }
  return commands;
}

/** True when at least one command in this turn writes somewhere other than scratch. */
function hasNonScratchWrite(commands: string[]): boolean {
  for (const cmd of commands) {
    for (const w of detectDeliverableWrites(cmd)) {
      if (!isScratchPath(w.path)) return true;
    }
  }
  return false;
}

// True when a write target names the same deliverable as `deliverablePath`,
// tolerating the relative-cwd forms the harness's own prompt prefix teaches
// the model to use ("Default working directory is /app" + "cd <path>
// persists") -- e.g. "gpt2.c" or "./gpt2.c" against "/app/gpt2.c". Exact
// equality still wins first; otherwise the two must share a basename AND
// the write target must not itself be a DIFFERENT absolute path (which
// would be a same-named file somewhere else, not this deliverable).
// Over-matching here only costs one extra `wc -c` proxy call --
// checkDeliverableSize always re-checks `deliverablePath` itself, never the
// matched string -- so a false positive is cheap and a false negative
// (the bug this replaces) is the only direction that actually mattered.
function refersToDeliverable(writtenPath: string, deliverablePath: string): boolean {
  if (writtenPath === deliverablePath) return true;
  if (writtenPath.startsWith("/")) return false;
  const deliverableBase = deliverablePath.split("/").pop();
  const writtenBase = writtenPath.split("/").pop();
  return !!deliverableBase && deliverableBase === writtenBase;
}

/** True when at least one command in this turn writes the given path specifically. */
function writesDeliverablePath(commands: string[], path: string): boolean {
  for (const cmd of commands) {
    for (const w of detectDeliverableWrites(cmd)) {
      if (refersToDeliverable(w.path, path)) return true;
    }
  }
  return false;
}

// A number plus a unit, allowing the phrasing this trigger is verified
// against ("must be <5000 bytes") and adjacent variants. The leading `<` is
// optional and absorbed by the numeric side of the match rather than gated
// behind a `\b` -- `<` is itself a non-word character, so `\b<` can never
// fire next to the space that almost always precedes it in prose.
//
// "less than" is excluded when preceded by "no"/"not" -- "no less than N
// bytes" states a FLOOR, and reading it as this trigger's ceiling would arm
// the guard backwards (steering the model to shrink a file that has no
// upper limit at all). The other alternatives don't have a floor-shaped
// negation in ordinary prose, so only this one needs the guard.
const BYTE_LIMIT_RE =
  /(?:\b(?:under|below|(?<!no\s)(?<!not\s)less\s+than|at\s+most|no\s+more\s+than|must\s+be)\s*|<\s*)<?\s*(\d+)\s*(bytes?|kb|kilobytes?)\b/i;

// An absolute path stated near an instruction verb, e.g. "Call your program
// /app/gpt2.c".
const DELIVERABLE_PATH_RE =
  /\b(?:call\s+your\s+\w+|save\s+(?:it|your\s+\w+)\s+(?:as|to)|write\s+(?:it|your\s+\w+)\s+to|program\s+(?:at|to))\s+(\/\S+)/i;

/**
 * The byte limit stated in the prompt, normalized to bytes (kb/kilobytes
 * x1024). Undefined when the pattern doesn't match at all. First-match, not
 * ambiguity-checked: a prompt stating two DIFFERENT size constraints (e.g. a
 * per-record size alongside the deliverable's own limit) resolves to
 * whichever the regex reaches first, not a detected conflict -- narrow
 * enough that no task in the current suite exercises it. Exported for the
 * smoke test against the real gpt2-codegolf prompt.
 */
export function parseByteLimit(prompt: string): number | undefined {
  const m = BYTE_LIMIT_RE.exec(prompt);
  if (!m) return undefined;
  const value = Number(m[1]);
  const unit = m[2].toLowerCase();
  return unit.startsWith("k") ? value * 1024 : value;
}

/**
 * The deliverable's absolute path stated in the prompt. Undefined when the
 * pattern doesn't match at all -- first-match, same caveat as
 * `parseByteLimit` above. Trailing sentence punctuation (a comma or period
 * immediately after the path, as in "Call your program /app/gpt2.c, I will
 * compile...") is stripped -- `\S+` has no way to distinguish it from a path
 * character, and no real deliverable path ends in one.
 */
export function parseDeliverablePath(prompt: string): string | undefined {
  const m = DELIVERABLE_PATH_RE.exec(prompt);
  return m ? m[1].replace(/[.,;:!?]+$/, "") : undefined;
}

export default function (pi: ExtensionAPI) {
  pi.on("session_start", async () => {
    triggerAFireCount = 0;
    triggerBFired = false;
    triggerCFireCount = 0;
    milestonesFired = new Set<number>();
    deliverableWriteEverSeen = false;
    lastKnownSize = undefined;
    consecutiveOverBudgetChecks = 0;
    byteLimitForRun = undefined;
    deliverablePathForRun = undefined;
  });

  pi.on("before_agent_start", async (event) => {
    turnsThisRun = 0;
    capForRun = resolveTurnCap(event);
    deadlineForRun = resolveDeadlineEpochMs(event);
    startForRun = resolveBudgetStartEpochMs(event);
    armed = false;
    armedAtTurn = 0;
    consecutiveNoWriteTurns = 0;
    lastTurnMessage = undefined;
    triggerDFiredAtTurn = 0;
    // Latch, don't overwrite: a continuation run's prompt is
    // COMPACTION_CONTINUE_PROMPT/ERROR_RETRY_PROMPT, not the task text --
    // see the state comment above for why re-parsing unconditionally here
    // would erase an already-resolved limit/path.
    if (byteLimitForRun === undefined) {
      byteLimitForRun = parseByteLimit((event as any).prompt ?? "");
    }
    if (deliverablePathForRun === undefined) {
      deliverablePathForRun = parseDeliverablePath((event as any).prompt ?? "");
    }
  });

  pi.on("turn_start", async (_event, ctx) => {
    turnsThisRun++;
    if (!isTerminalBench()) return;
    if (!armed && !triggerBFired && finalizeWarnWouldFire({ turnsThisRun, capForRun, deadlineForRun })) {
      armed = true;
      armedAtTurn = turnsThisRun;
    }
    // After the arming check, so a turn that arms finalize-warn's window is
    // already inside the endgame Trigger D defers to.
    maybeFireTriggerD(pi, ctx);
  });

  pi.on("turn_end", async (event, ctx) => {
    if (!isTerminalBench()) return;
    const message: any = (event as any)?.message;
    if (!message) return;
    lastTurnMessage = message;

    const { text, toolCallCount, toolCalls } = contentShape(message);

    // Shared instrumentation: log every terminal_bench turn_end's stopReason
    // and coarse content shape, regardless of trigger state. Diagnostic
    // only — not a harnessIntervention, so it doesn't count as an
    // intervention in the run's metrics.
    ctx.ui.notify(
      `tb-finalize-guard: turn_end stopReason=${String(message.stopReason)} ` +
        `hasText=${text.trim().length > 0} hasToolCalls=${toolCallCount > 0} ` +
        `isEmpty=${text.trim().length === 0 && toolCallCount === 0}`,
      "info",
    );

    // Computed for every turn, not just the ones Trigger B is armed for:
    // Trigger D's evidence flag has to be complete from turn 1, long before
    // Trigger B starts judging compliance.
    const commands = shellCommandsIn(toolCalls);
    const wroteDeliverable = hasNonScratchWrite(commands);
    if (wroteDeliverable) deliverableWriteEverSeen = true;

    const firedA = maybeFireTriggerA(pi, ctx, message, text, toolCallCount);
    if (firedA) return; // precedence: a toolless-quit turn is not also judged for Trigger B compliance

    maybeAdvanceTriggerB(pi, ctx, wroteDeliverable);
    await maybeCheckDeliverableSize(pi, ctx, commands);
  });

  pi.on("agent_end", async (_event, ctx) => {
    if (!isTerminalBench()) return;
    maybeFireTriggerC(pi, ctx);
  });
}

// Every caveat the baseline copy's own limits require, shared by Trigger A
// and C so their baseline mentions can't drift apart on what the copy
// actually guarantees -- mirrors _initial_snapshot_advertisement's own two
// caveats exactly (unconditional size/depth limit; the file-count-cap
// absence caveat only for "partial").
function baselineCaveats(baseline: InitialSnapshotOutcome): string {
  const partial =
    baseline === "partial"
      ? "That copy also hit a file-count cap, so a file missing from it may still " +
        "have existed at the start. "
      : "";
  return `It may not contain very large (>10MB) or deeply nested files. ${partial}`;
}

// Verbatim from _initial_snapshot_advertisement's own restraint sentence
// (minus its "never write into {path}" clause, which has no TS-side
// equivalent to point at), so the two recovery prompts can't drift apart on
// the exact wording: the dangerous misreading of "a reference copy exists"
// is a model near a stopping point restoring pristine originals over the
// solution it just finished writing.
const RESTORE_RESTRAINT =
  "Restore from there only a file you believe you corrupted — never over " +
  "your own completed solution. Treat it as read-only.";

/**
 * Trigger A's nudge text.
 *
 * Exported and pure so tests can assert against the builder's own output
 * instead of stitching prose regexes over the delivered message — the
 * shape `resolveFinalizeMessage` already uses.
 *
 * `baseline` is the start-of-trial copy's outcome, or undefined to say
 * nothing about it at all (see _shared/snapshot-paths.ts).
 */
export function buildTriggerAMessage(
  minutesLeft: number,
  baseline: InitialSnapshotOutcome | undefined,
): string {
  // (1) names a verification protocol rather than asking for a "spot-check":
  // both observed failures were re-checks incapable of failing. (2)/(3)
  // cover the other two ways a finished-looking trial still fails grading —
  // leftover files that weren't asked for, and an ambiguity resolved by
  // guessing at grader intent. "use ShellSession" stays explicit: this fires
  // on a toolless text turn, so a nudge answerable with another toolless
  // text turn would just burn the second fire on the same pattern.
  const baselineClause =
    baseline === undefined
      ? ""
      : "If some task-provided input your answer depends on could have been changed " +
        "this session — by your own edit, a script, or a command that died partway — " +
        `diff just those files against the untouched start-of-trial copy under ` +
        `${INITIAL_SNAPSHOT_APP_DIR}/ (${INITIAL_SNAPSHOT_APP_DIR}/somefile mirrors ` +
        "/app/somefile); it predates every change you made, unlike a backup of your " +
        `own. ${baselineCaveats(baseline)}${RESTORE_RESTRAINT}`;

  return (
    `You stopped without calling a tool, but roughly ${minutesLeft} minutes of budget ` +
    "remain and this task is graded by inspecting the container's files/state " +
    "afterward — not this chat. Re-read the task instructions above and use " +
    "ShellSession to re-check your work — not just that the required files exist: " +
    "(1) re-verify adversarially, not by re-reading your own output: first say what " +
    "result would prove your answer WRONG, then run a check that could actually " +
    "produce that result — re-running the same procedure that produced the answer, " +
    "or looking over the file you wrote, does not count. Recompute the result by a " +
    "different method, or test a consequence of it against the task's own data. " +
    "Never verify by comparing your output against another file you created this " +
    "session (a backup, an earlier copy, an intermediate): both can be wrong the " +
    "same way; " +
    "(2) if you created files the task didn't ask for (leftover scaffolding, " +
    "intermediate outputs), remove only ones you created yourself and that " +
    "nothing else needs — never anything that was already there; " +
    "(3) if you were ever unsure what's expected, resolve it " +
    "by the most literal reading of the task text. " +
    baselineClause +
    "If this recheck passes, say so " +
    "explicitly and stop. Otherwise fix what you found — you have plenty of time; " +
    "do not give up early."
  );
}

/**
 * Trigger C's calm recovery text — the branch for a dead run that is NOT
 * near its deadline. The near-deadline branch uses `resolveFinalizeMessage`
 * instead and is deliberately left alone: it tells the model to stop
 * verifying and save, the opposite regime from this one.
 *
 * Exported and pure for the same reason as `buildTriggerAMessage`.
 */
export function buildTriggerCRecoveryMessage(
  baseline: InitialSnapshotOutcome | undefined,
): string {
  // Ends the sentence itself (a period either way), rather than relying on
  // the closing " Remember..." to supply one -- RESTORE_RESTRAINT already
  // ends in a period, and a second one right after it would read as a typo.
  const sessionClause =
    baseline === undefined
      ? "."
      : "; if you suspect a task-provided file was damaged, the untouched " +
        `start-of-trial copy under ${INITIAL_SNAPSHOT_APP_DIR}/ is the one reference ` +
        `that predates your changes. ${baselineCaveats(baseline)}${RESTORE_RESTRAINT}`;

  return (
    "You still have ample time and turn budget remaining, so do not wrap up — " +
    "retry your last action or continue working from where you left off, " +
    "verifying and testing as you normally would — against the task's own data or " +
    "by a different method, never against another file you created this session" +
    sessionClause +
    " Remember the task is graded " +
    "by inspecting the container's files/state afterward, not this chat, so " +
    "make sure your results end up saved there."
  );
}

/**
 * Trigger D's nudge text, for a `fraction` of the budget spent and
 * `minutesLeft` remaining. Exported and pure for the same reason as
 * `buildTriggerAMessage`.
 *
 * The verification demand is shape-conditional on purpose: a TB deliverable
 * is often not a program, and "run it" would then be a demand the model
 * cannot satisfy — spending exactly the turns this nudge exists to save.
 */
export function buildTriggerDMessage(
  fraction: number,
  minutesLeft: number,
  hasDeliverableEvidence: boolean,
): string {
  const spent =
    `Progress check: about ${Math.round(fraction * 100)}% of this task's time budget ` +
    `is spent (~${minutesLeft} minutes remain)`;
  const check =
    "check it for real instead of assuming: if your deliverable is a program or " +
    "script, run it and read its actual output; otherwise compare the file or state " +
    "you produced against exactly what the task asked for.";

  if (hasDeliverableEvidence) {
    return (
      `${spent}. Make sure your best current version is saved at the task's required ` +
      `path right now, and ${check} Spend what is left improving what you have rather ` +
      "than restarting."
    );
  }
  return (
    `${spent} and nothing has been written outside /tmp yet. This task is graded by ` +
    "inspecting the container's files/state afterward, not this chat. Stop " +
    "investigating and write a first complete version of your deliverable to its real " +
    `path NOW, even if it is rough or incomplete — then ${check} Keep improving it in ` +
    "place from there: a rough deliverable on disk beats a perfect plan that never got " +
    "written."
  );
}

/**
 * Trigger E's escalation text, for a deliverable currently `overBy` bytes
 * past the stated `limit`. Exported and pure for the same reason as
 * `buildTriggerAMessage`.
 */
export function buildTriggerEMessage(overBy: number, limit: number): string {
  return (
    `Your deliverable is currently ~${overBy} bytes over the ${limit}-byte limit ` +
    "stated in the task. This must shrink before the file can be graded — keep " +
    "reducing size, and re-check with `wc -c` after each pass."
  );
}

function maybeFireTriggerA(
  pi: ExtensionAPI,
  ctx: any,
  message: any,
  text: string,
  toolCallCount: number,
): boolean {
  if (triggerAFireCount >= MAX_TRIGGER_A_FIRES) return false;
  if (message.stopReason === "aborted" || message.stopReason === "error") return false;
  // Trigger D already queued a steer for the turn this response answers --
  // firing another one here would stack two different nudges back to back.
  if (triggerDFiredAtTurn === turnsThisRun) return false;

  const hasText = text.trim().length > 0;
  const shapeMatches = hasText && toolCallCount === 0;
  if (!shapeMatches) return false;

  // Budget gate: no deadline known -> "early" is undefined, do nothing.
  if (deadlineForRun <= 0) return false;
  const remainingMs = deadlineForRun - Date.now();
  if (remainingMs < EARLY_QUIT_MIN_REMAINING_MS) return false;

  // Turn-cap headroom: a nudge queued now becomes the prompt for
  // turnsThisRun + 1; if that would exceed the cap, turn-cap aborts before
  // the model ever sees it (mirrors gaia-finalize-guard's own reasoning).
  if (capForRun > 0 && turnsThisRun >= capForRun) return false;

  const minutesLeft = Math.max(0, Math.round(remainingMs / 60000));
  const msg = buildTriggerAMessage(minutesLeft, initialSnapshotOutcome());

  try {
    pi.sendUserMessage(msg, { deliverAs: "steer" });
  } catch {
    // Don't burn the fire count or notify for a nudge that was never
    // actually delivered — mirrors gaia-finalize-guard's ordering.
    return false;
  }
  triggerAFireCount++;
  harnessIntervention(
    ctx,
    `turn ended without a tool call with ~${minutesLeft}m ` +
      "left on the wall-clock budget — telling the model not to give up early.",
  );
  return true;
}

function maybeAdvanceTriggerB(pi: ExtensionAPI, ctx: any, wroteDeliverable: boolean): void {
  if (triggerBFired) return;
  if (!armed) return;
  // The turn during which arming happened is the same turn finalize-warn's
  // steer message was queued for delivery on the NEXT turn — the model
  // hasn't seen it yet, so this turn can't be judged for compliance.
  if (turnsThisRun <= armedAtTurn) return;

  if (wroteDeliverable) {
    consecutiveNoWriteTurns = 0;
    return;
  }

  consecutiveNoWriteTurns++;
  if (consecutiveNoWriteTurns < NO_WRITE_TURNS_BEFORE_NUDGE) return;

  // Turn-cap headroom: a nudge queued now becomes the prompt for
  // turnsThisRun + 1; if that would exceed the cap, turn-cap aborts before
  // the model ever sees it (mirrors maybeFireTriggerA's own equivalent
  // guard above). Bookkeeping above (the counter reset/increment) still
  // happens even when suppressed here, matching Trigger A's placement.
  if (capForRun > 0 && turnsThisRun >= capForRun) return;

  const msg =
    "You still have not written your answer. Run exactly one command now that writes " +
    "your best current value to the deliverable path, then stop. If your previous " +
    "commands already wrote it, say 'done' and stop.";

  try {
    pi.sendUserMessage(msg, { deliverAs: "steer" });
  } catch {
    // Don't burn the one-shot latch or the counter for a nudge that was
    // never actually delivered — leave state as-is so a later turn_end can
    // still try.
    return;
  }
  triggerBFired = true;
  harnessIntervention(
    ctx,
    `${NO_WRITE_TURNS_BEFORE_NUDGE} consecutive turns since finalize-warn fired with no ` +
      "non-scratch write detected — telling the model to write its deliverable now.",
  );
}

function maybeFireTriggerC(pi: ExtensionAPI, ctx: any): void {
  if (triggerCFireCount >= MAX_TRIGGER_C_FIRES) return;

  const last = lastTurnMessage;
  if (!last) return;

  // Aborted runs are a harness decision (thinking-budget's ctx.abort,
  // turn-cap, a user Esc), not a dead run — and thinking-budget in
  // particular queues its own carefully sequenced commit-and-continue
  // recovery after aborting; stacking a second steer on top would
  // contradict it. Checked FIRST, before the shape checks: an abort while
  // only thinking tokens had streamed leaves a message with zero text and
  // zero tool calls (thinking blocks are invisible to contentShape), which
  // would otherwise slip through the isEmpty path below.
  if (last.stopReason === "aborted") return;

  const { text, toolCallCount } = contentShape(last);
  const isEmpty = text.trim().length === 0 && toolCallCount === 0;
  const isError = last.stopReason === "error";
  if (!isEmpty && !isError) return;

  // Turn-cap consistency: at settled time a steer starts a FRESH run
  // (before_agent_start re-fires, resetting turn-cap's counter), so unlike
  // Triggers A/B the nudge here WOULD be delivered — but delivering it
  // would hand a capped-out run an entire new turn budget, overriding
  // turn-cap's policy decision to end the run. Stand down at the cap, same
  // clause as Triggers A/B.
  if (capForRun > 0 && turnsThisRun >= capForRun) return;

  // Near-deadline framing check. `armed` is Trigger B's run-scoped latch,
  // set at turn_start when finalizeWarnWouldFire is true — reusing it here
  // matters because finalizeWarnWouldFire's turn trigger is edge-triggered
  // (exact-turn equality; callers are expected to latch), so re-calling it
  // at settle time would usually miss a window entered turns ago. The
  // direct call additionally catches a wall-clock deadline crossed since
  // the last turn_start (its time trigger is level-triggered).
  const nearDeadline =
    armed || finalizeWarnWouldFire({ turnsThisRun, capForRun, deadlineForRun });

  const prefix =
    "Your previous turn ended with an error or an empty response. The task is NOT " +
    "complete. ";
  const msg = nearDeadline
    ? prefix + resolveFinalizeMessage("terminal_bench")
    : prefix + buildTriggerCRecoveryMessage(initialSnapshotOutcome());

  try {
    pi.sendUserMessage(msg, { deliverAs: "steer" });
  } catch {
    // Don't burn the fire count or notify for a nudge that was never
    // actually delivered — mirrors Trigger A/B's own ordering.
    return;
  }
  triggerCFireCount++;
  harnessIntervention(
    ctx,
    `run settled with a final assistant message of stopReason=${String(last.stopReason)} ` +
      `isEmpty=${isEmpty} — the run ended on an error/empty message; ` +
      (nearDeadline
        ? "telling the model to save its best-effort result now."
        : "telling the model to retry and keep working."),
  );
}

function maybeFireTriggerD(pi: ExtensionAPI, ctx: any): void {
  const fraction = budgetFractionUsed({ startForRun, deadlineForRun });
  if (fraction === undefined) return; // no budget published -> no milestones

  if (armed || finalizeWarnWouldFire({ turnsThisRun, capForRun, deadlineForRun })) return;
  if (finalizeWarnTurnWindowOpen({ turnsThisRun, capForRun })) return;

  // Several milestones can come due at once — one very long turn, or a run
  // idle across a continuation — and the model needs where the clock is
  // now, not a backlog.
  let milestone: number | undefined;
  for (const m of PROGRESS_MILESTONES) {
    if (fraction >= m && !milestonesFired.has(m)) milestone = m;
  }
  if (milestone === undefined) return;

  // Spent silently rather than left unspent: the fraction never falls back
  // below a milestone, so an unspent one would come due again on every
  // later turn, duplicating the next milestone's job.
  if (milestone === PROGRESS_MILESTONES[0] && deliverableWriteEverSeen) {
    milestonesFired.add(milestone);
    return;
  }

  // Turn-cap headroom: same clause as Triggers A/B/C. Deliberately no
  // marking — past the cap there is no later turn for a retry to land on
  // anyway, and a new run re-opens the question honestly.
  if (capForRun > 0 && turnsThisRun >= capForRun) return;

  const minutesLeft = Math.max(0, Math.round((deadlineForRun - Date.now()) / 60000));
  const msg = buildTriggerDMessage(fraction, minutesLeft, deliverableWriteEverSeen);

  try {
    pi.sendUserMessage(msg, { deliverAs: "steer" });
  } catch {
    // Mark nothing: the condition is level-triggered (the fraction only
    // grows), so the next turn_start retries on its own -- no equivalent of
    // finalize-warn's dueThisRun latch needed.
    return;
  }
  // Smaller milestones go down with it: they are strictly less urgent
  // restatements of a message just delivered.
  for (const m of PROGRESS_MILESTONES) {
    if (m <= milestone) milestonesFired.add(m);
  }
  triggerDFiredAtTurn = turnsThisRun;
  harnessIntervention(
    ctx,
    `about ${Math.round(fraction * 100)}% of the wall-clock budget is spent with ` +
      (deliverableWriteEverSeen
        ? "a deliverable already written"
        : "nothing written outside /tmp") +
      " — sending a progress checkpoint.",
  );
}

/**
 * The deliverable's real byte count via the same tb-proxy channel
 * syntax-check uses to reach a file inside the TB container. Every failure
 * mode -- not in TB mode, a proxy error, a timeout, a non-zero exit, a
 * non-numeric response -- returns null rather than fabricating a size,
 * matching syntax-check's own "every failure mode is silence" discipline.
 */
async function checkDeliverableSize(ctx: ProxyUiCtx, path: string): Promise<number | null> {
  if (!inTbMode()) return null;
  let text: string | null;
  try {
    text = await tbProxyRun(ctx, `wc -c ${path}`, SIZE_CHECK_TIMEOUT_SEC, tbSessionId());
  } catch {
    return null;
  }
  if (text === null) return null;
  const { body, footer } = splitFooter(text);
  if (footer === null || footerTimedOut(footer)) return null;
  const exit = footerExit(footer);
  if (exit !== 0) return null;
  const m = /^\s*(\d+)/.exec(body);
  return m ? Number(m[1]) : null;
}

function maybeCheckDeliverableSize(pi: ExtensionAPI, ctx: any, commands: string[]): Promise<void> {
  return (async () => {
    if (deliverablePathForRun === undefined || byteLimitForRun === undefined) return;
    if (!writesDeliverablePath(commands, deliverablePathForRun)) return;

    const size = await checkDeliverableSize(ctx as ProxyUiCtx, deliverablePathForRun);
    if (size === null) return;
    lastKnownSize = size;

    if (size <= byteLimitForRun) {
      consecutiveOverBudgetChecks = 0;
      return;
    }

    consecutiveOverBudgetChecks++;
    // Free, non-harnessIntervention diagnostic on every over-budget check --
    // the escalated nudge below is what actually costs a session-metrics
    // intervention, gated by the consecutive-checks threshold so a run still
    // mid-golf isn't interrupted on every single edit.
    ctx.ui.notify(
      `[byte-limit] ${deliverablePathForRun} is ${size} bytes, over the ` +
        `${byteLimitForRun}-byte limit stated in the task.`,
      "info",
    );

    if (consecutiveOverBudgetChecks < OVER_BUDGET_CHECKS_BEFORE_NUDGE) return;
    // Turn-cap headroom: same clause as Triggers A/B/C/D.
    if (capForRun > 0 && turnsThisRun >= capForRun) return;

    const nearDeadline =
      armed || finalizeWarnWouldFire({ turnsThisRun, capForRun, deadlineForRun });
    const overBy = size - byteLimitForRun;
    const msg = nearDeadline
      ? resolveFinalizeMessage("terminal_bench")
      : buildTriggerEMessage(overBy, byteLimitForRun);

    try {
      pi.sendUserMessage(msg, { deliverAs: "steer" });
    } catch {
      // Don't burn the cooldown for a nudge that was never actually
      // delivered — mirrors every other trigger's ordering.
      return;
    }
    // Cooldown, not a one-shot latch: the next escalation needs another
    // OVER_BUDGET_CHECKS_BEFORE_NUDGE consecutive over-budget checks, rather
    // than firing on every check once the threshold is first crossed.
    consecutiveOverBudgetChecks = 0;
    harnessIntervention(
      ctx,
      `deliverable ${deliverablePathForRun} is ${size} bytes, ${overBy} over the ` +
        `${byteLimitForRun}-byte limit stated in the task` +
        (nearDeadline
          ? " — near deadline, telling the model to save its best version."
          : " — telling the model to keep golfing."),
    );
  })();
}
