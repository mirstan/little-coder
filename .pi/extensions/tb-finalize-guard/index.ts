import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { harnessIntervention } from "../_shared/intervention.ts";
import { resolveTurnCap } from "../_shared/turn-cap.ts";
import { resolveDeadlineEpochMs } from "../_shared/deadline.ts";
import { SHELL_TOOLS, detectDeliverableWrites, isScratchPath } from "../_shared/shell-write.ts";
import { finalizeWarnWouldFire } from "../_shared/finalize-warn-trigger.ts";

// tb-finalize-guard: a merged guard for Terminal-Bench with two independent
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
// session. The trade-off is a real limitation: a quit that manifests as a
// silent stopReason:"error" turn, rather than a toolless one with visible
// text, is not steerable by this trigger. It remains diagnosable from the
// run log via the turn_end instrumentation below, which is the point of
// keeping that logging unconditional.
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
// finalize-warn's message is delivered as deliverAs:"followUp", which lands
// on the model's *next* turn, not the turn during which the trigger fired
// (see finalize-warn/index.ts's own comment on this). So "armed" here also
// tracks the turn number at which arming happened, and compliance is judged
// starting from the turn AFTER that one — the model can't be faulted for not
// complying with a message it hasn't seen yet.
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
// Shared instrumentation
// ---------------------------------------------------------------------------
// On every terminal_bench turn_end, log the turn's stopReason and a coarse
// content-shape summary via ctx.ui.notify at "info"-but-diagnostic framing —
// NOT a harnessIntervention call, deliberately, so this doesn't inflate the
// intervention-count metric with pure diagnostics. This makes a silent
// stopReason:"error" turn — the shape Trigger A can't steer on, see above —
// diagnosable from the run log instead of invisible.

// WARN_REMAINING_MS lives in _shared/finalize-warn-trigger.ts now — see that
// module's header for why (this constant used to be hand-copied here and in
// finalize-warn/index.ts with nothing enforcing they stayed in lockstep).
const EARLY_QUIT_MIN_REMAINING_MS = 20 * 60 * 1000; // double finalize-warn's WARN_REMAINING_MS
const MAX_TRIGGER_A_FIRES = 2; // per session

const NO_WRITE_TURNS_BEFORE_NUDGE = 2; // consecutive non-compliant turns

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

export default function (pi: ExtensionAPI) {
  pi.on("session_start", async () => {
    triggerAFireCount = 0;
    triggerBFired = false;
  });

  pi.on("before_agent_start", async (event) => {
    turnsThisRun = 0;
    capForRun = resolveTurnCap(event);
    deadlineForRun = resolveDeadlineEpochMs(event);
    armed = false;
    armedAtTurn = 0;
    consecutiveNoWriteTurns = 0;
  });

  pi.on("turn_start", async () => {
    turnsThisRun++;
    if (!isTerminalBench()) return;
    if (!armed && !triggerBFired && finalizeWarnWouldFire({ turnsThisRun, capForRun, deadlineForRun })) {
      armed = true;
      armedAtTurn = turnsThisRun;
    }
  });

  pi.on("turn_end", async (event, ctx) => {
    if (!isTerminalBench()) return;
    const message: any = (event as any)?.message;
    if (!message) return;

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

    const firedA = maybeFireTriggerA(pi, ctx, message, text, toolCallCount);
    if (firedA) return; // precedence: a toolless-quit turn is not also judged for Trigger B compliance

    maybeAdvanceTriggerB(pi, ctx, toolCalls);
  });
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
  const msg =
    `You stopped without calling a tool, but roughly ${minutesLeft} minutes of budget ` +
    "remain and this task is graded by inspecting the container's files/state " +
    "afterward — not this chat. Re-read the task instructions above and use " +
    "ShellSession to verify every required file/state exists exactly as specified. " +
    "If everything is verified in place, say so explicitly and stop. Otherwise, keep " +
    "working — you have plenty of time; do not give up early.";

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
    `turn ended without a tool call (or errored with empty content) with ~${minutesLeft}m ` +
      "left on the wall-clock budget — telling the model not to give up early.",
  );
  return true;
}

function maybeAdvanceTriggerB(pi: ExtensionAPI, ctx: any, toolCalls: any[]): void {
  if (triggerBFired) return;
  if (!armed) return;
  // The turn during which arming happened is the same turn finalize-warn's
  // followUp message was queued for delivery on the NEXT turn — the model
  // hasn't seen it yet, so this turn can't be judged for compliance.
  if (turnsThisRun <= armedAtTurn) return;

  const commands = shellCommandsIn(toolCalls);
  if (hasNonScratchWrite(commands)) {
    consecutiveNoWriteTurns = 0;
    return;
  }

  consecutiveNoWriteTurns++;
  if (consecutiveNoWriteTurns < NO_WRITE_TURNS_BEFORE_NUDGE) return;

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
