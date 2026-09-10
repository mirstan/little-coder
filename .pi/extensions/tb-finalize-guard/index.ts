import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { harnessIntervention } from "../_shared/intervention.ts";
import { resolveTurnCap } from "../_shared/turn-cap.ts";
import { resolveDeadlineEpochMs } from "../_shared/deadline.ts";
import { SHELL_TOOLS, detectWriteTargets, isScratchPath } from "../_shared/shell-write.ts";

// tb-finalize-guard: a merged guard for Terminal-Bench with two independent
// trigger conditions (Plan 3 + Plan 4, reconciled after a Fable adversarial
// review — see the plan doc for the full history). Both are scoped to
// LITTLE_CODER_BENCHMARK === "terminal_bench" only; GAIA has its own
// separate gaia-finalize-guard, and the two must never fire on the same
// benchmark's sessions.
//
// This is a sibling of finalize-warn, not an addition to it: abort policy
// (turn-cap, thinking-budget), warn policy (finalize-warn), and guard policy
// (this extension) are kept in separate extensions so each can be tuned or
// disabled independently — matching finalize-warn's own header convention.
//
// ---------------------------------------------------------------------------
// Trigger A — early voluntary quit
// ---------------------------------------------------------------------------
// Motivated by trials that stopped calling tools with most of their wall-clock
// budget still unused (gpt2-codegolf, break-filter-js-from-html — see the plan
// doc). IMPORTANT scope-honesty note carried over from the plan: neither of
// those two specific trials is actually caught by Trigger A as implemented
// here. break-filter-js-from-html's last turns hit thinking-budget's abort
// (stopReason "aborted", excluded below). gpt2-codegolf's last turn was a
// *successful* tool call with no quality-monitor complaint, which points at a
// silent stopReason:"error" turn somewhere upstream of the visible
// transcript, not a toolless quit.
//
// Trigger A used to also match (empty content + stopReason "error") as a
// forward-looking net for a case shaped like that recurring. That clause was
// removed: stopReason "error" is a provider/transport failure, not a model
// decision — gaia-finalize-guard bails on `stopReason === "aborted" ||
// "error"` for the same reason, and quality-monitor carries the identical
// note ("can't fix a 400 by steering"). Worse, MAX_TRIGGER_A_FIRES is
// session-scoped, so a repeating provider error could burn both fires on
// turns the model never controlled, leaving the guard disarmed for the real
// early-quit it exists to catch. The trade-off: a genuine silent-error early
// quit (the gpt2-codegolf case) no longer gets steered — but it remains
// diagnosable from the run log via the turn_end instrumentation below, which
// is the point of keeping that logging unconditional.
//
// ---------------------------------------------------------------------------
// Trigger B — post-finalize-warn non-compliance
// ---------------------------------------------------------------------------
// Motivated by trials where finalize-warn's "save now" nudge fired correctly
// but the model kept investigating instead of writing its deliverable
// (overfull-hbox, mteb-leaderboard — see the plan doc).
//
// finalize-warn keeps no exported latch or state describing whether it has
// already fired this run (checked: finalize-warn/index.ts has no exports at
// all beyond the default extension function, and _shared/ has no shared
// "has finalize-warn fired" module). Per the plan's explicit preference for
// re-deriving over adding new cross-extension coupling, this guard
// independently re-derives finalize-warn's own trigger condition (turn-count
// OR wall-clock, computed the same way at turn_start) rather than reading
// finalize-warn's private state. The WARN_REMAINING / WARN_REMAINING_MS
// constants below are deliberately kept in lockstep with finalize-warn's
// (5 turns / 10 minutes) — if those ever change there, they must change here
// too, since this guard's "armed" condition needs to describe the same
// moment finalize-warn's nudge lands, not a different one.
//
// finalize-warn's message is delivered as deliverAs:"followUp", which lands
// on the model's *next* turn, not the turn during which the trigger fired
// (see finalize-warn/index.ts's own comment on this). So "armed" here also
// tracks the turn number at which arming happened, and compliance is judged
// starting from the turn AFTER that one — the model can't be faulted for not
// complying with a message it hasn't seen yet.
//
// Compliance is judged via _shared/shell-write.ts's existing write-command
// classification, extended with `isScratchPath` (added alongside this guard)
// so a write that only ever lands in /tmp does not count as having saved the
// real deliverable (Codex finding folded into the merged plan).
//
// ---------------------------------------------------------------------------
// Shared instrumentation
// ---------------------------------------------------------------------------
// On every terminal_bench turn_end, log the turn's stopReason and a coarse
// content-shape summary via ctx.ui.notify at "info"-but-diagnostic framing —
// NOT a harnessIntervention call, deliberately, so this doesn't inflate the
// intervention-count metric with pure diagnostics. This exists so a future
// occurrence of gpt2-codegolf's silent stopReason:"error" turn is
// diagnosable from the run log instead of invisible.

const EARLY_QUIT_MIN_REMAINING_MS = 20 * 60 * 1000; // double finalize-warn's WARN_REMAINING_MS
const MAX_TRIGGER_A_FIRES = 2; // per session

// Mirrors finalize-warn/index.ts's own constants exactly — see the header
// comment above for why these must stay in lockstep.
const WARN_REMAINING = 5; // turns
const WARN_REMAINING_MS = 10 * 60 * 1000; // wall-clock headroom before deadline

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

/** Re-derives finalize-warn's own turn-count-OR-wall-clock trigger condition. */
function finalizeWarnWouldFire(): boolean {
  const turnTrigger =
    capForRun > WARN_REMAINING && turnsThisRun === capForRun - WARN_REMAINING + 1;
  const remainingMs = deadlineForRun > 0 ? deadlineForRun - Date.now() : Infinity;
  const timeTrigger = deadlineForRun > 0 && remainingMs <= WARN_REMAINING_MS;
  return turnTrigger || timeTrigger;
}

/** Every ShellSession/bash command string found in this turn's tool calls. */
function shellCommandsIn(toolCalls: any[]): string[] {
  const commands: string[] = [];
  for (const c of toolCalls) {
    if (typeof c?.name !== "string" || !SHELL_TOOLS.has(c.name)) continue;
    const args = c.arguments ?? c.input ?? {};
    if (typeof args?.command === "string") commands.push(args.command);
  }
  return commands;
}

/** True when at least one command in this turn writes somewhere other than scratch. */
function hasNonScratchWrite(commands: string[]): boolean {
  for (const cmd of commands) {
    for (const w of detectWriteTargets(cmd)) {
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
    if (!armed && !triggerBFired && finalizeWarnWouldFire()) {
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
