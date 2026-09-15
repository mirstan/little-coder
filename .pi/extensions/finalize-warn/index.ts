import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { harnessIntervention } from "../_shared/intervention.ts";
import { resolveTurnCap } from "../_shared/turn-cap.ts";
import { resolveDeadlineEpochMs } from "../_shared/deadline.ts";
import { resolveFinalizeMessage } from "../_shared/finalize-message.ts";
import {
  WARN_REMAINING,
  WARN_REMAINING_MS,
  finalizeWarnWouldFire,
} from "../_shared/finalize-warn-trigger.ts";

// Pre-cap finalize-warn: when the agent is running low, inject a follow-up
// user message telling it to wrap up. Two independent triggers share the
// same single-shot warning:
//
//   - turn-count: WARN_REMAINING turns left (this turn included) before
//     the turn-cap would abort the run.
//   - wall-clock: WARN_REMAINING_MS left before a known deadline. This
//     exists because a trial can be turn-count-healthy but time-starved --
//     confirmed directly: a Terminal-Bench trial hit n_turns=74/80 with
//     stop_reason "deadline", meaning the model had turns to spare but ran
//     out of wall-clock time. The turn-count trigger alone structurally
//     cannot catch that case. Raised from 5 to 10 minutes after trajectory
//     review found trials that were close to a working answer when the
//     deadline hit but never persisted it -- raman-fitting's last logged
//     action computed valid curve fits, but the trial ended before
//     /app/results.json was ever written. 5 minutes wasn't reliably enough
//     wall-clock headroom to go from "have the answer" to "answer is on
//     disk," especially once resolveFinalizeMessage's terminal_bench text
//     also got more directive about saving immediately instead of verifying
//     further -- the model needs actual time to act on that instruction.
//
// Why this exists (turn-count side): a recurring small-model failure mode
// is "ran out of turns mid-thought, never produced a final answer, output
// extraction fell back to garbage." The warning fires once per agent run,
// only when the cap/deadline is large enough to give the model real
// headroom (cap > WARN_REMAINING).
//
// This is intentionally a separate extension from turn-cap so that the
// abort policy and the warn policy stay independent and can be tuned /
// disabled separately.
//
// pi.sendUserMessage(...,{deliverAs:"followUp"}) queues the message for the
// NEXT user turn — so a warning fired at turn 39 only reaches the model at
// turn 40, leaving 1 useful turn of headroom (then turn 41 = abort). Raised
// to 5 so the message lands ~4 turns before cap, giving the model real room.

// WARN_REMAINING / WARN_REMAINING_MS and the trigger condition itself now
// live in _shared/finalize-warn-trigger.ts — see that module's header for
// why (tb-finalize-guard independently re-derives this same condition and
// used to keep a hand-copied pair of these constants "in lockstep" with
// nothing enforcing it).

let turnsThisRun = 0;
let capForRun = 0;
let deadlineForRun = 0;
let warnedThisRun = false;

export default function (pi: ExtensionAPI) {
  pi.on("before_agent_start", async (event) => {
    turnsThisRun = 0;
    warnedThisRun = false;
    capForRun = resolveTurnCap(event);
    deadlineForRun = resolveDeadlineEpochMs(event);
  });

  pi.on("turn_start", async (_event, ctx) => {
    turnsThisRun++;
    if (warnedThisRun) return;

    if (!finalizeWarnWouldFire({ turnsThisRun, capForRun, deadlineForRun })) return;

    // Re-derive which of the two triggers fired, purely for message wording
    // (finalizeWarnWouldFire only reports whether — not which).
    const turnTrigger =
      capForRun > WARN_REMAINING &&
      turnsThisRun === capForRun - WARN_REMAINING + 1;
    const remainingMs = deadlineForRun > 0 ? deadlineForRun - Date.now() : Infinity;
    const timeTrigger = deadlineForRun > 0 && remainingMs <= WARN_REMAINING_MS;

    warnedThisRun = true;
    const msg = resolveFinalizeMessage(process.env.LITTLE_CODER_BENCHMARK);
    harnessIntervention(
      ctx,
      timeTrigger && !turnTrigger
        ? `~${Math.max(0, Math.round(remainingMs / 1000))}s left on the wall-clock budget — telling the model to finalize its answer now.`
        : `${WARN_REMAINING} turns left — telling the model to finalize its answer now.`,
    );
    try {
      pi.sendUserMessage(msg, { deliverAs: "followUp" });
    } catch {
      // SDK without sendUserMessage — silently no-op rather than break the run
    }
  });
}
