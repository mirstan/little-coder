import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { assessResponse, buildCorrectionMessage, phraseForUser, type ToolCall } from "./quality.ts";
import { harnessIntervention } from "../_shared/intervention.ts";

// Port of local/quality.py. Hooks turn_end, inspects the assistant message
// + previous turn's tool calls, and — if we detect a failure mode — sends
// a correction user message with deliverAs:"steer" so the model gets it
// immediately on its next turn rather than waiting for the next user input.

// Session-scoped state. Pi reuses extensions across turns within a session;
// a fresh extension instance is loaded per session via the session lifecycle.
let previousToolCalls: ToolCall[] = [];
let consecutiveFailures = 0;
// Tier-2 escalation state. Like previousToolCalls/consecutiveFailures these
// survive a mid-task compaction on purpose: compaction fires `session_compact`
// in place, not `session_start`, and the loop being broken outlives it.
let blockedCall: ToolCall | null = null;
let tier2Notified = false;
// Past this many consecutive failures, tier 1's plain correction has visibly
// failed and turn_end escalates instead (it used to go silent for the rest of
// the trial — a real run then looped for 334 more turns unopposed).
const MAX_CONSECUTIVE_CORRECTIONS = 2;

export default function (pi: ExtensionAPI) {
  // Populate the known-tools set lazily by observing tool_execution events.
  // This avoids needing to read pi's tool registry directly.
  const knownTools = new Set<string>();
  pi.on("tool_execution_start", async (event) => {
    const name = (event as any).toolName;
    if (typeof name === "string") knownTools.add(name);
  });

  pi.on("session_start", async () => {
    previousToolCalls = [];
    consecutiveFailures = 0;
    blockedCall = null;
    tier2Notified = false;
  });

  // Tier 2 for a repeat loop: reject the looping call outright. pi delivers a
  // `{block, reason}` result to the model as a tool error in the same turn, so
  // breaking the loop doesn't depend on the model heeding steered text. No
  // unblock path is needed — any difference in the input stops matching.
  //
  // Known gap: `event.input` is schema-validated (coerced), while blockedCall
  // came from the raw content-block arguments turn_end reads. Coercion is the
  // identity for string fields, so shell-command repeats — the loop shape this
  // targets — match; a repeat on a tool with a coercible numeric arg would not.
  pi.on("tool_call", async (event) => {
    if (!blockedCall) return;
    const name = (event as any).toolName;
    // ToolCallEvent carries only `input`; there is no `args`/`arguments` form.
    const input = (event as any).input;
    if (name === blockedCall.name && JSON.stringify(input) === JSON.stringify(blockedCall.input)) {
      return {
        block: true,
        reason:
          "This exact tool call is blocked -- it's a verbatim repeat of a call " +
          "already flagged as looping, with nothing else changing in between. " +
          "Take a materially different action.",
      };
    }
  });

  pi.on("turn_end", async (event, ctx) => {
    const message = (event as any).message;
    if (!message) return;

    // Skip turns that were interrupted/aborted — by the user pressing ESC OR by
    // a harness abort (thinking-budget, turn-cap). pi marks these with
    // stopReason "aborted"; their content is legitimately partial/empty, so
    // assessing them spuriously fires `empty_response` and steers a "your
    // previous response was empty" correction onto the user's NEXT prompt
    // (the escape-interrupt bug, and the second false warning in the
    // thinking-budget cascade). An aborted turn is not a model quality failure.
    if (message.stopReason === "aborted") return;

    // Skip turns that ended in a provider/transport error (pi marks these
    // stopReason "error"). An API 400 — e.g. an ollama model that rejects the
    // requested thinking level (issue #86) — leaves the content empty, which
    // `assessResponse` would read as `empty_response` and "correct" by re-sending
    // the same request 3× before backing off, burying pi's real error under
    // three "your previous response was empty" lines. Steering a correction
    // can't fix a 400. pi already renders the actual error; we add one targeted
    // hint for the thinking case, whose fix (lower the thinking level) isn't
    // obvious from the raw message, then stop.
    if (message.stopReason === "error") {
      const errMsg = typeof message.errorMessage === "string" ? message.errorMessage : "";
      if (/does not support thinking|reasoning setting .*is not supported/i.test(errMsg)) {
        harnessIntervention(
          ctx,
          "this model rejected the requested thinking level — lower it with " +
            "shift+tab (thinking cycle) or in /settings, or pick a model that supports thinking.",
        );
      }
      return;
    }

    // Extract assistant text + tool calls from pi's content-block format
    const content = Array.isArray(message.content) ? message.content : [];
    const text = content
      .filter((c: any) => c?.type === "text")
      .map((c: any) => c.text ?? "")
      .join("\n");
    const currentCalls: ToolCall[] = content
      .filter((c: any) => c?.type === "toolCall")
      .map((c: any) => ({ name: c.name, input: c.arguments ?? c.input ?? {} }));

    const verdict = assessResponse(text, currentCalls, previousToolCalls, knownTools);

    // Update rolling state for next turn regardless of verdict.
    previousToolCalls = currentCalls;

    if (verdict.ok) {
      consecutiveFailures = 0;
      blockedCall = null;
      tier2Notified = false;
      return;
    }

    consecutiveFailures++;
    if (consecutiveFailures > MAX_CONSECUTIVE_CORRECTIONS) {
      // Escalate once per streak; re-sending the same escalation every turn is
      // the "keep sending what already didn't work" failure this replaced.
      if (tier2Notified) return;
      // assessResponse names the exact offending call itself (its own
      // envChanged-aware match, not re-run independently here) — a multi-call
      // turn can have an earlier call that matches the previous turn but was
      // exempted (issue #81), so re-deriving this separately, without that
      // exemption, could name the wrong one.
      const offending = verdict.reason === "repeated_tool_call" ? verdict.offendingCall : undefined;
      tier2Notified = true;
      if (offending) {
        blockedCall = offending;
        harnessIntervention(
          ctx,
          `${phraseForUser(verdict.reason)} — blocking the exact call and escalating.`,
        );
        // No repeat count in this text: consecutiveFailures counts failures of
        // any reason, so "repeated N times" could be false on a mixed streak.
        pi.sendUserMessage(
          "You are stuck in a loop: a tool call has repeated verbatim with nothing " +
            "else changing in between. That exact call is now blocked -- it will be " +
            "rejected if you try it again. Explain what isn't working, then take a " +
            "materially different action.",
          { deliverAs: "steer" },
        );
      } else {
        // Nothing to block (empty response, unknown tool, malformed args). Keep
        // correcting rather than going silent.
        harnessIntervention(
          ctx,
          `${phraseForUser(verdict.reason)} — still correcting after ${consecutiveFailures} in a row.`,
        );
        pi.sendUserMessage(buildCorrectionMessage(verdict.reason), { deliverAs: "steer" });
      }
      return;
    }

    const correction = buildCorrectionMessage(verdict.reason);
    harnessIntervention(ctx, `${phraseForUser(verdict.reason)} — redirecting the model.`);
    // "steer" delivers the correction promptly to the in-flight loop. The
    // prior "followUp" mode parked the message until the *next* user input,
    // by which point it was no longer relevant (issue #16).
    pi.sendUserMessage(correction, { deliverAs: "steer" });
  });
}
