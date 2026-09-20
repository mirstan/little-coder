import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
  assessResponse,
  buildBlockedCallEscalationMessage,
  buildCorrectionMessage,
  buildFailureSignatureMessage,
  buildNearDuplicateLoopMessage,
  BLOCKED_CALL_REASON,
  phraseForUser,
  sameCall,
  type ToolCall,
} from "./quality.ts";
import { FuzzyLoopTracker, type NearDuplicateDetection } from "./similarity.ts";
import {
  boundResultText,
  FailureSignatureTracker,
  type FailureSignatureDetection,
} from "./failure-signature.ts";
import { harnessIntervention, type InterventionCtx } from "../_shared/intervention.ts";

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
// in place, not `session_start`, and the loop being broken outlives it. They
// do NOT survive a genuine new prompt in the same session — see the `input`
// handler below — since a fresh task shouldn't inherit a block or a failure
// streak from whatever the previous one left behind.
let blockedCall: ToolCall | null = null;
let tier2NotifiedKey: string | null = null;
// This turn's tool_call events, in pi's validated (schema-coerced) space —
// see the tool_call handler's comment on why blockedCall is armed from this,
// not from turn_end's raw content-block extraction. Cleared once turn_end
// consumes it for the turn that just ended.
let turnValidatedCalls: ToolCall[] = [];
// Two loop detectors the verbatim breaker above cannot see: near-identical
// (not identical) calls, and differing calls whose OUTPUT repeats. Both are
// steer-only and neither touches consecutiveFailures or blockedCall, so the
// hard block stays verbatim-exact. Their lifetime matches the state above --
// through a compaction, not through a new prompt -- and they are rebuilt
// rather than cleared so each reset re-reads the env knobs.
let fuzzyTracker = new FuzzyLoopTracker();
let failsigTracker = new FailureSignatureTracker();
// This turn's tool results. tool_result fires mid-turn, before the calls that
// produced it have been clustered, so they are correlated at turn_end.
let turnResults: BufferedResult[] = [];
// Advanced only on assessed turns, so an aborted turn cannot age entries out
// of either detector's window.
let assessedTurns = 0;
// Past this many consecutive failures, tier 1's plain correction has visibly
// failed and turn_end escalates instead of going silent for the rest of the trial.
const MAX_CONSECUTIVE_CORRECTIONS = 2;

// pi's raw content blocks carry the call id the matching tool_result reports;
// without one a call simply is not correlated, rather than every uncorrelated
// call sharing a single undefined key.
function callId(block: any): string | undefined {
  const id = block?.id ?? block?.toolCallId;
  return typeof id === "string" ? id : undefined;
}

// Outcome-repetition outranks input-similarity: "your changes aren't changing
// the outcome" is the more actionable of the two, and when the same attempts
// trigger both, it already says what the other would.
function steerLoopDetection(
  pi: ExtensionAPI,
  ctx: InterventionCtx,
  failsig: FailureSignatureDetection | null,
  fuzzy: NearDuplicateDetection | null,
): void {
  if (failsig) {
    failsigTracker.markNotified(failsig.sigKey);
    // Cross-suppression: the corroborated message already says the attempts
    // were near-identical, so the cluster must not say it again a turn later.
    if (failsig.clusterId !== undefined) fuzzyTracker.markNotified(failsig.clusterId);
    harnessIntervention(
      ctx,
      `${phraseForUser(failsig.reason)} — telling it to change approach.`,
    );
    pi.sendUserMessage(
      buildFailureSignatureMessage(failsig.toolName, failsig.count, {
        corroborated: failsig.corroborated,
        escalated: failsig.escalated,
        failed: failsig.failed,
      }),
      { deliverAs: "steer" },
    );
    return;
  }
  if (fuzzy) {
    fuzzyTracker.markNotified(fuzzy.clusterId);
    harnessIntervention(ctx, `${phraseForUser(fuzzy.reason)} — redirecting it.`);
    pi.sendUserMessage(buildNearDuplicateLoopMessage(fuzzy.count, fuzzy.escalated), {
      deliverAs: "steer",
    });
  }
}

interface BufferedResult {
  toolCallId?: string;
  toolName: string;
  input: unknown;
  text: string;
  isError: boolean;
}

function resetLoopDetectors(): void {
  fuzzyTracker = new FuzzyLoopTracker();
  failsigTracker = new FailureSignatureTracker();
  turnResults = [];
  assessedTurns = 0;
}

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
    tier2NotifiedKey = null;
    turnValidatedCalls = [];
    resetLoopDetectors();
  });

  // A genuinely new prompt in the same session is a fresh task, not a
  // continuation of whatever failure streak the previous one left behind —
  // without this, a block or streak from task A would carry into task B.
  // Mirrors thinking-budget's own input handler and its filter: our OWN
  // steer messages above ALSO route through prompt() and fire this event
  // with source "extension" — without excluding that, the escalation steer
  // we just sent would immediately wipe the state it exists to protect.
  // `streamingBehavior` catches what `source` alone can't: it's set only
  // when the user types while the agent is mid-turn (a steer of the CURRENT
  // task), which also isn't a new task.
  pi.on("input", async (event) => {
    if ((event as any)?.source === "extension" || (event as any)?.streamingBehavior !== undefined) {
      return;
    }
    previousToolCalls = [];
    consecutiveFailures = 0;
    blockedCall = null;
    tier2NotifiedKey = null;
    turnValidatedCalls = [];
    resetLoopDetectors();
  });

  // Tier 2 for a repeat loop: reject the looping call outright. pi delivers a
  // `{block, reason}` result to the model as a tool error in the same turn, so
  // breaking the loop doesn't depend on the model heeding steered text. No
  // unblock path is needed — any difference in the input stops matching.
  pi.on("tool_call", async (event) => {
    // ToolCallEvent carries only `toolName`/`input`, both properly typed on
    // every member of the union -- no `args`/`arguments` form, no cast
    // needed. `input` IS pi's validated (schema-coerced) space, which is why
    // turn_end records this same array to arm blockedCall from, rather than
    // its own raw content-block extraction -- a numeric arg submitted as a
    // string coerces here but not there, and comparing across the two spaces
    // would silently never match.
    const call: ToolCall = { name: event.toolName, input: event.input };
    turnValidatedCalls.push(call);

    // Deliberately no same-turn state-changed exemption (unlike
    // assessResponse's own, issue #81): turnValidatedCalls records every
    // call this hook SEES, not every call that actually ran, so a call
    // another guard rejected would still count as "environment changed" --
    // and worse, ANY trivial state-changing call (even a no-op) would,
    // letting a looping model permanently defeat the block by prefixing
    // every retry with one. Cost of staying unconditional: a genuine
    // same-turn fix-then-retry (Edit, then the blocked command) gets this
    // one retry rejected too, but blockedCall clears on that turn's own ok
    // verdict, so the very next turn succeeds.
    if (blockedCall && sameCall(call, blockedCall)) {
      return { block: true, reason: BLOCKED_CALL_REASON };
    }
  });

  // A blocked call emits no tool_result: pi answers it with an immediate
  // result that never reaches afterToolCall, this event's only caller. So a
  // model bouncing off the tier-2 block cannot feed the watchdog its own
  // rejection text as a repeated failure.
  pi.on("tool_result", async (event) => {
    const e = event as any;
    const content = Array.isArray(e.content) ? e.content : [];
    const text = content
      .filter((c: any) => c?.type === "text")
      .map((c: any) => c.text ?? "")
      .join("");
    if (!text) return;
    turnResults.push({
      toolCallId: typeof e.toolCallId === "string" ? e.toolCallId : undefined,
      toolName: String(e.toolName ?? ""),
      input: e.input,
      // Bounded here rather than at turn_end: this buffer holds every result
      // of a turn at once, and a turn can run several verbose calls.
      text: boundResultText(text),
      isError: e.isError === true,
    });
  });

  pi.on("turn_end", async (event, ctx) => {
    // Consumed unconditionally, before every early return below — the
    // missing-message one included: any tool_call events recorded while THIS
    // turn was in flight belong to this turn no matter how it concludes.
    // Leaving this cleared only on the normal path let an aborted/errored
    // turn's calls leak into the NEXT turn's offending-call lookup, misarming
    // the block on a stale entry from a turn that was never even assessed.
    const thisTurnValidatedCalls = turnValidatedCalls;
    turnValidatedCalls = [];
    // Same reasoning, and the same consequence for an aborted turn: results
    // that arrived while it was in flight are dropped with it rather than
    // counted toward a streak on a turn that was never assessed.
    const thisTurnResults = turnResults;
    turnResults = [];

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

    const content = Array.isArray(message.content) ? message.content : [];
    const text = content
      .filter((c: any) => c?.type === "text")
      .map((c: any) => c.text ?? "")
      .join("\n");
    const callBlocks = content.filter((c: any) => c?.type === "toolCall");
    const currentCalls: ToolCall[] = callBlocks.map((c: any) => ({
      name: c.name,
      input: c.arguments ?? c.input ?? {},
    }));

    const verdict = assessResponse(text, currentCalls, previousToolCalls, knownTools);

    // Update rolling state for next turn regardless of verdict.
    previousToolCalls = currentCalls;

    // Both detectors take every assessed turn, including failing ones: a
    // near-duplicate loop is a sequence of turns that each look individually
    // fine, so skipping the turns the verdict already flagged would leave
    // holes in a window that exists to span them. Only the SPEAKING is
    // arbitrated, below.
    assessedTurns++;
    const fuzzyDetection = fuzzyTracker.recordTurn(
      assessedTurns,
      currentCalls.map((call, i) => ({ call, id: callId(callBlocks[i]) })),
    );
    // After the fuzzy update, so this turn's results can be credited to the
    // cluster their own call just joined.
    let failsigDetection: FailureSignatureDetection | null = null;
    for (const r of thisTurnResults) {
      const clusterId = fuzzyTracker.clusterIdForToolCallId(r.toolCallId);
      const detection = failsigTracker.record(
        {
          toolName: r.toolName,
          input: r.input,
          text: r.text,
          isError: r.isError,
          clusterId,
          corroborated: fuzzyTracker.clusterSize(clusterId) >= 2,
        },
        assessedTurns,
      );
      if (detection) failsigDetection = detection;
    }

    if (verdict.ok) {
      consecutiveFailures = 0;
      blockedCall = null;
      tier2NotifiedKey = null;
      // An ok verdict deliberately does NOT reset the two trackers: their
      // whole subject matter is loops made of individually-ok turns.
      steerLoopDetection(pi, ctx, failsigDetection, fuzzyDetection);
      return;
    }

    consecutiveFailures++;
    if (consecutiveFailures > MAX_CONSECUTIVE_CORRECTIONS) {
      // assessResponse names the exact offending call itself (its own
      // envChanged-aware match, not re-run independently here) — a multi-call
      // turn can have an earlier call that matches the previous turn but was
      // exempted (issue #81), so re-deriving this separately, without that
      // exemption, could name the wrong one. Re-armed on every repeated_tool_
      // call verdict for as long as the streak continues, even after the
      // one-time notification below has already fired: if the model can't
      // retry the now-blocked call and starts looping on a DIFFERENT one
      // instead, that's still one continuous failure streak, and the block
      // needs to follow whichever call is looping right now, not freeze on
      // the first one.
      const offending = verdict.reason === "repeated_tool_call" ? verdict.offendingCall : undefined;
      if (offending) {
        // Arm from the validated-space recording of this same call, not the
        // raw offending object itself, so the tool_call handler's comparison
        // (also validated-space) can actually match it next time. Matched by
        // POSITION, not by name or by sameCall: a name-only match picks the
        // wrong call when a turn has two calls sharing a tool name, and
        // sameCall's input-inclusive comparison would defeat the exact case
        // this recording exists for -- raw "5000" vs. validated 5000 would
        // never satisfy it. tool_call fires once per call in the same order
        // the assistant's own content blocks list them, so `offending`'s
        // index within `currentCalls` is a reliable correspondence into
        // thisTurnValidatedCalls. Falls back to the raw value if not found
        // (defensive; should always be found in practice).
        const index = currentCalls.indexOf(offending);
        const validated = index >= 0 ? thisTurnValidatedCalls[index] : undefined;
        blockedCall = validated ?? offending;
      }

      // Notify once per distinct situation, not once per streak: a
      // DIFFERENT failure reason arising after an earlier tier-2
      // notification must still get its own message, or the model gets
      // zero feedback for it -- silence through a different door than the
      // one this PR closed. "blocked" is one key for the whole
      // repeated_tool_call case regardless of WHICH call is currently
      // armed, since that message's text never varies by call, and
      // re-arming (above) is not gated on this at all -- see its own
      // comment.
      const notifyKey = offending ? "blocked" : verdict.reason;
      if (tier2NotifiedKey === notifyKey) return;
      tier2NotifiedKey = notifyKey;
      if (offending) {
        harnessIntervention(
          ctx,
          `${phraseForUser(verdict.reason)} — blocking the exact call and escalating.`,
        );
        // No repeat count in this text: consecutiveFailures counts failures of
        // any reason, so "repeated N times" could be false on a mixed streak.
        // See buildBlockedCallEscalationMessage's own comment for why this
        // isn't a bare "try something different" push.
        pi.sendUserMessage(buildBlockedCallEscalationMessage(), { deliverAs: "steer" });
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
