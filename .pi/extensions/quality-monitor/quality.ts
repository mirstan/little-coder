// Port of local/quality.py::assess_response + build_correction_message.

export interface ToolCall {
  name: string;
  input: unknown;
}

export type QualityResult =
  | { ok: true }
  | { ok: false; reason: string; offendingCall?: ToolCall };

// The one place two ToolCalls are compared for "is this the same call" --
// exported so a caller (quality-monitor's tier-2 block) that needs the same
// notion of identity doesn't reimplement it inline and risk diverging from
// this module's own match (step 3 below).
export function sameCall(a: ToolCall, b: ToolCall): boolean {
  return a.name === b.name && JSON.stringify(a.input) === JSON.stringify(b.input);
}

// Tools that mutate state the environment then depends on. If the previous turn
// ran one of these *alongside* a repeated call, re-issuing that call is
// legitimate progress, not a loop — e.g. Edit a source file, then re-run the
// same build command (issue #81). Bash/ShellSession count because a shell
// command can change anything; matching is by lowercased tool name.
const STATE_CHANGING_TOOLS = new Set([
  "edit",
  "write",
  "multiedit",
  "notebookedit",
  "bash",
  "shellsession",
]);

export function assessResponse(
  text: string,
  toolCalls: ToolCall[],
  recentToolCalls: ToolCall[],
  knownTools: Set<string>,
): QualityResult {
  // 1. Empty response with no tool calls
  if (!text.trim() && toolCalls.length === 0) {
    return { ok: false, reason: "empty_response" };
  }

  // 2. Hallucinated tool names (only checked when registry populated)
  for (const tc of toolCalls) {
    if (!tc.name) return { ok: false, reason: "empty_tool_name" };
    if (knownTools.size > 0 && !knownTools.has(tc.name)) {
      return { ok: false, reason: `unknown_tool:${tc.name}` };
    }
  }

  // 3. Repeated tool call loop (exact name+input match with previous turn).
  //    A verbatim repeat is only a loop when nothing could have changed the
  //    outcome. If the previous turn ran a state-changing tool *other than* the
  //    repeated call itself (e.g. an Edit next to a re-run build command), the
  //    environment plausibly changed, so re-issuing the call is progress — not a
  //    loop (issue #81).
  if (toolCalls.length > 0 && recentToolCalls.length > 0) {
    for (const tc of toolCalls) {
      for (const prev of recentToolCalls) {
        if (sameCall(tc, prev)) {
          const envChanged = recentToolCalls.some(
            (r) => !sameCall(r, tc) && STATE_CHANGING_TOOLS.has(r.name.toLowerCase()),
          );
          if (envChanged) continue;
          // Surfaced so a caller acting on the specific offending call
          // (quality-monitor's tier-2 block) never has to re-run this same
          // envChanged-aware match independently and risk diverging from it.
          return { ok: false, reason: "repeated_tool_call", offendingCall: tc };
        }
      }
    }
  }

  // 4. Malformed arguments sentinel from repairJson fallback
  for (const tc of toolCalls) {
    if (tc.input && typeof tc.input === "object" && "_raw" in tc.input) {
      return { ok: false, reason: `malformed_args:${tc.name || "?"}` };
    }
  }

  return { ok: true };
}

export function buildCorrectionMessage(reason: string): string {
  const corrections: Record<string, string> = {
    empty_response:
      "Your previous response was empty. Please respond with either " +
      "text or a tool call to make progress on the task.",
    empty_tool_name:
      "Your tool call had an empty name. Please specify a valid tool name. " +
      "Available tools include: Read, Write, Edit, Bash, Glob, Grep.",
    repeated_tool_call:
      "You just made the exact same tool call as your previous turn. " +
      "This suggests you may be stuck in a loop. Please try a different " +
      "approach or explain what you're trying to accomplish.",
  };

  if (reason.startsWith("unknown_tool:")) {
    const toolName = reason.slice("unknown_tool:".length);
    return (
      `Tool '${toolName}' does not exist. ` +
      "Available tools are: Read, Write, Edit, Bash, Glob, Grep, " +
      "WebFetch, WebSearch. Please use one of these."
    );
  }
  if (reason.startsWith("malformed_args:")) {
    const toolName = reason.slice("malformed_args:".length);
    return (
      `The arguments for tool '${toolName}' were malformed (not valid JSON). ` +
      "Please provide the arguments as a proper JSON object."
    );
  }

  return corrections[reason] ?? `Issue detected: ${reason}. Please try again.`;
}

// Tier 2's model-facing text: sent once per failure streak, after tier 1's
// buildCorrectionMessage has already been steered MAX_CONSECUTIVE_CORRECTIONS
// times and visibly failed. Kept alongside buildCorrectionMessage/phraseForUser
// (this module's other model/harness-facing text) rather than inline in
// quality-monitor/index.ts, per the same convention.

// The exact tool call is now hard-blocked (see quality-monitor's tool_call
// handler) -- this explains that and, deliberately, does NOT say "try
// something else" unconditionally: the most likely real repeat loop is a
// model retrying a call permission-gate already refused (assessResponse's
// envChanged exemption doesn't cover a lone refused call re-issued with
// nothing else in between), and a bare "different action" push would read as
// license to hunt for another route to the same effect -- exactly what
// permission-gate's own refusal text exists to stop (issue #94: refused
// ./build.sh -> bash ./build.sh -> sh ./build.sh -> a successful python3 -c
// subprocess.run(...), since interpreters are themselves whitelisted). This
// carve-out mirrors that refusal's own wording so the two messages agree.
export function buildBlockedCallEscalationMessage(): string {
  return (
    "You are stuck in a loop: a tool call has repeated verbatim with nothing " +
    "else changing in between. That exact call is now blocked -- it will be " +
    "rejected if you try it again. If it was refused by a guardrail (a " +
    "permission or whitelist message), that refusal is the answer -- do not " +
    "re-run the same operation through python3 -c, node -e, env, sh, or an " +
    "-exec flag to reach the same effect another way; tell the user it was " +
    "refused and continue with the rest of the task. Otherwise, explain what " +
    "isn't working, then take a materially different action."
  );
}

// The model-facing reason on the block itself (see the point above: sent
// every time the blocked call is retried, not just once).
export const BLOCKED_CALL_REASON =
  "This exact tool call is blocked -- it's a verbatim repeat of a call " +
  "already flagged as looping, with nothing else changing in between. " +
  "Take a materially different action.";

// Short, user-facing phrasing for the harness-intervention line (distinct from
// buildCorrectionMessage, which is the verbose text sent to the model).
export function phraseForUser(reason: string): string {
  if (reason.startsWith("unknown_tool:")) {
    return `the model called a tool that doesn't exist (${reason.slice("unknown_tool:".length)})`;
  }
  if (reason.startsWith("malformed_args:")) {
    return `the model's tool arguments were malformed (${reason.slice("malformed_args:".length)})`;
  }
  const phrases: Record<string, string> = {
    empty_response: "the model returned an empty response",
    empty_tool_name: "the model emitted a tool call with no name",
    repeated_tool_call: "the model repeated its previous tool call verbatim",
    near_duplicate_loop: "the model is looping with near-identical attempts (varying only details)",
    repeated_failure_signature: "the model's differing attempts keep producing the identical failure",
    repeated_output_signature: "the model's differing attempts keep producing the identical output",
  };
  return phrases[reason] ?? `quality issue (${reason})`;
}

// The two output-of-the-loop detectors' model-facing text (see
// quality-monitor's similarity.ts and failure-signature.ts). Both are steered
// like a tier-1 correction and neither blocks anything: a similarity match
// can never prove the NEXT attempt is wrong.

// Counts are safe to state in these two, unlike the tier-2 escalation's: each
// tracker counts exactly the attempts it is describing, not a mixed-reason
// streak. Neither is a SPAN, though -- the fuzzy count is distinct turns in
// one cluster, the failure count is attempts matching one signature, and
// other turns can sit between them -- so neither opening says "your last N".
export function buildNearDuplicateLoopMessage(count: number, escalated: boolean): string {
  const opening = escalated
    ? `You are still repeating the same action: ${count} of your recent turns are now near-identical variations of each other.`
    : `Across ${count} of your recent turns you have made near-identical variations of the same action -- only small details changed each time, and this has not converged.`;
  return (
    `${opening} Stop varying constants or cosmetic details. State explicitly ` +
    "what hypothesis each attempt was testing and what you learned from it, " +
    "then take a structurally different approach: a different tool, a " +
    "different diagnostic, or a different reading of the problem."
  );
}

// Two outcomes, not one: the watchdog also counts a PASSING result whose
// fuzzy cluster corroborates it, and the failure text would tell a model to
// explain an error it never got. `some of those` because `corroborated` is
// sticky -- one near-identical pair in the streak sets it for the whole
// count. `the input` rather than `the command` because Write and Edit reach
// this too.
//
// The failure branch's carve-out mirrors buildBlockedCallEscalationMessage's
// (issue #94): a model probing a permission-gate refusal with varied
// phrasings is a differing-inputs/identical-output streak and will land
// there, so that text must not read as "find another route".
export function buildFailureSignatureMessage(
  toolName: string,
  count: number,
  opts: { corroborated?: boolean; escalated?: boolean; failed?: boolean } = {},
): string {
  const also = opts.corroborated ? " -- and some of those attempts were themselves near-identical" : "";
  const outcome = opts.failed === false ? "output" : "error or output";
  const opening = opts.escalated
    ? `That is now ${count} ${toolName} attempts with the same outcome${also}.`
    : `${count} of your recent ${toolName} attempts produced essentially the identical ${outcome}, each one after you changed the input${also}.`;
  if (opts.failed === false) {
    return (
      `${opening} These attempts are completing, but the result is not ` +
      "responding to what you change. Before the next attempt, state what " +
      "that output actually shows, what you have ruled out, and which of " +
      "your assumptions about the task might be wrong."
    );
  }
  return (
    `${opening} Changing details is not changing the outcome -- the approach ` +
    "itself is failing. Before the next attempt, state what the error " +
    "actually means, what you have ruled out, and which of your assumptions " +
    "might be wrong. If the repeated message is a guardrail refusal (a " +
    "permission or whitelist message), that refusal is the answer -- report " +
    "it and move on rather than seeking another route to the same effect."
  );
}
