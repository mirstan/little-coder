// Shared, stateless resolution of "what absolute wall-clock deadline (epoch
// ms) is this agent run allowed to run until, before a deadline-aware
// extension should intervene." Mirrors turn-cap.ts's precedence shape
// (Plan 5): an explicitly-SET LITTLE_CODER_DEADLINE_EPOCH_MS env var —
// including "0" — is AUTHORITATIVE over an event-carried override
// (systemPromptOptions.littleCoder.deadlineEpochMs, available for a future
// benchmark-profiles-style override, though nothing sets it yet); the event
// override applies only when the env var is absent. "No deadline" is 0
// (disabled), same convention as turn-cap's capForRun <= 0.
//
// The env var is set by little_coder_agent.py (harbor adapter) at PiRpc
// construction time, computed from the same wall-clock budget it passes to
// rpc.prompt_and_collect()'s timeout. That's deliberate: it's the one
// deadline value in the whole stack that is real, known, and entirely under
// little-coder's own control -- Harbor's own outer trial timeout is not
// observable from inside a custom agent. The adapter is likewise the only
// writer of either source today, so inverting the precedence to match
// turn-cap.ts is consistency-only and changes no observed behavior yet.
export function resolveDeadlineEpochMs(event: unknown): number {
  const raw = process.env.LITTLE_CODER_DEADLINE_EPOCH_MS;
  if (raw !== undefined) {
    // Number(), not parseInt(): parseInt accepts a malformed value with a
    // numeric prefix ("1700000000000garbage" -> 1700000000000) instead of
    // rejecting it outright, silently changing the warning deadline.
    const n = Number(raw);
    return Number.isInteger(n) && n > 0 ? n : 0;
  }
  const opts: any = (event as any)?.systemPromptOptions ?? {};
  const evDeadline = Number(opts?.littleCoder?.deadlineEpochMs);
  return Number.isInteger(evDeadline) && evDeadline > 0 ? evDeadline : 0;
}
