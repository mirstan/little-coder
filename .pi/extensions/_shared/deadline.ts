// Shared, stateless resolution of "what absolute wall-clock deadline (epoch
// ms) is this agent run allowed to run until, before a deadline-aware
// extension should intervene." Mirrors turn-cap.ts's precedence shape: an
// event-carried override (systemPromptOptions.littleCoder.deadlineEpochMs,
// available for a future benchmark-profiles-style override, though nothing
// sets it yet) wins over the env var, which wins over "no deadline" (0 =
// disabled, same convention as turn-cap's capForRun <= 0).
//
// The env var is set by little_coder_agent.py (harbor adapter) at PiRpc
// construction time, computed from the same wall-clock budget it passes to
// rpc.prompt_and_collect()'s timeout. That's deliberate: it's the one
// deadline value in the whole stack that is real, known, and entirely under
// little-coder's own control -- Harbor's own outer trial timeout is not
// observable from inside a custom agent.
export function resolveDeadlineEpochMs(event: unknown): number {
  const opts: any = (event as any)?.systemPromptOptions ?? {};
  const evDeadline = Number(opts?.littleCoder?.deadlineEpochMs);
  if (Number.isFinite(evDeadline) && evDeadline > 0) return evDeadline;
  const raw = process.env.LITTLE_CODER_DEADLINE_EPOCH_MS;
  if (!raw) return 0;
  const n = parseInt(raw, 10);
  return Number.isFinite(n) && n > 0 ? n : 0;
}
