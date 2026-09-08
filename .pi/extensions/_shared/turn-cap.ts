// Shared, stateless resolution of "how many turns is this agent run allowed
// before a turn-cap-style extension should intervene." turn-cap, finalize-warn,
// and gaia-finalize-guard each need this exact precedence — the per-run
// systemPromptOptions.littleCoder.maxTurns override (set by benchmark_overrides
// in .pi/settings.json via benchmark-profiles) wins over the env var, which
// wins over "no cap" (0). Each extension keeps its OWN turnsThisRun counter
// (that's per-extension mutable state, not shareable), but the cap VALUE
// itself is a pure function of the event/environment, so only that part
// lives here.
export function resolveTurnCap(event: unknown): number {
  const opts: any = (event as any)?.systemPromptOptions ?? {};
  const lcCap = Number(opts?.littleCoder?.maxTurns);
  if (Number.isInteger(lcCap) && lcCap > 0) return lcCap;
  const raw = process.env.LITTLE_CODER_MAX_TURNS;
  if (!raw) return 0;
  // Number(), not parseInt(): parseInt truncates at the first non-numeric
  // character ("40abc" -> 40, "3.7" -> 3) instead of rejecting the whole
  // malformed value, and a fractional cap would make the turn-count and
  // wall-clock triggers compare against inconsistent integer boundaries.
  const n = Number(raw);
  return Number.isInteger(n) && n > 0 ? n : 0;
}
