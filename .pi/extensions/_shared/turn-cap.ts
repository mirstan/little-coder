// Shared, stateless resolution of "how many turns is this agent run allowed
// before a turn-cap-style extension should intervene." turn-cap, finalize-warn,
// and gaia-finalize-guard each need this exact precedence. Each extension
// keeps its OWN turnsThisRun counter (that's per-extension mutable state,
// not shareable), but the cap VALUE itself is a pure function of the
// event/environment, so only that part lives here.
//
// Precedence (Plan 5 / Codex finding [high]): an explicitly-SET
// LITTLE_CODER_MAX_TURNS env var — including "0" — is now AUTHORITATIVE
// over the per-run systemPromptOptions.littleCoder.maxTurns override (set by
// benchmark_overrides in .pi/settings.json via benchmark-profiles); the
// profile value applies only when the env var is absent. This is a
// deliberate inversion of the prior precedence ("override wins over env"):
// a harness that constructs PiRpc with an explicit max_turns kwarg is the
// deliberate per-run authority and must not be silently overridden by a
// benchmark profile it doesn't control. Concretely, this closes the gap
// where little_coder_agent.py's max_turns=0 (no cap, wall-clock governs
// instead) was still being capped at 40 for any model whose profile
// publishes benchmark_overrides.terminal_bench.max_turns — including
// llamacpp/qwen3.6-35b-a3b, the pilot's own advertised default model.
// Interactive use (no env var set) is unaffected: the profile value still
// applies via the absent-env branch below.
//
// "Set" means set to a non-empty, non-whitespace-only string. An
// empty-but-exported var (`export LITTLE_CODER_MAX_TURNS=` in a wrapper
// script, a CI matrix that exports unset variables, `env VAR= cmd`) leaves
// `process.env.LITTLE_CODER_MAX_TURNS` defined as `""`, which is !==
// undefined but is not a deliberate "0" from anyone -- Number("") is 0, so
// treating it as set would silently produce "no cap" and invert exactly the
// precedence this fix exists to enforce, reached through an ambient-leak
// path instead of the profile-override path it was meant to close. Such a
// value is therefore treated as unset and falls through to the profile
// override below, same as a literal absent env var. A literal "0" (or any
// other in-range value) is unaffected and stays authoritative.
export function resolveTurnCap(event: unknown): number {
  const raw = process.env.LITTLE_CODER_MAX_TURNS;
  if (raw !== undefined && raw.trim() !== "") {
    // Number(), not parseInt(): parseInt truncates at the first non-numeric
    // character ("40abc" -> 40, "3.7" -> 3) instead of rejecting the whole
    // malformed value, and a fractional cap would make the turn-count and
    // wall-clock triggers compare against inconsistent integer boundaries.
    const n = Number(raw);
    return Number.isInteger(n) && n > 0 ? n : 0;
  }
  const opts: any = (event as any)?.systemPromptOptions ?? {};
  const lcCap = Number(opts?.littleCoder?.maxTurns);
  return Number.isInteger(lcCap) && lcCap > 0 ? lcCap : 0;
}
