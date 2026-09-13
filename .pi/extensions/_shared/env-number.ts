// Shared numeric env-var resolution for tuning knobs — fractions, durations,
// counts — where any finite value the operator writes is legitimate, including
// 0 and negatives.
//
// The convention matches context-watchdog's thresholdPercent(): absent,
// empty/whitespace-only, or non-numeric falls back to the caller's default;
// anything finite is returned as-is. Deliberately NOT deadline.ts's /
// turn-cap.ts's stricter `Number.isInteger(n) && n > 0` shape — those resolve
// an epoch-ms instant and a turn count, where a fraction is meaningless and
// a non-positive value already has a dedicated "disabled" meaning. A knob
// like a 0.25 guard fraction would be rejected outright by that convention.
//
// "Disabled" is left to the call site: this returns 0 for "0" rather than
// swallowing it into the fallback, so callers can test `<= 0` themselves and
// mean it. An empty-but-exported var (`export VAR=` in a wrapper, a CI matrix
// exporting unset variables) is treated as unset, not as a deliberate 0 —
// Number("") is 0, which would otherwise silently disable a knob nobody
// touched.
export function envNumber(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw.trim() === "") return fallback;
  const n = Number(raw);
  return Number.isFinite(n) ? n : fallback;
}
