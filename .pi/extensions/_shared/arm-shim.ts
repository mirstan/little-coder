// Env-gated loader for a vendored third-party compaction extension (compaction
// A/B experiment; vendor/compaction-ab/PROVENANCE.md). Off unless
// LC_COMPACTION_ARM names the arm. The vendored module is imported only inside
// the gate, so other arms never evaluate third-party code. Registration is
// recorded from session_start, because pi.appendEntry throws while extensions
// are still loading (pi loader.js createExtensionRuntime).
//
// A library, like telemetry.ts: no index.ts here.

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { emitTelemetry } from "./telemetry.ts";

export const ENV_ARM = "LC_COMPACTION_ARM";

export async function loadArm(
  pi: ExtensionAPI,
  arm: string,
  load: () => Promise<{ default: (pi: ExtensionAPI) => unknown }>,
): Promise<boolean> {
  if (process.env[ENV_ARM] !== arm) return false;
  const mod = await load();
  await mod.default(pi);
  pi.on("session_start", async () => {
    emitTelemetry(pi, "arm_registered", { arm });
    return undefined;
  });
  return true;
}
