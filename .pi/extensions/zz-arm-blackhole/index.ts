// Compaction A/B arm A2: pi-blackhole 0.5.12, deterministic (no model call).
// Active only with LC_COMPACTION_ARM=blackhole; config pinned in
// vendor/compaction-ab/config/pi-blackhole-config.json (copied by the driver).
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { loadArm } from "../_shared/arm-shim.ts";

export default async function (pi: ExtensionAPI) {
  await loadArm(pi, "blackhole", () => import("../../../vendor/compaction-ab/pi-blackhole@0.5.12/dist/index.js"));
}
