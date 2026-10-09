// Compaction A/B arm A3: pi-prefix-cache-compaction 0.3.0, warm-up off
// (.pi/pi-prefix-cache-compaction.json). Active only with
// LC_COMPACTION_ARM=prefix-cache.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { loadArm } from "../_shared/arm-shim.ts";

export default async function (pi: ExtensionAPI) {
  await loadArm(pi, "prefix-cache", () => import("../../../vendor/compaction-ab/pi-prefix-cache-compaction@0.3.0/src/index.ts"));
}
