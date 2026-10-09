// Offline-only payload capture for the compaction A/B multi-depth replay.
// Opt-in: inert unless LC_AB_CAPTURE_PAYLOAD_DIR is set. Writes each provider
// request body, byte for byte as the extension chain produced it, to
// <dir>/request-<n>.json. Never returns a value, so it changes no payload. Sorts
// after every payload rewriter (zzz-), like zzz-ab-observer.
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export const ENV_CAPTURE_DIR = "LC_AB_CAPTURE_PAYLOAD_DIR";

export default function (pi: ExtensionAPI) {
  const dir = process.env[ENV_CAPTURE_DIR];
  if (!dir) return;
  let n = 0;
  pi.on("before_provider_request", async (event: any) => {
    try {
      n += 1;
      writeFileSync(join(dir, `request-${n}.json`), JSON.stringify(event?.payload ?? null));
    } catch {
      // capture must never break a request
    }
    return undefined;
  });
}
