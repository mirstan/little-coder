import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { stubTruncatedMessages } from "./stub.ts";

// Hook wiring only; the rule lives in stub.ts.
//
// Load order matters: extensions load in directory-name order and context
// hooks chain in that order, so this runs before shell-retention. A stubbed
// pair is then usually too small for shell-retention to archive or demote,
// and the rendered prefix changes once instead of twice.
export default function (pi: ExtensionAPI) {
  if (process.env.LITTLE_CODER_NO_LENGTH_STUB === "1") return;

  pi.on("context", async (event) => {
    const { messages, stubbedCount } = stubTruncatedMessages((event as any).messages || []);
    if (stubbedCount > 0) return { messages };
  });
}
