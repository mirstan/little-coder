import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { echoesStubMarker, stubTruncatedMessages } from "./stub.ts";
import { harnessIntervention } from "../_shared/intervention.ts";

// Hook wiring only; the rule lives in stub.ts.
//
// Load order matters: extensions load in directory-name order and context
// hooks chain in that order, so this runs before shell-retention. A stubbed
// pair is then usually too small for shell-retention to archive or demote,
// and the rendered prefix changes once instead of twice.
export default function (pi: ExtensionAPI) {
  if (process.env.LITTLE_CODER_NO_LENGTH_STUB === "1") return;

  // A model told to re-issue a cut-off call may copy the stub it sees,
  // marker included, and head + marker + tail is small enough to run. The
  // reason must not quote the marker: copied into a file, it would trip
  // this guard again.
  pi.on("tool_call", async (event, ctx) => {
    const input = (event as any).input;
    if (input == null || !echoesStubMarker(input)) return;
    harnessIntervention(ctx, "blocked a tool call echoing a length-truncation stub.");
    return {
      block: true,
      reason:
        `This call contains the placeholder left in your earlier tool call that was cut off ` +
        `at the output token limit, not real content. Regenerate the content yourself ` +
        `instead of copying it, and split long content across several smaller calls so ` +
        `each fits the output limit.`,
    };
  });

  pi.on("context", async (event) => {
    const { messages, stubbedCount } = stubTruncatedMessages((event as any).messages || []);
    if (stubbedCount > 0) return { messages };
  });
}
