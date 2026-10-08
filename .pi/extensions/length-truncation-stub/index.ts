import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { echoesStubMarker, stubTruncatedMessages } from "./stub.ts";
import { harnessIntervention } from "../_shared/intervention.ts";
import { emitTelemetry } from "../_shared/telemetry.ts";

const jsonBytes = (v: unknown): number => Buffer.byteLength(JSON.stringify(v) ?? "", "utf-8");

// Hook wiring only; the rule lives in stub.ts.
//
// Load order matters: extensions load in directory-name order and context
// hooks chain in that order, so this runs before shell-retention. A stubbed
// pair is then usually too small for shell-retention to archive or demote,
// and the rendered prefix changes once instead of twice.
export default function (pi: ExtensionAPI) {
  if (process.env.LITTLE_CODER_NO_LENGTH_STUB === "1") return;

  // Set once the context hook below has stubbed a message, never cleared:
  // the stubbed message stays in history.
  let stubEmitted = false;

  // A model told to re-issue a cut-off call may copy the stub it sees,
  // marker included, and head + marker + tail is small enough to run. The
  // guard fires only after this registration emitted a stub: a model can
  // copy only a stub it was shown, and before that a call quoting the phrase
  // (a grep, a doc edit) is not an echo. The reason must not quote the
  // marker: copied into a file, it would trip this guard again.
  pi.on("tool_call", async (event, ctx) => {
    if (!stubEmitted) return;
    const input = (event as any).input;
    if (input == null || !echoesStubMarker(input)) return;
    emitTelemetry(pi, "echo_block", { source: "length_stub", tool: String((event as any).toolName ?? "") });
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
    const input: any[] = (event as any).messages || [];
    const { messages, stubbedCount } = stubTruncatedMessages(input);
    if (stubbedCount > 0) {
      stubEmitted = true;
      // stubTruncatedMessages returns every untouched message as the same
      // object, so identity picks out exactly the stubbed ones.
      let bytesBefore = 0;
      let bytesAfter = 0;
      messages.forEach((m, i) => {
        if (m !== input[i]) {
          bytesBefore += jsonBytes(input[i]);
          bytesAfter += jsonBytes(m);
        }
      });
      // pi hands this hook pristine stored history on every request, so
      // `stubbed` is how many stubbed messages this projection carries, not
      // how many are new; benchmarks/turn_ledger.py diffs the snapshots.
      emitTelemetry(pi, "length_stub", { stubbed: stubbedCount, bytesBefore, bytesAfter });
      return { messages };
    }
  });
}
