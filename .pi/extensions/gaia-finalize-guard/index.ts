import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { harnessIntervention } from "../_shared/intervention.ts";

// GAIA's prompt (benchmarks/gaia.py:_build_prompt) instructs the model to end
// its final reply with a single line: `Answer: <value>`. gaia_scorer.py's
// extract_final_answer() looks for that line first and falls back to the
// last non-empty line of the transcript if it's absent. Nothing currently
// checks whether the model actually followed the convention before pi's own
// agent loop ends the run (a turn with zero tool calls triggers agent_end)
// — so a model that trails off mid-plan ("Let me fetch it to get the zip
// codes.") gets scored on that fragment as its "answer."
//
// This mirrors finalize-warn's mechanism (inject a follow-up reminding the
// model to emit the Answer: line) but triggers on the actual signal that
// matters — a turn that ended with no tool call AND no Answer: line —
// rather than proximity to a turn cap. For GAIA on omlx/rapidmlx,
// finalize-warn never engages at all (no benchmark_overrides.gaia.max_turns
// configured, so capForRun is 0), and even if it did, every observed case
// trailed off far short of any plausible turn limit (3-7 turns in).
//
// Scoped to GAIA (LITTLE_CODER_BENCHMARK=gaia): aider_polyglot has no
// Answer: convention, and a coding turn legitimately ends without a tool
// call once the implementation is already written and tested in an earlier
// turn — this guard would misfire there.
//
// Fires at most once per task (reset on session_start, matching
// thinking-budget/finalize-warn's own convention) — a single nudge, not an
// open-ended retry loop. If the model still doesn't finalize, the run ends
// on whatever it says next, same as before this extension existed.

// Mirrors gaia_scorer.py's extract_final_answer() regex exactly — kept in
// lockstep by hand since there's no shared source of truth between the
// Python scorer and this TS extension.
const ANSWER_LINE_RE = /^(?:final\s+answer|answer)\s*[:\-]\s*(.+)$/im;

let nudgedThisRun = false;

export default function (pi: ExtensionAPI) {
  pi.on("session_start", async () => {
    nudgedThisRun = false;
  });

  pi.on("turn_end", async (event, ctx) => {
    if (process.env.LITTLE_CODER_BENCHMARK !== "gaia") return;
    if (nudgedThisRun) return;

    const message: any = (event as any).message;
    if (!message) return;
    // Aborted (thinking-budget/turn-cap abort) or errored turns are
    // legitimately partial — not the "model trailed off" case this guards.
    if (message.stopReason === "aborted" || message.stopReason === "error") return;

    const content = Array.isArray(message.content) ? message.content : [];
    const toolCallCount = content.filter((c: any) => c?.type === "toolCall").length;
    if (toolCallCount > 0) return; // more work queued — not the terminal turn

    const text = content
      .filter((c: any) => c?.type === "text")
      .map((c: any) => c.text ?? "")
      .join("\n");
    if (ANSWER_LINE_RE.test(text)) return; // followed the convention — nothing to do

    nudgedThisRun = true;
    harnessIntervention(
      ctx,
      "turn ended without a tool call or an Answer: line — asking the model to continue or finalize.",
    );
    try {
      pi.sendUserMessage(
        "Your last reply didn't call a tool or end with `Answer: <value>`. " +
          "If you're still working, call a tool to continue. If you have your answer, " +
          "restate it ending with that exact line.",
        { deliverAs: "steer" },
      );
    } catch {
      // SDK without sendUserMessage — let agent_end stand rather than break the run.
    }
  });
}
