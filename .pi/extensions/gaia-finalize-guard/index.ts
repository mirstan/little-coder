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

// A close mirror of gaia_scorer.py's extract_final_answer() regex, applied
// per physical line like the Python version (splitlines() + line.strip()) —
// NOT the naive line-blob version this extension shipped with initially,
// which used a bare `\s*` between the colon and the value. That let the
// value cross a newline (e.g. "Answer:\n42"), which the Python scorer never
// allows (it matches within one already-newline-free line), so this guard
// could wrongly treat a multi-line trailer as "answered" when the scorer
// would actually reject it and fall through to its own last-non-empty-line
// fallback. Fixed here:
//   - `^\s*` tolerates leading indentation, mirroring the scorer's
//     line.strip() before matching.
//   - `[ \t]*` (not `\s*`) between the colon and the value can't cross a
//     newline, matching the scorer's same-physical-line requirement.
//   - `(\S.*)$` requires a non-whitespace value, catching a bare "Answer:"
//     with nothing meaningful after it — worth nudging on even though the
//     scorer itself doesn't special-case it (it would match, .strip() the
//     captured value down to "", and return that empty string rather than
//     falling back — deliberately stricter here, not a scorer bug we need
//     to reproduce exactly, since a bare "Answer:" is exactly the kind of
//     unfinished reply this guard exists to catch).
const ANSWER_LINE_RE = /^\s*(?:final\s+answer|answer)\s*[:\-][ \t]*(\S.*)$/im;

let nudgedThisRun = false;
// Mirrors turn-cap's own turn counting (turnsThisRun/capForRun), duplicated
// rather than imported since turn-cap keeps this state module-private. A
// steer queued on turn N becomes the prompt for turn N+1 — but if N+1 would
// exceed turn-cap's limit, turn-cap aborts at that turn's turn_start before
// the model ever sees the nudge (turn-cap fires first: ctx.abort() there
// happens synchronously in the same before-generation window, so the
// steered message never reaches the model). Nudging in that situation is
// pure waste, so skip it once we're already at the cap.
let turnsThisRun = 0;
let capForRun = 0;

function envTurnCap(): number {
  const raw = process.env.LITTLE_CODER_MAX_TURNS;
  if (!raw) return 0;
  const n = parseInt(raw, 10);
  return Number.isFinite(n) && n > 0 ? n : 0;
}

export default function (pi: ExtensionAPI) {
  pi.on("session_start", async () => {
    nudgedThisRun = false;
  });

  pi.on("before_agent_start", async (event) => {
    turnsThisRun = 0;
    const opts: any = (event as any).systemPromptOptions ?? {};
    const lcCap = Number(opts?.littleCoder?.maxTurns);
    capForRun = Number.isFinite(lcCap) && lcCap > 0 ? lcCap : envTurnCap();
  });

  pi.on("turn_start", async () => {
    turnsThisRun++;
  });

  pi.on("turn_end", async (event, ctx) => {
    if (process.env.LITTLE_CODER_BENCHMARK !== "gaia") return;
    if (nudgedThisRun) return;
    // A nudge queued now would become the prompt for turnsThisRun + 1;
    // if that exceeds the cap, turn-cap aborts before the model sees it.
    if (capForRun > 0 && turnsThisRun >= capForRun) return;

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
    // A genuinely empty response (no text, no tool calls) is quality-monitor's
    // territory (its own "empty_response" steer) — stay out of its way
    // rather than queue a second, possibly conflicting correction for the
    // same turn.
    if (!text.trim()) return;
    // Check only the LAST non-blank physical line, not "does any line
    // anywhere match" — GAIA's own prompt tells the model not to append
    // prose after its Answer: line, so a well-formed reply has it as the
    // final line. Checking the whole blob let an earlier line that merely
    // *looks* like the convention (a mid-reasoning draft, e.g.
    // "Answer: <put final value here>") suppress the nudge on a turn that
    // still trails off afterward with real unresolved prose — the exact
    // unfinished-turn case this guard exists to catch.
    const lines = text.split("\n");
    let lastNonBlank = "";
    for (let i = lines.length - 1; i >= 0; i--) {
      if (lines[i].trim()) {
        lastNonBlank = lines[i];
        break;
      }
    }
    if (ANSWER_LINE_RE.test(lastNonBlank)) return; // followed the convention — nothing to do

    try {
      pi.sendUserMessage(
        "Your last reply didn't call a tool or end with `Answer: <value>`. " +
          "If you're still working, call a tool to continue. If you have your answer, " +
          "restate it ending with that exact line.",
        { deliverAs: "steer" },
      );
    } catch {
      // SDK without sendUserMessage — leave nudgedThisRun false so a later
      // turn_end can still try; if the SDK genuinely lacks this call for
      // the whole session, later attempts will fail identically and cost
      // nothing extra.
      return;
    }
    // Only claim the one-shot slot and surface the notification once the
    // send has actually succeeded — otherwise a failed send both burns the
    // only nudge this run gets and shows an intervention message for
    // something that was never delivered.
    nudgedThisRun = true;
    harnessIntervention(
      ctx,
      "turn ended without a tool call or an Answer: line — asking the model to continue or finalize.",
    );
  });
}
