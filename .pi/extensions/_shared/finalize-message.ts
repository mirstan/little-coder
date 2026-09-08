// Shared message text for the "you're running low, finalize now" nudge.
// finalize-warn's turn-count trigger and its wall-clock trigger both funnel
// through this so the two triggers can never present different instructions
// to the model.
//
// Benchmark-specific phrasing is keyed explicitly on the benchmark it's
// written for -- GAIA is scored by a chat-reply `Answer: <value>` line plus
// an EvidenceList convention; Terminal-Bench 2.0 is scored by inspecting the
// container's files/state afterward, not a chat reply. Anything else
// (undefined -- plain interactive pi use -- or a future third benchmark)
// gets a generic, convention-free fallback rather than defaulting to either
// benchmark's specific instructions.
export function resolveFinalizeMessage(benchmark: string | undefined): string {
  if (benchmark === "terminal_bench") {
    return (
      "You have very little time left — stop investigating or verifying " +
      "and save what you have right now. Re-check the task instructions " +
      "above for the exact file path(s), format, or state they specify, " +
      "then use the ShellSession tool to write your current best-effort " +
      "result there immediately (the task is graded by inspecting the " +
      "container's files/state afterward, not by this chat reply). Do not " +
      "run further checks, tests, or verification first — if your answer " +
      "is incomplete or uncertain, save it anyway: a rough answer on disk " +
      "beats a correct one that never got written. Then stop calling tools."
    );
  }
  if (benchmark === "gaia") {
    return (
      "Wrap up now. Produce your final reply, ending with a final line of " +
      "exactly: Answer: <value> (plain text, no code formatting or quotes " +
      "around the value). Do not start new tool chains; if you need a fact " +
      "you don't have, answer with your best supported guess from " +
      "EvidenceList rather than leaving it blank."
    );
  }
  return (
    "Wrap up now. Produce your final response and stop calling tools " +
    "rather than starting new tool chains or open-ended investigation."
  );
}
