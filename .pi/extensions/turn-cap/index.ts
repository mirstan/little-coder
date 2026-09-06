import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { harnessIntervention } from "../_shared/intervention.ts";
import { resolveTurnCap } from "../_shared/turn-cap.ts";

// Port of agent.py's max_turns early-break. Counts turn_start events per
// agent_start span; when the count exceeds LITTLE_CODER_MAX_TURNS (or the
// per-benchmark override injected via systemPromptOptions), calls ctx.abort()
// to halt the loop. Resets on agent_start.

let turnsThisRun = 0;
let capForRun = 0;

export default function (pi: ExtensionAPI) {
  pi.on("before_agent_start", async (event) => {
    turnsThisRun = 0;
    capForRun = resolveTurnCap(event);
  });

  pi.on("turn_start", async (_event, ctx) => {
    if (capForRun <= 0) return;
    turnsThisRun++;
    if (turnsThisRun > capForRun) {
      harnessIntervention(
        ctx,
        `the model hit the turn limit (${capForRun}) — stopping the run.`,
      );
      ctx.abort();
    }
  });
}
