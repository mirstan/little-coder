import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { annotate } from "./truncation.ts";

// Hook wiring for the partial-view note (see truncation.ts for the rules).
//
// Why `tool_result` rather than the ShellSession formatters: the note needs
// only the command and the returned text, which this event carries together
// for every tool, and its return value replaces what the model sees. That
// covers GAIA's built-in `bash` and all three ShellSession backends from one
// place, where a formatter change would have to be written three times and
// still miss GAIA.
//
// No `harnessIntervention` line: the note rides in the tool result the user
// already sees, and one UI line per qualifying command would flood it.

// Not _shared/shell-write.ts's SHELL_TOOLS: that set gates writes and includes
// ShellStart, whose result is a job-started acknowledgment, not command output.
const ANNOTATED_TOOLS = new Set(["bash", "Bash", "ShellSession"]);

type TextOrImage = { type: string; text?: string };

export default function (pi: ExtensionAPI) {
  pi.on("tool_result", async (event) => {
    const e = event as any;
    if (e.isError) return;
    if (!ANNOTATED_TOOLS.has(String(e.toolName ?? ""))) return;

    const content = (e.content ?? []) as TextOrImage[];
    if (content.length === 0) return;
    // An image block has no lines to count and nothing to append to.
    if (content.some((c) => c.type !== "text")) return;

    const annotated = annotate(
      String(e.input?.command ?? ""),
      content.map((c) => c.text ?? "").join(""),
    );
    if (annotated === null) return;
    return { content: [{ type: "text" as const, text: annotated }] };
  });
}
