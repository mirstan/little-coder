// Pure pieces of the cache-reuse compaction: the replay body, the anchors that
// tell the model where pi cut the history, the instruction, and the parse of
// the model's answer. No I/O here; index.ts wires it to pi and the network.

import {
  PI_SUMMARIZATION_PROMPT,
  PI_TURN_PREFIX_SUMMARIZATION_PROMPT,
  PI_UPDATE_SUMMARIZATION_PROMPT,
} from "./pi-compat.ts";

export type Payload = Record<string, unknown> & { messages: unknown[] };

export interface ReplayBudget {
  maxTokens: number;
  /** omlx's request-level thinking cap; 0 leaves the field out. */
  thinkingBudget: number;
}

/**
 * The captured payload with one user message appended and only the output
 * budget changed. Everything that reaches the chat template (system prompt,
 * tools, chat_template_kwargs, every earlier message) is left byte-identical,
 * and nothing that does reach it is added: omlx drops `tools` from the
 * template on tool_choice "none" and merges `reasoning_effort` into the
 * template kwargs, and either rewrites the first system block.
 */
export function buildReplayBody(captured: Payload, instruction: string, budget: ReplayBudget): Payload {
  const body = structuredClone(captured) as Payload;
  body.messages = [...body.messages, { role: "user", content: instruction }];
  if ("max_completion_tokens" in captured && !("max_tokens" in captured)) {
    body.max_completion_tokens = budget.maxTokens;
  } else {
    body.max_tokens = budget.maxTokens;
  }
  if (budget.thinkingBudget > 0) body.thinking_budget = budget.thinkingBudget;
  body.stream = true;
  body.stream_options = { ...((body.stream_options as object) ?? {}), include_usage: true };
  return body;
}

/**
 * Text the template must not see as a control sequence. The Qwen template
 * scans every user message for `<|think_*|>` and moves the thinking state
 * that renders at the top of the system block, and `<|im_end|>`-style text
 * would tokenize as a special token.
 */
export function sanitizeForTemplate(text: string): string {
  return text.replace(/<\|/g, "< |").replace(/\|>/g, "| >");
}

// ── Anchors ────────────────────────────────────────────────────────────────

/** A pi-ai Message (the shape convertToLlm returns). */
type LlmMessage = { role: string; content: unknown };

export interface AnchorSpec {
  /** The anchor message, already converted with convertToLlm. */
  message: LlmMessage;
  /** LLM messages that precede the anchor in what pi summarizes; same-text copies among them are skipped. */
  before: LlmMessage[];
}

export type AnchorResult = { index: number; excerpt: string } | { error: "anchor_missing" };

const SNIPPET_CHARS = 160;
const EXCERPT_CHARS = 120;

function normalize(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function partsText(content: unknown, types: string[]): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((b: any) => b && types.includes(b.type) && typeof (b.text ?? b.thinking) === "string")
    .map((b: any) => b.text ?? b.thinking)
    .join("\n");
}

function toolCallIds(m: LlmMessage): string[] {
  if (m.role !== "assistant" || !Array.isArray(m.content)) return [];
  return m.content.filter((b: any) => b?.type === "toolCall" && typeof b.id === "string").map((b: any) => b.id);
}

function payloadText(m: any): string {
  if (m.role === "assistant") {
    const text = partsText(m.content, ["text"]);
    return text || (typeof m.reasoning_content === "string" ? m.reasoning_content : "");
  }
  return partsText(m.content, ["text"]);
}

function messageText(m: LlmMessage): string {
  if (m.role === "assistant") return partsText(m.content, ["text"]) || partsText(m.content, ["thinking"]);
  return partsText(m.content, ["text"]);
}

function payloadRole(role: string): string {
  return role === "toolResult" ? "tool" : role;
}

function excerptOf(m: LlmMessage): string {
  const ids = toolCallIds(m);
  if (ids.length > 0) {
    const call: any = (m.content as any[]).find((b) => b?.type === "toolCall");
    const args = JSON.stringify(call.arguments ?? {});
    return sanitizeForTemplate(`your tool call ${call.name}(${args.slice(0, EXCERPT_CHARS)}${args.length > EXCERPT_CHARS ? "…" : ""})`);
  }
  const text = normalize(messageText(m));
  const who = m.role === "assistant" ? "your reply" : "the message";
  return sanitizeForTemplate(`${who} that begins "${text.slice(0, EXCERPT_CHARS)}${text.length > EXCERPT_CHARS ? "…" : ""}"`);
}

/**
 * Find the anchor message in the captured payload. An assistant turn with
 * tool calls matches on its call ids, anything else on its opening text; when
 * the same text occurs earlier in what pi summarizes, those occurrences are
 * skipped so the match is the anchor's own position. Not finding it means the
 * payload does not hold the history pi is about to summarize.
 */
export function locateAnchor(payloadMessages: unknown[], spec: AnchorSpec): AnchorResult {
  const ids = toolCallIds(spec.message);
  let matches: (m: any) => boolean;
  let skip = 0;
  if (ids.length > 0) {
    const want = [...ids].sort().join("\u0000");
    matches = (m) =>
      m?.role === "assistant" &&
      Array.isArray(m.tool_calls) &&
      m.tool_calls.map((t: any) => t?.id).sort().join("\u0000") === want;
  } else {
    const snippet = normalize(messageText(spec.message)).slice(0, SNIPPET_CHARS);
    if (!snippet) return { error: "anchor_missing" };
    const role = payloadRole(spec.message.role);
    matches = (m) => m?.role === role && normalize(payloadText(m)).includes(snippet);
    skip = spec.before.filter(
      (b) => payloadRole(b.role) === role && normalize(messageText(b)).includes(snippet),
    ).length;
  }
  let seen = 0;
  for (let i = 0; i < payloadMessages.length; i++) {
    if (!matches(payloadMessages[i])) continue;
    if (seen === skip) return { index: i, excerpt: excerptOf(spec.message) };
    seen++;
  }
  return { error: "anchor_missing" };
}

// ── Instruction ────────────────────────────────────────────────────────────

export interface InstructionSpec {
  /** The first message pi keeps verbatim. */
  cut: { index: number; excerpt: string };
  /** The message that starts the turn pi splits, on a split turn. */
  turnStart: { index: number; excerpt: string } | null;
  wantHistory: boolean;
  wantPrefix: boolean;
  /** A previous compaction summary opens the conversation. */
  hasPrevious: boolean;
  customInstructions?: string;
}

// pi's prompts open with "The messages above are …"; the scope paragraph
// below says which of the messages above, so that opening line is replaced.
function promptBody(prompt: string): string {
  return prompt.slice(prompt.indexOf("\n") + 1).trimStart();
}

/**
 * The one user message appended to the replayed conversation. pi's own
 * prompt text and section format, scoped to the span pi summarizes.
 */
export function buildInstruction(spec: InstructionSpec): string {
  const out: string[] = [];
  out.push(
    "STOP working on the task. This turn is a context checkpoint, not a continuation: do NOT call any tools, " +
      "do NOT continue the conversation, do NOT answer questions from it. Read the conversation above and output " +
      "ONLY the summary requested below. Keep your thinking brief.",
  );
  out.push(
    `Scope: ${spec.cut.excerpt} and everything after it is kept verbatim and must NOT be summarized. ` +
      "Summarize only what comes before it.",
  );
  if (spec.wantHistory) {
    const span = spec.turnStart
      ? `from the start of the conversation up to, but not including, ${spec.turnStart.excerpt}`
      : `from the start of the conversation up to, but not including, ${spec.cut.excerpt}`;
    let section: string;
    if (spec.hasPrevious) {
      section =
        `## Part 1: history summary\nSummarize the messages ${span}. The conversation opens with the summary of an ` +
        "earlier compaction (the message beginning \"The conversation history before this point was compacted\"); " +
        "treat it as the existing summary and the messages after it as the NEW messages.\n\n" +
        promptBody(PI_UPDATE_SUMMARIZATION_PROMPT);
    } else {
      section = `## Part 1: history summary\nSummarize the messages ${span}.\n\n${promptBody(PI_SUMMARIZATION_PROMPT)}`;
    }
    if (spec.customInstructions) section += `\n\nAdditional focus: ${spec.customInstructions}`;
    out.push(section);
  }
  if (spec.wantPrefix && spec.turnStart) {
    out.push(
      `## Part ${spec.wantHistory ? 2 : 1}: turn prefix summary\nThe turn that begins with ${spec.turnStart.excerpt} ` +
        `is being split: its messages from that one up to, but not including, ${spec.cut.excerpt} are the PREFIX ` +
        "and are summarized here; the rest of the turn is kept.\n\n" +
        PI_TURN_PREFIX_SUMMARIZATION_PROMPT,
    );
  }
  const tags: string[] = [];
  if (spec.wantHistory) tags.push("<history-summary>\n…Part 1…\n</history-summary>");
  if (spec.wantPrefix) tags.push(`<turn-prefix-summary>\n…Part ${spec.wantHistory ? 2 : 1}…\n</turn-prefix-summary>`);
  out.push(`Output format: exactly the following, and nothing else:\n\n${tags.join("\n\n")}`);
  return sanitizeForTemplate(out.join("\n\n"));
}

// ── Output ─────────────────────────────────────────────────────────────────

export type ParsedSummary = { history?: string; prefix?: string } | { error: "garbage" };

const MIN_SECTION_CHARS = 40;

function tagged(text: string, tag: string): string | null {
  const m = new RegExp(`<${tag}>([\\s\\S]*?)</${tag}>`).exec(text);
  return m ? m[1].trim() : null;
}

function validHistory(s: string | null): s is string {
  return !!s && s.length >= MIN_SECTION_CHARS && s.includes("## Goal") && s.includes("## Next Steps");
}

function validPrefix(s: string | null): s is string {
  return !!s && s.length >= MIN_SECTION_CHARS && s.includes("## Original Request");
}

/** The model's sections, or `garbage` when one asked for is missing, cut off or unstructured. */
export function parseSummaryOutput(content: string, want: { history: boolean; prefix: boolean }): ParsedSummary {
  const out: { history?: string; prefix?: string } = {};
  if (want.history) {
    let h = tagged(content, "history-summary");
    if (h === null && !want.prefix && !content.includes("<history-summary>")) h = content.trim();
    if (!validHistory(h)) return { error: "garbage" };
    out.history = h;
  }
  if (want.prefix) {
    const p = tagged(content, "turn-prefix-summary");
    if (!validPrefix(p)) return { error: "garbage" };
    out.prefix = p;
  }
  return out;
}
