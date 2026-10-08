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
 * that renders at the top of the system block, `<|im_end|>`-style text
 * tokenizes as a special token, and quoted `<tool_call>` / `<function=…>` /
 * `<think>` markup reads as the model's own structure.
 */
export function sanitizeForTemplate(text: string): string {
  return text
    .replace(/<\|/g, "< |")
    .replace(/\|>/g, "| >")
    .replace(/<(\/?)(tool_call|tool_response|think|function|parameter)\b/g, "< $1$2");
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

export type AnchorResult = { index: number; excerpt: string } | { error: "anchor_missing" | "anchor_ambiguous" };

const SNIPPET_CHARS = 160;
// Excerpt lengths tried in turn until the quoted text names few enough messages.
const EXCERPT_CHARS = [120, 300];
// More look-alikes than this and "the k-th of n" is not something a model can count reliably.
const MAX_NUMBERED = 3;

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

// What the model sees of a message, as (label, quoted text, does a payload message show the same).
interface Visible {
  describe: (quoted: string) => string;
  full: string;
  shows: (m: any, quoted: string) => boolean;
}

function argText(v: unknown): string {
  return typeof v === "string" ? v : JSON.stringify(v ?? "");
}

function parseArgs(raw: unknown): Record<string, unknown> {
  if (raw && typeof raw === "object") return raw as Record<string, unknown>;
  try {
    const v = JSON.parse(String(raw ?? "{}"));
    return v && typeof v === "object" ? v : {};
  } catch {
    return {};
  }
}

function visibleOf(m: LlmMessage): Visible {
  const call: any = Array.isArray(m.content) ? (m.content as any[]).find((b) => b?.type === "toolCall") : undefined;
  if (call) {
    // The template renders a call as <function=NAME><parameter=KEY>VALUE…, so
    // quote the first argument's value, never the JSON form.
    const [key, value] = Object.entries(call.arguments ?? {})[0] ?? ["", ""];
    const name = String(call.name);
    return {
      describe: (q) => (key ? `your ${name} call whose ${key} argument begins "${q}"` : `your ${name} call`),
      full: normalize(argText(value)),
      shows: (pm, q) =>
        pm?.role === "assistant" &&
        Array.isArray(pm.tool_calls) &&
        pm.tool_calls.some((t: any) => {
          const fn = t?.function ?? {};
          if (fn.name !== name) return false;
          if (!key) return true;
          return normalize(argText(parseArgs(fn.arguments)[key])).startsWith(q);
        }),
    };
  }
  const role = payloadRole(m.role);
  const who = m.role === "assistant" ? "your reply" : "the message";
  return {
    describe: (q) => `${who} that begins "${q}"`,
    full: normalize(messageText(m)),
    shows: (pm, q) => pm?.role === role && normalize(payloadText(pm)).startsWith(q),
  };
}

const ordinal = (n: number) => `${n}${n % 10 === 1 && n !== 11 ? "st" : n % 10 === 2 && n !== 12 ? "nd" : n % 10 === 3 && n !== 13 ? "rd" : "th"}`;

/**
 * The words that point the model at payload message `index`: a quote long
 * enough that few other messages look the same, numbered when a handful do.
 * Null when too many look alike for a model to pick the right one.
 */
function excerptFor(payloadMessages: unknown[], index: number, m: LlmMessage): string | null {
  const v = visibleOf(m);
  for (const chars of EXCERPT_CHARS) {
    const quoted = v.full.slice(0, chars);
    const same = payloadMessages.map((pm, i) => (v.shows(pm, quoted) ? i : -1)).filter((i) => i >= 0);
    const rank = same.indexOf(index) + 1;
    const text = sanitizeForTemplate(v.describe(quoted + (v.full.length > chars ? "…" : "")));
    if (same.length <= 1) return text;
    if (chars === EXCERPT_CHARS[EXCERPT_CHARS.length - 1] || v.full.length <= chars) {
      if (rank === 0 || same.length > MAX_NUMBERED) return null;
      return `${text} (the ${ordinal(rank)} of ${same.length} messages that look like that)`;
    }
  }
  return null;
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
    if (seen === skip) {
      const excerpt = excerptFor(payloadMessages, i, spec.message);
      return excerpt === null ? { error: "anchor_ambiguous" } : { index: i, excerpt };
    }
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
      "Summarize only what comes before it. If you are unsure exactly where that boundary is, include more " +
      "rather than less: anything left out of the summary before it is lost.",
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
