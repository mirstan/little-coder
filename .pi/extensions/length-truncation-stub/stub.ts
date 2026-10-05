// Projection-layer stub for assistant messages cut off at the output-token
// limit. pi-agent-core refuses to run their tool calls (it answers each with
// an error toolResult) but keeps the message itself, salvaged arguments and
// thinking included, in every later request. One hallucinated 80KB base64
// ShellSession call did that in a TB2.1 trial: context 45.7K -> 124.5K
// tokens, and each later cache-miss prefill re-processed ~90K of them.
//
// The attempt is worth keeping (the model learns the approach was too long);
// its payload is not. This must stay a pure function of the message list:
// it runs on every request, and a stub that rendered differently from one
// turn to the next would break the server's prefix cache it exists to help.

import { splitHeadTail } from "../shell-retention/retention.ts";

const byteLen = (s: string) => Buffer.byteLength(s, "utf-8");

export const KEEP_HEAD_BYTES = 512;
export const KEEP_TAIL_BYTES = 256;
// The smallest string worth stubbing. It sits well above head + marker +
// tail, so a stubbed string never qualifies again: idempotence follows from
// the sizes, with no marker sniffing.
export const STUB_MIN_BYTES = 2048;
// Many small strings (a 5,000-element array) slip under STUB_MIN_BYTES one
// at a time, so the serialized arguments get a cap of their own.
export const ARGS_CAP_BYTES = 8192;

// These APIs sign thinking blocks and reject an altered one, so their
// thinking is left as is.
const SIGNED_THINKING_APIS = new Set(["anthropic-messages", "bedrock-converse-stream"]);

function argsMarker(dropped: number): string {
  return (
    `[... ${dropped} bytes of this tool call's arguments omitted: the response hit ` +
    `the output token limit mid-call, so the call was not executed ...]`
  );
}

function thinkingMarker(dropped: number): string {
  return `[... ${dropped} bytes of reasoning omitted: this response hit the output token limit ...]`;
}

function stubString(s: string, marker: (dropped: number) => string): string {
  if (byteLen(s) <= STUB_MIN_BYTES) return s;
  const { head, tail, dropped } = splitHeadTail(s, KEEP_HEAD_BYTES, KEEP_TAIL_BYTES);
  if (dropped === 0) return s;
  return `${head}\n${marker(dropped)}\n${tail}`;
}

function stubValue(v: unknown): unknown {
  if (typeof v === "string") return stubString(v, argsMarker);
  if (Array.isArray(v)) return v.map(stubValue);
  if (v !== null && typeof v === "object") {
    return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, stubValue(x)]));
  }
  return v;
}

function stubArguments(args: unknown): unknown {
  const stubbed = stubValue(args);
  const size = byteLen(JSON.stringify(stubbed) ?? "");
  if (size <= ARGS_CAP_BYTES) return stubbed;
  return { omitted: argsMarker(byteLen(JSON.stringify(args) ?? "")) };
}

function stubBlock(block: any, api: unknown): any {
  // A Google-style thoughtSignature is replayed against the original
  // arguments; shell-retention leaves those calls alone for the same reason.
  if (block?.type === "toolCall" && block.thoughtSignature === undefined) {
    const args = stubArguments(block.arguments);
    return JSON.stringify(args) === JSON.stringify(block.arguments) ? block : { ...block, arguments: args };
  }
  if (
    block?.type === "thinking" &&
    !block.redacted &&
    typeof block.thinking === "string" &&
    !SIGNED_THINKING_APIS.has(String(api))
  ) {
    // Never empty: openai-completions drops the reasoning field when the
    // text trims to nothing.
    const thinking = stubString(block.thinking, thinkingMarker);
    return thinking === block.thinking ? block : { ...block, thinking };
  }
  return block;
}

/**
 * Stub the tool-call arguments and thinking of every assistant message that
 * stopped on "length" with tool calls in it. Everything else, including the
 * paired toolResult, is returned as the same object.
 */
export function stubTruncatedMessages(messages: any[]): { messages: any[]; stubbedCount: number } {
  let stubbedCount = 0;
  const out = messages.map((m) => {
    if (m?.role !== "assistant" || m.stopReason !== "length" || !Array.isArray(m.content)) return m;
    if (!m.content.some((c: any) => c?.type === "toolCall")) return m;
    const content = m.content.map((c: any) => stubBlock(c, m.api));
    if (content.every((c: any, i: number) => c === m.content[i])) return m;
    stubbedCount++;
    return { ...m, content };
  });
  return { messages: out, stubbedCount };
}
