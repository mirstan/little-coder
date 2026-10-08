// Minimal reader for an OpenAI-compatible chat-completions SSE stream.
//
// Only what the summary needs: the visible text, the reasoning text (so a
// summary written into the thinking channel can be told apart from none), how
// many tool calls the model started, the finish reason, the usage chunk, and
// when the first token of any kind arrived (TTFT).

export interface StreamUsage {
  promptTokens: number;
  completionTokens: number;
  cachedTokens: number;
}

export interface StreamResult {
  content: string;
  reasoning: string;
  toolCalls: number;
  finishReason: string | null;
  usage: StreamUsage | null;
  /** clock() value when the first content/reasoning/tool-call delta arrived. */
  firstTokenAt: number | null;
}

function num(v: unknown): number {
  return typeof v === "number" && Number.isFinite(v) ? v : 0;
}

export async function readChatCompletionStream(
  body: ReadableStream<Uint8Array>,
  clock: () => number,
  onFirstToken?: () => void,
): Promise<StreamResult> {
  const out: StreamResult = {
    content: "",
    reasoning: "",
    toolCalls: 0,
    finishReason: null,
    usage: null,
    firstTokenAt: null,
  };
  const toolIndexes = new Set<number>();
  const decoder = new TextDecoder();
  let buffer = "";

  const handle = (raw: string): void => {
    const lineText = raw.trimEnd();
    if (!lineText.startsWith("data:")) return;
    const data = lineText.slice(5).trim();
    if (data === "" || data === "[DONE]") return;
    let chunk: any;
    try {
      chunk = JSON.parse(data);
    } catch {
      return;
    }
    if (chunk?.usage && typeof chunk.usage === "object") {
      out.usage = {
        promptTokens: num(chunk.usage.prompt_tokens),
        completionTokens: num(chunk.usage.completion_tokens),
        cachedTokens: num(chunk.usage.prompt_tokens_details?.cached_tokens ?? chunk.usage.prompt_cache_hit_tokens),
      };
    }
    const choice = Array.isArray(chunk?.choices) ? chunk.choices[0] : undefined;
    if (!choice) return;
    const delta = choice.delta ?? {};
    let token = false;
    if (typeof delta.content === "string" && delta.content.length > 0) {
      out.content += delta.content;
      token = true;
    }
    const reasoning = delta.reasoning_content ?? delta.reasoning ?? delta.reasoning_text;
    if (typeof reasoning === "string" && reasoning.length > 0) {
      out.reasoning += reasoning;
      token = true;
    }
    if (Array.isArray(delta.tool_calls)) {
      for (const tc of delta.tool_calls) {
        toolIndexes.add(typeof tc?.index === "number" ? tc.index : toolIndexes.size);
      }
      token = token || delta.tool_calls.length > 0;
    }
    if (token && out.firstTokenAt === null) {
      out.firstTokenAt = clock();
      onFirstToken?.();
    }
    if (typeof choice.finish_reason === "string") out.finishReason = choice.finish_reason;
  };

  const reader = body.getReader();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl: number;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      handle(buffer.slice(0, nl));
      buffer = buffer.slice(nl + 1);
    }
  }
  buffer += decoder.decode();
  if (buffer.length > 0) handle(buffer);
  out.toolCalls = toolIndexes.size;
  return out;
}
