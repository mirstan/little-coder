import { describe, expect, it } from "vitest";
import { readChatCompletionStream } from "./sse.ts";

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(enc.encode(c));
      controller.close();
    },
  });
}

const line = (obj: unknown) => `data: ${JSON.stringify(obj)}\n\n`;

describe("readChatCompletionStream", () => {
  it("collects content, reasoning, finish reason and usage across split chunks", async () => {
    const body = [
      line({ choices: [{ delta: { reasoning_content: "think " } }] }),
      line({ choices: [{ delta: { reasoning_content: "more" } }] }),
      line({ choices: [{ delta: { content: "## Goal" } }] }),
      line({ choices: [{ delta: { content: "\nx" }, finish_reason: "stop" }] }),
      line({ choices: [], usage: { prompt_tokens: 1000, completion_tokens: 50, prompt_tokens_details: { cached_tokens: 900 } } }),
      "data: [DONE]\n\n",
    ].join("");
    // Split mid-line to prove the reader buffers partial lines.
    const parts = [body.slice(0, 17), body.slice(17, 90), body.slice(90)];
    const t0 = 1000;
    let now = t0;
    const out = await readChatCompletionStream(streamOf(parts), () => (now += 10));
    expect(out.content).toBe("## Goal\nx");
    expect(out.reasoning).toBe("think more");
    expect(out.finishReason).toBe("stop");
    expect(out.toolCalls).toBe(0);
    expect(out.usage).toEqual({ promptTokens: 1000, completionTokens: 50, cachedTokens: 900 });
    expect(out.firstTokenAt).not.toBeNull();
  });

  it("accepts the `reasoning` field and counts tool calls by index", async () => {
    const out = await readChatCompletionStream(
      streamOf([
        line({ choices: [{ delta: { reasoning: "r" } }] }),
        line({ choices: [{ delta: { tool_calls: [{ index: 0, id: "a", function: { name: "bash", arguments: "{" } }] } }] }),
        line({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: "}" } }] } }] }),
        line({ choices: [{ delta: { tool_calls: [{ index: 1, id: "b", function: { name: "read", arguments: "{}" } }] }, finish_reason: "tool_calls" }] }),
      ]),
      () => 0,
    );
    expect(out.reasoning).toBe("r");
    expect(out.toolCalls).toBe(2);
    expect(out.finishReason).toBe("tool_calls");
    expect(out.usage).toBeNull();
  });

  it("ignores comments, blank lines and unparseable data lines", async () => {
    const out = await readChatCompletionStream(
      streamOf([": keepalive\n\n", "data: {not json\n\n", "\n", line({ choices: [{ delta: { content: "ok" } }] })]),
      () => 0,
    );
    expect(out.content).toBe("ok");
  });

  it("reports the first token time only once any token arrived", async () => {
    let now = 0;
    const out = await readChatCompletionStream(streamOf([line({ choices: [{ delta: {} }] })]), () => ++now);
    expect(out.firstTokenAt).toBeNull();
  });
});
