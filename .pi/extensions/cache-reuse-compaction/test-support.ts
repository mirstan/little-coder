// Test fixtures built from pi's real code, so the tests exercise the shapes pi
// actually produces: a SessionManager session, prepareCompaction() on it, and
// the payload pi-ai's openai-completions provider builds for it. Not a test
// file itself (the vitest glob is *.test.ts).

import { convertToLlm, SessionManager } from "@earendil-works/pi-coding-agent";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const piDist = join(repoRoot, "node_modules", "@earendil-works", "pi-coding-agent", "dist");
const piAiDist = join(
  repoRoot,
  "node_modules",
  "@earendil-works",
  "pi-coding-agent",
  "node_modules",
  "@earendil-works",
  "pi-ai",
  "dist",
);

export async function piCompaction(): Promise<any> {
  return import(pathToFileURL(join(piDist, "core", "compaction", "compaction.js")).href);
}

export const MODEL: any = {
  id: "tiel-coder-oq6e-fp16",
  name: "test",
  api: "openai-completions",
  provider: "omlx",
  baseUrl: "http://127.0.0.1:1/v1",
  reasoning: true,
  input: ["text"],
  contextWindow: 262144,
  maxTokens: 80000,
  compat: { thinkingFormat: "chat-template", chatTemplateKwargs: { enable_thinking: true, preserve_thinking: true } },
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
};

export const SYSTEM_PROMPT = "You are little-coder.";
export const TOOLS = [
  { name: "bash", description: "Run a shell command", parameters: { type: "object", properties: { command: { type: "string" } } } },
  { name: "read", description: "Read a file", parameters: { type: "object", properties: { path: { type: "string" } } } },
];

let ts = 1_700_000_000_000;
const usage = (input: number) => ({
  input,
  output: 50,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: input + 50,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
});

export function userMsg(text: string): any {
  return { role: "user", content: text, timestamp: ts++ };
}

export function assistantCall(id: string, name: string, args: Record<string, unknown>, thinking = "thinking about it"): any {
  return {
    role: "assistant",
    content: [
      { type: "thinking", thinking, thinkingSignature: "reasoning_content" },
      { type: "toolCall", id, name, arguments: args },
    ],
    api: "openai-completions",
    provider: "omlx",
    model: MODEL.id,
    usage: usage(1000),
    stopReason: "toolUse",
    timestamp: ts++,
  };
}

export function assistantText(text: string): any {
  return {
    role: "assistant",
    content: [{ type: "text", text }],
    api: "openai-completions",
    provider: "omlx",
    model: MODEL.id,
    usage: usage(1000),
    stopReason: "stop",
    timestamp: ts++,
  };
}

export function toolResult(id: string, name: string, text: string): any {
  return { role: "toolResult", toolCallId: id, toolName: name, content: [{ type: "text", text }], isError: false, timestamp: ts++ };
}

/** One TB-shaped turn: a user prompt then `steps` tool calls with large outputs. */
export function longTurn(
  sm: SessionManager,
  prompt: string,
  steps: number,
  tag: string,
  outputChars = 12_000,
  telemetry = false,
): void {
  sm.appendMessage(userMsg(prompt));
  for (let i = 0; i < steps; i++) {
    // What a benchmark run's session looks like: shell-retention's context hook
    // appends an lc-telemetry entry before every request (LITTLE_CODER_TELEMETRY=1).
    if (telemetry) sm.appendCustomEntry("lc-telemetry", { kind: "shell_retention", v: 1 });
    const id = `call_${tag}_${i}`;
    const name = i % 3 === 0 ? "read" : "bash";
    const args = name === "read" ? { path: `/app/src/file_${tag}_${i}.c` } : { command: `make step_${tag}_${i}` };
    sm.appendMessage(assistantCall(id, name, args, `step ${i} of ${tag}`));
    sm.appendMessage(toolResult(id, name, `${tag}-${i} `.repeat(outputChars / 8)));
  }
}

export const SETTINGS = { enabled: true, reserveTokens: 16384, keepRecentTokens: 20000 };

/** The payload pi-ai's real openai-completions provider builds for this context. */
export async function realPayload(messages: any[], model: any = MODEL): Promise<any> {
  const { stream } = await import(pathToFileURL(join(piAiDist, "api", "openai-completions.js")).href);
  let captured: any;
  const s = stream(
    model,
    { systemPrompt: SYSTEM_PROMPT, messages: convertToLlm(messages), tools: TOOLS },
    {
      apiKey: "test",
      maxTokens: model.maxTokens,
      reasoningEffort: "high",
      onPayload: (p: any) => {
        captured = p;
        return undefined;
      },
      fetch: async () => {
        throw new Error("offline");
      },
      maxRetries: 0,
    },
  );
  await s.result();
  if (!captured) throw new Error("pi-ai built no payload");
  // benchmark-profiles' before_provider_request rewrite, as it lands in the capture.
  return { ...captured, temperature: 0.2 };
}

export function contextMessages(sm: SessionManager): any[] {
  return sm.buildSessionContext().messages;
}
