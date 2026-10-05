import { describe, it, expect } from "vitest";
import {
  stubTruncatedMessages,
  ARGS_CAP_BYTES,
  KEEP_HEAD_BYTES,
  KEEP_TAIL_BYTES,
  STUB_MIN_BYTES,
} from "./stub.ts";
import { demoteMessages, resolveOptions, type RetentionArchive } from "../shell-retention/retention.ts";

// Canned shapes mirror pi-ai's AssistantMessage / ToolCall / ThinkingContent /
// ToolResultMessage (node_modules/@earendil-works/pi-coding-agent/node_modules/
// @earendil-works/pi-ai/dist/types.d.ts).

const B64 = "QUJD".repeat(80_000 / 4); // ~80KB single-line base64, the TB2.1 shape
const THINKING = "Let me encode the binary as base64 and write it in one go.\n" + "x".repeat(20_000) + "\nDone thinking.";

function truncatedAssistant(over: Record<string, unknown> = {}) {
  return {
    role: "assistant",
    api: "openai-completions",
    provider: "omlx",
    model: "m",
    stopReason: "length",
    usage: {},
    timestamp: 1,
    content: [
      { type: "thinking", thinking: THINKING, thinkingSignature: "reasoning_content" },
      {
        type: "toolCall",
        id: "call_1",
        name: "ShellSession",
        arguments: { command: `echo '${B64}' | base64 -d > /app/out.bin`, timeout: 120 },
      },
    ],
    ...over,
  };
}

function errorResult(id = "call_1") {
  return {
    role: "toolResult",
    toolCallId: id,
    toolName: "ShellSession",
    content: [
      {
        type: "text",
        text: `Tool call "ShellSession" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.`,
      },
    ],
    isError: true,
    timestamp: 2,
  };
}

function convo(assistant = truncatedAssistant()) {
  return [{ role: "user", content: "write the binary", timestamp: 0 }, assistant, errorResult()];
}

const bytes = (v: unknown) => Buffer.byteLength(JSON.stringify(v), "utf-8");

describe("stubTruncatedMessages", () => {
  it("stubs a length-truncated tool call's oversized string argument, keeping head and tail", () => {
    const { messages, stubbedCount } = stubTruncatedMessages(convo());
    expect(stubbedCount).toBe(1);
    const call = (messages[1] as any).content[1];
    const cmd: string = call.arguments.command;
    expect(Buffer.byteLength(cmd)).toBeLessThan(STUB_MIN_BYTES);
    expect(cmd.startsWith("echo 'QUJD")).toBe(true);
    expect(cmd.endsWith("| base64 -d > /app/out.bin")).toBe(true);
    expect(cmd).toMatch(/bytes of this tool call's arguments omitted/);
    expect(cmd).toMatch(/output token limit/);
    expect(cmd).toMatch(/not executed/);
  });

  it("reports the exact dropped byte count in the marker", () => {
    const original = truncatedAssistant().content[1] as any;
    const origLen = Buffer.byteLength(original.arguments.command);
    const { messages } = stubTruncatedMessages(convo());
    const cmd: string = (messages[1] as any).content[1].arguments.command;
    const n = Number(cmd.match(/\[\.\.\. ([\d,]+) bytes/)![1].replace(/,/g, ""));
    const marker = cmd.match(/\n?\[\.\.\. [^\]]*\]\n?/)![0];
    expect(n).toBe(origLen - (Buffer.byteLength(cmd) - Buffer.byteLength(marker)));
  });

  it("keeps every key, the id, the name and short arguments verbatim", () => {
    const { messages } = stubTruncatedMessages(convo());
    const call = (messages[1] as any).content[1];
    expect(call.id).toBe("call_1");
    expect(call.name).toBe("ShellSession");
    expect(Object.keys(call.arguments)).toEqual(["command", "timeout"]);
    expect(call.arguments.timeout).toBe(120);
  });

  it("keeps arguments a plain JSON object that round-trips", () => {
    const { messages } = stubTruncatedMessages(convo());
    const args = (messages[1] as any).content[1].arguments;
    expect(typeof args).toBe("object");
    expect(Array.isArray(args)).toBe(false);
    expect(JSON.parse(JSON.stringify(args))).toEqual(args);
  });

  it("leaves the toolResult and tool_call/tool_result pairing intact", () => {
    const input = convo();
    const { messages } = stubTruncatedMessages(input);
    expect(messages).toHaveLength(3);
    expect(messages[2]).toBe(input[2]);
    expect((messages[1] as any).content[1].id).toBe((messages[2] as any).toolCallId);
  });

  it("never touches a message that did not stop on length", () => {
    for (const stopReason of ["stop", "toolUse", "error", "aborted"]) {
      const input = convo(truncatedAssistant({ stopReason }));
      const { messages, stubbedCount } = stubTruncatedMessages(input);
      expect(stubbedCount).toBe(0);
      expect(messages[1]).toBe(input[1]);
    }
  });

  it("leaves a length-stopped message without tool calls alone", () => {
    const msg = truncatedAssistant({ content: [{ type: "text", text: "y".repeat(50_000) }] });
    const { messages, stubbedCount } = stubTruncatedMessages(convo(msg));
    expect(stubbedCount).toBe(0);
    expect(messages[1]).toBe(msg);
  });

  it("leaves a small truncated call unchanged: the stub would not be smaller", () => {
    const msg = truncatedAssistant({
      content: [{ type: "toolCall", id: "call_1", name: "read", arguments: { path: "/app/a.py" } }],
    });
    const { messages, stubbedCount } = stubTruncatedMessages(convo(msg));
    expect(stubbedCount).toBe(0);
    expect(messages[1]).toBe(msg);
  });

  it("does not mutate its input", () => {
    const input = convo();
    const snapshot = structuredClone(input);
    stubTruncatedMessages(input);
    expect(input).toEqual(snapshot);
  });

  it("is idempotent: applying twice equals applying once", () => {
    const once = stubTruncatedMessages(convo()).messages;
    const twice = stubTruncatedMessages(once);
    expect(twice.stubbedCount).toBe(0);
    expect(twice.messages).toEqual(once);
  });

  it("is deterministic: the same history renders byte-identically every time", () => {
    const a = JSON.stringify(stubTruncatedMessages(convo()).messages);
    const b = JSON.stringify(stubTruncatedMessages(convo()).messages);
    expect(a).toBe(b);
  });

  it("stubs long strings nested inside arrays and objects", () => {
    const msg = truncatedAssistant({
      content: [
        {
          type: "toolCall",
          id: "call_1",
          name: "edit",
          arguments: { path: "/app/x.py", edits: [{ oldText: "a", newText: "z".repeat(30_000) }] },
        },
      ],
    });
    const { messages, stubbedCount } = stubTruncatedMessages(convo(msg));
    expect(stubbedCount).toBe(1);
    const args = (messages[1] as any).content[0].arguments;
    expect(args.path).toBe("/app/x.py");
    expect(args.edits[0].oldText).toBe("a");
    expect(Buffer.byteLength(args.edits[0].newText)).toBeLessThan(STUB_MIN_BYTES);
  });

  it("replaces arguments made of many small strings with one marker object", () => {
    const lines = Array.from({ length: 5_000 }, (_, i) => `line ${i}`);
    const msg = truncatedAssistant({
      content: [{ type: "toolCall", id: "call_1", name: "write", arguments: { path: "/app/x", lines } }],
    });
    const { messages, stubbedCount } = stubTruncatedMessages(convo(msg));
    expect(stubbedCount).toBe(1);
    const args = (messages[1] as any).content[0].arguments;
    expect(bytes(args)).toBeLessThan(ARGS_CAP_BYTES);
    expect(Object.keys(args)).toEqual(["omitted"]);
    expect(args.omitted).toMatch(/bytes of this tool call's arguments omitted/);
    expect(stubTruncatedMessages(messages).stubbedCount).toBe(0);
  });

  it("stubs every tool call in a truncated message", () => {
    const msg = truncatedAssistant({
      content: [
        { type: "toolCall", id: "call_1", name: "write", arguments: { path: "/a", content: "a".repeat(10_000) } },
        { type: "toolCall", id: "call_2", name: "write", arguments: { path: "/b", content: "b".repeat(10_000) } },
      ],
    });
    const input = [msg, errorResult("call_1"), errorResult("call_2")];
    const { messages } = stubTruncatedMessages(input);
    for (const c of (messages[0] as any).content) {
      expect(Buffer.byteLength(c.arguments.content)).toBeLessThan(STUB_MIN_BYTES);
    }
  });

  it("leaves a tool call carrying a thoughtSignature alone: the signature binds its arguments", () => {
    const msg = truncatedAssistant({ api: "google-generative-ai" });
    (msg.content as any)[1].thoughtSignature = "sig";
    const { messages } = stubTruncatedMessages(convo(msg));
    expect((messages[1] as any).content[1]).toBe((msg.content as any)[1]);
  });

  describe("thinking", () => {
    it("stubs an oversized thinking block to head + marker + tail, keeping its signature", () => {
      const { messages } = stubTruncatedMessages(convo());
      const block = (messages[1] as any).content[0];
      expect(block.type).toBe("thinking");
      expect(block.thinkingSignature).toBe("reasoning_content");
      expect(Buffer.byteLength(block.thinking)).toBeLessThan(STUB_MIN_BYTES);
      expect(block.thinking.startsWith("Let me encode the binary")).toBe(true);
      expect(block.thinking.endsWith("Done thinking.")).toBe(true);
      expect(block.thinking).toMatch(/bytes of reasoning omitted/);
      expect(block.thinking.trim().length).toBeGreaterThan(0);
    });

    it("leaves short thinking alone", () => {
      const msg = truncatedAssistant();
      (msg.content as any)[0].thinking = "short";
      const { messages } = stubTruncatedMessages(convo(msg));
      expect((messages[1] as any).content[0].thinking).toBe("short");
    });

    it("never rewrites redacted thinking", () => {
      const msg = truncatedAssistant();
      (msg.content as any)[0] = { type: "thinking", thinking: THINKING, thinkingSignature: "opaque", redacted: true };
      const { messages } = stubTruncatedMessages(convo(msg));
      expect((messages[1] as any).content[0]).toEqual((msg.content as any)[0]);
    });

    it("never rewrites thinking on an API that signs it, but still stubs the arguments", () => {
      for (const api of ["anthropic-messages", "bedrock-converse-stream"]) {
        const msg = truncatedAssistant({ api });
        const { messages, stubbedCount } = stubTruncatedMessages(convo(msg));
        expect(stubbedCount).toBe(1);
        expect((messages[1] as any).content[0]).toEqual((msg.content as any)[0]);
        expect(Buffer.byteLength((messages[1] as any).content[1].arguments.command)).toBeLessThan(STUB_MIN_BYTES);
      }
    });
  });

  it("leaves a stubbed ShellSession pair below shell-retention's demotion size", () => {
    const archive: RetentionArchive = { save: () => true, size: () => 0, readRange: () => undefined };
    const eager = { ...resolveOptions(), retainRaw: 0, staleDistance: 0 };
    const tail = [{ role: "user", content: "next", timestamp: 3 }];
    // Without thinking: shell-retention treats a thinkingSignature as signed
    // and would leave the command alone for that reason instead.
    const unsigned = () => convo(truncatedAssistant({ content: [truncatedAssistant().content[1]] }));
    expect(demoteMessages([...unsigned(), ...tail], archive, eager).demotedCount).toBe(1);
    const stubbed = stubTruncatedMessages(unsigned()).messages;
    expect(demoteMessages([...stubbed, ...tail], archive, eager).demotedCount).toBe(0);
  });

  it("stub constants leave head + marker + tail strictly below the stub threshold", () => {
    expect(KEEP_HEAD_BYTES + KEEP_TAIL_BYTES + 256).toBeLessThan(STUB_MIN_BYTES);
  });
});
