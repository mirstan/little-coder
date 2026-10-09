import { describe, expect, it } from "vitest";
import {
  buildInstruction,
  buildReplayBody,
  locateAnchor,
  parseSummaryOutput,
  sanitizeForTemplate,
  type AnchorSpec,
} from "./replay.ts";
import { PI_TURN_PREFIX_SUMMARIZATION_PROMPT, PI_UPDATE_SUMMARIZATION_PROMPT } from "./pi-compat.ts";

// An omlx-shaped payload as pi-ai's openai-completions buildParams emits it
// (thinkingFormat "chat-template", benchmark-profiles' temperature on top).
function capturedPayload() {
  return {
    model: "tiel-coder-oq6e-fp16",
    messages: [
      { role: "developer", content: "You are little-coder." },
      { role: "user", content: "Fix the build in /app." },
      {
        role: "assistant",
        content: "",
        reasoning_content: "look first",
        tool_calls: [{ id: "call_a", type: "function", function: { name: "bash", arguments: '{"command":"ls"}' } }],
      },
      { role: "tool", content: "Makefile", tool_call_id: "call_a" },
      {
        role: "assistant",
        reasoning_content: "now build",
        tool_calls: [{ id: "call_b", type: "function", function: { name: "bash", arguments: '{"command":"make"}' } }],
      },
      { role: "tool", content: "ok", tool_call_id: "call_b" },
    ],
    stream: true,
    stream_options: { include_usage: true },
    store: false,
    max_completion_tokens: 80000,
    tools: [{ type: "function", function: { name: "bash", description: "run", parameters: { type: "object" } } }],
    chat_template_kwargs: { enable_thinking: true, preserve_thinking: true },
    temperature: 0.2,
  };
}

describe("buildReplayBody", () => {
  it("appends exactly one user message and changes only the output budget fields", () => {
    const captured = capturedPayload();
    const frozen = JSON.parse(JSON.stringify(captured));
    const body = buildReplayBody(captured, "SUMMARIZE", { maxTokens: 9000, thinkingBudget: 2048 });

    // The capture itself is never mutated: a fallback or a second compaction reuses it.
    expect(captured).toEqual(frozen);

    const { messages, max_completion_tokens, thinking_budget, ...rest } = body as any;
    const { messages: cm, max_completion_tokens: _c, ...crest } = frozen;
    expect(rest).toEqual(crest);
    expect(messages.slice(0, -1)).toEqual(cm);
    expect(messages.at(-1)).toEqual({ role: "user", content: "SUMMARIZE" });
    expect(max_completion_tokens).toBe(9000);
    expect(thinking_budget).toBe(2048);
    // Fields that change the rendered template are never introduced.
    for (const k of ["tool_choice", "reasoning_effort", "enable_thinking", "max_tokens"]) expect(body).not.toHaveProperty(k);
  });

  it("serializes the captured messages as a byte prefix of the replayed ones", () => {
    const captured = capturedPayload();
    const body = buildReplayBody(captured, "SUMMARIZE", { maxTokens: 1, thinkingBudget: 0 });
    const a = JSON.stringify(captured.messages);
    const b = JSON.stringify((body as any).messages);
    expect(b.startsWith(a.slice(0, -1))).toBe(true);
    expect(b.slice(a.length - 1)).toBe(`,${JSON.stringify({ role: "user", content: "SUMMARIZE" })}]`);
  });

  it("uses max_tokens when that is the key the capture carries, and omits thinking_budget at 0", () => {
    const captured: any = capturedPayload();
    delete captured.max_completion_tokens;
    captured.max_tokens = 4096;
    const body: any = buildReplayBody(captured, "S", { maxTokens: 7, thinkingBudget: 0 });
    expect(body.max_tokens).toBe(7);
    expect(body).not.toHaveProperty("max_completion_tokens");
    expect(body).not.toHaveProperty("thinking_budget");
  });

  it("sets max_tokens when the capture has neither budget key", () => {
    const captured: any = capturedPayload();
    delete captured.max_completion_tokens;
    expect((buildReplayBody(captured, "S", { maxTokens: 5, thinkingBudget: 0 }) as any).max_tokens).toBe(5);
  });
});

describe("sanitizeForTemplate", () => {
  it("defuses chat-template control markers and special tokens", () => {
    const s = sanitizeForTemplate("a <|think_off|> b <|im_end|> c");
    expect(s).not.toContain("<|");
    expect(s).toContain("think_off");
  });

  it("defuses the template's structural tags", () => {
    const raw = "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call> <tool_response>x</tool_response> <think>y</think>";
    const s = sanitizeForTemplate(raw);
    for (const tag of ["<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter=", "</parameter>", "<tool_response>", "</tool_response>", "<think>", "</think>"]) {
      expect(s).not.toContain(tag);
    }
    // Our own output tags are untouched.
    expect(sanitizeForTemplate("<history-summary>")).toBe("<history-summary>");
  });
});

const assistantWithCalls = (ids: string[], text = "") => ({
  role: "assistant" as const,
  content: [
    ...(text ? [{ type: "text", text }] : []),
    ...ids.map((id) => ({ type: "toolCall", id, name: "bash", arguments: { command: `run ${id}` } })),
  ],
});
const user = (text: string) => ({ role: "user" as const, content: text });

describe("locateAnchor", () => {
  const msgs = capturedPayload().messages;

  it("finds an assistant turn by its tool-call ids and quotes it the way the model sees it", () => {
    const got = locateAnchor(msgs, { message: assistantWithCalls(["call_b"]) as any, before: [] });
    expect(got).toMatchObject({ index: 4 });
    // The template renders a call as <function=bash><parameter=command>…, so the
    // excerpt names the tool and quotes the argument value, not JSON.
    expect((got as any).excerpt).toBe('your bash call whose command argument begins "run call_b"');
  });

  it("refuses an excerpt the model could match to several tool calls", () => {
    const many = Array.from({ length: 6 }, (_, i) => ({
      role: "assistant",
      tool_calls: [{ id: `c${i}`, type: "function", function: { name: "bash", arguments: '{"command":"make"}' } }],
    }));
    const msg = { role: "assistant", content: [{ type: "toolCall", id: "c3", name: "bash", arguments: { command: "make" } }] };
    expect(locateAnchor(many, { message: msg as any, before: [] })).toEqual({ error: "anchor_ambiguous" });
  });

  it("numbers a repeated excerpt when only a few messages share it", () => {
    const two = [
      { role: "assistant", tool_calls: [{ id: "x1", type: "function", function: { name: "bash", arguments: '{"command":"make"}' } }] },
      { role: "tool", content: "ok", tool_call_id: "x1" },
      { role: "assistant", tool_calls: [{ id: "x2", type: "function", function: { name: "bash", arguments: '{"command":"make"}' } }] },
    ];
    const msg = { role: "assistant", content: [{ type: "toolCall", id: "x2", name: "bash", arguments: { command: "make" } }] };
    const got = locateAnchor(two, { message: msg as any, before: [] }) as any;
    expect(got.index).toBe(2);
    expect(got.excerpt).toContain("the 2nd of 2");
  });

  it("finds a user message by its opening text", () => {
    expect(locateAnchor(msgs, { message: user("Fix the build in /app.") as any, before: [] })).toMatchObject({ index: 1 });
  });

  it("reports a missing anchor", () => {
    expect(locateAnchor(msgs, { message: assistantWithCalls(["call_zzz"]) as any, before: [] })).toEqual({ error: "anchor_missing" });
    expect(locateAnchor(msgs, { message: user("never said") as any, before: [] })).toEqual({ error: "anchor_missing" });
  });

  it("picks the occurrence after the ones already summarized when text repeats", () => {
    const repeated = [
      { role: "developer", content: "sys" },
      { role: "user", content: "continue please" },
      { role: "assistant", content: "x" },
      { role: "user", content: "continue please" },
      { role: "assistant", content: "y" },
    ];
    const spec: AnchorSpec = { message: user("continue please") as any, before: [user("continue please") as any] };
    expect(locateAnchor(repeated, spec)).toMatchObject({ index: 3, excerpt: expect.stringContaining("the 2nd of 2") });
    expect(locateAnchor(repeated, { message: user("continue please") as any, before: [] })).toMatchObject({ index: 1 });
    const tooMany = { ...spec, before: [user("continue please") as any, user("continue please") as any] };
    expect(locateAnchor(repeated, tooMany)).toEqual({ error: "anchor_missing" });
  });

  it("matches user content given as text parts, and custom messages converted to user text", () => {
    const parts = [{ role: "user", content: [{ type: "text", text: "Hello   there\nfriend" }] }];
    expect(locateAnchor(parts, { message: user("Hello there friend") as any, before: [] })).toMatchObject({ index: 0 });
  });

  it("refuses a message with no text and no tool calls", () => {
    expect(locateAnchor(msgs, { message: { role: "user", content: [] } as any, before: [] })).toEqual({
      error: "anchor_missing",
    });
  });
});

describe("buildInstruction", () => {
  const cut = { index: 4, excerpt: 'your tool call bash({"command":"make"})' };
  const start = { index: 1, excerpt: '"Fix the build in /app."' };

  it("asks for a history summary up to the cut, with pi's initial prompt when there is no previous summary", () => {
    const text = buildInstruction({ cut, turnStart: null, wantHistory: true, wantPrefix: false, hasPrevious: false });
    expect(text).toContain("<history-summary>");
    expect(text).not.toContain("<turn-prefix-summary>");
    expect(text).toContain(cut.excerpt);
    expect(text).toContain("## Goal");
    expect(text).toMatch(/do not call any tools/i);
    expect(text).not.toContain("<|");
  });

  it("uses pi's update prompt when a previous summary exists", () => {
    const text = buildInstruction({ cut, turnStart: null, wantHistory: true, wantPrefix: false, hasPrevious: true });
    expect(text).toContain(PI_UPDATE_SUMMARIZATION_PROMPT.split("\n").at(-1));
    expect(text).toContain("PRESERVE all existing information from the previous summary");
  });

  it("asks for both sections on a split turn and names both boundaries", () => {
    const text = buildInstruction({ cut, turnStart: start, wantHistory: true, wantPrefix: true, hasPrevious: false });
    expect(text).toContain("<history-summary>");
    expect(text).toContain("<turn-prefix-summary>");
    expect(text).toContain(start.excerpt);
    expect(text).toContain("## Original Request");
    expect(text).toContain(PI_TURN_PREFIX_SUMMARIZATION_PROMPT.split("\n").at(-1));
  });

  it("asks only for the turn prefix on a split turn with no earlier history", () => {
    const text = buildInstruction({ cut, turnStart: start, wantHistory: false, wantPrefix: true, hasPrevious: false });
    expect(text).not.toContain("<history-summary>");
    expect(text).toContain("<turn-prefix-summary>");
  });

  it("carries custom instructions as pi's 'Additional focus'", () => {
    const text = buildInstruction({
      cut, turnStart: null, wantHistory: true, wantPrefix: false, hasPrevious: false, customInstructions: "the <|think_low|> bug",
    });
    expect(text).toContain("Additional focus: the ");
    expect(text).not.toContain("<|");
  });
});

const history =
  "## Goal\nShip it\n\n## Constraints & Preferences\n- (none)\n\n## Progress\n### Done\n- [x] a\n\n" +
  "## Key Decisions\n- **x**: y\n\n## Next Steps\n1. b\n\n## Critical Context\n- (none)";
const prefix = "## Original Request\nFix the build\n\n## Early Progress\n- ran ls\n\n## Context for Suffix\n- make next";

describe("parseSummaryOutput", () => {
  it("extracts both tagged sections", () => {
    const out = `<history-summary>\n${history}\n</history-summary>\n\n<turn-prefix-summary>\n${prefix}\n</turn-prefix-summary>`;
    expect(parseSummaryOutput(out, { history: true, prefix: true })).toEqual({ history, prefix });
  });

  it("accepts an untagged history summary when only history was asked for", () => {
    expect(parseSummaryOutput(`Here it is:\n${history}`, { history: true, prefix: false })).toEqual({
      history: `Here it is:\n${history}`,
    });
  });

  it("rejects a history summary missing any of pi's sections", () => {
    for (const heading of ["## Goal", "## Constraints & Preferences", "## Progress", "## Key Decisions", "## Next Steps", "## Critical Context"]) {
      expect(parseSummaryOutput(history.replace(heading, "## Other"), { history: true, prefix: false })).toEqual({ error: "garbage" });
    }
    for (const heading of ["## Original Request", "## Early Progress", "## Context for Suffix"]) {
      const out = `<history-summary>\n${history}\n</history-summary>\n<turn-prefix-summary>\n${prefix.replace(heading, "## X")}\n</turn-prefix-summary>`;
      expect(parseSummaryOutput(out, { history: true, prefix: true })).toEqual({ error: "garbage" });
    }
  });

  it("rejects empty, unstructured or truncated output", () => {
    expect(parseSummaryOutput("", { history: true, prefix: false })).toEqual({ error: "garbage" });
    expect(parseSummaryOutput("I will now run make.", { history: true, prefix: false })).toEqual({ error: "garbage" });
    expect(
      parseSummaryOutput(`<history-summary>\n${history}\n</history-summary>\n<turn-prefix-summary>\n## Orig`, {
        history: true,
        prefix: true,
      }),
    ).toEqual({ error: "garbage" });
    expect(parseSummaryOutput(`<turn-prefix-summary>\n${prefix}\n</turn-prefix-summary>`, { history: true, prefix: true })).toEqual({
      error: "garbage",
    });
  });
});
