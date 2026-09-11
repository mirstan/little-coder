import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";
import setupSkillInject, { looksLikeResearchTask, shouldInjectResearchDirective } from "./index.ts";
import setupKnowledgeInject from "../knowledge-inject/index.ts";

// End-to-end check of the #73 conversion: drive the real `before_agent_start`
// handlers of both injectors, against the real skills/ files, and assert the
// guidance still gets delivered — just at the conversation tail instead of
// stapled onto the system prompt.

type Handler = (event: any, ctx: any) => Promise<any>;

function handlerFor(setup: (pi: any) => void): Handler {
  let handler: Handler | undefined;
  setup({
    on(name: string, h: Handler) {
      if (name === "before_agent_start") handler = h;
    },
    registerCommand: () => {},
  });
  if (!handler) throw new Error("extension registered no before_agent_start handler");
  return handler;
}

/** Every handler an extension registers, so a test can drive tool_result too. */
function handlersFor(setup: (pi: any) => void): Record<string, Handler> {
  const handlers: Record<string, Handler> = {};
  setup({
    on: (name: string, h: Handler) => { handlers[name] = h; },
    registerCommand: () => {},
  });
  return handlers;
}

const ctx = { ui: { notify: () => {} } };

/** A turn event with the little-coder budgets the extensions expect. */
function turn(prompt: string, systemPrompt = "BASE SYSTEM PROMPT") {
  return {
    prompt,
    systemPrompt,
    systemPromptOptions: {
      littleCoder: {
        skillTokenBudget: 300,
        knowledgeTokenBudget: 200,
        contextLimit: 32768,
      } as Record<string, unknown>,
    },
  };
}

afterEach(() => {
  delete process.env.LITTLE_CODER_INJECT_MODE;
});

describe("skill-inject still injects after the #73 conversion", () => {
  it("delivers the tool skill cards as a hidden tail message", async () => {
    const handler = handlerFor(setupSkillInject);
    const result = await handler(turn("edit the parser to fix the bug"), ctx);

    expect(result?.message).toBeDefined();
    expect(result.message.customType).toBe("lc-skills");
    expect(result.message.display).toBe(false);
    expect(result.message.content).toContain("## Tool Usage Guidance");
    // The cached prefix must come through untouched.
    expect(result.systemPrompt).toBeUndefined();
  });

  it("injects the registered lowercase bash tool name", async () => {
    const handler = handlerFor(setupSkillInject);
    const result = await handler(turn("run ls please"), ctx);

    expect(result?.message.content).toContain('"name": "bash"');
    expect(result.message.content).not.toContain('"name": "Bash"');
  });

  // The half of the pi 0.83 rename nobody reported. The registry is keyed by
  // each card's target_tool, but recency and error-recovery look it up with the
  // name pi reports on tool events. Once pi went lowercase those lookups missed
  // every time, so priorities 1 and 2 of the selection algorithm contributed
  // nothing from v1.14.0 on and only intent prediction still fired. The prompts
  // below deliberately carry no INTENT_MAP keyword, so a card can only be
  // selected via the priority under test.
  // recentToolCalls / lastFailedTool are module-level, so each of these drives a
  // freshly imported copy of the extension rather than leaking recency state
  // into its neighbours.
  async function freshHandlers(): Promise<Record<string, Handler>> {
    vi.resetModules();
    const mod = await import("./index.ts");
    return handlersFor(mod.default);
  }

  it("selects a card by recency from pi's lowercase tool_result name", async () => {
    const h = await freshHandlers();
    await h.tool_result({ toolName: "bash", isError: false }, ctx);

    const result = await h.before_agent_start(turn("ok, on to the next thing"), ctx);

    expect(result?.message?.content ?? "").toContain("### bash");
  });

  it("selects a card by error recovery from pi's lowercase tool_result name", async () => {
    const h = await freshHandlers();
    await h.tool_result({ toolName: "edit", isError: true }, ctx);

    const result = await h.before_agent_start(turn("ok, carry on"), ctx);

    expect(result?.message?.content ?? "").toContain("### edit");
  });

  // Issue #97 / the pi 0.83 rename. The evidence step of the research directive
  // must not be handed to a process that cannot call the evidence tools.
  it("drops the EvidenceAdd step when evidence tools are gated out", async () => {
    const { SUBCODER_ALLOWED_TOOLS } = await import("../subagent/spawn.ts");
    const handler = handlerFor(setupSkillInject);
    const event = turn("research the history of the transistor online");
    event.systemPromptOptions.littleCoder.allowedTools =
      SUBCODER_ALLOWED_TOOLS.split(",");

    const result = await handler(event, ctx);

    expect(result?.message.content).toContain("## Research-first directive");
    expect(result.message.content).not.toContain("EvidenceAdd");
    expect(result.message.content).toContain("name its source URL inline");
  });

  it("keeps the EvidenceAdd step when nothing is gated", async () => {
    const handler = handlerFor(setupSkillInject);
    const result = await handler(
      turn("research the history of the transistor online, cite your sources"),
      ctx,
    );

    expect(result?.message.content).toContain("EvidenceAdd");
  });

  it("still appends the research directive last, ahead of nothing", async () => {
    const handler = handlerFor(setupSkillInject);
    const result = await handler(turn("research the history of the transistor online"), ctx);
    const content: string = result.message.content;
    expect(content).toContain("## Research-first directive");
    // Recency bias is the reason the directive is last; keep that ordering.
    expect(content.indexOf("## Research-first directive")).toBeGreaterThan(
      content.indexOf("## Tool Usage Guidance"),
    );
  });

  it("skips a repeat of the identical block on the next turn", async () => {
    const handler = handlerFor(setupSkillInject);
    const first = await handler(turn("edit the parser"), ctx);
    expect(first?.message).toBeDefined();
    // Same prompt shape → same selection → the copy from turn 1 is still there.
    const second = await handler(turn("edit the parser"), ctx);
    expect(second).toBeUndefined();
  });

  it("falls back to the system prompt under LITTLE_CODER_INJECT_MODE=system", async () => {
    process.env.LITTLE_CODER_INJECT_MODE = "system";
    const handler = handlerFor(setupSkillInject);
    const result = await handler(turn("edit the parser to fix the bug"), ctx);

    expect(result?.message).toBeUndefined();
    expect(result.systemPrompt.startsWith("BASE SYSTEM PROMPT")).toBe(true);
    expect(result.systemPrompt).toContain("## Tool Usage Guidance");
  });

  it("stays silent when nothing matches", async () => {
    const handler = handlerFor(setupSkillInject);
    expect(await handler(turn("zzzz"), ctx)).toBeUndefined();
  });
});

// The research directive should only fire when a browse tool is actually
// callable: the Harbor adapter's default allow-list is ShellSession-only, so
// a shell-only trial whose own prompt boilerplate happens to sound like a
// research task must not get told to call BrowserNavigate/BrowserExtract/
// websearch — tools tool-gating would refuse. shouldInjectResearchDirective
// gates on browse-tool availability rather than on prompt shape, and the
// adapter's own boilerplate is reworded (defense-in-depth) so it no longer
// smells like a research task by itself.
//
// These helpers read the real source files rather than mirroring their
// content as string literals, so drift in either the Harbor prompt template
// or its allow-list (or GAIA's) re-trips the tests below automatically.
function repoRoot(): string {
  return join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
}

/** Decode a Python double-quoted string body (the only escapes these files
 *  use are \n and \\, both valid inside a JSON string literal too). */
function decodePyString(body: string): string {
  return JSON.parse(`"${body}"`);
}

function extractQuotedStrings(block: string): string[] {
  const out: string[] = [];
  const re = /"((?:[^"\\]|\\.)*)"/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(block))) out.push(m[1]);
  return out;
}

/** Rebuild the exact string little_coder_agent.py's `prompt = (...)` produces,
 *  by concatenating its quoted segments the way Python's implicit adjacent-
 *  string-literal concatenation does. The `f"TASK:\n{instruction}\n\n"` segment
 *  is extracted like any other literal (the `{instruction}` placeholder is
 *  left as literal text, then substituted below). */
function harborPromptTemplate(): string {
  const src = readFileSync(
    join(repoRoot(), "benchmarks", "harbor_adapter", "little_coder_agent.py"),
    "utf-8",
  );
  const block = src.match(/prompt = \(\n([\s\S]*?)\n\s*\)\n/);
  if (!block) throw new Error("could not find `prompt = (...)` in little_coder_agent.py");
  return extractQuotedStrings(block[1]).map(decodePyString).join("");
}

function harborPrompt(instruction: string): string {
  return harborPromptTemplate().replace("{instruction}", instruction);
}

function harborDefaultAllowedTools(): string[] {
  const src = readFileSync(
    join(repoRoot(), "benchmarks", "harbor_adapter", "little_coder_agent.py"),
    "utf-8",
  );
  const block = src.match(/DEFAULT_ALLOWED_TOOLS = \[([^\]]*)\]/);
  if (!block) throw new Error("could not find DEFAULT_ALLOWED_TOOLS in little_coder_agent.py");
  return extractQuotedStrings(block[1]);
}

function gaiaAllowedTools(): string[] {
  const src = readFileSync(join(repoRoot(), "benchmarks", "gaia.py"), "utf-8");
  const block = src.match(/\nALLOWED_TOOLS = \[([\s\S]*?)\n\]\n/);
  if (!block) throw new Error("could not find ALLOWED_TOOLS in gaia.py");
  return extractQuotedStrings(block[1]);
}

describe("research directive gates on browse-tool availability", () => {
  it("DEFAULT_ALLOWED_TOOLS is ShellSession-only (sanity check on the extraction itself)", () => {
    expect(harborDefaultAllowedTools()).toEqual([
      "ShellSession",
      "ShellSessionCwd",
      "ShellSessionReset",
    ]);
  });

  it("does not inject the directive for a real Harbor trial (ShellSession-only allow-list), even when the task itself says 'research'", async () => {
    const handler = handlerFor(setupSkillInject);
    const prompt = harborPrompt("Research the article on Wikipedia and summarize its key points.");
    const event = turn(prompt);
    event.systemPromptOptions.littleCoder.allowedTools = harborDefaultAllowedTools();

    const result = await handler(event, ctx);

    expect(result?.message?.content ?? "").not.toContain("## Research-first directive");
  });

  it("still injects the directive for the same research-shaped task with no allow-list", async () => {
    const handler = handlerFor(setupSkillInject);
    const prompt = harborPrompt("Research the article on Wikipedia and summarize its key points.");
    const result = await handler(turn(prompt), ctx);

    expect(result?.message.content).toContain("## Research-first directive");
  });

  it("injects the directive with the EvidenceAdd steps for a GAIA-shaped prompt + GAIA's allow-list", async () => {
    const handler = handlerFor(setupSkillInject);
    const event = turn("GAIA research question: look up the answer on Wikipedia and cite your source.");
    event.systemPromptOptions.littleCoder.allowedTools = gaiaAllowedTools();

    const result = await handler(event, ctx);

    expect(result?.message.content).toContain("## Research-first directive");
    expect(result.message.content).toContain("EvidenceAdd");
  });

  it("the real Harbor boilerplate no longer smells like a research task by itself", () => {
    expect(harborPromptTemplate()).not.toContain("briefly research the task");
    expect(looksLikeResearchTask(harborPromptTemplate())).toBe(false);
  });

  it("looksLikeResearchTask still catches genuine research phrasing", () => {
    expect(looksLikeResearchTask("please research online for the answer")).toBe(true);
    expect(looksLikeResearchTask("look up the capital of France")).toBe(true);
    expect(looksLikeResearchTask("check wikipedia for details")).toBe(true);
  });

  it("requires websearch, or both browser tools, before injecting", () => {
    const shellOnly = new Set(harborDefaultAllowedTools());
    const gaia = new Set(gaiaAllowedTools());

    expect(shouldInjectResearchDirective("research this online", shellOnly)).toBe(false);
    expect(shouldInjectResearchDirective("research this online", gaia)).toBe(true);
    expect(shouldInjectResearchDirective("research this online", undefined)).toBe(true);
    // Partial availability (only websearch, say) is still enough to fire.
    expect(shouldInjectResearchDirective("research this online", new Set(["websearch"]))).toBe(true);
    expect(shouldInjectResearchDirective("edit the file", shellOnly)).toBe(false);
    // BrowserNavigate alone can't gather page text (no body in its output).
    expect(
      shouldInjectResearchDirective("research this online", new Set(["BrowserNavigate"])),
    ).toBe(false);
    // BrowserExtract alone reads an unnavigated about:blank session.
    expect(
      shouldInjectResearchDirective("research this online", new Set(["BrowserExtract"])),
    ).toBe(false);
    // The pair together is genuinely actionable.
    expect(
      shouldInjectResearchDirective(
        "research this online",
        new Set(["BrowserNavigate", "BrowserExtract"]),
      ),
    ).toBe(true);
    // Other browser tools alongside BrowserNavigate don't substitute for
    // BrowserExtract.
    expect(
      shouldInjectResearchDirective(
        "research this online",
        new Set(["BrowserNavigate", "BrowserClick", "BrowserScroll"]),
      ),
    ).toBe(false);
    // websearch plus a lone browser tool still fires, via the websearch leg.
    expect(
      shouldInjectResearchDirective("research this online", new Set(["websearch", "BrowserNavigate"])),
    ).toBe(true);
  });

  it("does not inject the directive for a browse allow-list that cannot actually browse (BrowserNavigate alone)", async () => {
    const handler = handlerFor(setupSkillInject);
    const prompt = "Research the article on Wikipedia and summarize its key points.";
    const event = turn(prompt);
    event.systemPromptOptions.littleCoder.allowedTools = ["BrowserNavigate", "ShellSession"];

    const result = await handler(event, ctx);

    expect(result?.message?.content ?? "").not.toContain("## Research-first directive");
  });
});

describe("knowledge-inject still injects after the #73 conversion", () => {
  // Scoring is word=1.0 / phrase=2.0 against MIN_SCORE_THRESHOLD=2.0, so the
  // prompt needs one phrase keyword or two single-word ones from a shipped
  // skills/knowledge entry. "dynamic programming" is a phrase keyword of
  // skills/knowledge/dynamic_programming.md.
  const PROMPT = "use dynamic programming to memoize this subproblem";

  async function inject(handler: Handler) {
    return handler(turn(PROMPT), ctx);
  }

  it("delivers algorithm reference entries as a hidden tail message", async () => {
    const handler = handlerFor(setupKnowledgeInject);
    const result = await inject(handler);

    expect(result, "no knowledge entry scored above threshold").toBeDefined();
    expect(result.message.customType).toBe("lc-knowledge");
    expect(result.message.display).toBe(false);
    expect(result.message.content).toContain("## Algorithm Reference");
    expect(result.systemPrompt).toBeUndefined();
  });

  it("falls back to the system prompt under LITTLE_CODER_INJECT_MODE=system", async () => {
    process.env.LITTLE_CODER_INJECT_MODE = "system";
    const handler = handlerFor(setupKnowledgeInject);
    const result = await inject(handler);

    expect(result, "no knowledge entry scored above threshold").toBeDefined();
    expect(result.message).toBeUndefined();
    expect(result.systemPrompt.startsWith("BASE SYSTEM PROMPT")).toBe(true);
    expect(result.systemPrompt).toContain("## Algorithm Reference");
  });
});
