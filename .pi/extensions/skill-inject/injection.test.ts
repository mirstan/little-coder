import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";
import setupSkillInject, {
  looksLikeResearchTask,
  looksLikeTemporalTask,
  shouldInjectResearchDirective,
  shouldInjectTemporalDirective,
} from "./index.ts";
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

// mteb-leaderboard trajectory: the model correctly noticed the task named a
// past date, but "answered" it by filtering today's live leaderboard by an
// unrelated proxy field (model release date) instead of finding an actual
// dated snapshot. The temporal directive exists to redirect that pattern
// toward a git revision / archived page instead.
describe("temporal directive triggers on phrasing that names a PAST state", () => {
  it("fires for genuine temporal-research phrasing (date/version-anchored)", () => {
    expect(looksLikeTemporalTask("what was the leaderboard as of August 2025")).toBe(true);
    expect(looksLikeTemporalTask("what was this repo as of the v2.0 release")).toBe(true);
    expect(looksLikeTemporalTask("the dataset's state at the time of the 2019 audit")).toBe(true);
    expect(looksLikeTemporalTask("as of 2024-05-01 what did the rankings look like")).toBe(true);
    expect(looksLikeTemporalTask("as of May 3, 2024, what were the standings")).toBe(true);
    expect(
      looksLikeTemporalTask("find an archived snapshot of the results page from 2024-01-15"),
    ).toBe(true);
    expect(looksLikeTemporalTask("what did the historical rankings look like?")).toBe(true);
    expect(looksLikeTemporalTask("what were the repositories' stars in 2021")).toBe(true);
    expect(looksLikeTemporalTask("as of commit abc123 what was in this repo")).toBe(true);
    expect(looksLikeTemporalTask("back in 2019 what was the leaderboard")).toBe(true);
    expect(looksLikeTemporalTask("leaderboard standings as of early 2024")).toBe(true);
    expect(looksLikeTemporalTask("as of mid-2022 what did the repo contain")).toBe(true);
    // Anaphoric case: no date in the "at the time" clause itself, but a real
    // date appears earlier in the prompt (the original mteb-leaderboard shape).
    expect(
      looksLikeTemporalTask(
        "The paper came out in June 2024. Which model led the leaderboard at the time?",
      ),
    ).toBe(true);
    expect(looksLikeTemporalTask("what was the price of bitcoin in March 2023")).toBe(true);
    // Must keep firing -- protects the existing gate tests below.
    expect(looksLikeTemporalTask("what was this as of last year?")).toBe(true);
    // Must keep firing both research+temporal directives (see the
    // both-directives test below).
    expect(looksLikeTemporalTask("please research the leaderboard as of March 2024")).toBe(true);
  });

  it("does not fire for dev-artifact snapshots, discourse adverbs, bare counts, or dateless anchors", () => {
    expect(looksLikeTemporalTask("snapshot of memory usage right now")).toBe(false);
    expect(looksLikeTemporalTask("let's take a snapshot of the docker image")).toBe(false);
    expect(looksLikeTemporalTask("the snapshot test in CI is flaky")).toBe(false);
    expect(looksLikeTemporalTask("as of today the build is green")).toBe(false);
    expect(looksLikeTemporalTask("as of now the tests pass")).toBe(false);
    expect(looksLikeTemporalTask("as of yesterday the CI is red, please fix the tests")).toBe(
      false,
    );
    expect(looksLikeTemporalTask("historically we used tabs not spaces")).toBe(false);
    expect(looksLikeTemporalTask("historic buildings from the 1800s")).toBe(false);
    // No date anywhere in the prompt -- the anaphoric disjunct requires a
    // prompt-level date, so a bare "at the time" alone does not fire.
    expect(looksLikeTemporalTask("at the time, GPT-4 was the top model")).toBe(false);
    // Inverted from an old assertion: bare adverb no longer fires.
    expect(looksLikeTemporalTask("historically, this model ranked lower")).toBe(false);
    // Inverted from an old assertion: no date, no qualifier.
    expect(looksLikeTemporalTask("give me a snapshot of the results")).toBe(false);
    expect(looksLikeTemporalTask("batches of 2048")).toBe(false);
    expect(looksLikeTemporalTask("port 2000")).toBe(false);
    expect(looksLikeTemporalTask("resize to 1920x1080")).toBe(false);
    expect(looksLikeTemporalTask("load the dataset in 2048 chunks")).toBe(false);
    expect(looksLikeTemporalTask("the 2021 census dataset")).toBe(false);
    expect(looksLikeTemporalTask("split the dataset in 2000 buckets")).toBe(false);
    expect(looksLikeTemporalTask("run the dataset in 2020 workers")).toBe(false);
    expect(looksLikeTemporalTask("the 2024 dataset loader has an off-by-one bug")).toBe(false);
    expect(looksLikeTemporalTask("add a retry to the 2024 dataset loader")).toBe(false);
    expect(looksLikeTemporalTask("at the time of writing this is broken")).toBe(false);
    expect(looksLikeTemporalTask("at the time of the crash")).toBe(false);
    expect(looksLikeTemporalTask("at the time of the incident")).toBe(false);
    expect(looksLikeTemporalTask("at the time of the outage")).toBe(false);
    expect(looksLikeTemporalTask("at the time of the deployment")).toBe(false);
    expect(looksLikeTemporalTask("at the time of the error")).toBe(false);
    expect(looksLikeTemporalTask("prune the archived snapshots on the ZFS pool")).toBe(false);
    // A bare year elsewhere in the prompt must NOT arm the anaphoric
    // "at the time" check.
    expect(
      looksLikeTemporalTask("allocate 2048 buffers; what was happening at the time?"),
    ).toBe(false);
  });

  // Deliberate near-miss: "historic" (an adjective describing an old
  // building) is not "historical"/"historically" and must not be treated as
  // a signal that the task wants a past snapshot of live/versioned data.
  it("does not fire for 'historic building' (proper adjective use, not a temporal-research signal)", () => {
    expect(
      looksLikeTemporalTask("This town has several historic buildings from the 1800s."),
    ).toBe(false);
  });

  it("anaphoric 'at the time' requires a real date/version somewhere in the prompt", () => {
    expect(
      looksLikeTemporalTask(
        "The paper came out in June 2024. Which model led the leaderboard at the time?",
      ),
    ).toBe(true);
    expect(
      looksLikeTemporalTask("allocate 2048 buffers; what was happening at the time?"),
    ).toBe(false);
  });

  it("does not inject the directive when no git-capable/browse tool is available, even though the trigger matches", async () => {
    const readOnly = new Set(["read", "edit"]);
    expect(shouldInjectTemporalDirective("what was this as of last year?", readOnly)).toBe(false);

    const handler = handlerFor(setupSkillInject);
    const event = turn("what was this as of last year?");
    event.systemPromptOptions.littleCoder.allowedTools = ["read", "edit"];
    const result = await handler(event, ctx);

    expect(result?.message?.content ?? "").not.toContain("## Temporal-research directive");
  });

  it("injects when a git-capable shell tool is available even without any browse tool", async () => {
    const shellOnly = new Set(["ShellSession", "ShellSessionCwd", "ShellSessionReset"]);
    expect(shouldInjectTemporalDirective("what was this as of last year?", shellOnly)).toBe(true);

    const handler = handlerFor(setupSkillInject);
    const event = turn("what was this as of last year?");
    event.systemPromptOptions.littleCoder.allowedTools = [
      "ShellSession",
      "ShellSessionCwd",
      "ShellSessionReset",
    ];
    const result = await handler(event, ctx);

    expect(result?.message?.content ?? "").toContain("## Temporal-research directive");
    // ShellSession-only can't reach an archived web page, so the directive
    // should only mention the git route, not the web.archive.org one.
    expect(result?.message?.content ?? "").toContain("git log");
    expect(result?.message?.content ?? "").not.toContain("web.archive.org");
  });

  it("injects when only browse tools are available even without a shell tool", async () => {
    const browseOnly = new Set(["websearch"]);
    expect(shouldInjectTemporalDirective("what was this as of last year?", browseOnly)).toBe(true);

    const handler = handlerFor(setupSkillInject);
    const event = turn("what was this as of last year?");
    event.systemPromptOptions.littleCoder.allowedTools = ["websearch"];
    const result = await handler(event, ctx);

    expect(result?.message?.content ?? "").toContain("## Temporal-research directive");
    expect(result?.message?.content ?? "").toContain("web.archive.org");
    expect(result?.message?.content ?? "").not.toContain("git log");
  });

  it("fires both the research and temporal directives on a prompt that trips both, temporal last", async () => {
    const handler = handlerFor(setupSkillInject);
    const result = await handler(
      turn("please research the leaderboard as of March 2024"),
      ctx,
    );
    const content: string = result?.message?.content ?? "";

    expect(content).toContain("## Research-first directive");
    expect(content).toContain("## Temporal-research directive");
    // The more specific, corrective directive wins the recency argument.
    expect(content.indexOf("## Temporal-research directive")).toBeGreaterThan(
      content.indexOf("## Research-first directive"),
    );
  });

  it("skips a repeat of the identical temporal-directive block on the next turn", async () => {
    const handler = handlerFor(setupSkillInject);
    const first = await handler(turn("what was the leaderboard as of March 2024?"), ctx);
    expect(first?.message?.content ?? "").toContain("## Temporal-research directive");

    const second = await handler(turn("what was the leaderboard as of March 2024?"), ctx);
    expect(second).toBeUndefined();
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
