import { describe, it, expect, beforeEach } from "vitest";
import { assessResponse, buildCorrectionMessage, phraseForUser } from "./quality.ts";
import setupQualityMonitor from "./index.ts";

const known = new Set(["Read", "Write", "Edit", "Bash", "Glob", "Grep"]);

describe("assessResponse", () => {
  it("accepts text-only assistant response", () => {
    expect(assessResponse("here's my thinking", [], [], known)).toEqual({ ok: true });
  });
  it("accepts valid tool calls", () => {
    const calls = [{ name: "Read", input: { file_path: "/a" } }];
    expect(assessResponse("", calls, [], known)).toEqual({ ok: true });
  });
  it("detects empty response (no text, no calls)", () => {
    expect(assessResponse("", [], [], known)).toEqual({
      ok: false, reason: "empty_response",
    });
  });
  it("detects empty tool name", () => {
    expect(assessResponse("", [{ name: "", input: {} }], [], known)).toEqual({
      ok: false, reason: "empty_tool_name",
    });
  });
  it("detects hallucinated tool name", () => {
    const result = assessResponse("", [{ name: "FakeTool", input: {} }], [], known);
    expect(result).toEqual({ ok: false, reason: "unknown_tool:FakeTool" });
  });
  it("skips hallucination check when registry empty", () => {
    expect(
      assessResponse("", [{ name: "Anything", input: {} }], [], new Set()),
    ).toEqual({ ok: true });
  });
  it("detects repeated tool call", () => {
    const now = [{ name: "Read", input: { file_path: "/a" } }];
    const prev = [{ name: "Read", input: { file_path: "/a" } }];
    expect(assessResponse("", now, prev, known)).toEqual({
      ok: false, reason: "repeated_tool_call", offendingCall: now[0],
    });
  });
  it("does not flag as repeat when inputs differ", () => {
    const now = [{ name: "Read", input: { file_path: "/a" } }];
    const prev = [{ name: "Read", input: { file_path: "/b" } }];
    expect(assessResponse("", now, prev, known)).toEqual({ ok: true });
  });
  it("does not flag a re-run build command when the prior turn also edited a file (#81)", () => {
    const build = { name: "Bash", input: { command: "npm run build" } };
    const now = [build];
    // Prior turn: fixed a source file AND ran the build — re-running the build
    // now is progress, not a loop, because the Edit changed the environment.
    const prev = [{ name: "Edit", input: { file_path: "/src/main.ts" } }, build];
    expect(assessResponse("", now, prev, known)).toEqual({ ok: true });
  });
  it("still flags a verbatim repeat when nothing else changed", () => {
    const build = { name: "Bash", input: { command: "npm run build" } };
    expect(assessResponse("", [build], [build], known)).toEqual({
      ok: false, reason: "repeated_tool_call", offendingCall: build,
    });
  });
  it("still flags a repeat when the only other calls are read-only (#81)", () => {
    const build = { name: "Bash", input: { command: "npm run build" } };
    const now = [build];
    const prev = [{ name: "Read", input: { file_path: "/log" } }, build];
    expect(assessResponse("", now, prev, known)).toEqual({
      ok: false, reason: "repeated_tool_call", offendingCall: build,
    });
  });
  it("names the correct offender, not just the first array match, when an earlier same-turn call is exempted (#81 interaction)", () => {
    // prev turn ran Bash(X) then Read(Y). Current turn repeats both, in the
    // OPPOSITE order (Read(Y) checked first). Read(Y)'s repeat is exempted --
    // Bash(X) is present in prev, is state-changing, and isn't Read(Y) itself,
    // so the environment plausibly changed since Read(Y) last ran. Bash(X)'s
    // own repeat is NOT exempted: Read(Y) isn't state-changing, and Bash(X) is
    // excluded from exempting itself. So assessResponse's real control flow
    // (checks Read(Y) first, finds it exempted, continues to Bash(X), flags
    // that) must return Bash(X) as the offender -- a naive re-derivation that
    // just finds the first tc with ANY exact match in prev, ignoring the
    // exemption, would wrongly name Read(Y) instead (it's first in `now`).
    const bashX = { name: "Bash", input: { command: "run-x" } };
    const readY = { name: "Read", input: { file_path: "/y" } };
    const prev = [bashX, readY];
    const now = [readY, bashX]; // readY checked first; bashX is the real offender
    const result = assessResponse("", now, prev, known);
    expect(result).toEqual({ ok: false, reason: "repeated_tool_call", offendingCall: bashX });
  });
  it("detects malformed args sentinel", () => {
    const calls = [{ name: "Read", input: { _raw: "garbage" } }];
    expect(assessResponse("", calls, [], known)).toEqual({
      ok: false, reason: "malformed_args:Read",
    });
  });
});

describe("buildCorrectionMessage", () => {
  it("generates empty-response message", () => {
    const m = buildCorrectionMessage("empty_response");
    expect(m).toContain("empty");
  });
  it("generates unknown-tool message with tool name", () => {
    const m = buildCorrectionMessage("unknown_tool:FakeTool");
    expect(m).toContain("'FakeTool'");
    expect(m).toContain("does not exist");
  });
  it("generates malformed-args message", () => {
    const m = buildCorrectionMessage("malformed_args:Read");
    expect(m).toContain("'Read'");
    expect(m).toContain("malformed");
  });
  it("generates repeated-tool-call message", () => {
    const m = buildCorrectionMessage("repeated_tool_call");
    expect(m).toContain("loop");
  });
  it("falls back to generic on unknown reason", () => {
    expect(buildCorrectionMessage("weird_thing")).toContain("weird_thing");
  });
});

describe("phraseForUser", () => {
  it("phrases known reasons in plain language", () => {
    expect(phraseForUser("empty_response")).toMatch(/empty response/i);
    expect(phraseForUser("repeated_tool_call")).toMatch(/repeated/i);
  });
  it("includes the tool name for parameterized reasons", () => {
    expect(phraseForUser("unknown_tool:Frobnicate")).toContain("Frobnicate");
    expect(phraseForUser("malformed_args:Edit")).toContain("Edit");
  });
});

// ── turn_end handler: must skip interrupted/aborted turns ───────────────────
function harness() {
  const handlers: Record<string, ((e: any, c: any) => any)[]> = {};
  const followUps: { msg: string; opts: any }[] = [];
  const pi = {
    handlers,
    on(name: string, h: (e: any, c: any) => any) {
      (handlers[name] ??= []).push(h);
    },
    sendUserMessage(msg: string, opts: any) {
      followUps.push({ msg, opts });
    },
  };
  const notifies: string[] = [];
  const ctx = { ui: { notify: (m: string) => notifies.push(m) } };
  setupQualityMonitor(pi as any);
  return { pi, ctx, followUps, notifies };
}
async function fire(h: any, name: string, event: any) {
  for (const fn of h.pi.handlers[name] ?? []) await fn(event, h.ctx);
}
// Mirrors pi's emitToolCall dispatch: every handler runs, the first truthy
// `block` short-circuits.
async function fireToolCall(h: any, toolName: string, input: unknown) {
  for (const fn of h.pi.handlers["tool_call"] ?? []) {
    const r = await fn({ type: "tool_call", toolCallId: "t1", toolName, input }, h.ctx);
    if (r?.block) return r;
  }
  return undefined;
}
// Fires the tool_call events a real turn making `calls` would fire BEFORE
// its turn_end -- needed for anything that exercises blockedCall, since it's
// armed from the validated-space recording tool_call builds, not from
// turn_end's own raw extraction. `overrideInputs[i]`, if given, is what the
// tool_call event reports for calls[i] (its "validated/coerced" input),
// letting a test simulate coercion diverging from the raw content-block
// value turn(calls) below will separately carry.
async function fireTurn(
  h: any,
  calls: { name: string; input: unknown }[],
  text = "",
  overrideInputs: unknown[] = [],
) {
  for (let i = 0; i < calls.length; i++) {
    await fireToolCall(h, calls[i].name, overrideInputs[i] ?? calls[i].input);
  }
  return fire(h, "turn_end", turn(calls, text));
}
// turn_end event for a completed turn making the given tool calls. pi's
// content blocks carry raw `arguments`, which is what the extension reads.
function turn(calls: { name: string; input: unknown }[], text = "") {
  return {
    message: {
      stopReason: "stop",
      content: [
        ...(text ? [{ type: "text", text }] : []),
        ...calls.map((c) => ({ type: "toolCall", name: c.name, arguments: c.input })),
      ],
    },
  };
}

describe("quality-monitor turn_end", () => {
  let h: ReturnType<typeof harness>;
  beforeEach(async () => {
    h = harness();
    await fire(h, "session_start", {}); // reset session-scoped counters
  });

  it("skips an aborted/interrupted turn — no empty_response correction", async () => {
    // An ESC interrupt or harness abort produces a partial/empty message with
    // stopReason "aborted". This is the escape-interrupt bug: it must NOT steer
    // a 'your previous response was empty' correction onto the next prompt.
    await fire(h, "turn_end", { message: { stopReason: "aborted", content: [] } });
    expect(h.followUps).toHaveLength(0);
    expect(h.notifies).toHaveLength(0);
  });

  it("flags a genuinely empty COMPLETED turn and steers a correction", async () => {
    await fire(h, "turn_end", { message: { stopReason: "stop", content: [] } });
    expect(h.followUps).toHaveLength(1);
    expect(h.followUps[0].opts).toEqual({ deliverAs: "steer" });
    expect(h.notifies[0]).toMatch(/harness intervention:/i);
  });

  it("passes a normal text turn without intervention", async () => {
    await fire(h, "turn_end", {
      message: { stopReason: "stop", content: [{ type: "text", text: "done." }] },
    });
    expect(h.followUps).toHaveLength(0);
    expect(h.notifies).toHaveLength(0);
  });

  it("does not steer an empty-response correction on a provider error turn (#86)", async () => {
    // A 400 from the backend arrives as stopReason "error" with empty content.
    // It must NOT be treated as an empty model response and re-sent.
    await fire(h, "turn_end", {
      message: {
        stopReason: "error",
        errorMessage: '400 "gemma-4-26B" does not support thinking',
        content: [],
      },
    });
    expect(h.followUps).toHaveLength(0);
  });

  it("adds a one-time hint when the error is a thinking rejection (#86)", async () => {
    await fire(h, "turn_end", {
      message: {
        stopReason: "error",
        errorMessage: '400 "gemma-4-26B" does not support thinking',
        content: [],
      },
    });
    // No correction is steered, but a user-facing hint about the thinking level fires.
    expect(h.followUps).toHaveLength(0);
    expect(h.notifies.join("\n")).toMatch(/thinking level/i);
  });

  it("stays silent on an unrelated provider error (#86)", async () => {
    await fire(h, "turn_end", {
      message: { stopReason: "error", errorMessage: "500 upstream timeout", content: [] },
    });
    expect(h.followUps).toHaveLength(0);
    expect(h.notifies).toHaveLength(0);
  });
});

// ── tier-2 escalation: block the looping call instead of going silent ───────
describe("quality-monitor tier-2 escalation", () => {
  let h: ReturnType<typeof harness>;
  const bash = { name: "Bash", input: { command: "gcc encode.c" } };

  beforeEach(async () => {
    h = harness();
    await fire(h, "session_start", {});
  });

  // Drive `n` consecutive verbatim repeats of `bash` (the first turn seeds the
  // previous-turn state and is itself ok), firing realistic tool_call events
  // ahead of each turn_end the way pi actually would.
  async function repeatLoop(n: number) {
    await fireTurn(h, [bash]);
    for (let i = 0; i < n; i++) await fireTurn(h, [bash]);
  }

  it("keeps tier 1 unchanged for the first two failures", async () => {
    await repeatLoop(2);
    expect(h.followUps).toHaveLength(2);
    expect(h.followUps[0].msg).toBe(buildCorrectionMessage("repeated_tool_call"));
    expect(h.followUps[1].msg).toBe(buildCorrectionMessage("repeated_tool_call"));
    expect(await fireToolCall(h, bash.name, bash.input)).toBeUndefined();
  });

  it("arms the block on the repeated call, not another call in the same turn", async () => {
    // Seed, then two tier-1 failures (not yet past the cap), then the turn
    // that first crosses it makes the repeat AND an unrelated new call -- so
    // this is the FIRST time the offending-call derivation runs, not a
    // re-arm of a block a single-call turn already set (which wouldn't
    // exercise the derivation at all: it would already be armed from before
    // this turn ran).
    await fireTurn(h, [bash]); // seed
    await fireTurn(h, [bash]); // fail 1 (tier 1)
    await fireTurn(h, [bash]); // fail 2 (tier 1)
    const other = { name: "Read", input: { file_path: "/new" } };
    await fireTurn(h, [bash, other]); // fail 3: crosses the cap

    expect(await fireToolCall(h, bash.name, bash.input)).toMatchObject({ block: true });
    expect(await fireToolCall(h, other.name, other.input)).toBeUndefined();
  });

  it("blocks the exact repeated call with a reason once tier 2 fires", async () => {
    await repeatLoop(3);
    const result = await fireToolCall(h, bash.name, bash.input);
    expect(result?.block).toBe(true);
    expect(typeof result?.reason).toBe("string");
    expect(result?.reason).toMatch(/repeat/i);
    expect(h.notifies.join("\n")).toMatch(/blocking the exact call/i);
    expect(h.followUps[2].msg).toMatch(/blocked/i);
    expect(h.followUps[2].opts).toEqual({ deliverAs: "steer" });
  });

  it("does not block a call with the same tool name but different input", async () => {
    await repeatLoop(3);
    expect(await fireToolCall(h, "Bash", { command: "ls" })).toBeUndefined();
  });

  it("matches in pi's validated-argument space, not turn_end's raw extraction", async () => {
    // A numeric arg submitted as the string "5000" and schema-coerced to the
    // number 5000 is what the tool_call event actually reports -- the space
    // blockedCall must be armed from, and the space the tool_call handler's
    // own comparison runs in. If arming used turn_end's raw ("5000") value
    // instead, this match would silently never fire.
    const coerced = { name: "Bash", input: { retries: 5000 } };
    const raw = { name: "Bash", input: { retries: "5000" } };
    await fireTurn(h, [raw], "", [coerced.input]); // seed
    await fireTurn(h, [raw], "", [coerced.input]); // fail 1 (tier 1)
    await fireTurn(h, [raw], "", [coerced.input]); // fail 2 (tier 1)
    await fireTurn(h, [raw], "", [coerced.input]); // fail 3 (crosses the cap)
    expect(await fireToolCall(h, coerced.name, coerced.input)).toMatchObject({ block: true });
  });

  it("escalates exactly once, however long the loop continues", async () => {
    await repeatLoop(3);
    const followUps = h.followUps.length;
    const notifies = h.notifies.length;
    await fireTurn(h, [bash]);
    await fireTurn(h, [bash]);
    expect(h.followUps).toHaveLength(followUps);
    expect(h.notifies).toHaveLength(notifies);
    // The block itself keeps firing per call.
    expect(await fireToolCall(h, bash.name, bash.input)).toMatchObject({ block: true });
  });

  it("keeps re-arming the block for whichever call is looping, even after the one-time message fired, within ONE continuous streak", async () => {
    // Two distinct Grep calls (not state-changing, unlike Bash) so a shift in
    // offender is reachable without an intervening `ok` verdict resetting
    // tier2Notified -- with Bash, the previous offender's mere presence in
    // the prior turn would itself exempt anything else from being flagged
    // (STATE_CHANGING_TOOLS's envChanged check), making a same-streak shift
    // impossible to construct realistically. This is why this case is
    // distinct from "re-arms for a different loop later in the same
    // session" below: no `ok` turn happens anywhere in this sequence, so
    // tier2Notified must stay true throughout, and only the fix under test
    // (re-arming blockedCall outside the tier2Notified gate) makes the block
    // follow the new offender instead of freezing on the first one.
    // NOTE: no assertion-only fireToolCall() calls between the fireTurn()s
    // below -- fireToolCall's own tool_call firing pushes into
    // turnValidatedCalls same as a real one would, and an extra probe call
    // for the same tool NAME as the next turn's real call would leave a
    // stale entry `.find` could match instead of the real one. Assert only
    // once, after the full sequence.
    const callA = { name: "Grep", input: { pattern: "foo" } };
    const callB = { name: "Grep", input: { pattern: "bar" } };
    await fireTurn(h, [callA]); // seed
    await fireTurn(h, [callA]); // fail 1 (tier 1)
    await fireTurn(h, [callA]); // fail 2 (tier 1)
    await fireTurn(h, [callA, callB]); // fail 3: A repeats (crosses cap) AND B appears for the first time
    const followUpsAfterTier2 = h.followUps.length;

    // Now the model repeats ONLY B (drops A) -- B matches the B from the
    // turn just above, and is not exempted (the only other entry in that
    // prior turn, A, isn't state-changing) -- so this is STILL a failure
    // (verdict.ok is never true across this whole sequence), and B becomes
    // the new offendingCall.
    await fireTurn(h, [callB]);

    expect(h.followUps).toHaveLength(followUpsAfterTier2); // no new message
    expect(await fireToolCall(h, callA.name, callA.input)).toBeUndefined(); // no longer armed
    expect(await fireToolCall(h, callB.name, callB.input)).toMatchObject({ block: true }); // now armed
  });

  it("clears the block and the escalation on a recovered turn", async () => {
    await repeatLoop(3);
    await fire(h, "turn_end", turn([], "I'll try something else."));
    expect(await fireToolCall(h, bash.name, bash.input)).toBeUndefined();

    // consecutiveFailures reset too: the next repeat gets tier 1 again.
    const before = h.followUps.length;
    await fireTurn(h, [bash]);
    await fireTurn(h, [bash]);
    expect(h.followUps.slice(before).map((f) => f.msg)).toEqual([
      buildCorrectionMessage("repeated_tool_call"),
    ]);
  });

  it("re-arms for a different loop later in the same session", async () => {
    await repeatLoop(3);
    await fire(h, "turn_end", turn([], "changing approach."));

    const other = { name: "Bash", input: { command: "make test" } };
    await fireTurn(h, [other]);
    await fireTurn(h, [other]);
    await fireTurn(h, [other]);
    await fireTurn(h, [other]);

    expect(await fireToolCall(h, other.name, other.input)).toMatchObject({ block: true });
    expect(await fireToolCall(h, bash.name, bash.input)).toBeUndefined();
  });

  it("arms the block on the real repeat, not another call sharing its tool name", async () => {
    // The previous turn's only call was bash ("gcc encode.c"). The turn that
    // crosses the cap issues [Bash("ls -la"), Bash("gcc encode.c")] -- same
    // tool name, only the second is the genuine verbatim repeat. A name-only
    // lookup for the validated counterpart would find "ls -la" (first in
    // array order) instead of the real offender.
    await fireTurn(h, [bash]); // seed
    await fireTurn(h, [bash]); // fail 1 (tier 1)
    await fireTurn(h, [bash]); // fail 2 (tier 1)
    const decoy = { name: "Bash", input: { command: "ls -la" } };
    await fireTurn(h, [decoy, bash]); // fail 3: decoy is new, bash is the real repeat

    expect(await fireToolCall(h, bash.name, bash.input)).toMatchObject({ block: true });
    expect(await fireToolCall(h, decoy.name, decoy.input)).toBeUndefined();
  });

  it("does not let an aborted or errored turn's calls leak into the next turn's offending-call lookup", async () => {
    // The arming turn must follow the abort IMMEDIATELY, with no ordinary
    // turn_end in between: an ordinary turn already consumes-and-clears
    // turnValidatedCalls on its own (both before and after this fix), so an
    // intervening one would flush the leak before it could matter and this
    // test would pass even against the pre-fix code.
    await fireTurn(h, [bash]); // seed
    await fireTurn(h, [bash]); // fail 1 (tier 1)
    await fireTurn(h, [bash]); // fail 2 (tier 1)
    // A turn that fires tool_call for an unrelated command, then aborts --
    // its recording must not survive into the next turn's lookup.
    const decoy = { name: "Bash", input: { command: "ls -la" } };
    await fireToolCall(h, decoy.name, decoy.input);
    await fire(h, "turn_end", { message: { stopReason: "aborted", content: [] } });

    await fireTurn(h, [bash]); // fail 3: crosses the cap, right after the abort

    expect(await fireToolCall(h, bash.name, bash.input)).toMatchObject({ block: true });
    expect(await fireToolCall(h, decoy.name, decoy.input)).toBeUndefined();
  });

  it("still notifies once for a different failure reason arising after an earlier tier-2 notification", async () => {
    await repeatLoop(3); // tier 2 fires on bash; one escalation message sent
    const followUpsAfterTier2 = h.followUps.length;

    // Model can't retry bash; instead it emits an empty response 3 times in
    // a row -- a different failure reason within the same never-ok streak.
    const empty = { message: { stopReason: "stop", content: [] } };
    await fire(h, "turn_end", empty);
    await fire(h, "turn_end", empty);
    await fire(h, "turn_end", empty); // crosses the cap again, for a NEW reason

    expect(h.followUps).toHaveLength(followUpsAfterTier2 + 1); // one fresh notification
    expect(h.followUps[h.followUps.length - 1].msg).toBe(buildCorrectionMessage("empty_response"));

    // A repeat of the SAME (now-notified) reason still doesn't re-notify.
    await fire(h, "turn_end", empty);
    expect(h.followUps).toHaveLength(followUpsAfterTier2 + 1);
  });

  it("blocks unconditionally -- no same-turn state-changing call exempts it", async () => {
    // A same-turn Edit (or any other call) before the retry must NOT exempt
    // it: any exemption based only on "some state-changing call happened
    // this turn" can't verify that call was relevant or even that it
    // succeeded, and would let a looping model permanently defeat the block
    // by prefixing every retry with a trivial one (even a no-op).
    await repeatLoop(3); // tier 2 fires and blocks bash
    const edit = { name: "Edit", input: { file_path: "/src/main.c" } };
    expect(await fireToolCall(h, edit.name, edit.input)).toBeUndefined();
    expect(await fireToolCall(h, bash.name, bash.input)).toMatchObject({ block: true });
    // And the block survives indefinitely -- prefixing every subsequent
    // attempt with its own fresh state-changing call doesn't wear it down.
    const noop = { name: "Bash", input: { command: "echo hi" } };
    expect(await fireToolCall(h, noop.name, noop.input)).toBeUndefined();
    expect(await fireToolCall(h, bash.name, bash.input)).toMatchObject({ block: true });
  });

  it("does not push toward a guardrail-bypass route in the escalation message", async () => {
    // The escalation message must not read as an unqualified "find another
    // way" -- that is exactly the interpreter-bypass hunt permission-gate's
    // own refusal text exists to stop (issue #94). This pins the presence of
    // the carve-out, not just that SOME escalation text was sent.
    await repeatLoop(3);
    const escalation = h.followUps[h.followUps.length - 1].msg;
    expect(escalation).toMatch(/refus/i);
    expect(escalation).toMatch(/python3|node -e|env|-exec/i);
  });

  it("resets the block and streak on a genuinely new prompt", async () => {
    await repeatLoop(3);
    // A real new prompt: no `source`/`streamingBehavior` marking it as our
    // own steer or a mid-turn interjection.
    await fire(h, "input", { text: "let's do something else" });
    expect(await fireToolCall(h, bash.name, bash.input)).toBeUndefined();

    const before = h.followUps.length;
    await fireTurn(h, [bash]);
    await fireTurn(h, [bash]);
    expect(h.followUps.slice(before).map((f) => f.msg)).toEqual([
      buildCorrectionMessage("repeated_tool_call"),
    ]);
  });

  it("does not reset on its own steer message's input event", async () => {
    // pi.sendUserMessage() (ours, tier 2's own escalation) routes through the
    // same prompt() pi's real "input" event fires from, tagged source
    // "extension" -- without excluding that, the escalation we just sent
    // would immediately undo the block it exists to set.
    await repeatLoop(3);
    await fire(h, "input", { source: "extension", text: "..." });
    expect(await fireToolCall(h, bash.name, bash.input)).toMatchObject({ block: true });
  });

  it("does not reset on a mid-turn steer from the user", async () => {
    await repeatLoop(3);
    await fire(h, "input", { source: "interactive", streamingBehavior: "queue", text: "..." });
    expect(await fireToolCall(h, bash.name, bash.input)).toMatchObject({ block: true });
  });

  it("keeps correcting past the cap for reasons with no call to block", async () => {
    const empty = { message: { stopReason: "stop", content: [] } };
    await fire(h, "turn_end", empty);
    await fire(h, "turn_end", empty);
    await fire(h, "turn_end", empty); // 3rd failure: used to go silent here
    expect(h.followUps).toHaveLength(3);
    expect(h.followUps[2].msg).toBe(buildCorrectionMessage("empty_response"));
    expect(h.followUps[2].opts).toEqual({ deliverAs: "steer" });
    expect(h.notifies[2]).toMatch(/still correcting/i);
    expect(await fireToolCall(h, "Bash", {})).toBeUndefined();

    // Still only one escalation.
    await fire(h, "turn_end", empty);
    expect(h.followUps).toHaveLength(3);
  });

  it("claims no repeat count when the streak mixed other failure reasons", async () => {
    // knownTools is populated by observed executions; without it the
    // unknown-tool check is skipped.
    await fire(h, "tool_execution_start", { toolName: "Bash" });
    const ghost = { name: "Ghost", input: {} };
    await fire(h, "turn_end", turn([bash]));
    await fire(h, "turn_end", turn([ghost, bash])); // unknown_tool
    await fire(h, "turn_end", turn([ghost, bash])); // unknown_tool
    await fire(h, "turn_end", turn([bash])); // first actual repeat -> tier 2

    const escalation = h.followUps[h.followUps.length - 1].msg;
    expect(escalation).toMatch(/blocked/i);
    // consecutiveFailures is 3 here but the call repeated once: no count
    // claim (not a bare digit check -- the message's own "python3" mention
    // would false-positive on that).
    expect(escalation).not.toMatch(/times|in a row/i);
    expect(await fireToolCall(h, bash.name, bash.input)).toMatchObject({ block: true });
  });
});
