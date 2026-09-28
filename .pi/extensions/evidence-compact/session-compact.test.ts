import { describe, it, expect, vi, beforeEach } from "vitest";

// Drive the actual session_compact handler (not just the pure BRIDGE_TEMPLATE
// function covered in bridge.test.ts).

const getSessionStoreMock = vi.fn();
vi.mock("../evidence/index.ts", () => ({
  getSessionStore: (...args: unknown[]) => getSessionStoreMock(...args),
}));

import setupEvidenceCompact from "./index.ts";

function setup() {
  const handlers: Record<string, Function> = {};
  const sendUserMessage = vi.fn();
  const pi = {
    on: (evt: string, h: Function) => {
      handlers[evt] = h;
    },
    sendUserMessage,
  };
  setupEvidenceCompact(pi as any);
  return { handlers, sendUserMessage };
}

describe("evidence-compact session_compact guard", () => {
  beforeEach(() => {
    getSessionStoreMock.mockReset();
  });

  it("RPC + manual compaction: does not resume (the harness drives its own continuation)", async () => {
    getSessionStoreMock.mockReturnValue([{ id: "e1" }]);
    const { handlers, sendUserMessage } = setup();
    const ctx = { mode: "rpc", ui: { notify: vi.fn() } };

    await handlers.session_compact({ reason: "manual" }, ctx);

    expect(sendUserMessage).not.toHaveBeenCalled();
  });

  it("RPC + threshold compaction: still resumes (automatic compaction needs the bridge message)", async () => {
    getSessionStoreMock.mockReturnValue([{ id: "e1" }]);
    const { handlers, sendUserMessage } = setup();
    const ctx = { mode: "rpc", ui: { notify: vi.fn() } };

    await handlers.session_compact({ reason: "threshold" }, ctx);

    expect(sendUserMessage).toHaveBeenCalledTimes(1);
  });

  it("non-RPC modes: resumes regardless of reason (unchanged, pre-existing behavior)", async () => {
    getSessionStoreMock.mockReturnValue([{ id: "e1" }]);

    for (const mode of ["tui", "print"]) {
      const { handlers, sendUserMessage } = setup();
      const ctx = { mode, ui: { notify: vi.fn() } };

      await handlers.session_compact({ reason: "manual" }, ctx);

      expect(sendUserMessage, `mode=${mode}`).toHaveBeenCalledTimes(1);
    }
  });

  it("empty evidence store: never resumes regardless of mode (pre-existing behavior)", async () => {
    getSessionStoreMock.mockReturnValue([]);
    const { handlers, sendUserMessage } = setup();
    const ctx = { mode: "rpc", ui: { notify: vi.fn() } };

    await handlers.session_compact({ reason: "manual" }, ctx);

    expect(sendUserMessage).not.toHaveBeenCalled();
  });
});
