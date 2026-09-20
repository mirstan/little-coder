import { describe, it, expect, afterEach } from "vitest";
import {
  DEADLINE_SNAPSHOT_DIR,
  INITIAL_SNAPSHOT_APP_DIR,
  initialSnapshotOutcome,
} from "./snapshot-paths.ts";

describe("snapshot paths", () => {
  it("match the paths little_coder_agent.py publishes to", () => {
    // Hand-kept lockstep with that file's SNAPSHOT_PUBLISH_PATH and
    // INITIAL_SNAPSHOT_PUBLISH_PATH + "/app" (its stage command's
    // `cp --parents` is what keeps the /app prefix).
    expect(DEADLINE_SNAPSHOT_DIR).toBe("/tmp/.lc-snapshot");
    expect(INITIAL_SNAPSHOT_APP_DIR).toBe("/tmp/.lc-initial/app");
  });
});

describe("initialSnapshotOutcome", () => {
  afterEach(() => {
    delete process.env.LITTLE_CODER_INITIAL_SNAPSHOT;
  });

  it("returns the outcome for the two values that mean files were staged", () => {
    for (const outcome of ["succeeded", "partial"] as const) {
      process.env.LITTLE_CODER_INITIAL_SNAPSHOT = outcome;
      expect(initialSnapshotOutcome()).toBe(outcome);
    }
  });

  it("is undefined when unset — a TB1.0 trial, or plain interactive use", () => {
    expect(initialSnapshotOutcome()).toBeUndefined();
  });

  it("is undefined for the outcomes that staged nothing", () => {
    for (const outcome of ["failed", "refused"]) {
      process.env.LITTLE_CODER_INITIAL_SNAPSHOT = outcome;
      expect(initialSnapshotOutcome()).toBeUndefined();
    }
  });

  it("is undefined for an empty or unrecognized value", () => {
    // An empty-but-exported var is not a deliberate anything, same reading
    // deadline.ts's own resolver gives it.
    for (const raw of ["", "   ", "SUCCEEDED", "yes", "1"]) {
      process.env.LITTLE_CODER_INITIAL_SNAPSHOT = raw;
      expect(initialSnapshotOutcome()).toBeUndefined();
    }
  });
});
