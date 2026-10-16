// The two container-side snapshot locations benchmarks/harbor_adapter/
// little_coder_agent.py stages, plus the signal saying whether the
// start-of-trial one actually exists on this trial.
//
// The path strings mirror that file's SNAPSHOT_PUBLISH_PATH and
// INITIAL_SNAPSHOT_PUBLISH_PATH. Nothing spans the Python/TypeScript
// boundary to enforce that, so the lockstep is comment-level only -- the
// same arrangement as SNAPSHOT_LEAD_SEC <-> WARN_REMAINING_MS. Collecting
// them here at least means every TS mention of a snapshot path moves in one
// edit instead of drifting file by file.
export const DEADLINE_SNAPSHOT_DIR = "/tmp/.lc-snapshot";
export const INITIAL_SNAPSHOT_APP_DIR = "/tmp/.lc-initial/app";

/** The two outcomes for which the harbor adapter confirmed files were staged. */
export type InitialSnapshotOutcome = "succeeded" | "partial";

/**
 * The start-of-trial copy's classified outcome, or undefined when no copy is
 * known to be there.
 *
 * The adapter sets LITTLE_CODER_INITIAL_SNAPSHOT non-empty only for an
 * outcome it classified from a real container-side file count, and only for
 * the two outcomes that mean files were staged; it exports the var as
 * explicitly empty otherwise, so a value leaked from an earlier trial in the
 * same process can't stand. Undefined here covers TB1.0 (whose
 * adapter stages no initial snapshot at all), interactive pi use, and any
 * harbor trial whose staging refused or failed. Callers must stay silent
 * about the path then: pointing a model at a directory that may not exist
 * costs it turns for nothing, which is the same rule
 * _initial_snapshot_advertisement applies to the initial prompt.
 *
 * Read at call time rather than at module load, mirroring deadline.ts, so a
 * test can set the var after the extension has been imported.
 */
export function initialSnapshotOutcome(): InitialSnapshotOutcome | undefined {
  const raw = process.env.LITTLE_CODER_INITIAL_SNAPSHOT;
  // Exact match on the two staged-something outcomes -- the adapter's
  // explicit empty "no copy staged" value falls through without a separate
  // check, as does a stray empty export.
  return raw === "succeeded" || raw === "partial" ? raw : undefined;
}
