// Test support. Extension tuning knobs are read from the environment when the
// object that uses them is constructed, so a suite asserting against their
// defaults has to neutralize whatever the surrounding shell or CI exports.
// Shared because more than one suite needs the same save/clear/restore, and a
// partial copy of it silently retunes the code under test instead of failing.

/**
 * Handle over `names`: `clear()` records and removes them, `restore()` puts
 * back exactly what was there — including "was not set".
 */
export function pinEnv(names: readonly string[]): { clear(): void; restore(): void } {
  let saved: (string | undefined)[] = [];
  return {
    clear() {
      saved = names.map((n) => process.env[n]);
      for (const n of names) delete process.env[n];
    },
    restore() {
      names.forEach((n, i) => {
        const value = saved[i];
        if (value === undefined) delete process.env[n];
        else process.env[n] = value;
      });
    },
  };
}
