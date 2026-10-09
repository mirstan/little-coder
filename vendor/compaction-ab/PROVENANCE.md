# Vendored third-party pi extensions (compaction A/B experiment only)

The files are copied verbatim from the npm tarballs. Each tarball's sha512 matches the registry's `dist.integrity` (checked 2026-10-09):

| Package | Registry integrity |
|---|---|
| pi-blackhole@0.5.12 | sha512-Py9Dv01rv6Fz4pnVqNJWGemmUralYbFnkQC1BRyoEddsBtGCfPRiIKX5BYmLZG3ukSwY2PXLItw4k284kuySuQ== |
| pi-prefix-cache-compaction@0.3.0 | sha512-Ju2crTiGiY7opOZ7mq3omd3L0z7I1NXp88KLIczwp0InQ5Utz3+1H6VL5Sd1NvoL5zItJKMR8Kj7tPnT4XABhA== |

- Only the files pi loads are kept. For blackhole that is `dist/index.js`; for prefix-cache-compaction it is `src/index.ts` and `src/core.ts`. Each package also keeps its package.json and LICENSE (MIT).
- `SHA256SUMS` covers every vendored file. `npm install` is never run for these packages.
- They are loaded only through the env-gated shims in `.pi/extensions/zz-arm-*`, and only when `LC_COMPACTION_ARM` names that arm.
- `config/` holds the experiment's pinned configs.
