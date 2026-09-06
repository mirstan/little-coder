"""Historical-log ingestion + free trajectory-scoring report
(VALIDATION_PLAN.md Layers 2-3) -- moved out of run_gepa.py, which no
longer scores from frozen historical data at all (see
polyglot_adapter.py's module docstring for why that design was replaced by
a live-execution GEPAAdapter). This module still serves real ingestion
validation and reporting, independent of the live-eval loop.

Usage:
    python -m benchmarks.self_improve.report_trajectories \\
        --log-roots aider=<log_root>,<results.json> gaia=<log_root> \\
        --components-config benchmarks/self_improve/config/components.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from benchmarks.self_improve.components import load_components
from benchmarks.self_improve.ingest import aider_polyglot_ingest, gaia_ingest, harbor_tb_ingest
from benchmarks.self_improve.schema import NormalizedTrajectory

logger = logging.getLogger(__name__)

_KNOWN_LOG_ROOT_KEYS = {"aider", "gaia", "harbor", "tb"}
DEFAULT_WEIGHTS = {"aider_polyglot": 0.4, "gaia": 0.3, "harbor": 0.15, "tb": 0.15}


def parse_log_roots(entries: list[str]) -> dict[str, str]:
    """Parse --log-roots KEY=VALUE tokens, validating both the key and that
    a value is present. Real bug, confirmed by review: a malformed or
    unrecognized token was previously silently absorbed into log_roots under
    whatever partition() produced, then silently never matched any real
    source, with zero warning -- a confirmed run could believe a source was
    ingested when it was never even attempted."""
    log_roots: dict[str, str] = {}
    for entry in entries:
        key, sep, value = entry.partition("=")
        if not sep or not value:
            raise ValueError(f"Invalid --log-roots entry {entry!r}: expected KEY=VALUE")
        if key not in _KNOWN_LOG_ROOT_KEYS:
            raise ValueError(
                f"Invalid --log-roots entry {entry!r}: unknown key {key!r}, "
                f"expected one of {sorted(_KNOWN_LOG_ROOT_KEYS)}"
            )
        if key in log_roots:
            raise ValueError(
                f"Invalid --log-roots entry {entry!r}: duplicate key {key!r} "
                f"(already given as {log_roots[key]!r}) -- the earlier value would be "
                "silently dropped, omitting requested data from the report with no error"
            )
        log_roots[key] = value
    return log_roots


def ingest_all(
    log_roots: dict[str, str], repo_root: Path | None = None,
) -> tuple[list[NormalizedTrajectory], list[str]]:
    """log_roots keys: 'aider' (value: 'log_root,results_json'), 'gaia', 'harbor', 'tb'.
    repo_root resolves knowledge-inject component usage for aider/gaia (see
    ingest.common.build_knowledge_topic_index()).

    Returns (trajectories, empty_sources) -- empty_sources lists every
    REQUESTED key that yielded zero trajectories, whether because ingest
    raised or because it legitimately found nothing."""
    trajectories: list[NormalizedTrajectory] = []
    empty_sources: list[str] = []

    if "aider" in log_roots:
        raw = log_roots["aider"]
        log_root_str, sep, results_json_str = raw.partition(",")
        if not sep or not log_root_str or not results_json_str:
            logger.warning(
                "aider_polyglot ingest failed: --log-roots aider=%r is missing the "
                "required ',<results.json path>' suffix", raw,
            )
            empty_sources.append("aider")
        else:
            try:
                trajs = aider_polyglot_ingest.load(
                    Path(log_root_str), Path(results_json_str), repo_root=repo_root,
                )
                trajectories.extend(trajs)
                logger.info("ingested %d aider_polyglot trajectories", len(trajs))
                if not trajs:
                    empty_sources.append("aider")
            except Exception as e:
                logger.warning("aider_polyglot ingest failed: %s", e, exc_info=True)
                empty_sources.append("aider")

    if "gaia" in log_roots:
        try:
            trajs = gaia_ingest.load(Path(log_roots["gaia"]), repo_root=repo_root)
            trajectories.extend(trajs)
            logger.info("ingested %d gaia trajectories", len(trajs))
            if not trajs:
                empty_sources.append("gaia")
        except Exception as e:
            logger.warning("gaia ingest failed: %s", e, exc_info=True)
            empty_sources.append("gaia")

    for key in ("harbor", "tb"):
        if key in log_roots:
            try:
                trajs = harbor_tb_ingest.load(Path(log_roots[key]), benchmark=key)
                trajectories.extend(trajs)
                logger.info("ingested %d %s trajectories", len(trajs), key)
                if not trajs:
                    empty_sources.append(key)
            except Exception as e:
                logger.warning("%s ingest failed: %s", key, e, exc_info=True)
                empty_sources.append(key)

    return trajectories, empty_sources


def _score_for_benchmark(traj: NormalizedTrajectory) -> float:
    if traj.partial_score is not None:
        return traj.partial_score
    return 1.0 if traj.success else 0.0


def weighted_aggregate(per_benchmark_avg: dict[str, float], weights: dict[str, float]) -> float:
    """Combine per-benchmark average scores using weights, renormalized over
    only the benchmarks actually present in per_benchmark_avg."""
    present = {k: weights.get(k, 0.0) for k in per_benchmark_avg}
    total_weight = sum(present.values())
    if total_weight == 0:
        return sum(per_benchmark_avg.values()) / len(per_benchmark_avg) if per_benchmark_avg else 0.0
    return sum(per_benchmark_avg[k] * (present[k] / total_weight) for k in per_benchmark_avg)


def dry_run_report(
    trajectories: list[NormalizedTrajectory], components: dict[str, str], weights: dict[str, float],
) -> None:
    """Ingest-and-score report over frozen historical data: per-benchmark
    average score, the weighted aggregate, and per-component usage counts.
    Pure reporting -- no GEPA, no LM, no live execution."""
    print(f"\n=== Trajectory report: {len(trajectories)} trajectories, {len(components)} components ===\n")

    if not trajectories:
        print("WARNING: no trajectories ingested -- nothing to score.")
        return

    per_benchmark_scores: dict[str, list[float]] = {}
    per_component_usage_counts: dict[str, int] = {}

    for traj in trajectories:
        per_benchmark_scores.setdefault(traj.benchmark, []).append(_score_for_benchmark(traj))
        for usage in traj.components_used:
            if usage.pred_name not in components:
                continue
            per_component_usage_counts[usage.pred_name] = per_component_usage_counts.get(usage.pred_name, 0) + 1

    per_benchmark_avg = {b: sum(scores) / len(scores) for b, scores in per_benchmark_scores.items()}
    aggregate = weighted_aggregate(per_benchmark_avg, weights)

    print("Per-benchmark trajectory counts:")
    for b, scores in sorted(per_benchmark_scores.items()):
        print(f"  {b}: {len(scores)} trajectories, avg score {per_benchmark_avg[b]:.3f}")

    print(f"\nWeighted aggregate score: {aggregate:.3f}")
    assert 0.0 <= aggregate <= 1.0, f"aggregate out of [0,1] range: {aggregate}"

    print("\nPer-component usage counts:")
    if per_component_usage_counts:
        for name, count in sorted(per_component_usage_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {name}: {count}")
    else:
        print("  (none -- no trajectory used any known component)")

    print("\n=== Report complete (no live execution, no GEPA, no LM calls) ===")


def _resolve_components_yaml(repo_root: Path, components_config: str) -> Path:
    """components.yaml paths are conventionally repo-relative -- resolving
    against `cwd` instead silently breaks whenever --repo-root points
    somewhere other than the caller's own working directory."""
    rel = Path(components_config)
    if rel.is_absolute():
        rel = rel.relative_to(repo_root)
    return repo_root / rel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-roots", nargs="+", default=[],
                     help="key=value pairs, e.g. aider=<log_root>,<results.json> gaia=<log_root>")
    ap.add_argument("--components-config", default="benchmarks/self_improve/config/components.yaml")
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        log_roots = parse_log_roots(args.log_roots)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    repo_root = Path(args.repo_root).resolve()
    components_yaml = _resolve_components_yaml(repo_root, args.components_config)

    trajectories, _empty_sources = ingest_all(log_roots, repo_root=repo_root)
    components = load_components(components_yaml, repo_root=repo_root)

    weights_path = components_yaml.parent / "benchmark_weights.yaml"
    weights = DEFAULT_WEIGHTS
    if weights_path.exists():
        weights = yaml.safe_load(weights_path.read_text()) or DEFAULT_WEIGHTS

    dry_run_report(trajectories, components, weights)
    return 0


if __name__ == "__main__":
    sys.exit(main())
