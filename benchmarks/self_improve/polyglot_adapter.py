"""PolyglotGEPAAdapter: implements gepa.core.adapter.GEPAAdapter's contract
against PolyglotLiveRunner. GEPAAdapter is a typing.Protocol -- structural
conformance only, never isinstance-checked by the engine, so this class does
not (and need not) inherit from it.

Replaces the entire benchmarks/self_improve/metric.py + components.py
HarnessProgram apparatus: that whole design (ScoreWithFeedback's dual
calling convention, the dspy.Predict/DummyLM scaffolding, GEPA's trace-
completeness requirement) existed only to satisfy dspy.GEPA's wrapper
around a fundamentally different (and, confirmed by review, permanently
broken) frozen-historical-data scoring design. Here, scores are plain floats
straight out of a REAL live run, and feedback text is built directly from
that run's own real diff/pytest-output/transcript -- no vestigial trace
machinery required.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from gepa.core.adapter import EvaluationBatch

from benchmarks.self_improve.exercises import ExerciseSpec, describe_exercise
from benchmarks.self_improve.ingest.common import merge_component_usage
from benchmarks.self_improve.live_eval import LiveRunResult, PolyglotLiveRunner

_SCORING_RULE = (
    "Scoring: pass on attempt 1 = 1.00, attempt 2 = 0.70, attempt 3+ = 0.40, "
    "any failure/timeout/error = 0.00. Fewer attempts is strictly better."
)


@dataclass
class PolyglotTrajectory:
    spec: ExerciseSpec
    result: LiveRunResult


@dataclass
class PolyglotRolloutOutput:
    task_id: str
    status: str
    score: float
    from_cache: bool


def _component_feedback(pred_name: str, result: LiveRunResult, knowledge_topic_index: Mapping[str, str]) -> str:
    """Assembled from real measured data, in order: outcome, injection
    evidence (the one genuinely good idea from the old metric.py, now
    grounded in a real measurement instead of a guess about frozen data),
    test/diff evidence, and the scoring rule stated explicitly so the
    reflection LM optimizes the right objective."""
    parts = [
        f"{result.task_id} scored {result.score:.2f} (status={result.status}, "
        f"{result.attempts} attempt(s), stop_reasons={result.stop_reasons}, "
        f"{result.elapsed_s:.1f}s, {result.turn_count} agent turns)."
    ]

    if pred_name == "agents_md":
        parts.append(f"{pred_name} is the master system prompt -- always active in every run.")
    else:
        usages = merge_component_usage(result.notifications, knowledge_topic_index=knowledge_topic_index)
        usage = next((u for u in usages if u.pred_name == pred_name), None)
        if usage is not None:
            error_note = " (including right after a tool error)" if usage.was_error_context else ""
            parts.append(f"{pred_name} was injected {usage.invocation_count} time(s) during this run{error_note}.")
        else:
            parts.append(f"{pred_name} was NOT injected during this run -- its text had no effect on this outcome.")

    if result.status == "harness_error":
        parts.append(
            f"The benchmark harness itself failed before the agent could be properly scored "
            f"({result.error}). This outcome says NOTHING about the quality of {pred_name}; "
            f"do not rewrite it in response to this record."
        )
    elif result.success:
        parts.append(f"Tests passed on attempt {result.attempts}.")
    else:
        if result.test_output_tail:
            parts.append(f"The tests failed. pytest output:\n```\n{result.test_output_tail}\n```")
        if result.diff_summary:
            parts.append(f"The agent's actual code changes:\n```diff\n{result.diff_summary}\n```")

    parts.append(_SCORING_RULE)
    return " ".join(parts)


class PolyglotGEPAAdapter:
    """GEPAAdapter[ExerciseSpec, PolyglotTrajectory, PolyglotRolloutOutput]."""

    #: Real bug, confirmed against a live run: the real GEPAAdapter Protocol
    #: declares `propose_new_texts: ProposalFn | None = None` as a
    #: class-level default (gepa/core/adapter.py), but this class does NOT
    #: inherit from that Protocol (structural typing only -- see the module
    #: docstring), so it never gets that default for free. Without declaring
    #: it explicitly, gepa/proposer/reflective_mutation/reflective_mutation.py
    #: does `self.adapter.propose_new_texts is not None` -- a direct
    #: attribute access, not a defensive getattr() -- which raised
    #: AttributeError on EVERY reflection attempt. reflective_mutation.py
    #: catches that internally and logs "no candidate proposed this
    #: iteration" rather than propagating it (raise_on_exception=True never
    #: fired), so a real run just kept selecting the same seed program and
    #: retrying reflection forever -- ~9,000 iterations in a few minutes,
    #: 100% CPU, zero real reflection LM calls, bounded only by
    #: --max-wall-clock-s rather than --max-metric-calls (which never grows
    #: past the seed valset size in this failure mode, since no real
    #: evaluate() call ever happens). gepa.optimize()'s own top-level check
    #: (api.py) IS correctly guarded with hasattr(); only this specific
    #: internal path was not.
    propose_new_texts = None

    def __init__(
        self,
        runner: PolyglotLiveRunner,
        *,
        component_paths: Mapping[str, str],
        practice_dir_path: Path,
        knowledge_topic_index: Mapping[str, str] | None = None,
    ):
        self.runner = runner
        self.component_paths = dict(component_paths)
        self.practice_dir_path = Path(practice_dir_path)
        self.knowledge_topic_index = dict(knowledge_topic_index or {})

    def evaluate(
        self, batch: Sequence[ExerciseSpec], candidate: dict[str, str], capture_traces: bool = False,
    ) -> EvaluationBatch:
        specs = list(batch)
        # Audit logging (run_gepa.py's SpendLog) happens via runner.on_result,
        # called per-result INSIDE run_batch() -- not here after the fact --
        # so a later exercise in this same batch raising doesn't erase the
        # audit trail for exercises that already genuinely ran.
        results = self.runner.run_batch(candidate, specs)  # candidate IS read -- this is the fix.

        outputs = [
            PolyglotRolloutOutput(task_id=r.task_id, status=r.status, score=r.score, from_cache=r.from_cache)
            for r in results
        ]
        scores = [float(r.score) for r in results]
        trajectories = (
            [PolyglotTrajectory(spec=s, result=r) for s, r in zip(specs, results, strict=True)]
            if capture_traces else None
        )
        # Only actual live runs count against max_metric_calls -- a fully
        # cached re-evaluation must charge nothing, so max_metric_calls
        # means "real agent runs I'm willing to pay for."
        num_metric_calls = sum(1 for r in results if not r.from_cache)

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories,
                                num_metric_calls=num_metric_calls)

    def make_reflective_dataset(
        self, candidate: dict[str, str], eval_batch: EvaluationBatch, components_to_update: list[str],
    ) -> Mapping[str, list[dict[str, Any]]]:
        dataset: dict[str, list[dict[str, Any]]] = {}
        trajectories = eval_batch.trajectories or []
        for component in components_to_update:
            records = []
            for traj in trajectories:
                spec, result = traj.spec, traj.result
                records.append({
                    "Inputs": {
                        "exercise": spec.task_id,
                        "task": describe_exercise(spec, self.practice_dir_path),
                        "component_file": self.component_paths.get(component, "?"),
                        "current_component_text": candidate.get(component, ""),
                    },
                    "Generated Outputs": {
                        "status": result.status,
                        "attempts": result.attempts,
                        "transcript_excerpt": result.transcript_excerpt,
                    },
                    "Feedback": _component_feedback(component, result, self.knowledge_topic_index),
                    "score": result.score,
                })
            dataset[component] = records
        return dataset

    # propose_new_texts deliberately NOT implemented -- GEPA's default
    # reflection-LM-based proposer is used. get_adapter_state/set_adapter_state
    # also not implemented: the on-disk live_cache already provides
    # crash-resume, and duck typing means their absence is a no-op.
