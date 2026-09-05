"""Read-modify-write logic for the optimizable harness artifacts (AGENTS.md,
PRINCIPLES.md, skill files). Frontmatter is split from body and NEVER
re-serialized through a YAML dumper -- string concatenation only, so a
component whose optimized body is byte-identical to the original round-trips
exactly (TDD_SPEC.md §7.2).

config/components.yaml is the single source of truth mapping pred_name ->
repo-relative file path.
"""
from pathlib import Path

import dspy
import yaml

_FRONTMATTER_DELIM = "---\n"


def split_frontmatter(text: str) -> tuple[str | None, str]:
    """Split a file's leading '---\\n...\\n---\\n' YAML block from its body.
    Returns (frontmatter_including_delimiters, body). If the file does not
    start with the delimiter, returns (None, text) unchanged."""
    if not text.startswith(_FRONTMATTER_DELIM):
        return None, text
    end = text.find("\n---\n", len(_FRONTMATTER_DELIM))
    if end == -1:
        return None, text
    end_of_block = end + len("\n---\n")
    frontmatter = text[:end_of_block]
    body = text[end_of_block:]
    return frontmatter, body


def reattach_frontmatter(frontmatter: str | None, body: str) -> str:
    """Exact inverse of split_frontmatter: string concatenation only."""
    if frontmatter is None:
        return body
    return frontmatter + body


def _load_components_yaml(components_yaml_path: Path) -> dict[str, str]:
    return yaml.safe_load(Path(components_yaml_path).read_text()) or {}


def load_components(components_yaml_path: Path, repo_root: Path) -> dict[str, str]:
    """Load every component's body text (frontmatter stripped) keyed by
    pred_name, per components.yaml."""
    mapping = _load_components_yaml(components_yaml_path)
    components = {}
    for pred_name, rel_path in mapping.items():
        text = (Path(repo_root) / rel_path).read_text()
        _frontmatter, body = split_frontmatter(text)
        components[pred_name] = body
    return components


class HarnessProgram(dspy.Module):
    """Addressable text container for GEPA: one dspy.Predict per artifact.

    forward() calls EVERY predictor once per rollout so a real trace entry
    exists for each one GEPA might attribute credit/blame to during
    reflection (dspy's GEPA requires at least one real (predictor, inputs,
    prediction) trace entry per predictor being optimized -- an empty trace
    for any predictor crashes make_reflective_dataset(), confirmed against
    installed dspy-ai 3.3.1 source, dspy/teleprompt/gepa/gepa_utils.py).

    forward() does NOT take pred_name as an input: GEPA calls
    program(**example.inputs()) using only the example's declared input
    fields, and determines which predictor produced which trace entry by
    inspecting dspy.settings.trace afterward -- pred_name is supplied to
    OUR metric() by GEPA's own machinery, never by the caller of forward().

    Scoring never depends on forward()'s actual output text -- metric()
    reads gold.trajectory (historical, already-collected data) regardless
    of what any predictor returns here. Run under dspy.settings.configure(
    lm=DummyLM(...)) so every one of these calls is free and makes no
    network request; see run_gepa.py.
    """
    def __init__(self, components: dict[str, str]):
        super().__init__()
        self.predictors: dict[str, dspy.Predict] = {}
        for pred_name, body_text in components.items():
            sig = dspy.make_signature(
                "task_context -> guidance",
                instructions=body_text,
                signature_name=f"{pred_name}_signature",
            )
            self.predictors[pred_name] = dspy.Predict(sig)

    def forward(self, task_context: str = ""):
        outputs = {
            name: predictor(task_context=task_context).guidance
            for name, predictor in self.predictors.items()
        }
        return dspy.Prediction(**outputs)


def build_harness_program(components: dict[str, str]) -> HarnessProgram:
    return HarnessProgram(components)


def write_components_back(
    components_yaml_path: Path, repo_root: Path, optimized: dict[str, str]
) -> list[Path]:
    """Write optimized body text back into each component's real file,
    preserving frontmatter untouched. Skips (does not write) any file whose
    optimized body is identical to its current body. Returns the list of
    file paths actually changed."""
    mapping = _load_components_yaml(components_yaml_path)
    changed: list[Path] = []
    for pred_name, new_body in optimized.items():
        rel_path = mapping.get(pred_name)
        if rel_path is None:
            continue
        file_path = Path(repo_root) / rel_path
        current_text = file_path.read_text()
        frontmatter, current_body = split_frontmatter(current_text)
        if new_body == current_body:
            continue
        file_path.write_text(reattach_frontmatter(frontmatter, new_body))
        changed.append(file_path)
    return changed
