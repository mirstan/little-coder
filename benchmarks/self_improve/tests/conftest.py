import json
from pathlib import Path

import pytest


@pytest.fixture
def gaia_run(tmp_path) -> Path:
    """A minimal gaia benchmark run directory: two task dirs, matching the
    real layout confirmed in TDD_SPEC.md §0 (result.json, tool_calls.jsonl,
    notifications.txt, transcript.txt, prompt.txt per task)."""
    t1 = tmp_path / "task-001"
    t1.mkdir()
    (t1 / "result.json").write_text(json.dumps({
        "model_answer": "42", "gold": "42", "correct": True, "elapsed_s": 12.3,
    }))
    (t1 / "tool_calls.jsonl").write_text(
        json.dumps({"name": "bash", "args": {}, "result_text": "ok", "is_error": False}) + "\n"
    )
    (t1 / "notifications.txt").write_text("[info] skill-inject: +1 [bash]\n")
    (t1 / "transcript.txt").write_text("final answer: 42")
    (t1 / "prompt.txt").write_text("solve this task")

    t2 = tmp_path / "task-002"
    t2.mkdir()
    (t2 / "result.json").write_text(json.dumps({
        "model_answer": "", "gold": "7", "correct": False, "elapsed_s": 900.0,
    }))
    (t2 / "tool_calls.jsonl").write_text("")
    (t2 / "notifications.txt").write_text("")
    (t2 / "transcript.txt").write_text("")
    (t2 / "stderr.log").write_text("Traceback ...\nRuntimeError: pi exited\n")

    return tmp_path
