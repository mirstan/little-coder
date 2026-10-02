"""harbor_pilot.sh task-name prefixing, run against a stub `harbor` on PATH.

Bare names get the dataset's org prefix only for an org/name registry
dataset; a legacy name@version dataset has no org, so bare names must pass
through unchanged.
"""

import os
import stat
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "harbor_pilot.sh"

# Prints each --include-task-name value on its own line and nothing else
# runs: no docker, no network.
STUB = """#!/usr/bin/env bash
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--include-task-name" ]]; then
    echo "TASK=$2"
    shift
  fi
  shift
done
"""


def _run(tmp_path, *tasks, dataset=None):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "harbor"
    stub.write_text(STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env.pop("TB_DATASET", None)
    if dataset is not None:
        env["TB_DATASET"] = dataset
    proc = subprocess.run(
        ["bash", str(SCRIPT), *tasks],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return [
        line[len("TASK="):]
        for line in proc.stdout.splitlines()
        if line.startswith("TASK=")
    ]


def test_default_dataset_prefixes_bare_name(tmp_path):
    assert _run(tmp_path, "fix-git") == ["terminal-bench/fix-git"]


def test_prefixed_name_passes_through(tmp_path):
    assert _run(tmp_path, "terminal-bench/fix-git") == ["terminal-bench/fix-git"]


def test_legacy_name_at_version_dataset_keeps_bare_name(tmp_path):
    assert _run(tmp_path, "fix-git", dataset="terminal-bench@2.0") == ["fix-git"]
