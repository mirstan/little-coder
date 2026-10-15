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

# The script picks between running harbor directly and wrapping it in
# `sg docker -c ...` from `groups` and the presence of `sg`, which vary by
# machine. Both are stubbed so each test takes one branch deterministically.
GROUPS_WITH_DOCKER = "#!/usr/bin/env bash\necho \"staff docker\"\n"
GROUPS_WITHOUT_DOCKER = "#!/usr/bin/env bash\necho \"staff\"\n"
# Tripwire: the direct branch must never reach sg.
SG_FORBIDDEN = "#!/usr/bin/env bash\necho \"sg must not be called\" >&2\nexit 97\n"
# Runs the command sg was handed, after checking the call shape.
SG_PASSTHROUGH = '#!/usr/bin/env bash\n[[ $1 == docker && $2 == -c ]] || exit 97\nexec bash -c "$3"\n'


def _write_stub(bin_dir, name, body):
    stub = bin_dir / name
    stub.write_text(body)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _run(tmp_path, *tasks, dataset=None, groups=GROUPS_WITH_DOCKER, sg=SG_FORBIDDEN):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    _write_stub(bin_dir, "harbor", STUB)
    _write_stub(bin_dir, "groups", groups)
    _write_stub(bin_dir, "sg", sg)
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


def test_sg_branch_passes_task_names_through(tmp_path):
    """Not in the docker group, sg present: the command goes through
    `sg docker -c` with %q quoting and must arrive intact."""
    assert _run(
        tmp_path, "fix-git", "terminal-bench/other-task",
        groups=GROUPS_WITHOUT_DOCKER, sg=SG_PASSTHROUGH,
    ) == ["terminal-bench/fix-git", "terminal-bench/other-task"]
