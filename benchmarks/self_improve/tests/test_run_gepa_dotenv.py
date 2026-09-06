"""run_gepa.py should load a .env file (via python-dotenv) from its own
directory so REFLECTION_LM_API_KEY can live in a gitignored file instead of
requiring the user to export it into every shell they invoke the script
from. README.md already documents this as the design intent; this closes
the gap between that documentation and the actual implementation.

Tested via a real subprocess import (not importlib.reload(), which
re-executes the module's own `from dotenv import load_dotenv` line and
clobbers any monkeypatch on it before the effect can be observed) -- this
is the only reliable way to test an import-time side effect.

CONFIRMED REAL RISK (caught by review, not hypothetical): an earlier version
of these tests wrote fake content directly into the real, gitignored
benchmarks/self_improve/.env file (which can hold a real API key) and
restored it in a `finally` block. `finally` does not survive SIGKILL, a
hard crash, or parallel test workers -- and the file is untracked, with no
git history to recover from. A crash mid-test could permanently destroy a
real key. Fixed by making the loaded path injectable via
SELF_IMPROVE_DOTENV, so these tests write ONLY to a disposable tmp_path
fixture and never touch the real file at all.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent


def _run_with_dotenv(env_file: Path, unset: tuple[str, ...] = (), **extra_env: str) -> str:
    """Run `import benchmarks.self_improve.run_gepa` in a fresh subprocess
    with SELF_IMPROVE_DOTENV pointed at env_file, and print the resulting
    REFLECTION_LM_API_KEY. `unset` names real-environment vars to remove
    first (so a test can observe the .env value actually taking effect,
    rather than an inherited real value masking it)."""
    env = {k: v for k, v in os.environ.items() if k not in unset}
    env.update(extra_env)
    env["SELF_IMPROVE_DOTENV"] = str(env_file)
    result = subprocess.run(
        [sys.executable, "-c",
         "import benchmarks.self_improve.run_gepa; import os; "
         "print(os.environ.get('REFLECTION_LM_API_KEY'))"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        env=env,
    )
    return result.stdout.strip()


def test_dotenv_file_next_to_run_gepa_is_loaded_on_import(tmp_path):
    """load_dotenv(override=False) treats an env var set to "" as already
    present and will NOT load the .env value over it -- the key must be
    genuinely ABSENT from the subprocess env, not merely empty, to observe
    the .env file actually taking effect."""
    env_file = tmp_path / ".env"
    env_file.write_text("REFLECTION_LM_API_KEY=sk-fake-from-dotenv-test-not-real\n")

    result = _run_with_dotenv(env_file, unset=("REFLECTION_LM_API_KEY",))
    assert result == "sk-fake-from-dotenv-test-not-real"


def test_dotenv_does_not_override_an_already_exported_env_var(tmp_path):
    """An explicitly exported env var must win over a stale .env file value
    -- load_dotenv()'s default override=False behavior, verified rather
    than assumed."""
    env_file = tmp_path / ".env"
    env_file.write_text("REFLECTION_LM_API_KEY=sk-from-dotenv-should-be-overridden\n")

    result = _run_with_dotenv(env_file, REFLECTION_LM_API_KEY="sk-explicitly-exported")
    assert result == "sk-explicitly-exported"


def test_dotenv_missing_file_does_not_crash_import(tmp_path):
    """SELF_IMPROVE_DOTENV pointing at a nonexistent file must not raise --
    load_dotenv() already handles a missing path gracefully; this pins that
    behavior for the injectable-path mechanism specifically."""
    missing = tmp_path / "does-not-exist" / ".env"
    result = _run_with_dotenv(missing, unset=("REFLECTION_LM_API_KEY",))
    assert result == "None"
