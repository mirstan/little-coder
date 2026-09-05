"""run_gepa.py should load a .env file (via python-dotenv) from its own
directory so REFLECTION_LM_API_KEY can live in a gitignored file instead of
requiring the user to export it into every shell they invoke the script
from. README.md already documents this as the design intent; this closes
the gap between that documentation and the actual implementation.

Tested via a real subprocess import (not importlib.reload(), which
re-executes the module's own `from dotenv import load_dotenv` line and
clobbers any monkeypatch on it before the effect can be observed) -- this
is the only reliable way to test an import-time side effect.
"""
import subprocess
import sys
from pathlib import Path

RUN_GEPA_DIR = Path(__file__).parent.parent
REPO_ROOT = RUN_GEPA_DIR.parent.parent


def test_dotenv_file_next_to_run_gepa_is_loaded_on_import(tmp_path, monkeypatch):
    """load_dotenv(override=False) treats an env var set to "" as already
    present and will NOT load the .env value over it -- the key must be
    genuinely ABSENT from the subprocess env, not merely empty, to observe
    the .env file actually taking effect."""
    env_file = RUN_GEPA_DIR / ".env"
    backup = env_file.read_text() if env_file.exists() else None
    try:
        env_file.write_text("REFLECTION_LM_API_KEY=sk-fake-from-dotenv-test-not-real\n")
        import os
        clean_env = {k: v for k, v in os.environ.items() if k != "REFLECTION_LM_API_KEY"}
        result = subprocess.run(
            [sys.executable, "-c",
             "import benchmarks.self_improve.run_gepa; import os; "
             "print(os.environ.get('REFLECTION_LM_API_KEY'))"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
            env=clean_env,
        )
        assert result.stdout.strip() == "sk-fake-from-dotenv-test-not-real"
    finally:
        if backup is not None:
            env_file.write_text(backup)
        else:
            env_file.unlink(missing_ok=True)


def test_dotenv_does_not_override_an_already_exported_env_var(tmp_path):
    """An explicitly exported env var must win over a stale .env file value
    -- load_dotenv()'s default override=False behavior, verified rather
    than assumed."""
    env_file = RUN_GEPA_DIR / ".env"
    backup = env_file.read_text() if env_file.exists() else None
    try:
        env_file.write_text("REFLECTION_LM_API_KEY=sk-from-dotenv-should-be-overridden\n")
        import os
        result = subprocess.run(
            [sys.executable, "-c",
             "import benchmarks.self_improve.run_gepa; import os; "
             "print(os.environ.get('REFLECTION_LM_API_KEY'))"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
            env={**os.environ, "REFLECTION_LM_API_KEY": "sk-explicitly-exported"},
        )
        assert result.stdout.strip() == "sk-explicitly-exported"
    finally:
        if backup is not None:
            env_file.write_text(backup)
        else:
            env_file.unlink(missing_ok=True)
