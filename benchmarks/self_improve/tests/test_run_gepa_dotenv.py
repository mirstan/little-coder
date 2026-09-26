"""run_gepa.py should load a .env file (via python-dotenv) from its own
directory so REFLECTION_LM_API_KEY can live in a gitignored file instead of
requiring the user to export it into every shell they invoke the script
from. README.md documents this as the design intent.

Tested via a real subprocess import, not importlib.reload(), which
re-executes the module's own `from dotenv import load_dotenv` line and
clobbers any monkeypatch on it before the effect can be observed.

These tests write ONLY to a disposable tmp_path, never to the real
benchmarks/self_improve/.env -- that file is untracked, can hold a real API
key, and has no git history to recover from, so a `finally`-block restore
(which does not survive SIGKILL, a hard crash, or parallel workers) is not
good enough. SELF_IMPROVE_DOTENV exists to make the loaded path injectable
for exactly that reason.
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


def _run_import_and_print(env_file: Path, expr: str, unset: tuple[str, ...] = (), **extra_env: str) -> str:
    """Like _run_with_dotenv, but prints an arbitrary expression evaluated
    after importing run_gepa (bound as `run_gepa`)."""
    env = {k: v for k, v in os.environ.items() if k not in unset}
    env.update(extra_env)
    env["SELF_IMPROVE_DOTENV"] = str(env_file)
    result = subprocess.run(
        [sys.executable, "-c",
         "import benchmarks.self_improve.run_gepa as run_gepa; import os; "
         f"print({expr})"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        env=env,
    )
    return result.stdout.strip()


_CONSTANT_NAMES = ["REFLECTION_LM_API_KEY", "SELF_IMPROVE_DOTENV"]


def test_every_dotenv_name_is_orchestrator_only(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("REFLECTION_LM_API_KEY=sk-fake-not-real\nOTHER_SECRET=other-fake\n")
    out = _run_import_and_print(env_file, "sorted(run_gepa._orchestrator_only_env_names())",
                                unset=("REFLECTION_LM_API_KEY", "OTHER_SECRET"))
    assert out == repr(sorted({"OTHER_SECRET", *_CONSTANT_NAMES}))


def test_missing_dotenv_withholds_only_the_constants(tmp_path):
    missing = tmp_path / "does-not-exist" / ".env"
    out = _run_import_and_print(missing, "sorted(run_gepa._orchestrator_only_env_names())")
    assert out == repr(sorted(_CONSTANT_NAMES))


def test_dotenv_interpolation_and_disable_flag_match_load_dotenv(tmp_path):
    """The single-read loader keeps load_dotenv(override=False) semantics:
    an exported value wins inside ${} expansion, and PYTHON_DOTENV_DISABLED
    turns loading off."""
    env_file = tmp_path / ".env"
    env_file.write_text("BASE_X=fromfile\nKEY_X=${BASE_X}-x\n")
    out = _run_import_and_print(env_file, "os.environ.get('KEY_X')", unset=("KEY_X",), BASE_X="fromshell")
    assert out == "fromshell-x"
    out = _run_import_and_print(env_file, "os.environ.get('KEY_X')", unset=("KEY_X", "BASE_X"),
                                PYTHON_DOTENV_DISABLED="1")
    assert out == "None"


def test_import_sets_litellm_mode_production_so_litellm_loads_no_dotenv(tmp_path):
    """litellm runs its own load_dotenv() on import unless LITELLM_MODE is
    set to something other than DEV; run_gepa must stay the only loader."""
    out = _run_import_and_print(tmp_path / "missing.env", "os.environ.get('LITELLM_MODE')",
                                unset=("LITELLM_MODE",))
    assert out == "PRODUCTION"


def test_dotenv_refusals_name_knobs_the_child_reads(tmp_path):
    import benchmarks.aider_polyglot as aider_polyglot
    shared = ["ATTEMPT_TIMEOUT_S", "CODEX_TIMEOUT_S", "LITTLE_CODER_ANYTHING", "PI_ANYTHING",
              *aider_polyglot._ENV_KNOBS]
    env_file = tmp_path / ".env"
    env_file.write_text("".join(f"{n}=1\n" for n in shared)
                        + "POLYGLOT_RESULTS_FILE=/tmp/x\nOTHER_SECRET=s\n")
    out = _run_import_and_print(env_file, "run_gepa._dotenv_refusals()",
                                unset=tuple(shared) + ("POLYGLOT_RESULTS_FILE", "OTHER_SECRET"))
    for name in shared:
        assert (f"{name} is set in benchmarks/self_improve/.env, which is orchestrator-only; "
                "export it in your shell instead.") in out
    assert "POLYGLOT_RESULTS_FILE" not in out
    assert "OTHER_SECRET" not in out
