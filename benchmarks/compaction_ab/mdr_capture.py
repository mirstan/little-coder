"""Multi-depth replay, step 1: capture the trial's exact request template offline.

This starts the fake OpenAI server and a headless pi through PiRpc with the
harbor adapter's arguments (TB mode, the harbor tool allowlist, benchmark
terminal_bench, thinking high, and the trial's arm env). The omlx provider is
pointed at the fake server through a temporary LITTLE_CODER_MODELS_FILE, a
copy of the real config with only baseUrl changed. It then sends the trial's
first user message and lets zzz-ab-capture write the request body. The
template's system message and tools are checked against the live trial's
observer fingerprints (sha1 of messages[0] and of tools).

No omlx and no GPU: every request goes to 127.0.0.1:<port>.
Usage: python3 mdr_capture.py <session.jsonl> <ab-observer.jsonl> <outdir> [--arm blackhole]
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "benchmarks" / "harbor_adapter"))
import rpc_client as RC  # noqa: E402

REAL_MODELS = Path.home() / ".config" / "little-coder" / "models.json"
MODEL = "omlx/tiel-coder-oq6e-fp16"
PORT = 18765


def sha1_16(value) -> str:
    # Same as zzz-ab-observer: sha1 over JSON.stringify(value).
    return hashlib.sha1(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()[:16]


def main() -> int:
    session, observer, outdir = sys.argv[1:4]
    arm = sys.argv[sys.argv.index("--arm") + 1] if "--arm" in sys.argv else "blackhole"
    out = Path(outdir).resolve()
    cap = out / "capture"
    if cap.exists():
        shutil.rmtree(cap)
    cap.mkdir(parents=True)
    entries = [json.loads(l) for l in Path(session).read_text().splitlines()]
    first_user = next(e for e in entries if e.get("type") == "message" and e["message"]["role"] == "user")
    content = first_user["message"]["content"]
    prompt = content if isinstance(content, str) else "".join(b.get("text", "") for b in content if b.get("type") == "text")
    obs = [json.loads(l) for l in Path(observer).read_text().splitlines()]
    req0 = next(o for o in obs if o["kind"] == "request")

    models = json.loads(REAL_MODELS.read_text())
    models["providers"]["omlx"]["baseUrl"] = f"http://127.0.0.1:{PORT}/v1"
    mfile = out / "models.fake.json"
    mfile.write_text(json.dumps(models))

    if arm == "blackhole":
        bh = ROOT / ".cache" / "pi-bench-agent" / "pi-blackhole"
        bh.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "vendor/compaction-ab/config/pi-blackhole-config.json", bh / "pi-blackhole-config.json")
    server = subprocess.Popen([sys.executable, str(HERE / "fake_openai.py"), str(PORT), str(out / "fake-requests.jsonl")])
    time.sleep(1.0)
    env = {
        "LITTLE_CODER_MODELS_FILE": str(mfile),
        "LC_AB_CAPTURE_PAYLOAD_DIR": str(cap),
        "LITTLE_CODER_PERMISSION_MODE": "accept-all",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    for k in ("LC_COMPACTION_ARM", "LITTLE_CODER_CACHE_REUSE_COMPACTION"):
        os.environ.pop(k, None)
    if arm in ("blackhole", "prefix-cache"):
        env["LC_COMPACTION_ARM"] = arm
    elif arm == "reuse84":
        env["LITTLE_CODER_CACHE_REUSE_COMPACTION"] = "1"
    try:
        import little_coder_agent as LCA  # harbor adapter constants (allowlist)
        tools = LCA.DEFAULT_ALLOWED_TOOLS
    except Exception:  # noqa: BLE001 - harbor not importable from this python
        tools = ["ShellSession", "ShellSessionCwd", "ShellSessionReset", "ShellRecall"]
    rpc = RC.PiRpc(MODEL, benchmark="terminal_bench", allowed_tools=tools, tb_mode=True,
                   thinking="high", env=env, tb_shell_handler=lambda req: "[exit=0 cwd=/app]")
    try:
        rpc.prompt_and_collect(prompt, timeout=120)
    finally:
        rpc.close(timeout=10)
        server.terminate()
    reqs = sorted(cap.glob("request-*.json"), key=lambda p: int(p.stem.split("-")[1]))
    # pi writes its cwd into the system prompt. The live trial ran from the
    # session header's cwd (the frozen exp worktree), and this capture runs
    # from the mdr worktree, so substitute the live path back in.
    live_cwd = entries[0].get("cwd") or ""
    raw = reqs[0].read_text()
    if live_cwd and str(ROOT) != live_cwd:
        raw = raw.replace(json.dumps(str(ROOT))[1:-1], json.dumps(live_cwd)[1:-1])
    template = json.loads(raw)
    got = {"system_sha1": sha1_16(template["messages"][0]), "tools_sha1": sha1_16(template.get("tools"))}
    want = {"system_sha1": req0["system_sha1"], "tools_sha1": req0["tools_sha1"]}
    (out / "template.json").write_text(json.dumps(template))
    report = {"got": got, "want": want, "match": got == want, "n_requests": len(reqs),
              "template_keys": sorted(template.keys())}
    (out / "capture-report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))
    return 0 if report["match"] else 1


if __name__ == "__main__":
    sys.exit(main())
