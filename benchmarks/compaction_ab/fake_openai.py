"""Minimal local OpenAI-compatible chat-completions server for offline dry runs.

It streams one canned answer per request as SSE chunks, followed by usage and
[DONE], and logs every request body to REQUEST_LOG (JSONL). The canned text
carries every section the compaction arms validate (#84's tagged sections, the
pi summary headings), so each arm's parser accepts it. No model is involved.

Usage: python3 fake_openai.py PORT REQUEST_LOG [--delay SECONDS]
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1])
LOG = sys.argv[2]
DELAY = float(sys.argv[sys.argv.index("--delay") + 1]) if "--delay" in sys.argv else 0.0

SUMMARY = (
    "## Goal\nDry-run goal.\n\n## Constraints & Preferences\n- (none)\n\n## Progress\n### Done\n- [x] dry run\n\n"
    "### In Progress\n- [ ] nothing\n\n### Blocked\n- (none)\n\n## Key Decisions\n- **dry**: run\n\n"
    "## Next Steps\n1. continue\n\n## Critical Context\n- (none)\n"
)
ANSWER = (
    "<history-summary>\n" + SUMMARY + "</history-summary>\n"
    "<turn-prefix-summary>\n## Original Request\nDry-run request.\n\n## Early Progress\n- none\n\n"
    "## Context for Suffix\n- none\n</turn-prefix-summary>\n"
)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(n)
        try:
            body = json.loads(raw)
        except ValueError:
            body = {}
        msgs = body.get("messages") or []
        with open(LOG, "a") as fh:
            fh.write(json.dumps({
                "t": time.time(), "path": self.path, "n_messages": len(msgs),
                "max_tokens": body.get("max_tokens"), "thinking_budget": body.get("thinking_budget"),
                "has_tools": bool(body.get("tools")), "bytes": len(raw),
            }) + "\n")
        if DELAY:
            time.sleep(DELAY)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        text = "OK" if body.get("max_tokens") == 1 else ANSWER
        prompt = max(1, len(raw) // 4)

        def chunk(delta, finish=None):
            ev = {"id": "x", "object": "chat.completion.chunk", "model": body.get("model"),
                  "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
            self.wfile.flush()

        chunk({"role": "assistant", "reasoning_content": "thinking briefly"})
        for i in range(0, len(text), 200):
            chunk({"content": text[i:i + 200]})
        chunk({}, "length" if body.get("max_tokens") == 1 else "stop")
        usage = {"id": "x", "object": "chat.completion.chunk", "choices": [],
                 "usage": {"prompt_tokens": prompt, "completion_tokens": len(text) // 4 + 3,
                           "total_tokens": prompt + len(text) // 4 + 3,
                           "prompt_tokens_details": {"cached_tokens": 0}}}
        self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())
        self.wfile.flush()


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
