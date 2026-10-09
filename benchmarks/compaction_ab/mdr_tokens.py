"""Multi-depth replay, step 3: tokenizer checks (CPU only, no server).

It renders exactly as omlx 0.7.0 does: the request is parsed with omlx's own
ChatCompletionRequest, then passed through extract_text_content,
prepare_system_messages_for_template, convert_tools_for_template and the
model's HF tokenizer apply_chat_template. It must run under omlx's Python:
  /opt/homebrew/Cellar/omlx/0.7.0/libexec/bin/python -I mdr_tokens.py <model_dir> <points_dir> <out.json>
For each point directory (from mdr_build.mjs):
  - request-N and request-N1 token counts vs the live turns' prompt_tokens
    (must be within 16);
  - shared prefix of a1.json with request-N, and of a3.json with request-N
    (the warm prefix) and with request-N1 (its own capture): whole 4096 blocks.
"""
import json
import sys
from pathlib import Path

from omlx.api.openai_models import ChatCompletionRequest
from omlx.api.tool_calling import convert_tools_for_template
from omlx.api.utils import extract_text_content, prepare_system_messages_for_template
from transformers import AutoTokenizer


def make_encoder(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)

    def enc(body):
        req = ChatCompletionRequest(**body)
        tools = convert_tools_for_template(req.tools)
        kw = body.get("chat_template_kwargs") or {}
        msgs = extract_text_content(req.messages, None, tok, consolidate_system_messages=False)
        msgs = prepare_system_messages_for_template(msgs, tok, tools=tools, chat_template_kwargs=kw)
        text = tok.apply_chat_template(msgs, tools=convert_tools_for_template(tools), tokenize=False,
                                       add_generation_prompt=True, **kw)
        return tok.encode(text)

    return enc


def shared(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    model_dir, points_dir, out_path = sys.argv[1:4]
    enc = make_encoder(model_dir)
    report = json.loads(Path(points_dir, "build-report.json").read_text())
    out = []
    for p in report["points"]:
        if p.get("prep", True) is None:
            continue
        d = Path(points_dir, f"turn-{p['turn']}")
        rn = enc(json.loads((d / "request-N.json").read_text()))
        row = {"turn": p["turn"], "tokens_N": len(rn), "recorded_N": p["recorded_prompt_tokens"],
               "delta_N": len(rn) - p["recorded_prompt_tokens"]}
        rn1 = None
        if (d / "request-N1.json").exists():
            rn1 = enc(json.loads((d / "request-N1.json").read_text()))
            row |= {"tokens_N1": len(rn1), "recorded_N1": p["recorded_prompt_tokens_next"],
                    "delta_N1": len(rn1) - p["recorded_prompt_tokens_next"]}
        if (d / "a1.json").exists():
            a1 = enc(json.loads((d / "a1.json").read_text()))
            s = shared(rn, a1)
            row["a1"] = {"tokens": len(a1), "shared_with_N": s, "N_not_shared": len(rn) - s,
                         "blocks": f"{s // 4096}/{len(rn) // 4096}"}
        if (d / "a3.json").exists():
            a3 = enc(json.loads((d / "a3.json").read_text()))
            s = shared(rn, a3)
            row["a3"] = {"tokens": len(a3), "shared_with_N": s, "N_not_shared": len(rn) - s,
                         "blocks_vs_N": f"{s // 4096}/{len(rn) // 4096}"}
            if rn1 is not None:
                s1 = shared(rn1, a3)
                row["a3"] |= {"shared_with_N1": s1, "blocks_vs_N1": f"{s1 // 4096}/{len(rn1) // 4096}"}
        row["ok_tokens"] = abs(row["delta_N"]) <= 16 and abs(row.get("delta_N1", 0)) <= 16
        out.append(row)
        print(json.dumps(row), flush=True)
    Path(out_path).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
