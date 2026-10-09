"""Render two OpenAI chat-completions bodies through the model's chat template
and tokenizer, offline, and report how many leading tokens they share.

Usage: python3 -I render_prefix.py <model_dir> <pairs.json> <out.json>

pairs.json: [{"name": str, "captured": <body>, "replay": <body>}, ...]

Mirrors the parts of omlx's request preprocessing that can change the
rendered prompt: tool_choice "none" drops tools, reasoning_effort is merged
into chat_template_kwargs, consecutive same-role user/system messages are
merged with "\\n\\n", tool-call arguments are parsed from JSON strings. This
proves relative prefix sharing under one renderer, not omlx's exact count.
"""
import json
import sys

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer


def raise_exception(msg):
    raise jinja2.exceptions.TemplateError(msg)


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def to_template_messages(body):
    out = []
    for m in body["messages"]:
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                fn = dict(tc.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args)
                    except ValueError:
                        pass
                tcs.append({**tc, "function": fn})
            m["tool_calls"] = tcs
        out.append(m)
    merged = []
    for m in out:
        if merged and m["role"] == merged[-1]["role"] and m["role"] in ("user", "system"):
            a, b = merged[-1].get("content") or "", m.get("content") or ""
            if isinstance(a, list) or isinstance(b, list):
                a = a if isinstance(a, list) else [{"type": "text", "text": a}]
                b = b if isinstance(b, list) else [{"type": "text", "text": b}]
                merged[-1]["content"] = a + b
            else:
                merged[-1]["content"] = a + "\n\n" + b
            continue
        merged.append(m)
    return merged


def render(template, body):
    kwargs = dict(body.get("chat_template_kwargs") or {})
    if body.get("reasoning_effort") is not None:
        kwargs.setdefault("reasoning_effort", body["reasoning_effort"])
    tools = None if body.get("tool_choice") == "none" else body.get("tools")
    tools = [t.get("function", t) for t in tools] if tools else None
    return template.render(messages=to_template_messages(body), tools=tools, add_generation_prompt=True, **kwargs)


def main():
    model_dir, pairs_path, out_path = sys.argv[1:4]
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    with open(f"{model_dir}/chat_template.jinja", encoding="utf-8") as fh:
        template = env.from_string(fh.read())
    tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
    with open(pairs_path, encoding="utf-8") as fh:
        pairs = json.load(fh)
    results = []
    for p in pairs:
        a_text, b_text = render(template, p["captured"]), render(template, p["replay"])
        a = tok.encode(a_text, add_special_tokens=False).ids
        b = tok.encode(b_text, add_special_tokens=False).ids
        common = 0
        for x, y in zip(a, b):
            if x != y:
                break
            common += 1
        # The captured prompt's own generation prompt ("<|im_start|>assistant\n<think>\n")
        # is never part of the replay; the cacheable prefix is the captured prompt
        # minus that suffix.
        results.append({
            "name": p["name"],
            "captured_tokens": len(a),
            "replay_tokens": len(b),
            "common_prefix_tokens": common,
            "captured_not_shared": len(a) - common,
            "whole_4096_blocks_shared": common // 4096,
            "whole_4096_blocks_in_captured": len(a) // 4096,
            "replay_new_tokens": len(b) - common,
            "text_prefix_equal": b_text.startswith(a_text[: a_text.rfind("<|im_start|>assistant")]),
        })
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    for r in results:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
