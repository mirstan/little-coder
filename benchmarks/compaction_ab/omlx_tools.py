"""Mirror of omlx 0.7.0's convert_tools_for_template (omlx/api/tool_calling.py:3459)
and _copy_schema_with_template_defaults (:46), reimplemented so the offline
renderer feeds the chat template the same tool structure omlx does: wrapped
{"type":"function","function":{name, description, parameters}}, with `strict`
dropped and every nested schema given a string description."""

_SCHEMA_KEYS = {"items", "additionalProperties", "contains", "propertyNames", "not", "if", "then", "else"}


def _desc(v):
    if v is None:
        return ""
    return v if isinstance(v, str) else str(v)


def _copy(value, is_schema):
    if isinstance(value, dict):
        out = {}
        for k, child in value.items():
            if k == "properties" and isinstance(child, dict):
                out[k] = {n: _copy(s, True) for n, s in child.items()}
            elif k in _SCHEMA_KEYS:
                out[k] = _copy(child, True)
            elif k in {"oneOf", "anyOf", "allOf", "prefixItems"} and isinstance(child, list):
                out[k] = [_copy(i, True) for i in child]
            else:
                out[k] = _copy(child, False)
        if is_schema:
            out["description"] = _desc(out.get("description"))
        return out
    if isinstance(value, list):
        return [_copy(i, False) for i in value]
    return value


def convert_tools_for_template(tools):
    if not tools:
        return None
    out = []
    for t in tools:
        f = t.get("function") if isinstance(t, dict) else None
        if t.get("type") == "function" and f:
            params = f.get("parameters", {"type": "object", "properties": {}})
            if params is None:
                params = {"type": "object", "properties": {}}
            out.append({"type": "function", "function": {
                "name": f.get("name", ""), "description": _desc(f.get("description", "")),
                "parameters": _copy(params, False)}})
    return out or None
