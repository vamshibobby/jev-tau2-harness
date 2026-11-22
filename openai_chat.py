"""Streaming client for OpenAI-compatible chat endpoints, used for every agent model that is not Gemini-native:

  * Vertex AI Model-as-a-Service, e.g. Gemma 4 26B A4B: model "google/gemma-4-26b-a4b-it-maas" (global endpoint).
  * OpenRouter: model "openrouter/<openrouter id>", e.g. "openrouter/deepseek/deepseek-v4-flash" (OPENROUTER_API_KEY).
    The cost of each call is the amount OpenRouter reports as billed.

Same return shape as gemini.generate, so the agents treat all models alike:
    (text, function_calls, raw_parts, usage, t_first_token, error)
with function calls as {"name", "args"} and usage in Gemini's field names (promptTokenCount, candidatesTokenCount),
plus "cost" when the provider reports it.
"""
from __future__ import annotations

import json
import os
import time

import httpx

import gemini

# $ per 1M tokens, for providers that do not report a per-call cost (Vertex AI)
PRICE = {"google/gemma-4-26b-a4b-it-maas": {"in": 0.07, "out": 0.34}}
OPENROUTER = "openrouter/"

_client = httpx.Client(timeout=300, limits=httpx.Limits(max_connections=96, max_keepalive_connections=48))


def _endpoint(model: str) -> tuple[str, dict, str]:
    """(url, headers, model id to send) for a model name."""
    if model.startswith(OPENROUTER):
        return ("https://openrouter.ai/api/v1/chat/completions",
                {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, model[len(OPENROUTER):])
    loc = os.environ.get("VERTEX_LOCATION", "global")
    host = "aiplatform.googleapis.com" if loc == "global" else f"{loc}-aiplatform.googleapis.com"
    url = f"https://{host}/v1/projects/{os.environ['VERTEX_PROJECT']}/locations/{loc}/endpoints/openapi/chat/completions"
    return url, {"Authorization": f"Bearer {gemini._Token.get()}"}, model


def to_messages(msgs: list[dict], system: str, cache: bool = False) -> list[dict]:
    """The agents' OpenAI-format history -> request messages (private keys such as "_parts" dropped).

    The model's reasoning blocks from earlier turns (kept under "_parts") are sent back as `reasoning_details`, which
    Anthropic requires on tool-use turns when reasoning is on. With `cache`, Anthropic cache breakpoints go on the system
    prompt (caches tools + system) and on the newest message with text (caches the conversation so far)."""
    out = [{"role": "system", "content": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            if cache else system}]
    for m in msgs:
        if m["role"] == "assistant":
            d = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                d["tool_calls"] = m["tool_calls"]
            if isinstance(m.get("_parts"), list) and m["_parts"]:
                d["reasoning_details"] = m["_parts"]
            out.append(d)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            out.append({"role": "user", "content": m["content"] or " "})
    if len(out) > 1 and out[1]["role"] == "assistant":   # tau2 opens with an agent greeting
        out.insert(1, {"role": "user", "content": "[conversation started]"})
    if cache:
        for m in reversed(out[1:]):
            if isinstance(m.get("content"), str) and m["content"]:
                m["content"] = [{"type": "text", "text": m["content"], "cache_control": {"type": "ephemeral"}}]
                break
    return out


def _json_objects(s: str) -> list[dict]:
    """One or more JSON objects back to back. Some providers stream parallel calls of the same tool under one index
    and one id, so their arguments arrive concatenated: '{...}{...}'."""
    dec, out, i = json.JSONDecoder(), [], 0
    while i < len(s):
        if s[i].isspace():
            i += 1
            continue
        obj, i = dec.raw_decode(s, i)
        out.append(obj)
    return out


def generate(model: str, msgs: list[dict], *, system: str, tools: list[dict], reasoning: str | None = None):
    """`reasoning` (OpenRouter only): "low" | "medium" | "high" effort, sent as OpenRouter's `reasoning.effort`."""
    url, headers, model_id = _endpoint(model)
    cache = model.startswith(OPENROUTER + "anthropic/")  # Anthropic models cache only with explicit breakpoints
    body = {"model": model_id, "messages": to_messages(msgs, system, cache), "stream": True,
            "stream_options": {"include_usage": True}}
    if model.startswith(OPENROUTER):
        body["usage"] = {"include": True}            # OpenRouter adds the billed cost to the final usage chunk
        if reasoning:
            body["reasoning"] = _reasoning(model_id, reasoning)
        if os.environ.get("AGENT_PROVIDER"):        # pin one provider, so both arms of a comparison see the same one
            body["provider"] = {"order": [os.environ["AGENT_PROVIDER"]], "allow_fallbacks": False}
    if tools:
        body["tools"] = tools
    text, calls, usage, first, rdetails = "", {}, {}, None, []
    try:
        with _client.stream("POST", url, headers=headers, json=body) as r:
            if r.status_code != 200:
                return "", [], [], {}, None, f"HTTP {r.status_code} {r.read()[:400]!r}"
            for line in r.iter_lines():
                if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                    continue
                d = json.loads(line[6:])
                if d.get("error"):                       # OpenRouter reports mid-stream provider errors this way
                    return "", [], [], {}, None, f"stream error {str(d['error'])[:300]}"
                if d.get("usage"):
                    u = d["usage"]
                    usage = {"promptTokenCount": u.get("prompt_tokens"), "candidatesTokenCount": u.get("completion_tokens"),
                             "cachedContentTokenCount": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                             "thoughtsTokenCount": (u.get("completion_tokens_details") or {}).get("reasoning_tokens")}
                    if u.get("cost") is not None:
                        usage["cost"] = u["cost"]
                for ch in d.get("choices", []):
                    delta = ch.get("delta") or {}
                    if first is None and (delta.get("content") or delta.get("tool_calls")):
                        first = time.time()
                    text += delta.get("content") or ""
                    rdetails += delta.get("reasoning_details") or []
                    for tc in delta.get("tool_calls") or []:          # tool calls arrive in fragments, keyed by index
                        key = tc.get("index", 0)
                        c = calls.get(key)
                        # some providers send parallel calls under one index; a new id starts a new call
                        if c is not None and tc.get("id") and c.get("id") and tc["id"] != c["id"]:
                            calls[(key, tc["id"])] = calls.pop(key)
                            c = None
                        if c is None:
                            c = calls[key] = {"name": "", "arguments": "", "id": tc.get("id")}
                        c["id"] = c["id"] or tc.get("id")
                        fn = tc.get("function") or {}
                        c["name"] += fn.get("name") or ""
                        c["arguments"] += fn.get("arguments") or ""
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        return "", [], [], {}, None, f"{type(e).__name__}: {e}"
    out = []
    for c in calls.values():
        try:
            out += [{"name": c["name"], "args": a} for a in _json_objects(c["arguments"] or "{}")]
        except json.JSONDecodeError:
            return "", [], [], {}, None, f"bad tool-call arguments: {c['arguments'][:200]}"
    return text, out, _merge_reasoning(rdetails) or None, usage, first, None


def _reasoning(model_id: str, level: str) -> dict:
    """OpenRouter `reasoning` setting for a level. DeepSeek V4 ignores effort levels, so for it "low" switches thinking
    off and anything higher switches it on; other models get the effort level as is."""
    if model_id.startswith("deepseek/"):
        return {"enabled": level != "low"}
    return {"effort": level}


def _merge_reasoning(chunks: list[dict]) -> list[dict]:
    """Streamed reasoning_details arrive in fragments; join consecutive fragments of the same block (same type and
    index) so they can be sent back whole, signatures included."""
    out: list[dict] = []
    for c in chunks:
        if out and out[-1].get("type") == c.get("type") and out[-1].get("index") == c.get("index"):
            prev = out[-1]
            for k in ("text", "summary", "data"):
                if c.get(k):
                    prev[k] = (prev.get(k) or "") + c[k]
            if c.get("signature"):
                prev["signature"] = c["signature"]
        else:
            out.append(dict(c))
    return out


def cost_of(usage: dict, model: str) -> float:
    if "cost" in usage:                                  # billed amount reported by the provider (OpenRouter)
        return float(usage["cost"])
    p = PRICE[model]
    return ((usage.get("promptTokenCount") or 0) * p["in"] + (usage.get("candidatesTokenCount") or 0) * p["out"]) / 1e6
