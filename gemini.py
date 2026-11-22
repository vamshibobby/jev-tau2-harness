"""Minimal REST client for Gemini, used by both agents.

Why not an SDK: we need the time to the first token of every call, the raw response parts (Gemini 3 attaches "thought
signatures" to function calls and rejects a function call sent back without its signature), and nothing else.

Backends (same request and response shapes):
  * Vertex AI, when GOOGLE_APPLICATION_CREDENTIALS points at a service-account key. Set VERTEX_PROJECT, and
    VERTEX_LOCATION if not "global" (Gemini 3.8 Flash is served from the global endpoint).
  * Gemini API, when GEMINI_API_KEY is set instead.

History is kept in OpenAI chat format by the agents; `to_contents` converts it to Gemini `contents`.
Credentials are never printed or logged.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import timezone

import httpx

SKIP_SIG = "skip_thought_signature_validator"   # documented bypass for function calls Gemini did not produce itself

# $ per 1M tokens (standard tier). Thinking tokens bill as output.
# "cached": input tokens Vertex served from its automatic (implicit) prompt cache, billed at 10% of the input price
PRICE = {"gemini-3.8-flash": {"in": 0.75, "cached": 0.075, "out": 3.75},
         "gemini-3.5-flash-lite": {"in": 0.30, "cached": 0.03, "out": 2.50}}

_client = httpx.Client(timeout=180, limits=httpx.Limits(max_connections=96, max_keepalive_connections=48))


def _vertex() -> bool:
    return bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))


class _Token:
    """Thread-safe OAuth token for the service account, refreshed 5 minutes before it expires."""
    lock = threading.Lock()
    creds = None

    @classmethod
    def get(cls) -> str:
        with cls.lock:
            if cls.creds is None:
                from google.oauth2 import service_account
                cls.creds = service_account.Credentials.from_service_account_file(
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"], scopes=["https://www.googleapis.com/auth/cloud-platform"])
            exp = getattr(cls.creds, "expiry", None)   # google-auth returns a *naive* datetime in UTC
            left = (exp.replace(tzinfo=timezone.utc).timestamp() - time.time()) if exp else -1
            if not cls.creds.token or left < 300:
                import google.auth.transport.requests
                cls.creds.refresh(google.auth.transport.requests.Request())
            return cls.creds.token

    @classmethod
    def invalidate(cls) -> None:
        """Force a refresh on the next call (used after an HTTP 401)."""
        with cls.lock:
            if cls.creds is not None:
                cls.creds.token = None


def _headers() -> dict:
    if _vertex():
        return {"Authorization": f"Bearer {_Token.get()}", "Content-Type": "application/json"}
    return {"x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"}


def _model_url(model: str) -> str:
    if _vertex():
        loc = os.environ.get("VERTEX_LOCATION", "global")
        host = "aiplatform.googleapis.com" if loc == "global" else f"{loc}-aiplatform.googleapis.com"
        return f"https://{host}/v1/projects/{os.environ['VERTEX_PROJECT']}/locations/{loc}/publishers/google/models/{model}"
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}"


def tool_decls(openai_tools: list[dict]) -> list[dict]:
    """tau2's OpenAI-style tool schemas -> Gemini function declarations (JSON Schema passed through unchanged)."""
    fns = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
            "parametersJsonSchema": t["function"].get("parameters") or {"type": "object", "properties": {}}}
           for t in openai_tools]
    return [{"functionDeclarations": fns}]


def to_contents(msgs: list[dict]) -> list[dict]:
    """OpenAI-format history (no system message) -> Gemini `contents`.

    * Model turns that Gemini produced keep their raw parts (stored under the private key "_parts"), signatures included.
    * Model turns the harness produced itself (code-issued tool calls) carry the bypass signature.
    * Consecutive tool results merge into one user turn of function responses.
    """
    out: list[dict] = []
    names = {tc["id"]: tc["function"]["name"] for m in msgs for tc in m.get("tool_calls") or []}
    for m in msgs:
        if m["role"] == "user":
            out.append({"role": "user", "parts": [{"text": m["content"] or " "}]})
        elif m["role"] == "assistant":
            if m.get("_parts"):
                out.append({"role": "model", "parts": m["_parts"]})
                continue
            parts = [{"text": m["content"]}] if m.get("content") else []
            for tc in m.get("tool_calls") or []:
                parts.append({"functionCall": {"name": tc["function"]["name"], "args": json.loads(tc["function"]["arguments"] or "{}")},
                              "thoughtSignature": SKIP_SIG})
            out.append({"role": "model", "parts": parts or [{"text": " "}]})
        elif m["role"] == "tool":
            try:
                resp = json.loads(m["content"])
                resp = resp if isinstance(resp, dict) else {"result": resp}
            except (json.JSONDecodeError, TypeError):
                resp = {"result": m["content"]}
            part = {"functionResponse": {"name": names.get(m.get("tool_call_id"), "tool"), "response": resp}}
            if out and out[-1]["role"] == "user" and all("functionResponse" in p for p in out[-1]["parts"]):
                out[-1]["parts"].append(part)
            else:
                out.append({"role": "user", "parts": [part]})
    if out and out[0]["role"] == "model":   # tau2 opens with an agent greeting; Gemini wants a user turn first
        out.insert(0, {"role": "user", "parts": [{"text": "[conversation started]"}]})
    return out


def generate(model: str, contents: list[dict], *, system: str, tools: list[dict], thinking: str = "low"):
    """One streamed call. Returns (text, function_calls, raw_parts, usage, t_first_token, error).

    `t_first_token` is the wall-clock time of the first visible output: answer text or a function call (thoughts don't
    count). `error` is None on success, otherwise a short description (the caller retries).
    """
    body = {"contents": contents, "generationConfig": {"thinkingConfig": {"thinkingLevel": thinking}},
            "systemInstruction": {"parts": [{"text": system}]}}
    if tools:
        body["tools"] = tools
    text, calls, parts, usage, first = "", [], [], {}, None
    try:
        with _client.stream("POST", f"{_model_url(model)}:streamGenerateContent?alt=sse", headers=_headers(), json=body) as r:
            if r.status_code != 200:
                return "", [], [], {}, None, f"HTTP {r.status_code} {r.read()[:400]!r}"
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                d = json.loads(line[6:])
                usage = d.get("usageMetadata") or usage
                for c in d.get("candidates", []):
                    for p in (c.get("content") or {}).get("parts", []):
                        if first is None and ((p.get("text") and not p.get("thought")) or p.get("functionCall")):
                            first = time.time()
                        parts.append(p)
                        if p.get("functionCall"):
                            calls.append(p["functionCall"])
                        elif p.get("text") and not p.get("thought"):
                            text += p["text"]
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        return "", [], [], {}, None, f"{type(e).__name__}: {e}"
    return text, calls, _merge_text_parts(parts), usage, first, None


def _merge_text_parts(parts: list[dict]) -> list[dict]:
    """Streaming splits text into many parts; merge adjacent plain-text chunks (signed parts stay separate)."""
    out: list[dict] = []
    for p in parts:
        if set(p) <= {"text"} and out and set(out[-1]) <= {"text"}:
            out[-1] = {"text": out[-1]["text"] + p["text"]}
        else:
            out.append(dict(p))
    return out


def cost_of(usage: dict, model: str) -> float:
    """Dollar cost of one call from its usage metadata, at list price (implicitly cached input at the cached rate)."""
    p = PRICE[model]
    prompt = usage.get("promptTokenCount") or 0
    cached = usage.get("cachedContentTokenCount") or 0
    out = (usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0)
    return ((prompt - cached) * p["in"] + cached * p["cached"] + out * p["out"]) / 1e6
