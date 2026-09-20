"""Calls to the remote OpenAI-compatible API (stdlib, no SDK required)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Iterator


class ApiError(RuntimeError):
    pass


def _request(state: dict, path: str, payload: dict | None = None, timeout: float = 600):
    url = state["url"].rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json", "Accept-Encoding": "identity"}
    if state.get("api_key"):
        headers["Authorization"] = f"Bearer {state['api_key']}"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:400]
        if exc.code == 401:
            raise ApiError(
                "API key rejected — did the notebook restart? Run `gpufree connect` again."
            ) from exc
        raise ApiError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ApiError(f"could not reach {url} ({exc}). Is the notebook still running?") from exc


def models(state: dict) -> list[str]:
    with _request(state, "/models", timeout=30) as resp:
        data = json.loads(resp.read())
    return [item["id"] for item in data.get("data", [])]


def chat_stream(
    state: dict, messages: list[dict], model: str | None = None, **params
) -> Iterator[str]:
    payload = {
        "model": model or state.get("model_id") or state.get("model") or "default",
        "messages": messages,
        "stream": True,
        **params,
    }
    with _request(state, "/chat/completions", payload) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                return
            try:
                delta = json.loads(chunk)["choices"][0].get("delta", {})
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if delta.get("content"):
                yield delta["content"]
