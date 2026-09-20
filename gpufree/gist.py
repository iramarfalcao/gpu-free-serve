"""Publish/discover the endpoint through a secret GitHub Gist.

The Colab/Kaggle tunnel gets a new URL on every run. Instead of copy-pasting it,
the notebook writes `{url, key, model}` into a secret gist and your laptop reads
that same gist with `gpufree connect --gist`.

Needs a token with the `gist` scope on both ends (locally: `gh auth token`).
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request

API = "https://api.github.com"
FILENAME = "gpu-free-serve.json"
MARKER = "gpu-free-serve endpoint (auto-generated)"


class GistError(RuntimeError):
    pass


def find_token(explicit: str | None = None) -> str:
    """GitHub token: argument > env > `gh auth token` > Colab/Kaggle secrets."""
    if explicit:
        return explicit
    for var in ("GPUFREE_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        out = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, timeout=10, check=False
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    for getter in (_colab_secret, _kaggle_secret):
        token = getter()
        if token:
            return token
    raise GistError(
        "no GitHub token found. Set GITHUB_TOKEN (scope `gist`) or run `gh auth login` locally."
    )


def _colab_secret() -> str | None:
    try:
        from google.colab import userdata  # type: ignore

        return userdata.get("GITHUB_TOKEN")
    except Exception:
        return None


def _kaggle_secret() -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        return UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception:
        return None


def _call(method: str, path: str, token: str, payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        API + path,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "gpu-free-serve",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise GistError(f"GitHub returned {exc.code} for {method} {path}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise GistError(f"network failure talking to the GitHub API: {exc}") from exc


def _find_gist_id(token: str) -> str | None:
    for gist in _call("GET", "/gists?per_page=100", token):
        if gist.get("description") == MARKER and FILENAME in (gist.get("files") or {}):
            return gist["id"]
    return None


def publish(payload: dict, token: str | None = None) -> str:
    """Write the endpoint to the gist (created on first run, updated afterwards)."""
    token = find_token(token)
    body = {
        "description": MARKER,
        "files": {FILENAME: {"content": json.dumps(payload, indent=2, ensure_ascii=False)}},
    }
    gist_id = _find_gist_id(token)
    if gist_id:
        result = _call("PATCH", f"/gists/{gist_id}", token, body)
    else:
        body["public"] = False
        result = _call("POST", "/gists", token, body)
    return result.get("html_url", "")


def discover(token: str | None = None) -> dict:
    """Read the endpoint published by the notebook."""
    token = find_token(token)
    gist_id = _find_gist_id(token)
    if not gist_id:
        raise GistError(
            "nothing published yet. Run the notebook with `--publish gist`, "
            "or use `gpufree connect --url ... --key ...`."
        )
    gist = _call("GET", f"/gists/{gist_id}", token)
    content = (gist.get("files") or {}).get(FILENAME, {}).get("content")
    if not content:
        raise GistError("the gist exists but is empty.")
    return json.loads(content)
