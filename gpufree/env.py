"""Dependency-free `.env` reader.

Secrets (GitHub token, Hugging Face token, ngrok authtoken, the server API key)
always live in environment variables or in a local `.env` — never in the code,
never in the model catalog, never in a committed notebook. `.env` is gitignored;
what ships in the repo is `.env.example`.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load KEY=VALUE pairs into the environment. Returns what was read."""
    candidates = [Path(path)] if path else [Path.cwd() / ".env", REPO_ROOT / ".env"]
    loaded: dict[str, str] = {}
    for candidate in candidates:
        if not candidate.is_file():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip().removeprefix("export ").strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            loaded[key] = value
            if override or key not in os.environ:
                os.environ[key] = value
        break  # first .env found wins
    return loaded


def redact(value: str | None, keep: int = 4) -> str:
    """Format a secret for logs: only the tail is shown."""
    if not value:
        return "(empty)"
    if len(value) <= keep:
        return "*" * len(value)
    return "*" * (len(value) - keep) + value[-keep:]
