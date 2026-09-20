"""Local state: where the tunnel URL and the session API key are kept."""

from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("GPUFREE_HOME") or (Path.home() / ".config" / "gpu-free-serve"))
CONFIG_FILE = CONFIG_DIR / "config.json"


class NotConnected(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "no endpoint saved. Run `gpufree connect --gist` or "
            "`gpufree connect --url https://... --key ...`."
        )


def save(data: dict) -> Path:
    """Write the endpoint with mode 600 — the file holds the API key."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    CONFIG_FILE.chmod(0o600)
    return CONFIG_FILE


def load() -> dict:
    # Environment wins over the file: handy for CI or a throwaway session.
    env_url = os.environ.get("GPUFREE_URL")
    if env_url:
        return {"url": env_url, "api_key": os.environ.get("GPUFREE_API_KEY", "")}
    if not CONFIG_FILE.exists():
        raise NotConnected
    return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))


def clear() -> None:
    CONFIG_FILE.unlink(missing_ok=True)
