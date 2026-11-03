"""UI preferences and session clock (provider, model, idle countdown)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .state import CONFIG_DIR

PREFS_FILE = CONFIG_DIR / "ui_prefs.json"

# Documented free-tier ceilings (README). Colab disconnects on idle ~90 min;
# Kaggle sessions top out around 12 hours.
PROVIDER_LIMITS_SEC = {
    "colab": 90 * 60,
    "kaggle": 12 * 60 * 60,
}

NOTEBOOK_URLS = {
    "colab": (
        "https://colab.research.google.com/github/iramarfalcao/"
        "gpu-free-serve/blob/main/notebooks/colab_serve.ipynb"
    ),
    "kaggle": (
        "https://www.kaggle.com/code?newNotebook=true"
        # User uploads notebooks/kaggle_serve.ipynb or clones the repo notebook.
    ),
}

KAGGLE_NOTEBOOK_HINT = (
    "https://github.com/iramarfalcao/gpu-free-serve/blob/main/notebooks/kaggle_serve.ipynb"
)


def defaults() -> dict:
    return {
        "provider": "colab",
        "model": "qwen2.5-7b-awq",
        "session_started_at": None,
        "proxy_host": "127.0.0.1",
        "proxy_port": 8787,
    }


def load() -> dict:
    data = defaults()
    if PREFS_FILE.exists():
        try:
            saved = json.loads(PREFS_FILE.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                data.update(saved)
        except (json.JSONDecodeError, OSError):
            pass
    return data


def save(data: dict) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    merged = defaults()
    merged.update({k: v for k, v in data.items() if k in merged})
    PREFS_FILE.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    try:
        PREFS_FILE.chmod(0o600)
    except OSError:
        pass
    return PREFS_FILE


def mark_session_started(prefs: dict | None = None) -> dict:
    data = prefs or load()
    if not data.get("session_started_at"):
        data["session_started_at"] = time.time()
        save(data)
    return data


def clear_session_clock(prefs: dict | None = None) -> dict:
    data = prefs or load()
    data["session_started_at"] = None
    save(data)
    return data


def countdown(prefs: dict | None = None) -> dict:
    """Return remaining seconds until the free-tier idle/session ceiling."""
    data = prefs or load()
    provider = data.get("provider") or "colab"
    limit = PROVIDER_LIMITS_SEC.get(provider, PROVIDER_LIMITS_SEC["colab"])
    started = data.get("session_started_at")
    now = time.time()
    if not started:
        return {
            "provider": provider,
            "limit_sec": limit,
            "started_at": None,
            "elapsed_sec": 0,
            "remaining_sec": limit,
            "running": False,
            "label": _label(provider, limit),
        }
    elapsed = max(0, int(now - float(started)))
    remaining = max(0, int(limit - elapsed))
    return {
        "provider": provider,
        "limit_sec": limit,
        "started_at": started,
        "elapsed_sec": elapsed,
        "remaining_sec": remaining,
        "running": True,
        "exhausted": remaining <= 0,
        "label": _label(provider, limit),
    }


def _label(provider: str, limit: int) -> str:
    if provider == "kaggle":
        return f"Limite de sessão Kaggle (~{limit // 3600} h)"
    return f"Idle Colab (~{limit // 60} min sem uso)"
