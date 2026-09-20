"""Model catalog loading (models.yaml)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG = REPO_ROOT / "models.yaml"

VALID_ENGINES = ("vllm", "ollama")


@dataclass
class ModelSpec:
    name: str
    engine: str
    id: str
    description: str = ""
    args: dict = field(default_factory=dict)


class CatalogError(RuntimeError):
    pass


def _parse(text: str) -> dict:
    try:
        import yaml
    except ImportError:
        pass  # no pyyaml around: the same catalog written as JSON still works
    else:
        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise CatalogError(
            "reading a YAML catalog needs pyyaml: `pip install pyyaml` "
            "(already available on Colab/Kaggle)."
        ) from None


def load_catalog(path: str | Path | None = None) -> tuple[dict[str, ModelSpec], str | None]:
    """Return ({name: ModelSpec}, default_name)."""
    path = Path(path or os.environ.get("GPUFREE_CATALOG") or DEFAULT_CATALOG)
    if not path.exists():
        raise CatalogError(f"catalog not found: {path}")

    data = _parse(path.read_text(encoding="utf-8")) or {}
    raw_models = data.get("models") or {}
    if not raw_models:
        raise CatalogError(f"{path} has no entries under `models:`")

    models: dict[str, ModelSpec] = {}
    for name, entry in raw_models.items():
        entry = entry or {}
        engine = str(entry.get("engine", "vllm")).lower()
        if engine not in VALID_ENGINES:
            raise CatalogError(
                f"model '{name}': unknown engine '{engine}' (use {' or '.join(VALID_ENGINES)})"
            )
        model_id = entry.get("id")
        if not model_id:
            raise CatalogError(f"model '{name}': missing `id`")
        models[name] = ModelSpec(
            name=name,
            engine=engine,
            id=str(model_id),
            description=str(entry.get("description", "")),
            args=dict(entry.get("args") or {}),
        )

    default = data.get("default")
    if default is not None and default not in models:
        raise CatalogError(f"default '{default}' is not defined under `models:`")
    return models, default


def resolve(name: str | None, path: str | Path | None = None) -> ModelSpec:
    """Resolve a catalog name, or an ad-hoc model id.

    Also accepts models that are not in the catalog:
      `org/model`          -> vLLM
      `ollama:llama3.1:8b` -> Ollama
    """
    models, default = load_catalog(path)
    name = name or os.environ.get("GPUFREE_MODEL") or default
    if not name:
        raise CatalogError("no model given and the catalog defines no `default:`")

    if name in models:
        return models[name]

    if name.startswith("ollama:"):
        model_id = name.split(":", 1)[1]
        return ModelSpec(name=model_id, engine="ollama", id=model_id, description="ad-hoc")
    if "/" in name:
        return ModelSpec(name=name, engine="vllm", id=name, description="ad-hoc")

    available = ", ".join(sorted(models))
    raise CatalogError(f"model '{name}' is not in the catalog. Available: {available}")
