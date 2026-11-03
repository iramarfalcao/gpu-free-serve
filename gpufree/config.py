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


def _parse_catalog_yaml_lite(text: str) -> dict:
    """Minimal parser for our models.yaml shape (no pyyaml on the client)."""
    import re

    default = None
    models: dict[str, dict] = {}
    current: str | None = None
    in_models = False
    in_args = False
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw.startswith("default:"):
            default = raw.split(":", 1)[1].strip().strip("\"'")
            continue
        if raw.startswith("models:"):
            in_models = True
            continue
        if not in_models:
            continue
        m_name = re.match(r"^  ([A-Za-z0-9._-]+):\s*$", raw)
        if m_name:
            current = m_name.group(1)
            models[current] = {"engine": "vllm", "id": "", "description": "", "args": {}}
            in_args = False
            continue
        if current is None:
            continue
        if re.match(r"^    args:\s*$", raw):
            in_args = True
            continue
        if in_args:
            m_arg = re.match(r"^      ([A-Za-z0-9._-]+):\s*(.+)\s*$", raw)
            if m_arg:
                key, val = m_arg.group(1), m_arg.group(2).strip().strip("\"'")
                try:
                    models[current]["args"][key] = int(val) if val.isdigit() else float(val)
                except ValueError:
                    models[current]["args"][key] = val
            elif raw.startswith("  ") and not raw.startswith("    "):
                in_args = False
            continue
        m_field = re.match(r'^    (engine|id|description):\s*(.+)\s*$', raw)
        if m_field:
            key, val = m_field.group(1), m_field.group(2).strip()
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            models[current][key] = val
    if not models:
        raise CatalogError("lite YAML parse found no models")
    return {"default": default, "models": models}


def _parse(text: str) -> dict:
    try:
        import yaml
    except ImportError:
        pass  # no pyyaml around: JSON or the lite catalog parser
    else:
        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return _parse_catalog_yaml_lite(text)
        except CatalogError as exc:
            raise CatalogError(
                "reading a YAML catalog needs pyyaml: `pip install pyyaml` "
                "(already available on Colab/Kaggle)."
            ) from exc


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
