"""Generate notebooks/colab_serve.ipynb and notebooks/kaggle_serve.ipynb.

Keeping the notebooks generated from one script means the two platforms never
drift apart, and the committed .ipynb files always have empty outputs — no
tunnel URL and no API key ever land in git.

    python scripts/build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_URL = "https://github.com/iramarfalcao/gpu-free-serve.git"
OUT = Path(__file__).resolve().parent.parent / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


def secrets_cell(platform: str) -> dict:
    if platform == "colab":
        reader = """
from google.colab import userdata

def read_secret(name):
    try:
        return userdata.get(name)
    except Exception:
        return None
"""
        where = "Colab: click the key icon in the left sidebar and add them there."
    else:
        reader = """
from kaggle_secrets import UserSecretsClient

_client = UserSecretsClient()

def read_secret(name):
    try:
        return _client.get_secret(name)
    except Exception:
        return None
"""
        where = "Kaggle: Add-ons > Secrets (and turn Internet ON in the right sidebar)."

    return code(
        f"""
# Secrets are read from the platform's secret manager and never written into
# this notebook. {where}
#
#   GITHUB_TOKEN     scope `gist` -> lets `gpufree connect --gist` work
#   HF_TOKEN         only for gated models (Llama, Gemma, ...)
#   NGROK_AUTHTOKEN  only if TUNNEL = "ngrok"
#   GPUFREE_API_KEY  optional fixed API key; otherwise a random one is generated
import os
{reader.strip()}

for name in ("GITHUB_TOKEN", "HF_TOKEN", "NGROK_AUTHTOKEN", "GPUFREE_API_KEY"):
    value = read_secret(name)
    if value:
        os.environ[name] = value
        print(f"{{name}}: loaded")
    else:
        print(f"{{name}}: not set (fine if you don't need it)")
"""
    )


def build(platform: str) -> dict:
    workdir = "/content" if platform == "colab" else "/kaggle/working"
    accel = (
        "Runtime > Change runtime type > T4 GPU"
        if platform == "colab"
        else "Settings > Accelerator > GPU T4 x2 (and Internet: On)"
    )
    title = "Colab" if platform == "colab" else "Kaggle"

    cells = [
        md(
            f"""
# gpu-free-serve — {title}

Boots an LLM on this notebook's free GPU and exposes an OpenAI-compatible
endpoint you can call from your own machine.

**Before running:** {accel}

Then run the cells top to bottom. The last cell keeps running — that is the
server. Closing this tab kills the endpoint.
"""
        ),
        code("!nvidia-smi"),
        code(
            f"""
# Clone the repo (or update it if it is already here).
import os, subprocess

os.chdir("{workdir}")
if os.path.isdir("gpu-free-serve"):
    subprocess.run(["git", "-C", "gpu-free-serve", "pull", "--ff-only"], check=False)
else:
    subprocess.run(["git", "clone", "--depth", "1", "{REPO_URL}"], check=True)
os.chdir("{workdir}/gpu-free-serve")
print(os.getcwd())
"""
        ),
        secrets_cell(platform),
        md(
            """
## Pick a model

The catalog below comes from `models.yaml` in the repo. Edit that file (and push)
to add your own models — or set `MODEL` to a raw id:

* `Qwen/Qwen2.5-7B-Instruct-AWQ` → served by vLLM
* `ollama:llama3.1:8b` → served by Ollama
"""
        ),
        code("!python -m gpufree.server.serve --list"),
        code(
            f"""
MODEL = "qwen2.5-7b-awq"   # a catalog name, `org/model`, or `ollama:tag`
TUNNEL = "cloudflare"      # "cloudflare" (no account) or "ngrok" (needs authtoken)
PUBLISH = "gist"           # "gist" publishes the endpoint; "none" just prints it

# This cell blocks while the server is alive. First run installs vLLM and
# downloads the weights (10-15 min).
!python -m gpufree.server.serve --model {{MODEL}} --tunnel {{TUNNEL}} --publish {{PUBLISH}}
"""
        ),
        md(
            """
## Use it from your machine

```bash
pipx install git+https://github.com/iramarfalcao/gpu-free-serve.git   # once

gpufree connect --gist          # or: gpufree connect --url <url> --key <key>
gpufree status
gpufree chat "explain attention in two sentences"
gpufree proxy                   # http://127.0.0.1:8787/v1 — no key needed
```

Point any OpenAI client at the proxy:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="not-needed")
```

**Do not commit this notebook with its output** — the banner prints the API key.
Edit > Clear all outputs first.
"""
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for platform in ("colab", "kaggle"):
        path = OUT / f"{platform}_serve.ipynb"
        path.write_text(json.dumps(build(platform), indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
