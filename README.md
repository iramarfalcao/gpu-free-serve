# gpu-free-serve

[![License: MIT](https://img.shields.io/github/license/iramarfalcao/gpu-free-serve)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![Engines: vLLM, Ollama](https://img.shields.io/badge/engines-vLLM%20%7C%20Ollama-informational)
![Runs on: Colab, Kaggle](https://img.shields.io/badge/runs%20on-Colab%20%7C%20Kaggle-orange)

Serve open LLMs on the free GPUs of Google Colab and Kaggle and call them from your own machine through an OpenAI-compatible API.

## What it is

A Colab/Kaggle notebook starts a model with vLLM or Ollama, puts an API-key gateway in front of it and opens a public HTTPS tunnel. A small stdlib-only Python client (`gpufree`) finds that endpoint and re-exposes it as `http://127.0.0.1:8787/v1`, so any OpenAI-compatible tool can use it. You choose the model in `models.yaml`.

## What it does

It lets you run 7B-class (and, on Kaggle, up to ~32B 4-bit) models for chat, code assistance and embeddings without owning a GPU or paying for an inference API. It is aimed at personal, interactive use: trying models, plugging a coding model into an editor, prototyping against an OpenAI-style endpoint.

```bash
gpufree chat "explain attention in two sentences"
# or, from any OpenAI client:
curl http://127.0.0.1:8787/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-7b-awq","messages":[{"role":"user","content":"hi"}]}'
```

## Features

- Notebooks for Colab (`notebooks/colab_serve.ipynb`) and Kaggle (`notebooks/kaggle_serve.ipynb`).
- Model catalog in `models.yaml` with two engines: vLLM (Hugging Face models) and Ollama (GGUF). You can also pass a Hub id or `ollama:<tag>` directly with `--model`.
- Catalog entries for general chat models and for coding models (Qwen2.5-Coder 1.5B, 7B, 14B and 32B, Devstral 24B, among others).
- API-key gateway in front of the engine; the engine binds only to `127.0.0.1`.
- Tunnel through cloudflared (no account) or ngrok (authtoken, optional fixed domain).
- Endpoint discovery through a secret GitHub Gist: `gpufree connect --gist`.
- Local reverse proxy (`gpufree proxy`) that injects the key, so clients use `api_key="not-needed"`.
- CLI: `connect`, `status`, `models`, `chat`, `proxy`, `env`, `ui`, `disconnect`.
- Local control panel (`gpufree ui`): pick Colab (Google login) or Kaggle, choose a model, idle/session countdown, start the local proxy and copy OpenAI base URL snippets for OpenCode, Pi, Cursor, Continue.
- Routes exposed by the engine, with vLLM: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`, with streaming.
- Automatic handling of T4 limits (`dtype: half`).
- Zero runtime dependencies on the client side (`pyyaml` and `pyngrok` are only needed on the notebook side, and the notebook installs them).

## How to run

Requirements: a Google or Kaggle account for the notebook; Python 3.9+ on your machine. Optional: GitHub token with `gist` scope (for `--gist`), Hugging Face token (gated models), ngrok authtoken.

**1. Install the client**

```bash
pipx install git+https://github.com/iramarfalcao/gpu-free-serve.git
# or
git clone https://github.com/iramarfalcao/gpu-free-serve.git
cd gpu-free-serve && pip install -e .
```

**2. Start the model in the notebook.** Open one of the notebooks and run the cells top to bottom:

- Colab: `notebooks/colab_serve.ipynb`. Runtime > Change runtime type > T4 GPU.
- Kaggle: `notebooks/kaggle_serve.ipynb`. Settings > Accelerator > GPU T4 x2, and Internet: On.

The last cell blocks while the server is alive and prints the model, the endpoint, the API key and the gist URL. The first run takes about 10 to 15 minutes with vLLM (install plus weights download); Ollama models are up in about a minute.

**3. Connect from your machine**

```bash
gpufree connect --gist        # reads the URL and key from the secret gist
# or, without a GitHub token:
gpufree connect --url https://vast-pink-wolf.trycloudflare.com --key 8Kq2...

gpufree status                # is it alive? which models?
gpufree chat "write a haiku about GPUs"
gpufree proxy                 # http://127.0.0.1:8787/v1, no key needed locally

# or use the local panel (browser):
gpufree ui                    # http://127.0.0.1:8765/
```

To list the catalog: `python -m gpufree.server.serve --list`.

The repository has no test suite and no CI workflow. The notebooks are generated: edit `scripts/build_notebooks.py` and re-run it instead of editing the `.ipynb` files.

## What is missing

No roadmap, backlog or TODO is recorded in the repository. Limitations that exist today:

- Sessions end (Colab on idle and after a few hours; Kaggle after 12 hours, about 30 GPU hours per week). Each restart changes the tunnel URL and, unless `GPUFREE_API_KEY` is set, the API key. The UI countdown uses those published ceilings; it cannot prevent Colab/Kaggle from reclaiming the GPU.
- “Login Google/Kaggle” in `gpufree ui` opens the platform notebooks so you sign in there. There is no OAuth API that starts a free GPU from this app alone.
- `gpufree connect` stores one endpoint at a time. Using a large chat model and a small autocomplete model together means two notebooks and `--url`/`--key` for each.
- Not meant for production traffic.

## Store status

Not applicable. This is a developer tool, not a store app.

---

## More details

Contents: [How it works](#how-it-works) · [Choosing models](#choosing-models-modelsyaml) · [Capacity](#capacity-what-fits-where) · [Coding models](#coding-models) · [Secrets and .env](#secrets-and-env) · [CLI reference](#cli-reference) · [Server options](#server-options-notebook-side) · [Using it from code and editors](#using-it-from-your-code-and-editor) · [Cloudflare vs ngrok](#cloudflare-vs-ngrok) · [Free-tier limits](#free-tier-limits) · [Troubleshooting](#troubleshooting) · [Security notes](#security-notes) · [Project layout](#project-layout) · [License](#license)

### How it works

```
  Colab / Kaggle notebook (free GPU)              your machine
  ┌──────────────────────────────────┐          ┌───────────────────────────┐
  │ vLLM or Ollama on 127.0.0.1      │          │ gpufree proxy             │
  │            ↑ localhost only      │          │ http://127.0.0.1:8787/v1  │
  │ gateway (API key) on :8500       │          │            ↑              │
  │            ↑                     │          │ OpenAI SDK, curl, Cursor, │
  │ cloudflared / ngrok tunnel  ─────┼── https ──┼──→ Continue, LangChain... │
  └──────────────────────────────────┘          └───────────────────────────┘
```

1. The notebook installs the engine, downloads the model and serves it on localhost.
2. A small gateway (stdlib only) sits in front of it and requires an API key.
3. A tunnel gives that gateway a public HTTPS URL.
4. The URL and key are written to a secret GitHub Gist, so your machine finds them with `gpufree connect --gist` and you do not copy a new URL on every run.
5. `gpufree proxy` re-exposes the endpoint on `http://127.0.0.1:8787/v1` with the key injected.

The engine itself is never exposed. Only the authenticated gateway is.

### Choosing models (`models.yaml`)

Each entry becomes a `--model <name>` you can run:

```yaml
default: qwen2.5-7b-awq        # used when --model is omitted

models:
  qwen2.5-7b-awq:              # the name you type
    engine: vllm               # vllm | ollama
    id: Qwen/Qwen2.5-7B-Instruct-AWQ   # HF repo (vllm) or Ollama tag (ollama)
    description: "Good all-rounder. Fits a single 16GB T4."
    args:                      # passed straight to the engine
      max-model-len: 8192
      gpu-memory-utilization: 0.90
```

| Field | Required | Meaning |
|---|---|---|
| `engine` | no (default `vllm`) | `vllm` for Hugging Face models, `ollama` for GGUF |
| `id` | yes | `org/model` on the Hub, or an Ollama tag like `llama3.1:8b` |
| `description` | no | Shown by `--list` |
| `args` | no | vLLM flags without the `--` (`max-model-len`, `tensor-parallel-size`, `dtype`, `quantization`, and so on). Booleans become bare flags. |

You can also skip the catalog:

```bash
python -m gpufree.server.serve --model Qwen/Qwen2.5-7B-Instruct-AWQ   # vLLM
python -m gpufree.server.serve --model ollama:llama3.1:8b             # Ollama
```

Which engine:

| | vLLM | Ollama |
|---|---|---|
| Startup | slow (installs and compiles, ~10 min the first time) | fast (~1 min) |
| Throughput | high, real batching | moderate |
| Models | anything on the Hub (prefer AWQ/GPTQ on a T4) | GGUF from the Ollama library |
| Best for | serving a model for hours | quick tests, tiny GPUs, embeddings |

A 7 to 8B model in 4-bit AWQ is the usual choice on a free T4.

### Capacity: what fits where

What the free tiers give you:

| | Colab (free) | Kaggle |
|---|---|---|
| GPU | 1x T4, 16 GB VRAM (~15.0 GiB usable) | 2x T4 (16 GB each, not pooled) or 1x P100 16 GB |
| System RAM | ~12.7 GB | ~30 GB |
| Disk | ~100 GB under `/content` | ~70 GB, with `/kaggle/working` capped near 20 GB |
| vCPU | 2 | 4 |

VRAM decides whether a model runs, not system RAM:

```
weights ≤ VRAM × gpu-memory-utilization − KV cache − ~1 GiB (activations, CUDA graphs)
```

| Setup | Total VRAM | Budget at `gpu-memory-utilization: 0.90` | Practical weight ceiling |
|---|---|---|---|
| Colab free (1x T4) | 15.0 GiB | ~13.5 GiB | ~10-11 GiB (leaving ~2 GiB of KV cache) |
| Kaggle, 1 GPU (T4 or P100) | 15.0 GiB | ~13.5 GiB | ~10-11 GiB |
| Kaggle, 2x T4 with `tensor-parallel-size: 2` | 30.0 GiB | ~27 GiB | ~21-22 GiB |

Roughly how much VRAM one billion parameters costs:

| Precision | GiB per 1B params | Ceiling on one T4 | Ceiling on Kaggle with TP=2 |
|---|---|---|---|
| fp16 / bf16 | ~2.0 | ~5B (a 7B in fp16 is 14 GiB and does not fit) | ~10B |
| 8-bit (GPTQ/AWQ int8) | ~1.0 | ~10B | ~21B |
| 4-bit (AWQ/GPTQ) | ~0.55-0.65 | ~16-18B | ~32-34B |

Real models:

| Model | Weights | 1x T4 | Kaggle 2x T4 |
|---|---|---|---|
| Phi-3.5-mini 3.8B fp16 | 7.6 GiB | yes | yes |
| Qwen2.5 7B / Llama 3.1 8B AWQ | ~5.5 GiB | yes, with room for long context | yes |
| Qwen2.5 7B fp16 | ~15 GiB | no | yes |
| Qwen2.5 14B AWQ | ~9.5 GiB | tight, short KV cache | yes, comfortable |
| Qwen2.5 32B AWQ | ~19.5 GiB | no | tight, context around 4-8k |
| Llama 3.1 70B AWQ | ~39 GiB | no | no |

Two things that cause trouble in practice:

1. The KV cache takes what is left. On a 7B with GQA it costs ~60-100 MiB per 1k tokens per sequence, so `max-model-len: 8192` already reserves ~0.5-0.8 GiB. If the engine runs out of memory at boot, lower `max-model-len` before switching models.
2. TP=2 is not unified memory. Kaggle's 32 GiB become one pool only with `tensor-parallel-size: 2`. Without it you get 16 GiB and the second GPU idles.

Colab's ~12.7 GB of RAM and ~100 GB of disk handle any download in the table above. On Kaggle the Hugging Face cache lives in `~/.cache` (on the larger disk), not in the ~20 GB `/kaggle/working`, so a 32B AWQ (19 GB) downloads fine, slowly.

Ollama has a different ceiling. GGUF offloads layers to system RAM, so on Kaggle (30 GB RAM + 16 GB VRAM) even a 70B Q4 will load, at a few tokens per second because the CPU layers dominate. vLLM never spills to RAM: it fits in VRAM or fails.

### Coding models

These fit one T4, so they run on free Colab and on Kaggle:

| Catalog name | Weights | Usable context on a T4 | Notes |
|---|---|---|---|
| `qwen2.5-coder-7b-awq` | ~5.5 GiB | 16-32k | Best code/VRAM trade-off. FIM support, room left for KV cache. Start here. |
| `qwen2.5-coder-14b-awq` | ~9.5 GiB | 4-8k | Better on refactors and multi-file work. Tight, hence `max-model-len: 4096`. |
| `qwen2.5-coder-1.5b` | ~3.1 GiB | 16k+ | Fast responses; use for editor autocomplete (FIM), not chat. |
| `qwen2.5-coder-7b-gguf` | ~4.7 GiB | 8k | Same family through Ollama; up in a minute. |

Kaggle only, using both GPUs:

| Catalog name | Weights | Notes |
|---|---|---|
| `qwen2.5-coder-32b-awq-2gpu` | ~19.5 GiB | Needs `tensor-parallel-size: 2`, so Colab free cannot run it. |
| `devstral-24b-2gpu` | ~14 GiB | Devstral Small 24B, tuned for agentic coding. Served through Ollama, which spreads layers over both T4s. |

Not in the catalog but worth trying: `google/codegemma-7b-it` (gated, needs `HF_TOKEN`), `Qwen/Qwen2.5-Coder-3B-Instruct` (fp16, ~6.2 GiB), and community 4-bit builds of StarCoder2-15B or DeepSeek-Coder-V2-Lite. Check file sizes first, since third-party quantizations vary in quality.

Caveats:

1. A T4 is Turing (sm75). AWQ runs on the older kernel: Marlin needs Ampere (sm80+), and FP8 does not exist here. Expect roughly 15-30 tok/s on a 7B, well below A100 benchmark numbers.
2. One model per session. For a big chat model and a small autocomplete model at once, run two notebooks (for example the 1.5B on Colab and the 32B on Kaggle) and connect to each with `--url`/`--key`.
3. Autocomplete needs `/v1/completions`, not only chat. vLLM serves both, so Continue's `tabAutocompleteModel` works against the same endpoint.

### Secrets and `.env`

Nothing sensitive is committed. Secrets come from the environment, a local `.env` (gitignored) or the platform's secret manager.

```bash
cp .env.example .env    # then fill only what you need
```

| Variable | Where | Required | Purpose |
|---|---|---|---|
| `GPUFREE_API_KEY` | notebook | no | Fixed API key clients must send. If empty, a random one is generated and printed on every run. |
| `HF_TOKEN` | notebook | only for gated models | Hugging Face access token ([huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)). Needed for official Llama/Gemma/Mistral repos. |
| `NGROK_AUTHTOKEN` | notebook | only with `--tunnel ngrok` | [dashboard.ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken) |
| `NGROK_DOMAIN` | notebook | no | A reserved ngrok domain, so the URL does not change. |
| `GITHUB_TOKEN` | both | only for `--gist` | Token with the `gist` scope ([classic](https://github.com/settings/tokens) or fine-grained with Gists: read and write). Locally the CLI falls back to `gh auth token`. |
| `GPUFREE_URL` | your machine | no | Point the client at an endpoint without `gpufree connect`. |
| `GPUFREE_MODEL` | notebook | no | Default model, instead of `--model`. |
| `GPUFREE_CATALOG` | notebook | no | Path to an alternative `models.yaml`. |
| `GPUFREE_HOME` | your machine | no | Where the client stores its config (default `~/.config/gpu-free-serve`). |

On Colab, open the key icon in the left sidebar, add secrets with exactly these names and turn on Notebook access. On Kaggle, use Add-ons > Secrets. The notebooks read them automatically, so do not paste tokens into cells.

The local config (`~/.config/gpu-free-serve/config.json`) holds the API key and is written with `600` permissions.

### CLI reference

```
gpufree connect [--gist | --url URL [--key KEY]] [--token TOKEN]
gpufree status                 # endpoint, key (redacted) and a live probe
gpufree models                 # model ids being served
gpufree chat [PROMPT...] [-m MODEL] [-s SYSTEM] [-t TEMP] [--max-tokens N]
gpufree proxy [--port 8787] [--host 127.0.0.1] [-v]
gpufree env [--local]          # prints OPENAI_BASE_URL / OPENAI_API_KEY
gpufree ui [--port 8765]       # local panel: Colab/Kaggle, model, idle clock, local link
gpufree disconnect             # forget the saved endpoint
```

```bash
gpufree chat -s "You are terse." "why is the sky blue?"
git diff | gpufree chat -s "Write a commit message for this diff."
eval "$(gpufree env --local)"    # export OPENAI_* pointing at the local proxy
```

### Server options (notebook side)

```
python -m gpufree.server.serve [options]

  --model NAME        catalog name, `org/model` or `ollama:tag`
  --catalog PATH      alternative models.yaml
  --tunnel {cloudflare,ngrok,none}   default: cloudflare
  --port N            gateway port (default 8500)
  --engine-port N     internal engine port
  --api-key KEY       overrides GPUFREE_API_KEY
  --publish {gist,none}   publish the endpoint to a secret gist (default: gist)
  --no-install        skip installing the engine
  --ready-timeout S   how long to wait for the engine (default 1800s)
  --list              print the catalog and exit
```

### Using it from your code and editor

With `gpufree proxy` running:

OpenAI SDK (Python):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="not-needed")
print(client.chat.completions.create(
    model="qwen2.5-7b-awq",
    messages=[{"role": "user", "content": "hello"}],
).choices[0].message.content)
```

Without the proxy (straight to the tunnel), use the real key:

```python
client = OpenAI(base_url="https://xxx.trycloudflare.com/v1", api_key="<api key>")
```

LangChain:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(base_url="http://127.0.0.1:8787/v1", api_key="not-needed",
                 model="qwen2.5-7b-awq")
```

Continue (VS Code / JetBrains), `~/.continue/config.json`:

```json
{ "models": [{ "title": "Colab GPU", "provider": "openai", "model": "qwen2.5-coder-7b-awq",
               "apiBase": "http://127.0.0.1:8787/v1", "apiKey": "not-needed" }] }
```

Aider:

```bash
OPENAI_API_BASE=http://127.0.0.1:8787/v1 OPENAI_API_KEY=not-needed \
  aider --model openai/qwen2.5-coder-7b-awq
```

### Cloudflare vs ngrok

| | Cloudflare (`--tunnel cloudflare`) | ngrok (`--tunnel ngrok`) |
|---|---|---|
| Account | not needed | free account + authtoken |
| URL | random `*.trycloudflare.com`, new on every run | random, or fixed with a reserved domain |
| Setup | binary downloaded automatically | `pyngrok` installed automatically |
| Good for | the default, no setup | a stable URL |

With `--publish gist` you never need to know the URL: your machine reads it from the gist.

### Free-tier limits

- These are free tiers, not a hosting platform. Colab disconnects on idle (roughly 90 min) and caps sessions at a few hours; Kaggle gives around 30 GPU hours per week with a 12-hour session limit.
- Every restart changes the tunnel URL and the random API key. Run `gpufree connect --gist` again, or set a fixed `GPUFREE_API_KEY` plus an ngrok domain.
- Keep the notebook tab open. Closing it kills the endpoint.
- Respect the platforms' terms. These GPUs are for interactive, personal work: no production traffic, no mining.
- Cold start dominates: ~10-15 min for vLLM the first time (pip and weights), then the model stays loaded while the session lives.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `no GPU visible` | Colab: Runtime > Change runtime type. Kaggle: Settings > Accelerator, and Internet: On. |
| Engine dies right after start | Out of GPU memory. Lower `max-model-len` / `gpu-memory-utilization`, or use a 4-bit (AWQ) model. |
| `Cannot use bfloat16` / dtype error | Handled automatically on T4 (`dtype: half`). If you overrode `dtype`, set it back to `half`. |
| `401 invalid API key` | The notebook restarted with a new random key. Run `gpufree connect --gist` again, or set `GPUFREE_API_KEY` in secrets. |
| `could not reach ...` | The notebook stopped or the tunnel dropped. Re-run the serve cell. |
| `no GitHub token found` | Add `GITHUB_TOKEN` (scope `gist`) to the notebook secrets, or use `--publish none` and connect with `--url`/`--key`. |
| Gated model 401/403 on download | Accept the license on the model page and set `HF_TOKEN`. |
| `reading a YAML catalog needs pyyaml` | Only happens locally: `pip install pyyaml`. Colab and Kaggle already have it. |
| Kaggle: tunnel never comes up | Internet must be enabled in the notebook settings (phone-verified account). |

### Security notes

- The engine binds to `127.0.0.1`; only the API-key gateway goes through the tunnel.
- The API key is random per session unless you set `GPUFREE_API_KEY`.
- The gist used for discovery is secret (unlisted), but a secret gist is not encrypted: anyone with the link can read it. Treat the endpoint as short-lived and prefer a fresh random key per session.
- `.env`, `*.pem` and `*.key` are gitignored. Clear notebook outputs before committing, since the banner prints the API key. The committed notebooks have no outputs.
- A public tunnel URL is reachable by anyone who learns it; the key is what protects it.

### Project layout

```
models.yaml                   the file you edit to choose models
notebooks/
  colab_serve.ipynb           run this on Colab
  kaggle_serve.ipynb          run this on Kaggle
gpufree/
  config.py                   catalog loading
  env.py                      .env reader (stdlib)
  gist.py                     publish/discover the endpoint
  _http.py                    reverse proxy shared by both ends
  server/
    serve.py                  notebook entrypoint
    engines.py                vLLM / Ollama
    tunnel.py                 cloudflared / ngrok
  client/
    cli.py                    the `gpufree` command
    api.py                    OpenAI-compatible calls
    proxy.py                  local proxy
scripts/build_notebooks.py    regenerates both notebooks
```

### License

MIT. See [LICENSE](LICENSE).
