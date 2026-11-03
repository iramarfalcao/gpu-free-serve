"""Local control panel: pick Colab/Kaggle, model, idle clock, local OpenAI link.

  gpufree ui [--port 8765]

Login is the platform's own browser login (Google for Colab, Kaggle account).
This UI does not OAuth into Google/Kaggle — it opens the notebook and connects
to the tunnel/gist afterward. Stdlib only.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .. import config, gist
from ..env import load_dotenv, redact
from . import api, prefs, proxy, state

DEFAULT_UI_PORT = 8765
_proxy_thread: threading.Thread | None = None
_proxy_lock = threading.Lock()
_proxy_running = False
_proxy_error: str | None = None


def _catalog_payload() -> dict[str, Any]:
    try:
        models, default = config.load_catalog()
        return {
            "default": default,
            "models": [
                {
                    "name": m.name,
                    "engine": m.engine,
                    "id": m.id,
                    "description": m.description,
                }
                for m in models.values()
            ],
        }
    except config.CatalogError as exc:
        return {"default": None, "models": [], "error": str(exc)}


def _connection_status() -> dict[str, Any]:
    out: dict[str, Any] = {
        "connected": False,
        "online": False,
        "url": None,
        "model": None,
        "models": [],
        "key_redacted": None,
        "error": None,
    }
    try:
        data = state.load()
    except state.NotConnected:
        out["error"] = "nenhum endpoint salvo"
        return out
    out["connected"] = True
    out["url"] = data.get("url")
    out["model"] = data.get("model") or data.get("model_id")
    out["key_redacted"] = redact(data.get("api_key"))
    out["updated_at"] = data.get("updated_at")
    try:
        served = api.models(data)
        out["online"] = True
        out["models"] = served
        if served and not out["model"]:
            out["model"] = served[0]
        prefs.mark_session_started()
    except api.ApiError as exc:
        out["online"] = False
        out["error"] = str(exc)
    return out


def _client_snippets(prefs_data: dict) -> dict[str, Any]:
    host = prefs_data.get("proxy_host") or "127.0.0.1"
    port = int(prefs_data.get("proxy_port") or 8787)
    local = f"http://{host}:{port}/v1"
    model = prefs_data.get("model") or "qwen2.5-7b-awq"
    tunnel = None
    try:
        tunnel = state.load().get("url")
    except state.NotConnected:
        pass
    return {
        "local_base_url": local,
        "local_api_key": "not-needed",
        "model": model,
        "tunnel_url": tunnel,
        "snippets": {
            "env": (
                f"export OPENAI_BASE_URL={local}\n"
                f"export OPENAI_API_KEY=not-needed\n"
                f"export OPENAI_MODEL={model}"
            ),
            "opencode": (
                f"# OpenCode / OpenAI-compatible providers\n"
                f"baseURL: {local}\n"
                f"apiKey: not-needed\n"
                f"model: {model}"
            ),
            "pi": (
                f"# Pi / qualquer cliente OpenAI\n"
                f"OPENAI_BASE_URL={local}\n"
                f"OPENAI_API_KEY=not-needed\n"
                f"model={model}"
            ),
            "curl": (
                f"curl {local}/chat/completions \\\n"
                f"  -H 'Content-Type: application/json' \\\n"
                f"  -d '{{\"model\":\"{model}\","
                f"\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]}}'"
            ),
            "cursor_continue": (
                f"OpenAI Compatible\n"
                f"API Base URL: {local}\n"
                f"API Key: not-needed\n"
                f"Model: {model}"
            ),
        },
        "proxy_cmd": f"gpufree proxy --host {host} --port {port}",
        "proxy_running": _proxy_running,
        "proxy_error": _proxy_error,
    }


def _start_proxy(host: str, port: int) -> dict[str, Any]:
    global _proxy_thread, _proxy_running, _proxy_error
    with _proxy_lock:
        if _proxy_running:
            return {"ok": True, "running": True, "message": "proxy já ativo"}
        try:
            data = state.load()
        except state.NotConnected as exc:
            return {"ok": False, "error": str(exc)}

        def _run() -> None:
            global _proxy_running, _proxy_error
            _proxy_running = True
            _proxy_error = None
            try:
                proxy.run(data, host=host, port=port, quiet=True)
            except OSError as exc:
                _proxy_error = str(exc)
            finally:
                _proxy_running = False

        _proxy_thread = threading.Thread(target=_run, name="gpufree-proxy", daemon=True)
        _proxy_thread.start()
        return {
            "ok": True,
            "running": True,
            "base_url": f"http://{host}:{port}/v1",
        }


def _do_connect(body: dict) -> dict[str, Any]:
    mode = body.get("mode") or "gist"
    if mode == "gist":
        data = gist.discover(body.get("token"))
    else:
        url = (body.get("url") or "").rstrip("/")
        if not url:
            return {"ok": False, "error": "informe a URL do túnel"}
        if not url.endswith("/v1"):
            url += "/v1"
        data = {"url": url, "api_key": body.get("key") or ""}
    try:
        served = api.models(data)
        if served:
            data["model_id"] = served[0]
    except api.ApiError:
        pass
    path = state.save(data)
    prefs.mark_session_started()
    return {
        "ok": True,
        "path": str(path),
        "url": data.get("url"),
        "model": data.get("model_id") or data.get("model"),
        "key_redacted": redact(data.get("api_key")),
    }


def _page() -> bytes:
    return _HTML.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "gpufree-ui/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter default
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def _json(self, code: int, payload: dict | list) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            body = _page()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/state":
            p = prefs.load()
            self._json(
                200,
                {
                    "prefs": p,
                    "catalog": _catalog_payload(),
                    "connection": _connection_status(),
                    "countdown": prefs.countdown(p),
                    "clients": _client_snippets(p),
                    "notebooks": {
                        "colab": prefs.NOTEBOOK_URLS["colab"],
                        "kaggle_new": prefs.NOTEBOOK_URLS["kaggle"],
                        "kaggle_source": prefs.KAGGLE_NOTEBOOK_HINT,
                    },
                },
            )
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        try:
            body = self._read_json()
        except json.JSONDecodeError:
            self._json(400, {"error": "JSON inválido"})
            return

        if path == "/api/prefs":
            current = prefs.load()
            if "provider" in body:
                current["provider"] = body["provider"]
            if "model" in body:
                current["model"] = body["model"]
            if body.get("reset_clock"):
                current["session_started_at"] = None
            if body.get("start_clock"):
                current["session_started_at"] = current.get("session_started_at") or time.time()
            prefs.save(current)
            self._json(200, {"ok": True, "prefs": prefs.load(), "countdown": prefs.countdown()})
            return

        if path == "/api/connect":
            try:
                result = _do_connect(body)
            except gist.GistError as exc:
                self._json(400, {"ok": False, "error": str(exc)})
                return
            code = 200 if result.get("ok") else 400
            self._json(code, result)
            return

        if path == "/api/disconnect":
            state.clear()
            prefs.clear_session_clock()
            self._json(200, {"ok": True})
            return

        if path == "/api/proxy/start":
            p = prefs.load()
            host = body.get("host") or p.get("proxy_host") or "127.0.0.1"
            port = int(body.get("port") or p.get("proxy_port") or 8787)
            p["proxy_host"] = host
            p["proxy_port"] = port
            prefs.save(p)
            result = _start_proxy(host, port)
            code = 200 if result.get("ok") else 400
            self._json(code, result)
            return

        self._json(404, {"error": "not found"})


def run(host: str = "127.0.0.1", port: int = DEFAULT_UI_PORT, open_browser: bool = True) -> None:
    load_dotenv()
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.verbose = False  # type: ignore[attr-defined]
    url = f"http://{host}:{port}/"
    print(f"gpu-free-serve UI on {url}")
    print("  Colab/Kaggle login happens in the browser tabs the UI opens.")
    print("  Local OpenAI link after proxy: http://127.0.0.1:8787/v1")
    if open_browser:
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nUI stopped.")


_HTML = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>gpu-free-serve</title>
<style>
  :root {
    --bg: #0f1419;
    --panel: #1a222c;
    --ink: #e7ecf1;
    --muted: #8b9aab;
    --line: #2a3542;
    --accent: #3d9cfd;
    --ok: #3ecf8e;
    --warn: #f0b429;
    --bad: #f07178;
    --colab: #f9ab00;
    --kaggle: #20beff;
    --radius: 10px;
    --font: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
    --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh; background: var(--bg); color: var(--ink);
    font: 15px/1.45 var(--font);
    background-image:
      radial-gradient(ellipse 80% 50% at 10% -10%, #1e3a5f55, transparent),
      radial-gradient(ellipse 60% 40% at 100% 0%, #1a403555, transparent);
  }
  main { max-width: 920px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }
  h1 { font-size: 1.55rem; font-weight: 650; letter-spacing: -0.02em; margin: 0 0 .35rem; }
  .sub { color: var(--muted); margin: 0 0 1.75rem; }
  section {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: var(--radius); padding: 1.1rem 1.2rem; margin-bottom: 1rem;
  }
  section h2 {
    margin: 0 0 .85rem; font-size: .72rem; text-transform: uppercase;
    letter-spacing: .08em; color: var(--muted); font-weight: 600;
  }
  .row { display: flex; flex-wrap: wrap; gap: .75rem; }
  .provider {
    flex: 1 1 200px; border: 1px solid var(--line); border-radius: 8px;
    padding: 1rem; background: #121820; cursor: pointer; text-align: left;
    color: inherit; font: inherit; transition: border-color .15s, background .15s;
  }
  .provider:hover { border-color: #456; }
  .provider.active { border-color: var(--accent); background: #152033; }
  .provider .name { font-weight: 650; display: block; margin-bottom: .25rem; }
  .provider.colab.active { border-color: var(--colab); }
  .provider.kaggle.active { border-color: var(--kaggle); }
  .provider .hint { color: var(--muted); font-size: .85rem; }
  label { display: block; font-size: .8rem; color: var(--muted); margin-bottom: .35rem; }
  select, input, textarea, button.primary, button.ghost, button.danger {
    font: inherit; border-radius: 7px; border: 1px solid var(--line);
    background: #121820; color: var(--ink); padding: .55rem .7rem;
  }
  select, input { width: 100%; }
  button.primary {
    background: var(--accent); border-color: transparent; color: #041018;
    font-weight: 650; cursor: pointer;
  }
  button.primary:hover { filter: brightness(1.08); }
  button.ghost { cursor: pointer; background: transparent; }
  button.ghost:hover { border-color: var(--accent); }
  button.danger { cursor: pointer; border-color: var(--bad); color: var(--bad); background: transparent; }
  .actions { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .85rem; }
  .status-pill {
    display: inline-flex; align-items: center; gap: .4rem;
    padding: .2rem .55rem; border-radius: 999px; font-size: .78rem; font-weight: 600;
    background: #243040;
  }
  .dot { width: .55rem; height: .55rem; border-radius: 50%; background: var(--muted); }
  .dot.on { background: var(--ok); box-shadow: 0 0 8px #3ecf8e88; }
  .dot.off { background: var(--bad); }
  .clock {
    font-family: var(--mono); font-size: 2rem; font-weight: 600;
    letter-spacing: .04em; margin: .4rem 0;
  }
  .clock.warn { color: var(--warn); }
  .clock.bad { color: var(--bad); }
  .meta { color: var(--muted); font-size: .85rem; }
  pre {
    margin: .5rem 0 0; padding: .75rem .85rem; background: #0c1015;
    border-radius: 7px; border: 1px solid var(--line); overflow-x: auto;
    font: .8rem/1.4 var(--mono); white-space: pre-wrap;
  }
  .tabs { display: flex; flex-wrap: wrap; gap: .35rem; margin-bottom: .5rem; }
  .tabs button {
    font: inherit; font-size: .8rem; padding: .35rem .65rem; border-radius: 6px;
    border: 1px solid var(--line); background: transparent; color: var(--muted); cursor: pointer;
  }
  .tabs button.on { color: var(--ink); border-color: var(--accent); background: #152033; }
  .msg { margin-top: .75rem; font-size: .85rem; color: var(--muted); }
  .msg.err { color: var(--bad); }
  .msg.ok { color: var(--ok); }
  .steps { margin: 0; padding-left: 1.15rem; color: var(--muted); font-size: .9rem; }
  .steps li { margin: .25rem 0; }
  code { font-family: var(--mono); font-size: .86em; }
</style>
</head>
<body>
<main>
  <h1>gpu-free-serve</h1>
  <p class="sub">Sobe a GPU free no Colab (Google) ou Kaggle, escolhe o modelo, acompanha o idle e gera o link local OpenAI para OpenCode, Pi e outros.</p>

  <section>
    <h2>1 · Plataforma (login no browser)</h2>
    <div class="row" id="providers">
      <button type="button" class="provider colab" data-provider="colab">
        <span class="name">Google Colab</span>
        <span class="hint">Login Google · T4 · ~90 min idle</span>
      </button>
      <button type="button" class="provider kaggle" data-provider="kaggle">
        <span class="name">Kaggle</span>
        <span class="hint">Login Kaggle · T4×2 · ~12 h sessão</span>
      </button>
    </div>
    <ol class="steps" id="launch-steps"></ol>
    <div class="actions">
      <button type="button" class="primary" id="btn-open">Abrir notebook e entrar</button>
      <button type="button" class="ghost" id="btn-copy-model">Copiar linha MODEL=</button>
    </div>
  </section>

  <section>
    <h2>2 · Modelo</h2>
    <label for="model">Catálogo (models.yaml)</label>
    <select id="model"></select>
    <p class="meta" id="model-desc"></p>
  </section>

  <section>
    <h2>3 · Conexão</h2>
    <div class="row" style="align-items:center;gap:1rem">
      <span class="status-pill"><span class="dot" id="dot"></span><span id="status-label">—</span></span>
      <span class="meta" id="endpoint"></span>
    </div>
    <div class="actions">
      <button type="button" class="primary" id="btn-gist">Conectar via Gist</button>
      <button type="button" class="ghost" id="btn-manual">Conectar URL + key</button>
      <button type="button" class="danger" id="btn-disconnect">Desconectar</button>
    </div>
    <div id="manual-box" style="display:none;margin-top:.75rem">
      <label>URL do túnel</label>
      <input id="manual-url" placeholder="https://….trycloudflare.com" />
      <label style="margin-top:.5rem">API key</label>
      <input id="manual-key" placeholder="chave impressa no notebook" />
      <div class="actions"><button type="button" class="primary" id="btn-manual-go">Salvar</button></div>
    </div>
    <p class="msg" id="conn-msg"></p>
  </section>

  <section>
    <h2>4 · Contador (quando a GPU pode cair)</h2>
    <div class="clock" id="clock">--:--:--</div>
    <p class="meta" id="clock-label"></p>
    <div class="actions">
      <button type="button" class="ghost" id="btn-reset-clock">Zerar relógio</button>
      <button type="button" class="ghost" id="btn-start-clock">Marcar GPU ativa agora</button>
    </div>
  </section>

  <section>
    <h2>5 · Link local (OpenCode, Pi, Cursor…)</h2>
    <p class="meta">Rode o proxy local e aponte qualquer cliente OpenAI-compatible para a URL abaixo.</p>
    <pre id="local-url">http://127.0.0.1:8787/v1</pre>
    <div class="actions">
      <button type="button" class="primary" id="btn-proxy">Iniciar proxy local</button>
      <button type="button" class="ghost" id="btn-copy-url">Copiar URL</button>
    </div>
    <p class="msg" id="proxy-msg"></p>
    <div class="tabs" id="snippet-tabs"></div>
    <pre id="snippet"></pre>
    <div class="actions"><button type="button" class="ghost" id="btn-copy-snip">Copiar snippet</button></div>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
let state = null;
let snipKey = "opencode";

function fmt(sec) {
  sec = Math.max(0, sec|0);
  const h = String(Math.floor(sec/3600)).padStart(2,"0");
  const m = String(Math.floor((sec%3600)/60)).padStart(2,"0");
  const s = String(sec%60).padStart(2,"0");
  return `${h}:${m}:${s}`;
}

async function api(path, opts) {
  const r = await fetch(path, {
    headers: {"Content-Type":"application/json"},
    ...opts,
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

function renderProviders() {
  const p = state.prefs.provider;
  document.querySelectorAll(".provider").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.provider === p);
  });
  const steps = $("launch-steps");
  if (p === "kaggle") {
    steps.innerHTML = `
      <li>Faça login no Kaggle no browser.</li>
      <li>Settings → Accelerator → <strong>GPU T4 x2</strong>, Internet On.</li>
      <li>Importe <code>notebooks/kaggle_serve.ipynb</code> (ou abra o link do fonte).</li>
      <li>Na célula serve, defina <code>MODEL = "${state.prefs.model}"</code> e rode tudo.</li>
      <li>Volte aqui e use <strong>Conectar via Gist</strong>.</li>`;
  } else {
    steps.innerHTML = `
      <li>Faça login com Google no Colab.</li>
      <li>Runtime → Change runtime type → <strong>T4 GPU</strong>.</li>
      <li>Secrets: GITHUB_TOKEN (gist), opcional HF_TOKEN / NGROK.</li>
      <li>Na célula serve, defina <code>MODEL = "${state.prefs.model}"</code> e rode tudo.</li>
      <li>Volte aqui e use <strong>Conectar via Gist</strong>.</li>`;
  }
}

function renderCatalog() {
  const sel = $("model");
  const models = state.catalog.models || [];
  sel.innerHTML = models.map(m =>
    `<option value="${m.name}">${m.name} · ${m.engine}</option>`
  ).join("");
  const want = state.prefs.model || state.catalog.default;
  if (want) sel.value = want;
  const cur = models.find(m => m.name === sel.value);
  $("model-desc").textContent = cur ? cur.description : (state.catalog.error || "");
}

function renderConnection() {
  const c = state.connection;
  const dot = $("dot");
  const label = $("status-label");
  dot.className = "dot " + (c.online ? "on" : "off");
  if (!c.connected) label.textContent = "sem endpoint";
  else if (c.online) label.textContent = "ONLINE";
  else label.textContent = "OFFLINE";
  $("endpoint").textContent = c.url
    ? `${c.url} · model ${c.model || "?"} · key ${c.key_redacted || "—"}`
    : "Conecte depois que o notebook publicar o túnel.";
}

function renderClock() {
  const cd = state.countdown;
  const el = $("clock");
  el.textContent = fmt(cd.remaining_sec);
  el.className = "clock" + (cd.remaining_sec < 600 ? " bad" : cd.remaining_sec < 1800 ? " warn" : "");
  $("clock-label").textContent = cd.running
    ? `${cd.label} · decorrido ${fmt(cd.elapsed_sec)}`
    : `${cd.label} · relógio para ao marcar GPU ativa ou ao conectar online`;
}

function renderClients() {
  const cl = state.clients;
  $("local-url").textContent = cl.local_base_url;
  const tabs = $("snippet-tabs");
  const keys = Object.keys(cl.snippets);
  tabs.innerHTML = keys.map(k =>
    `<button type="button" data-k="${k}" class="${k===snipKey?"on":""}">${k}</button>`
  ).join("");
  $("snippet").textContent = cl.snippets[snipKey] || "";
  $("proxy-msg").textContent = cl.proxy_running
    ? "Proxy ativo."
    : (cl.proxy_error || `Depois: ${cl.proxy_cmd}`);
  $("proxy-msg").className = "msg " + (cl.proxy_running ? "ok" : "");
}

function renderAll() {
  renderProviders();
  renderCatalog();
  renderConnection();
  renderClock();
  renderClients();
}

async function refresh() {
  state = await api("/api/state");
  renderAll();
}

document.querySelectorAll(".provider").forEach(btn => {
  btn.addEventListener("click", async () => {
    await api("/api/prefs", { method:"POST", body: JSON.stringify({ provider: btn.dataset.provider }) });
    await refresh();
  });
});

$("model").addEventListener("change", async (e) => {
  await api("/api/prefs", { method:"POST", body: JSON.stringify({ model: e.target.value }) });
  await refresh();
});

$("btn-open").addEventListener("click", () => {
  const p = state.prefs.provider;
  if (p === "kaggle") {
    window.open(state.notebooks.kaggle_source, "_blank");
    window.open(state.notebooks.kaggle_new, "_blank");
  } else {
    window.open(state.notebooks.colab, "_blank");
  }
});

$("btn-copy-model").addEventListener("click", async () => {
  const line = `MODEL = "${state.prefs.model}"`;
  await navigator.clipboard.writeText(line);
  $("conn-msg").textContent = "Copiado: " + line;
  $("conn-msg").className = "msg ok";
});

$("btn-gist").addEventListener("click", async () => {
  $("conn-msg").textContent = "Conectando via gist…";
  $("conn-msg").className = "msg";
  try {
    const r = await api("/api/connect", { method:"POST", body: JSON.stringify({ mode: "gist" }) });
    $("conn-msg").textContent = "Conectado: " + r.url;
    $("conn-msg").className = "msg ok";
    await refresh();
  } catch (e) {
    $("conn-msg").textContent = e.message;
    $("conn-msg").className = "msg err";
  }
});

$("btn-manual").addEventListener("click", () => {
  $("manual-box").style.display = $("manual-box").style.display === "none" ? "block" : "none";
});

$("btn-manual-go").addEventListener("click", async () => {
  try {
    const r = await api("/api/connect", {
      method:"POST",
      body: JSON.stringify({
        mode: "manual",
        url: $("manual-url").value.trim(),
        key: $("manual-key").value.trim(),
      }),
    });
    $("conn-msg").textContent = "Conectado: " + r.url;
    $("conn-msg").className = "msg ok";
    await refresh();
  } catch (e) {
    $("conn-msg").textContent = e.message;
    $("conn-msg").className = "msg err";
  }
});

$("btn-disconnect").addEventListener("click", async () => {
  await api("/api/disconnect", { method:"POST", body: "{}" });
  await refresh();
});

$("btn-reset-clock").addEventListener("click", async () => {
  await api("/api/prefs", { method:"POST", body: JSON.stringify({ reset_clock: true }) });
  await refresh();
});

$("btn-start-clock").addEventListener("click", async () => {
  await api("/api/prefs", { method:"POST", body: JSON.stringify({ start_clock: true }) });
  await refresh();
});

$("btn-proxy").addEventListener("click", async () => {
  try {
    const r = await api("/api/proxy/start", { method:"POST", body: "{}" });
    $("proxy-msg").textContent = r.base_url ? "Proxy em " + r.base_url : (r.message || "ok");
    $("proxy-msg").className = "msg ok";
    await refresh();
  } catch (e) {
    $("proxy-msg").textContent = e.message;
    $("proxy-msg").className = "msg err";
  }
});

$("btn-copy-url").addEventListener("click", async () => {
  await navigator.clipboard.writeText(state.clients.local_base_url);
});

$("snippet-tabs").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-k]");
  if (!b) return;
  snipKey = b.dataset.k;
  renderClients();
});

$("btn-copy-snip").addEventListener("click", async () => {
  await navigator.clipboard.writeText($("snippet").textContent);
});

refresh();
setInterval(async () => {
  try { await refresh(); } catch (_) {}
}, 15000);
setInterval(() => {
  if (!state || !state.countdown || !state.countdown.running) return;
  state.countdown.remaining_sec = Math.max(0, state.countdown.remaining_sec - 1);
  state.countdown.elapsed_sec += 1;
  renderClock();
}, 1000);
</script>
</body>
</html>
"""
