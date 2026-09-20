"""Notebook entrypoint: `python -m gpufree.server.serve --model <name>`.

    [engine: vLLM/Ollama on 127.0.0.1]
              ^
              | (localhost only, never exposed)
    [API-key gateway on 0.0.0.0:8500]
              ^
              | (tunnel)
    [https://something.trycloudflare.com]  <- your machine talks to this
"""

from __future__ import annotations

import argparse
import os
import secrets
import signal
import sys
import threading
import time
from datetime import datetime, timezone

from .. import _http, gist
from ..config import CatalogError, load_catalog, resolve
from ..env import load_dotenv, redact
from . import engines, tunnel as tunnels


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gpufree-serve",
        description="Boot a model on the Colab/Kaggle GPU and publish its API.",
    )
    parser.add_argument("--model", help="catalog name, `org/model` (vLLM) or `ollama:tag`")
    parser.add_argument("--catalog", help="path to an alternative models.yaml")
    parser.add_argument(
        "--tunnel", default="cloudflare", choices=["cloudflare", "ngrok", "none"],
        help="how to expose the gateway (default: cloudflare)",
    )
    parser.add_argument("--port", type=int, default=8500, help="gateway port")
    parser.add_argument("--engine-port", type=int, default=None, help="internal engine port")
    parser.add_argument(
        "--api-key",
        default=None,
        help="key required from clients; defaults to GPUFREE_API_KEY from .env, or a random one",
    )
    parser.add_argument(
        "--publish", default="gist", choices=["gist", "none"],
        help="publish the endpoint to a secret gist for `gpufree connect --gist` (default: gist)",
    )
    parser.add_argument("--no-install", action="store_true", help="skip installing the engine")
    parser.add_argument("--list", action="store_true", help="print the catalog and exit")
    parser.add_argument(
        "--ready-timeout", type=float, default=1800,
        help="seconds to wait for the engine, weight download included",
    )
    return parser.parse_args(argv)


def _print_catalog(catalog_path: str | None) -> None:
    models, default = load_catalog(catalog_path)
    print("Available models:\n")
    for name, spec in models.items():
        mark = " (default)" if name == default else ""
        print(f"  {name}{mark}")
        print(f"      engine: {spec.engine}  |  id: {spec.id}")
        if spec.description:
            print(f"      {spec.description}")
    print("\nEdit models.yaml to add your own.")


def _banner(url: str | None, api_key: str, spec, gist_url: str) -> None:
    print("\n" + "=" * 72)
    print("  MODEL IS UP")
    print("=" * 72)
    print(f"  model    : {spec.name}  ({spec.engine} / {spec.id})")
    if url:
        print(f"  endpoint : {url}/v1")
    else:
        print("  endpoint : no tunnel (--tunnel none) — reachable only inside the notebook")
    print(f"  api key  : {api_key}")
    if gist_url:
        print(f"  gist     : {gist_url}")
    print("-" * 72)
    print("  On your machine:\n")
    if gist_url:
        print("      gpufree connect --gist")
    if url:
        print(f"      gpufree connect --url {url} --key <api key above>")
        print('      gpufree chat "hello there"')
        print("      gpufree proxy        # http://127.0.0.1:8787/v1, no key needed")
    print("=" * 72 + "\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    env_loaded = load_dotenv()

    try:
        if args.list:
            _print_catalog(args.catalog)
            return 0
        spec = resolve(args.model, args.catalog)
    except CatalogError as exc:
        print(f"[gpufree] error: {exc}", file=sys.stderr)
        return 2

    api_key = args.api_key or os.environ.get("GPUFREE_API_KEY") or secrets.token_urlsafe(24)
    if env_loaded:
        print(f"[gpufree] .env loaded: {', '.join(sorted(env_loaded))}", flush=True)
    if os.environ.get("HF_TOKEN"):
        print(f"[gpufree] HF_TOKEN detected ({redact(os.environ['HF_TOKEN'])})", flush=True)

    gpus = engines.gpu_info()
    if gpus:
        for index, gpu in enumerate(gpus):
            print(
                f"[gpufree] GPU {index}: {gpu['name']} / {gpu['memory_mib']} MiB / "
                f"compute {gpu['compute_cap']}",
                flush=True,
            )
    else:
        print(
            "[gpufree] WARNING: no GPU visible. Colab: Runtime > Change runtime type. "
            "Kaggle: Settings > Accelerator.",
            flush=True,
        )

    engine = engines.build(spec, args.engine_port)
    tunnel = tunnels.build(args.tunnel, args.port)
    httpd = None

    try:
        if not args.no_install:
            engine.install()
        engine.start()
        print(f"[gpufree] Waiting for the engine on {engine.health_url} ...", flush=True)
        engine.wait_ready(timeout=args.ready_timeout)
        print("[gpufree] Engine ready.", flush=True)

        handler = _http.build_handler(engine.base_url, require_key=api_key)
        httpd = _http.serve("0.0.0.0", args.port, handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print(f"[gpufree] Authenticated gateway on port {args.port}.", flush=True)

        url = None
        if args.tunnel != "none":
            url = tunnel.open()

        gist_url = ""
        if args.publish == "gist" and url:
            try:
                gist_url = gist.publish(
                    {
                        "url": f"{url}/v1",
                        "api_key": api_key,
                        "model": spec.name,
                        "engine": spec.engine,
                        "model_id": spec.id,
                        "tunnel": tunnel.provider,
                        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    }
                )
                print(f"[gpufree] Endpoint published to a secret gist: {gist_url}", flush=True)
            except gist.GistError as exc:
                print(
                    f"[gpufree] Gist not published ({exc}). Use the URL and key below instead.",
                    file=sys.stderr,
                    flush=True,
                )

        _banner(url, api_key, spec, gist_url)

        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        started = time.time()
        while not stop.is_set():
            if engine.process and engine.process.poll() is not None:
                print("[gpufree] The engine exited. Shutting down.", file=sys.stderr)
                return 1
            stop.wait(30)
        print(f"[gpufree] Stopping after {int(time.time() - started)}s.", flush=True)
        return 0
    except (engines.EngineError, tunnels.TunnelError) as exc:
        print(f"[gpufree] error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        if httpd:
            httpd.shutdown()
        tunnel.close()
        engine.stop()


if __name__ == "__main__":
    raise SystemExit(main())
