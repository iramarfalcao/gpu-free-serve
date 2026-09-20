"""Local proxy: http://127.0.0.1:8787/v1 -> notebook tunnel, key injected.

Any OpenAI-speaking tool (SDK, Continue, Cursor, Aider, curl) can then point at
localhost without knowing the tunnel URL or the API key.
"""

from __future__ import annotations

import threading

from .. import _http


def base_root(url: str) -> str:
    """Strip the /v1 suffix — the full path comes from the caller."""
    url = url.rstrip("/")
    return url[: -len("/v1")] if url.endswith("/v1") else url


def run(state: dict, host: str = "127.0.0.1", port: int = 8787, quiet: bool = True) -> None:
    handler = _http.build_handler(
        base_root(state["url"]), inject_key=state.get("api_key"), quiet=quiet
    )
    httpd = _http.serve(host, port, handler)
    print(f"Local proxy on http://{host}:{port}/v1  ->  {state['url']}")
    print(f"  OPENAI_BASE_URL=http://{host}:{port}/v1  OPENAI_API_KEY=not-needed")
    print("Ctrl+C to stop.")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        thread.join()
    except KeyboardInterrupt:
        print("\nStopping the proxy.")
    finally:
        httpd.shutdown()
