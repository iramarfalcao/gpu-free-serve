"""Minimal reverse HTTP proxy, stdlib only.

Used on both ends:

  notebook  ->  authenticated gateway in front of the engine (vLLM/Ollama)
  laptop    ->  local proxy that injects the API key and talks to the tunnel

No third-party dependency: the local client installs in seconds and the notebook
does not burn GPU minutes installing a web framework.
"""

from __future__ import annotations

import json
import socket
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Headers that must not cross a proxy (RFC 9110 section 7.6.1).
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

CHUNK = 8192
UPSTREAM_TIMEOUT = 900  # long streamed generations are normal


def build_handler(
    upstream: str,
    *,
    require_key: str | None = None,
    inject_key: str | None = None,
    public_paths: tuple[str, ...] = ("/healthz",),
    quiet: bool = True,
):
    """Build the handler class that forwards everything to `upstream`.

    require_key: when set, demands `Authorization: Bearer <key>`.
    inject_key:  when set, replaces the Authorization sent upstream.
    """
    base = upstream.rstrip("/")

    class ProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "gpufree-proxy"

        # --- helpers ----------------------------------------------------
        def log_message(self, fmt, *args):  # noqa: D102
            if not quiet:
                sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if not require_key:
                return True
            if self.path.split("?")[0] in public_paths:
                return True
            sent = self.headers.get("Authorization", "")
            if sent.lower().startswith("bearer "):
                sent = sent[7:]
            return sent.strip() == require_key

        # --- HTTP methods -----------------------------------------------
        def do_OPTIONS(self):  # noqa: N802
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):  # noqa: N802
            self._forward("GET")

        def do_POST(self):  # noqa: N802
            self._forward("POST")

        def do_DELETE(self):  # noqa: N802
            self._forward("DELETE")

        # --- forwarding ---------------------------------------------------
        def _forward(self, method: str) -> None:
            if self.path.split("?")[0] == "/healthz" and require_key:
                self._json(200, {"status": "ok"})
                return
            if not self._authorized():
                self._json(401, {"error": {"message": "invalid API key", "type": "unauthorized"}})
                return

            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None

            headers = {
                k: v
                for k, v in self.headers.items()
                if k.lower() not in HOP_BY_HOP | {"host", "content-length", "accept-encoding"}
            }
            headers["Accept-Encoding"] = "identity"
            if inject_key:
                headers["Authorization"] = f"Bearer {inject_key}"

            req = urllib.request.Request(base + self.path, data=body, headers=headers, method=method)
            try:
                resp = urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT)
            except urllib.error.HTTPError as exc:  # engine error: pass it through as-is
                self._relay(exc, exc.code)
                return
            except (urllib.error.URLError, socket.timeout, ConnectionError) as exc:
                self._json(
                    502,
                    {"error": {"message": f"upstream unreachable: {exc}", "type": "bad_gateway"}},
                )
                return
            with resp:
                self._relay(resp, resp.status)

        def _relay(self, resp, status: int) -> None:
            """Relay the response in chunks — works for plain JSON and for SSE."""
            self.send_response(status)
            for key, value in resp.headers.items():
                if key.lower() in HOP_BY_HOP | {"content-length"}:
                    continue
                self.send_header(key, value)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # client walked away mid-stream

    return ProxyHandler


def serve(host: str, port: int, handler) -> ThreadingHTTPServer:
    """Create the server. Call serve_forever() on it (in a thread if needed)."""
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    return httpd


def free_port(preferred: int) -> int:
    """Return `preferred` if it is free, otherwise any free port."""
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
