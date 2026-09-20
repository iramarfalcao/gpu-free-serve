"""Public tunnel for the gateway: Cloudflare (default) or ngrok."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-{os}-{arch}"
URL_PATTERN = re.compile(r"https://[-\w.]+\.trycloudflare\.com")


class TunnelError(RuntimeError):
    pass


class Tunnel:
    provider = "base"

    def __init__(self, port: int):
        self.port = port
        self.url: str | None = None

    def open(self) -> str:  # pragma: no cover - network
        raise NotImplementedError

    def close(self) -> None:
        pass


class CloudflareTunnel(Tunnel):
    """cloudflared quick tunnel: no account, no token, fresh URL per session."""

    provider = "cloudflare"

    def __init__(self, port: int):
        super().__init__(port)
        self.process: subprocess.Popen | None = None

    def _binary(self) -> str:
        existing = shutil.which("cloudflared")
        if existing:
            return existing
        target = Path.home() / ".local" / "bin" / "cloudflared"
        if target.exists():
            return str(target)
        system = platform.system().lower()
        machine = platform.machine().lower()
        arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(
            machine, "amd64"
        )
        url = CLOUDFLARED_URL.format(os="darwin" if system == "darwin" else "linux", arch=arch)
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[gpufree] Downloading cloudflared from {url}", flush=True)
        urllib.request.urlretrieve(url, target)
        target.chmod(0o755)
        return str(target)

    def open(self, timeout: float = 90) -> str:
        cmd = [
            self._binary(),
            "tunnel",
            "--no-autoupdate",
            "--url",
            f"http://127.0.0.1:{self.port}",
        ]
        print(f"[gpufree] $ {' '.join(cmd)}", flush=True)
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )

        found: list[str] = []

        def reader() -> None:
            assert self.process and self.process.stdout
            for line in self.process.stdout:
                match = URL_PATTERN.search(line)
                if match and not found:
                    found.append(match.group(0))
                if "ERR" in line or "error" in line.lower():
                    sys.stderr.write(f"[cloudflared] {line}")

        threading.Thread(target=reader, daemon=True).start()

        deadline = time.time() + timeout
        while time.time() < deadline:
            if found:
                self.url = found[0]
                return self.url
            if self.process.poll() is not None:
                raise TunnelError("cloudflared exited before publishing a URL.")
            time.sleep(1)
        raise TunnelError("cloudflared did not publish a URL in time.")

    def close(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()


class NgrokTunnel(Tunnel):
    """ngrok: needs a (free) authtoken, but supports a stable domain."""

    provider = "ngrok"

    def __init__(self, port: int, authtoken: str | None = None, domain: str | None = None):
        super().__init__(port)
        self.authtoken = authtoken or os.environ.get("NGROK_AUTHTOKEN")
        self.domain = domain or os.environ.get("NGROK_DOMAIN")
        self._tunnel = None

    def open(self, timeout: float = 60) -> str:
        if not self.authtoken:
            raise TunnelError(
                "NGROK_AUTHTOKEN is not set. Grab yours at "
                "https://dashboard.ngrok.com/get-started/your-authtoken and put it in .env "
                "(or in the Colab/Kaggle secrets)."
            )
        try:
            from pyngrok import ngrok  # type: ignore
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyngrok"], check=True)
            from pyngrok import ngrok  # type: ignore

        ngrok.set_auth_token(self.authtoken)
        options = {"bind_tls": True}
        if self.domain:
            options["domain"] = self.domain
        self._tunnel = ngrok.connect(self.port, "http", **options)
        self.url = self._tunnel.public_url
        return self.url

    def close(self) -> None:
        if self._tunnel is None:
            return
        try:
            from pyngrok import ngrok  # type: ignore

            ngrok.disconnect(self._tunnel.public_url)
        except Exception:  # pragma: no cover - best-effort cleanup
            pass


def build(provider: str, port: int) -> Tunnel:
    provider = (provider or "cloudflare").lower()
    if provider in ("cloudflare", "cloudflared"):
        return CloudflareTunnel(port)
    if provider == "ngrok":
        return NgrokTunnel(port)
    if provider == "none":
        return Tunnel(port)
    raise TunnelError(f"unknown tunnel: {provider} (use cloudflare, ngrok or none)")
