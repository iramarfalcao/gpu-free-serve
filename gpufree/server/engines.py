"""Supported inference engines, all exposing an OpenAI-compatible API."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

from ..config import ModelSpec


class EngineError(RuntimeError):
    pass


def _run(cmd: list[str] | str, **kwargs) -> subprocess.CompletedProcess:
    shell = isinstance(cmd, str)
    printable = cmd if shell else " ".join(cmd)
    print(f"[gpufree] $ {printable}", flush=True)
    return subprocess.run(cmd, shell=shell, check=True, **kwargs)


def gpu_info() -> list[dict]:
    """Name, memory (MiB) and compute capability of every visible GPU."""
    if not shutil.which("nvidia-smi"):
        return []
    query = "name,memory.total,compute_cap"
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            memory = int(float(parts[1]))
            capability = float(parts[2])
        except ValueError:
            continue
        gpus.append({"name": parts[0], "memory_mib": memory, "compute_cap": capability})
    return gpus


class Engine:
    """Common contract: install, start the process, report when it is ready."""

    name = "base"
    default_port = 8000

    def __init__(self, spec: ModelSpec, port: int):
        self.spec = spec
        self.port = port
        self.process: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/v1/models"

    def install(self) -> None:  # pragma: no cover - touches the environment
        raise NotImplementedError

    def start(self) -> subprocess.Popen:  # pragma: no cover
        raise NotImplementedError

    def wait_ready(self, timeout: float = 1800) -> None:
        """Wait for the engine. The first boot downloads weights, so be patient."""
        deadline = time.time() + timeout
        last_error = ""
        while time.time() < deadline:
            if self.process and self.process.poll() is not None:
                raise EngineError(
                    f"the {self.name} engine died with code {self.process.returncode}. "
                    "Check the log above — usually it is GPU out-of-memory: lower "
                    "`max-model-len` or pick a smaller model."
                )
            try:
                with urllib.request.urlopen(self.health_url, timeout=5) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, OSError) as exc:
                last_error = str(exc)
            time.sleep(3)
        raise EngineError(f"the {self.name} engine was not ready within {timeout}s ({last_error})")

    def stop(self) -> None:
        if not self.process or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()


class VLLMEngine(Engine):
    """vLLM: best throughput, Hugging Face models."""

    name = "vllm"

    def install(self) -> None:
        if importlib.util.find_spec("vllm"):
            print("[gpufree] vLLM already installed.", flush=True)
            return
        print("[gpufree] Installing vLLM (several minutes on the first run)...", flush=True)
        _run([sys.executable, "-m", "pip", "install", "-q", "vllm"])

    def _flags(self) -> list[str]:
        args = dict(self.spec.args)

        gpus = gpu_info()
        # Turing (T4, cc 7.5) has no bfloat16: without this vLLM fails at startup.
        if gpus and min(g["compute_cap"] for g in gpus) < 8.0:
            args.setdefault("dtype", "half")
        if gpus and int(args.get("tensor-parallel-size", 1)) > len(gpus):
            raise EngineError(
                f"this model asks for tensor-parallel-size={args['tensor-parallel-size']} "
                f"but only {len(gpus)} GPU(s) are available in this session."
            )

        flags: list[str] = []
        for key, value in args.items():
            flag = f"--{str(key).lstrip('-')}"
            if isinstance(value, bool):
                if value:
                    flags.append(flag)
            else:
                flags += [flag, str(value)]
        return flags

    def start(self) -> subprocess.Popen:
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.spec.id,
            "--served-model-name",
            self.spec.name,
            self.spec.id,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            *self._flags(),
        ]
        print(f"[gpufree] $ {' '.join(cmd)}", flush=True)
        self.process = subprocess.Popen(cmd, env=os.environ.copy())
        return self.process


class OllamaEngine(Engine):
    """Ollama: up in under a minute, GGUF models."""

    name = "ollama"
    default_port = 11434

    def install(self) -> None:
        if shutil.which("ollama"):
            print("[gpufree] Ollama already installed.", flush=True)
            return
        print("[gpufree] Installing Ollama...", flush=True)
        _run("curl -fsSL https://ollama.com/install.sh | sh")

    def _env(self) -> dict:
        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"127.0.0.1:{self.port}"
        return env

    def start(self) -> subprocess.Popen:
        env = self._env()
        print(f"[gpufree] $ ollama serve (port {self.port})", flush=True)
        self.process = subprocess.Popen(["ollama", "serve"], env=env)
        self._wait_daemon(env)
        print(f"[gpufree] Pulling {self.spec.id}...", flush=True)
        _run(["ollama", "pull", self.spec.id], env=env)
        return self.process

    def _wait_daemon(self, env: dict, timeout: float = 120) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process and self.process.poll() is not None:
                raise EngineError("the Ollama daemon exited right after starting.")
            try:
                with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3):
                    return
            except (urllib.error.URLError, OSError):
                time.sleep(2)
        raise EngineError("the Ollama daemon did not answer in time.")


ENGINES = {"vllm": VLLMEngine, "ollama": OllamaEngine}


def build(spec: ModelSpec, port: int | None = None) -> Engine:
    try:
        cls = ENGINES[spec.engine]
    except KeyError:  # pragma: no cover - already validated by the catalog
        raise EngineError(f"unknown engine: {spec.engine}") from None
    return cls(spec, port or cls.default_port)
