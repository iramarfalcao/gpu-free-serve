"""`gpufree` CLI: connect to, test and proxy the model running in the notebook."""

from __future__ import annotations

import argparse
import sys

from .. import gist
from ..env import load_dotenv, redact
from . import api, proxy, state


def _connect(args: argparse.Namespace) -> int:
    if args.gist:
        data = gist.discover(args.token)
    else:
        if not args.url:
            print("pass --url (and --key), or use --gist", file=sys.stderr)
            return 2
        url = args.url.rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        data = {"url": url, "api_key": args.key or ""}
    _fill_model(data)
    path = state.save(data)
    print(f"Connected to {data['url']}")
    if data.get("model"):
        print(f"  model: {data['model']} ({data.get('engine', '?')})")
    print(f"  key  : {redact(data.get('api_key'))}")
    print(f"  saved: {path}")
    return 0


def _status(_: argparse.Namespace) -> int:
    data = state.load()
    print(f"endpoint: {data['url']}")
    print(f"key     : {redact(data.get('api_key'))}")
    if data.get("updated_at"):
        print(f"published: {data['updated_at']}")
    try:
        available = api.models(data)
    except api.ApiError as exc:
        print(f"status  : OFFLINE ({exc})")
        return 1
    print(f"status  : ONLINE — models: {', '.join(available) or '(none)'}")
    return 0


def _models(_: argparse.Namespace) -> int:
    for name in api.models(state.load()):
        print(name)
    return 0


def _fill_model(data: dict) -> str | None:
    """Ask the endpoint which model it serves, so `chat` needs no -m."""
    if data.get("model_id"):
        return data["model_id"]
    try:
        served = api.models(data)
    except api.ApiError:
        return None  # offline right now; chat will resolve it later
    if served:
        data["model_id"] = served[0]
    return data.get("model_id")


def _chat(args: argparse.Namespace) -> int:
    data = state.load()
    prompt = " ".join(args.prompt) if args.prompt else sys.stdin.read().strip()
    if not prompt:
        print("nothing to send", file=sys.stderr)
        return 2
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})
    model = args.model or _fill_model(data)
    if not model:
        print("could not work out which model to use; pass -m", file=sys.stderr)
        return 1
    for piece in api.chat_stream(
        data, messages, model=model, temperature=args.temperature, max_tokens=args.max_tokens
    ):
        sys.stdout.write(piece)
        sys.stdout.flush()
    print()
    return 0


def _proxy(args: argparse.Namespace) -> int:
    proxy.run(state.load(), host=args.host, port=args.port, quiet=not args.verbose)
    return 0


def _env(args: argparse.Namespace) -> int:
    data = state.load()
    if args.local:
        print("export OPENAI_BASE_URL=http://127.0.0.1:8787/v1")
        print("export OPENAI_API_KEY=not-needed")
    else:
        print(f"export OPENAI_BASE_URL={data['url']}")
        print(f"export OPENAI_API_KEY={data.get('api_key', '')}")
    return 0


def _disconnect(_: argparse.Namespace) -> int:
    state.clear()
    print("saved endpoint removed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpufree", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    connect = sub.add_parser("connect", help="save the endpoint published by the notebook")
    connect.add_argument("--url", help="https://....trycloudflare.com (with or without /v1)")
    connect.add_argument("--key", help="API key printed by the notebook")
    connect.add_argument("--gist", action="store_true", help="discover it from the secret gist")
    connect.add_argument("--token", help="GitHub token (default: env or `gh auth token`)")
    connect.set_defaults(func=_connect)

    sub.add_parser("status", help="show and probe the current endpoint").set_defaults(func=_status)
    sub.add_parser("models", help="list the served models").set_defaults(func=_models)
    sub.add_parser("disconnect", help="forget the saved endpoint").set_defaults(func=_disconnect)

    chat = sub.add_parser("chat", help="send a prompt (streamed) to the model")
    chat.add_argument("prompt", nargs="*", help="the text; reads stdin when empty")
    chat.add_argument("-m", "--model", help="model name (default: whatever the notebook booted)")
    chat.add_argument("-s", "--system", help="system prompt")
    chat.add_argument("-t", "--temperature", type=float, default=0.7)
    chat.add_argument("--max-tokens", type=int, default=1024)
    chat.set_defaults(func=_chat)

    local = sub.add_parser("proxy", help="expose the model on http://127.0.0.1:8787/v1, no key")
    local.add_argument("--port", type=int, default=8787)
    local.add_argument("--host", default="127.0.0.1")
    local.add_argument("-v", "--verbose", action="store_true", help="log every request")
    local.set_defaults(func=_proxy)

    env = sub.add_parser("env", help="print OPENAI_* variables for your shell")
    env.add_argument("--local", action="store_true", help="pointing at the local proxy")
    env.set_defaults(func=_env)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (state.NotConnected, api.ApiError, gist.GistError) as exc:
        print(f"gpufree: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
