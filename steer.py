#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from steering.ollama_client import OllamaClient, OllamaError
from steering.prompt import build_steering_system_prompt
from steering.state import (
    SteerItem,
    SteeringError,
    clear_state,
    default_state_path,
    load_state,
    parse_layers,
    update_state,
)


DEFAULT_MODEL = "gemma3:270m"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (SteeringError, OllamaError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Steer a local Ollama-backed LLM with shared live state.",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=None,
        help="Path to steering state JSON. Defaults to .steering/state.json.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    update = subparsers.add_parser("update", help="replace or append a steer")
    update.add_argument("--feature-id", required=True, type=int)
    update.add_argument("--strength", required=True, type=float)
    update.add_argument("--layers", required=True, help="comma-separated layer list")
    update.add_argument("--label", default=None, help="optional semantic label for Ollama prompt steering")
    update.add_argument("--append", action="store_true", help="append instead of replacing current state")
    update.add_argument("--json", action="store_true", help="print JSON state")
    update.set_defaults(func=cmd_update)

    show = subparsers.add_parser("show", help="print current steering state")
    show.add_argument("--json", action="store_true", help="print raw JSON state")
    show.set_defaults(func=cmd_show)

    clear = subparsers.add_parser("clear", help="drop all steers")
    clear.add_argument("--json", action="store_true", help="print JSON state")
    clear.set_defaults(func=cmd_clear)

    models = subparsers.add_parser("models", help="list local Ollama models")
    models.add_argument("--base-url", default=None, help="Ollama API URL, defaults to OLLAMA_HOST or localhost")
    models.set_defaults(func=cmd_models)

    generate = subparsers.add_parser("generate", help="generate through Ollama using current steering state")
    generate.add_argument("prompt", nargs="?", help="prompt text; stdin is used when omitted")
    generate.add_argument("--model", default=DEFAULT_MODEL)
    generate.add_argument("--base-url", default=None, help="Ollama API URL, defaults to OLLAMA_HOST or localhost")
    generate.add_argument("--max-tokens", type=int, default=120)
    generate.add_argument("--temperature", type=float, default=0.7)
    generate.add_argument(
        "--no-stream",
        action="store_true",
        help="wait for the full response before printing",
    )
    generate.set_defaults(func=cmd_generate)

    chat = subparsers.add_parser("chat", help="interactive Ollama chat that rereads steering state each turn")
    chat.add_argument("--model", default=DEFAULT_MODEL)
    chat.add_argument("--base-url", default=None, help="Ollama API URL, defaults to OLLAMA_HOST or localhost")
    chat.add_argument("--max-tokens", type=int, default=160)
    chat.add_argument("--temperature", type=float, default=0.7)
    chat.set_defaults(func=cmd_chat)

    return parser


def cmd_update(args: argparse.Namespace) -> int:
    item = SteerItem(
        feature_id=args.feature_id,
        strength=args.strength,
        layers=parse_layers(args.layers),
        label=args.label,
    )
    state = update_state(item, append=args.append, path=args.state_path)
    if args.json:
        print(json.dumps(state.to_dict(), indent=2, sort_keys=True))
    else:
        mode = "appended" if args.append else "updated"
        print(f"{mode} steering state at {resolved_state_path(args)}")
        print(format_state(state))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    state = load_state(args.state_path)
    if args.json:
        print(json.dumps(state.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_state(state))
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    state = clear_state(args.state_path)
    if args.json:
        print(json.dumps(state.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"cleared steering state at {resolved_state_path(args)}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    client = OllamaClient.from_env(args.base_url)
    data = client.tags()
    models = data.get("models", [])
    if not models:
        print("no local Ollama models found")
        return 0
    for model in models:
        name = model.get("name", "<unknown>")
        size = model.get("size", 0)
        print(f"{name}\t{human_size(size)}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    prompt = args.prompt if args.prompt is not None else sys.stdin.read().strip()
    if not prompt:
        raise SteeringError("prompt is required")
    state = load_state(args.state_path)
    system_prompt = build_steering_system_prompt(state)
    client = OllamaClient.from_env(args.base_url)
    for chunk in client.generate(
        model=args.model,
        prompt=prompt,
        system=system_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stream=not args.no_stream,
    ):
        print(chunk, end="", flush=True)
    print()
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    client = OllamaClient.from_env(args.base_url)
    print(f"model: {args.model}")
    print(f"state: {resolved_state_path(args)}")
    print("type /show, /clear, or /exit")
    while True:
        try:
            prompt = input("\n> ").strip()
        except EOFError:
            print()
            return 0
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            return 0
        if prompt == "/show":
            print(format_state(load_state(args.state_path)))
            continue
        if prompt == "/clear":
            clear_state(args.state_path)
            print("cleared")
            continue

        state = load_state(args.state_path)
        system_prompt = build_steering_system_prompt(state)
        for chunk in client.generate(
            model=args.model,
            prompt=prompt,
            system=system_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=True,
        ):
            print(chunk, end="", flush=True)
        print()


def format_state(state) -> str:
    if state.is_empty:
        return "no active steers"

    lines = [f"updated_at: {state.updated_at}", "active steers:"]
    for index, item in enumerate(state.items, start=1):
        layers = ",".join(str(layer) for layer in item.layers)
        label = f" label={item.label!r}" if item.label else ""
        lines.append(
            f"  {index}. feature_id={item.feature_id} strength={item.strength:g} "
            f"layers={layers}{label}"
        )
    return "\n".join(lines)


def resolved_state_path(args: argparse.Namespace) -> Path:
    return args.state_path or default_state_path()


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


if __name__ == "__main__":
    raise SystemExit(main())
