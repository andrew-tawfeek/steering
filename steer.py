#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from steering.local_client import LocalServerClient, LocalServerError
from steering.neuronpedia_client import (
    DEFAULT_STEERING_MODEL,
    NeuronpediaClient,
    NeuronpediaError,
    summarize_feature,
)
from steering.state import (
    SteerItem,
    SteeringError,
    clear_state,
    default_state_path,
    load_state,
    parse_layers,
    update_state,
)


DEFAULT_SERVER_URL = "http://127.0.0.1:8000"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (SteeringError, LocalServerError, NeuronpediaError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live SAE feature steering for a local hookable LLM backend.",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=None,
        help="Path to steering state JSON. Defaults to .steering/state.json.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    update = subparsers.add_parser("update", help="replace or append a feature steer")
    update.add_argument("--feature-id", required=True, type=int)
    update.add_argument("--strength", required=True, type=float)
    update.add_argument("--layers", default=None, help="comma-separated layer list, such as 6 or 6,8,10")
    update.add_argument("--sae-id", default=None, help="explicit SAE id, such as blocks.6.hook_resid_pre")
    update.add_argument("--model-id", default=None, help="source model id for metadata, such as gpt2-small")
    update.add_argument("--label", default=None, help="optional human-readable note from Neuronpedia")
    update.add_argument("--append", action="store_true", help="append instead of replacing current state")
    update.add_argument("--json", action="store_true", help="print JSON state")
    update.set_defaults(func=cmd_update)

    show = subparsers.add_parser("show", help="print current steering state")
    show.add_argument("--json", action="store_true", help="print raw JSON state")
    show.set_defaults(func=cmd_show)

    clear = subparsers.add_parser("clear", help="drop all steers")
    clear.add_argument("--json", action="store_true", help="print JSON state")
    clear.set_defaults(func=cmd_clear)

    generate = subparsers.add_parser("generate", help="generate through the local TransformerLens server")
    generate.add_argument("prompt", nargs="?", help="prompt text; stdin is used when omitted")
    generate.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    generate.add_argument("--max-tokens", type=int, default=60)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--seed", type=int, default=None)
    generate.add_argument("--no-stream", action="store_true", help="wait for the full response before printing")
    generate.set_defaults(func=cmd_generate)

    chat = subparsers.add_parser("chat", help="interactive chat through the local TransformerLens server")
    chat.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    chat.add_argument("--max-tokens", type=int, default=80)
    chat.add_argument("--temperature", type=float, default=0.8)
    chat.add_argument("--seed", type=int, default=None)
    chat.set_defaults(func=cmd_chat)

    health = subparsers.add_parser("health", help="check the local TransformerLens server")
    health.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    health.set_defaults(func=cmd_health)

    feature = subparsers.add_parser("feature", help="look up feature metadata from Neuronpedia")
    feature.add_argument("--feature-id", required=True, type=int)
    feature.add_argument("--model-id", default=DEFAULT_STEERING_MODEL)
    feature.add_argument("--sae-id", default=None, help="Neuronpedia SAE id, such as 6-res-jb")
    feature.add_argument("--layer", type=int, default=None, help="layer shorthand for GPT-2 res-jb, such as 6")
    feature.add_argument("--base-url", default=None)
    feature.add_argument("--json", action="store_true")
    feature.set_defaults(func=cmd_feature)

    return parser


def cmd_update(args: argparse.Namespace) -> int:
    layers = parse_layers(args.layers)
    item = SteerItem(
        feature_id=args.feature_id,
        strength=args.strength,
        layers=layers,
        label=args.label,
        model_id=args.model_id,
        sae_id=args.sae_id,
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


def cmd_generate(args: argparse.Namespace) -> int:
    prompt = args.prompt if args.prompt is not None else sys.stdin.read().strip()
    if not prompt:
        raise SteeringError("prompt is required")
    client = LocalServerClient.from_env(args.server_url)
    for chunk in client.generate(
        prompt=prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        stream=not args.no_stream,
    ):
        print(chunk, end="", flush=True)
    print()
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    client = LocalServerClient.from_env(args.server_url)
    print(f"server: {client.base_url}")
    print(f"state: {resolved_state_path(args)}")
    print("type /show, /clear, /health, or /exit")
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
        if prompt == "/health":
            print(json.dumps(client.health(), indent=2, sort_keys=True))
            continue

        for chunk in client.generate(
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
            stream=True,
        ):
            print(chunk, end="", flush=True)
        print()


def cmd_health(args: argparse.Namespace) -> int:
    client = LocalServerClient.from_env(args.server_url)
    print(json.dumps(client.health(), indent=2, sort_keys=True))
    return 0


def cmd_feature(args: argparse.Namespace) -> int:
    sae_id = args.sae_id
    if sae_id is None:
        if args.layer is None:
            raise SteeringError("provide either --sae-id or --layer")
        sae_id = f"{args.layer}-res-jb"

    client = NeuronpediaClient.from_env(args.base_url)
    data = client.feature(args.model_id, sae_id, args.feature_id)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(summarize_feature(data))
    return 0


def format_state(state) -> str:
    if state.is_empty:
        return "no active steers"

    lines = [f"updated_at: {state.updated_at}", "active steers:"]
    for index, item in enumerate(state.items, start=1):
        layers = ",".join(str(layer) for layer in item.layers) if item.layers else "-"
        model = f" model={item.model_id}" if item.model_id else ""
        sae = f" sae={item.sae_id}" if item.sae_id else ""
        label = f" label={item.label!r}" if item.label else ""
        lines.append(
            f"  {index}. feature_id={item.feature_id} strength={item.strength:g} "
            f"layers={layers}{model}{sae}{label}"
        )
    return "\n".join(lines)


def resolved_state_path(args: argparse.Namespace) -> Path:
    return args.state_path or default_state_path()


if __name__ == "__main__":
    raise SystemExit(main())
