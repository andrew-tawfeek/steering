# Steering CLI

A local steering CLI for experimenting with live control of an Ollama-backed
LLM on macOS.

The current implementation keeps steering state in a shared JSON file and lets
one terminal update that state while another terminal generates or chats with a
local Ollama model.

## Current Backend

Ollama does not expose transformer residual streams, SAE decoder vectors, or
per-layer forward hooks through its public API. Because of that, this backend
uses **prompt-level steering**: the CLI preserves the intended steering shape
of `feature_id`, `strength`, and `layers`, then converts active steers into a
system prompt for Ollama.

This gives the project a working local control loop now, while keeping the
state format ready for a later true SAE backend.

## Requirements

- macOS
- Python 3.11 or newer
- Ollama installed and running
- At least one local Ollama model

The current CLI uses only the Python standard library.

## Quick Start

Start Ollama if it is not already running:

```bash
ollama serve
```

In this repo, list available local models:

```bash
python3 steer.py models
```

Add a steer:

```bash
python3 steer.py update --feature-id 204 --strength 30 --layers 6 --label "Python code"
```

Show active steering state:

```bash
python3 steer.py show
```

Generate through Ollama using the active steering state:

```bash
python3 steer.py generate --model gemma3:270m "Say hello in one short sentence."
```

Clear all steers:

```bash
python3 steer.py clear
```

## Two-Terminal Usage

Terminal A runs Ollama:

```bash
ollama serve
```

Terminal B runs the steering CLI:

```bash
python3 steer.py update --feature-id 204 --strength 30 --layers 6 --label "Python code"
python3 steer.py generate --model gemma3:270m "Write a short weather report."
python3 steer.py clear
```

For an interactive workflow, run chat in one terminal:

```bash
python3 steer.py chat --model gemma3:270m
```

Then update steering state from another terminal or tmux pane:

```bash
python3 steer.py update --feature-id 204 --strength 50 --layers 6 --label "Python code"
```

`chat` rereads `.steering/state.json` before each user turn, so changes apply
on the next prompt.

## tmux Workflow

Useful sessions for parallel work:

```bash
tmux attach -t steering-ollama
tmux attach -t steering-impl
tmux attach -t steering-tests
```

Suggested lane usage:

| Session | Purpose |
|---------|---------|
| `steering-ollama` | Keep `python3 steer.py chat --model gemma3:270m` or Ollama checks running. |
| `steering-impl` | Edit code and inspect state. |
| `steering-tests` | Run tests and command smoke checks. |

## Commands

### `update`

Replace the current steering state with one steer:

```bash
python3 steer.py update --feature-id <N> --strength <F> --layers <L1,L2,...>
```

Add `--append` to stack a new steer with the current state:

```bash
python3 steer.py update --feature-id 7 --strength 10 --layers 8 --label "Legal text" --append
```

Options:

| Option | Description |
|--------|-------------|
| `--feature-id` | Integer feature identifier. |
| `--strength` | Numeric steering strength. Positive values steer toward the label; negative values steer away. |
| `--layers` | Comma-separated layer list, such as `6` or `6,8,10`. |
| `--label` | Optional human-readable concept used by the Ollama prompt backend. |
| `--append` | Append instead of replacing active steers. |
| `--json` | Print the resulting state as JSON. |

### `show`

Print current steering state:

```bash
python3 steer.py show
```

Print raw JSON:

```bash
python3 steer.py show --json
```

### `clear`

Drop all active steers:

```bash
python3 steer.py clear
```

### `models`

List local Ollama models:

```bash
python3 steer.py models
```

Use a custom Ollama endpoint:

```bash
python3 steer.py models --base-url http://127.0.0.1:11434
```

### `generate`

Generate one response using the current steering state:

```bash
python3 steer.py generate --model gemma3:270m "Your prompt here"
```

Read the prompt from stdin:

```bash
printf "Your prompt here" | python3 steer.py generate --model gemma3:270m
```

Common options:

```bash
python3 steer.py generate \
  --model gemma3:270m \
  --max-tokens 120 \
  --temperature 0.7 \
  "Write a concise explanation of local LLM steering."
```

Use `--no-stream` to print only after the full response is complete.

### `chat`

Start an interactive loop:

```bash
python3 steer.py chat --model gemma3:270m
```

Chat commands:

| Command | Effect |
|---------|--------|
| `/show` | Print active steering state. |
| `/clear` | Clear active steering state. |
| `/exit` or `/quit` | Leave chat. |

## State File

By default, steering state is written to:

```bash
.steering/state.json
```

Use a different path by placing the global `--state-path` option before the
subcommand:

```bash
python3 steer.py --state-path /tmp/steering-state.json update --feature-id 204 --strength 30 --layers 6
python3 steer.py --state-path /tmp/steering-state.json show
```

You can also set:

```bash
export STEERING_STATE_PATH=/tmp/steering-state.json
```

State shape:

```json
{
  "version": 1,
  "updated_at": "2026-04-30T18:14:35+00:00",
  "items": [
    {
      "feature_id": 204,
      "strength": 30.0,
      "layers": [6],
      "label": "Python code"
    }
  ]
}
```

## Ollama Configuration

The CLI connects to `http://127.0.0.1:11434` by default.

Override the endpoint per command:

```bash
python3 steer.py generate --base-url http://127.0.0.1:11434 --model gemma3:270m "Hello"
```

Or use `OLLAMA_HOST`:

```bash
export OLLAMA_HOST=http://127.0.0.1:11434
```

## Testing

Run the test suite:

```bash
python3 -m unittest discover -s tests
```

Run basic command checks:

```bash
python3 steer.py --help
python3 steer.py models
```

## Project Layout

| Path | Purpose |
|------|---------|
| `steer.py` | CLI entrypoint. |
| `steering/state.py` | Steering state models, validation, loading, and saving. |
| `steering/ollama_client.py` | Minimal Ollama HTTP client. |
| `steering/prompt.py` | Converts steering state into an Ollama system prompt. |
| `tests/` | Unit tests for state and prompt behavior. |

## Roadmap

- Add a true SAE steering server for models that expose activation hooks.
- Reuse the same `.steering/state.json` format from the Ollama backend.
- Add end-to-end tests for a long-running generation or chat workflow.
