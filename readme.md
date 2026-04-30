# Steering CLI

A local CLI and server for live SAE feature steering.

The backend loads a TransformerLens model and SAE Lens sparse autoencoders,
then mutates the residual stream during generation by adding:

```text
strength * SAE.W_dec[feature_id]
```

Neuronpedia is used as the feature catalog.

## Current Default Stack

| Piece | Default |
|-------|---------|
| Model runtime | TransformerLens |
| Model | `gpt2-small` |
| SAE library | SAE Lens |
| SAE release | `gpt2-small-res-jb` |
| State file | `.steering/state.json` |
| Server | `http://127.0.0.1:8000` |

Layer shorthand maps to SAE Lens hook ids. For example:

| CLI layer | SAE Lens id | Neuronpedia source |
|-----------|-------------|--------------------|
| `--layers 6` | `blocks.6.hook_resid_pre` | `gpt2-small/6-res-jb` |
| `--layers 8` | `blocks.8.hook_resid_pre` | `gpt2-small/8-res-jb` |
| `--layers 10` | `blocks.10.hook_resid_pre` | `gpt2-small/10-res-jb` |

## Setup

Use Python 3.12 on this Mac:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

The current environment has been verified with:

```bash
python -m unittest discover -s tests
python steer.py feature --layer 6 --feature-id 204
python steer.py feature-cache sources --model-id gpt2-small --contains res-jb
```

## Start The Backend

Terminal A:

```bash
source .venv/bin/activate
uvicorn server:app --host 127.0.0.1 --port 8000
```

The first startup downloads GPT-2 small. The first steered generation for a
layer downloads that layer's SAE weights.

Check that the server is ready:

```bash
source .venv/bin/activate
python steer.py health
```

Expected shape:

```json
{
  "backend": "transformer-lens",
  "device": "mps",
  "model_name": "gpt2-small",
  "sae_id_template": "blocks.{layer}.hook_resid_pre",
  "sae_release": "gpt2-small-res-jb"
}
```

## Launch The Terminal Interface

The primary demo path is the split-pane terminal UI:

```bash
./start.sh
```

`start.sh` creates/uses `.venv`, installs missing dependencies from
`requirements.txt`, starts the TransformerLens backend if it is not already
running, waits for `/health`, then opens the interface.

The left pane is a text-completion surface for the current backend model. The
default backend is raw `gpt2-small`, not a chat-tuned assistant, so use prompts
that look like text to continue rather than questions expecting a helpful reply.
The right pane shows active steers, feature lookup controls, and the local
feature-label cache. Steering changes are written to `.steering/state.json`;
the backend reads that state during generation.

Useful keys:

| Key | Action |
|-----|--------|
| `F2` | Focus the completion prompt. |
| `F3` | Focus active steers. |
| `F6` | Continue the current prompt. |
| `F8` | Clear all active steers after confirmation. |
| `F10` | List cache sources for the selected model. |
| `F11` | Search cached labels. |
| `F12` | Apply the selected cached feature to the steer form. |
| `Ctrl+C` | Quit the interface. |

## CLI Interface Demo

Start from the repo root:

```bash
cd /Users/atawfeek/GitHub/steering
git switch main
./start.sh
```

The first run may download GPT-2 small and the first steered generation may
download SAE Lens weights.

Inside the interface:

1. In the left completion prompt, enter:

   ```text
   Today the weather report says
   ```

2. In the right pane, set a manual steer:

   ```text
   Feature: 204
   Strength: 10
   Layers: 6
   ```

   Press `Set Only`.

3. Continue the same weather prompt again from the left pane and compare the
   continuation with the baseline.

4. Try the cache workflow:

   ```text
   Model: gpt2-small
   Source: 6-res-jb
   Search: time phrases
   ```

   Press `Download` once if that source is not cached yet, press `Search`,
   select a result, then press `Apply` to copy its model/source/feature label
   into the steer form.

5. Press `Clear All` before ending the demo so no steer is left active.

## Smoke Test

Terminal B:

```bash
source .venv/bin/activate
python steer.py clear
python steer.py generate "The weather today is" --max-tokens 20 --temperature 0
python steer.py update --feature-id 204 --strength 10 --layers 6
python steer.py show
python steer.py generate "The weather today is" --max-tokens 20 --temperature 0
python steer.py clear
```

What this does:

1. Clears old state.
2. Generates a baseline continuation.
3. Adds feature `204` at layer `6`.
4. Generates again with the residual-stream intervention active.
5. Clears state when finished.

## Neuronpedia Feature Lookup

Feature ids are only meaningful for a specific model and SAE source. Use
Neuronpedia lookup before steering:

```bash
python steer.py feature --layer 6 --feature-id 204
```

Equivalent explicit lookup:

```bash
python steer.py feature --model-id gpt2-small --sae-id 6-res-jb --feature-id 204
```

The output includes the Neuronpedia explanation, default steer strength, top
logit effects, an activation snippet, and the feature URL.

## Feature Label Cache

Neuronpedia's docs expose both the feature API and full public exports. This
CLI uses the exports for bulk label caching so local search does not hammer the
interactive API.

Cache identity is:

```text
model_id/source_id/feature_id
```

For example, `gpt2-small/6-res-jb/204` and `gpt2-small/8-res-jb/204` are
different features.

Search works against the local SQLite cache only. Run `download` first, then
use `search` or `show` without waiting on Neuronpedia. Search is
case-insensitive, splits the query into words, and returns labels whose
description contains every query word. It is keyword search over cached
descriptions, not embedding or semantic search.

List models and sources available in the public export:

```bash
python steer.py feature-cache models
python steer.py feature-cache sources --model-id gpt2-small --contains res-jb
```

Download labels for one source:

```bash
python steer.py feature-cache download --model-id gpt2-small --source 6-res-jb
```

Download labels for every matching source. This can take a while and can create
a large cache:

```bash
python steer.py feature-cache download \
  --model-id gpt2-small \
  --all-sources \
  --source-contains res-jb
```

Search cached labels:

```bash
python steer.py feature-cache search "time phrases" \
  --model-id gpt2-small \
  --source 6-res-jb \
  --limit 10
```

Leave off `--source` to search all cached sources for a model, or leave off
both `--model-id` and `--source` to search the whole local cache:

```bash
python steer.py feature-cache search "calendar dates" --model-id gpt2-small
python steer.py feature-cache search "induction heads" --limit 25
```

Show the cached labels for a specific feature id:

```bash
python steer.py feature-cache show \
  --model-id gpt2-small \
  --source 6-res-jb \
  --feature-id 204
```

Check what is cached:

```bash
python steer.py feature-cache status
```

The default cache path is:

```bash
.steering/feature-cache.sqlite3
```

Use a custom cache path with `--cache-path`, or set
`STEERING_FEATURE_CACHE_PATH`.

## Commands

### `update`

Replace active steering state:

```bash
python steer.py update --feature-id 204 --strength 10 --layers 6
```

Append another steer:

```bash
python steer.py update --feature-id 1200 --strength 15 --layers 8 --append
```

Use an explicit SAE Lens hook id:

```bash
python steer.py update \
  --feature-id 204 \
  --strength 10 \
  --sae-id blocks.6.hook_resid_pre
```

Options:

| Option | Meaning |
|--------|---------|
| `--feature-id` | SAE feature index. |
| `--strength` | Scalar multiplier for the SAE decoder vector. |
| `--layers` | Comma-separated layer shorthand, such as `6` or `6,8,10`. |
| `--sae-id` | Explicit SAE Lens id. Use this instead of `--layers` for non-default SAEs. |
| `--append` | Stack this steer on top of existing state. |
| `--label` | Optional local note stored with the state. |
| `--json` | Print resulting state as JSON. |

### `show`

```bash
python steer.py show
python steer.py show --json
```

### `clear`

```bash
python steer.py clear
```

### `generate`

```bash
python steer.py generate "The weather today is" --max-tokens 40
```

Common deterministic test form:

```bash
python steer.py generate "The weather today is" --max-tokens 20 --temperature 0
```

Use a custom backend URL:

```bash
python steer.py generate --server-url http://127.0.0.1:8000 "Hello"
```

### `chat`

```bash
python steer.py chat
```

This command is a raw completion loop for the TransformerLens backend. It is
named `chat` for convenience, but `gpt2-small` is not instruction tuned.

Inside chat:

| Command | Effect |
|---------|--------|
| `/show` | Print active steering state. |
| `/clear` | Clear active steering state. |
| `/health` | Check backend status. |
| `/exit` or `/quit` | Leave chat. |

### `ui`

Open the split-pane TUI directly if the backend is already running:

```bash
python steer.py ui
```

Common options:

```bash
python steer.py ui --server-url http://127.0.0.1:8000 --max-tokens 80 --temperature 0
```

Use `--temperature 0` for deterministic demos with the raw GPT-2 backend.

### `feature-cache`

```bash
python steer.py feature-cache models
python steer.py feature-cache sources --model-id gpt2-small
python steer.py feature-cache download --model-id gpt2-small --source 6-res-jb
python steer.py feature-cache search "calendar dates" --model-id gpt2-small --source 6-res-jb
python steer.py feature-cache show --model-id gpt2-small --source 6-res-jb --feature-id 204
python steer.py feature-cache status
```

For development smoke tests, limit the export download:

```bash
python steer.py feature-cache download \
  --model-id gpt2-small \
  --source 6-res-jb \
  --max-files 1 \
  --cache-path /tmp/feature-cache.sqlite3
```

## tmux Workflow

Optional tmux sessions:

```bash
tmux attach -t steering-backend
```

Recommended usage:

| Session | Purpose |
|---------|---------|
| `steering-backend` | Keep `uvicorn server:app --host 127.0.0.1 --port 8000` running. |

`./start.sh` starts the backend automatically when it is not already running,
so tmux is optional for normal demos.

## Configuration

Environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `STEERING_DEVICE` | `auto` | Chooses `mps`, then `cuda`, then `cpu`. |
| `STEERING_MODEL_NAME` | `gpt2-small` | TransformerLens model name. |
| `STEERING_SAE_RELEASE` | `gpt2-small-res-jb` | SAE Lens release. |
| `STEERING_SAE_ID_TEMPLATE` | `blocks.{layer}.hook_resid_pre` | Maps `--layers` values to SAE Lens ids. |
| `STEERING_STATE_PATH` | `.steering/state.json` | Shared steering state path. |
| `STEERING_FEATURE_CACHE_PATH` | `.steering/feature-cache.sqlite3` | SQLite cache for Neuronpedia labels. |
| `STEERING_SERVER_URL` | `http://127.0.0.1:8000` | CLI target server. |

TransformerLens currently warns that MPS may produce incorrect results with
PyTorch 2.11.0. For correctness-first testing, start the backend on CPU:

```bash
source .venv/bin/activate
STEERING_DEVICE=cpu uvicorn server:app --host 127.0.0.1 --port 8000
```

## State File

Default path:

```bash
.steering/state.json
```

Example state:

```json
{
  "version": 1,
  "updated_at": "2026-04-30T18:40:06+00:00",
  "items": [
    {
      "feature_id": 204,
      "strength": 10.0,
      "layers": [6]
    }
  ]
}
```

Use a custom state path by putting `--state-path` before the command:

```bash
python steer.py --state-path /tmp/steering.json update --feature-id 204 --strength 10 --layers 6
python steer.py --state-path /tmp/steering.json show
```

## Tests

```bash
source .venv/bin/activate
python -m unittest discover -s tests
python -m py_compile steer.py server.py steering/*.py tests/*.py
python steer.py feature --layer 6 --feature-id 204
tmpdir=$(mktemp -d)
python steer.py feature-cache download --model-id gpt2-small --source 6-res-jb --max-files 1 --cache-path "$tmpdir/cache.sqlite3"
python steer.py feature-cache search time --model-id gpt2-small --source 6-res-jb --cache-path "$tmpdir/cache.sqlite3" --limit 3
rm -rf "$tmpdir"
```

TUI smoke check:

```bash
python - <<'PY'
import asyncio
from steering.tui import SteeringTUI

async def main():
    app = SteeringTUI(server_url="http://127.0.0.1:8000", max_tokens=1, temperature=0, state_path=None)
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        assert app.query_one("#prompt")
        assert app.query_one("#steer-table")

asyncio.run(main())
PY
```

## Backend Notes

TransformerLens is the first backend because its hook names line up directly
with SAE Lens GPT-2 residual-stream SAEs.

NNsight is a good next backend for Hugging Face models and NDIF-style remote
execution, but it is not required for the current GPT-2/SAE Lens path.
