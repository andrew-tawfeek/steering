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

## Quick Start

From a fresh clone, run one command:

```bash
./start.sh
```

`start.sh` is the production startup path. It:

1. Finds Python 3.10+; Python 3.12 is preferred when available.
2. Creates `.venv` if it does not exist.
3. Installs or updates the Python package dependencies from `requirements.txt`
   and the editable local package from `pyproject.toml`.
4. Reuses the existing environment on future runs when dependencies are already
   present and the dependency files have not changed.
5. Starts the FastAPI web server in the foreground.

Open `http://127.0.0.1:8000/` after the server logs that it is running. Stop
the server with `Ctrl+C`. To keep it alive outside your current shell, run it
inside tmux:

```bash
tmux new -s steering './start.sh'
```

The first startup downloads the default GPT-2 small model through
TransformerLens. The first steered generation for a layer downloads that
layer's SAE weights through SAE Lens. Later startups reuse the local package
environment and the model/SAE caches.

If Python is not auto-detected, set:

```bash
STEERING_PYTHON=/path/to/python ./start.sh
```

Use another port by passing normal `serve` options through `start.sh`:

```bash
./start.sh --port 8080
```

Or via environment variables:

```bash
STEERING_SERVER_HOST=127.0.0.1 STEERING_SERVER_PORT=8080 ./start.sh
```

## Manual Setup

Manual setup is only needed for development or debugging the bootstrap process:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
python steer.py serve
```

After editable install, the `steer` console command is equivalent to
`python steer.py`. Check the installed version with `steer --version`.

Check that the server is ready:

```bash
source .venv/bin/activate
python steer.py health
```

Expected shape:

```json
{
  "backend": "transformer-lens",
  "busy": false,
  "device": "cpu",
  "model_name": "gpt2-small",
  "sae_id_template": "blocks.{layer}.hook_resid_pre",
  "sae_release": "gpt2-small-res-jb"
}
```

For a broader setup check, including dependency imports, writable state/cache
paths, and backend reachability, run:

```bash
python steer.py doctor
```

Use `--skip-server` before starting the backend, or `--json` when integrating
the check into scripts.

After the backend is running, run the local generation smoke test:

```bash
scripts/backend_smoke.sh
```

This uses an isolated temporary state file, checks `/health`, runs a baseline
completion, applies feature `204` at layer `6`, runs a steered completion, then
clears the temporary state.

## Web Interface

The FastAPI backend serves the browser UI at the server root:

```bash
./start.sh
```

Open `http://127.0.0.1:8000/`.

The web UI can:

- Generate raw completions through the active backend model.
- List Neuronpedia export models and source ids.
- Switch/download a TransformerLens model with an explicit SAE Lens release and
  layer SAE id template.
- Download Neuronpedia explanation labels for a selected model/source into the
  local feature cache.
- Search cached labels and apply selected features to the shared steering state.

For Neuronpedia residual JB sources such as `6-res-jb`, the UI maps the source
to layer `6` and uses `STEERING_SAE_ID_TEMPLATE` for SAE Lens loading. For other
source ids, the UI stores the source id as an explicit `sae_id`, so the selected
SAE Lens release must contain a matching SAE id.

Useful keys:

| Key | Action |
|-----|--------|
| `F2` | Focus the completion prompt. |
| `F3` | Focus active steers. |
| `F6` | Continue the current prompt. |
| `F8` | Clear all active steers after confirmation. |
| `F10` | Cache compatible residual JB layers for the selected model. |
| `F11` | Search cached labels across compatible layers. |
| `F12` | Apply the selected cached feature to the steer form. |
| `Ctrl+C` | Stop the server from the terminal where `./start.sh` is running. |

## Terminal Interface

The terminal UI is still available after the backend is running:

```bash
source .venv/bin/activate
python steer.py ui
```

The terminal UI is a split-pane completion and steering surface. The default
backend is raw `gpt2-small`, not a chat-tuned assistant, so use prompts that
look like text to continue rather than questions expecting a helpful reply.

## Web Demo

Start from the repo root:

```bash
./start.sh
```

The first run may download GPT-2 small and the first steered generation may
download SAE Lens weights.

In the browser at `http://127.0.0.1:8000/`:

1. In the prompt field, enter:

   ```text
   Today the weather report says
   ```

2. In `Active Steer`, set a manual steer:

   ```text
   Feature id: 204
   Strength: 10
   Layers: 6
   ```

   Press `Set only`.

3. Press `Baseline`, then `Generate`, and compare the saved research runs.

4. Try the cache workflow:

   ```text
   Model: gpt2-small
   Search: time phrases
   ```

   Press `Cache Layers`, or search directly after caching a source. Results
   show feature ids, SAE sources, and local cached labels. Select a result, then
   press `Apply selected label` to copy its model/source/feature label into the
   steer form.

5. Press `Clear` before ending the demo so no steer is left active.

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

In the UI, the normal cache path is model-first. Enter `gpt2-small`, then press
`F10`/`Cache Layers` or search directly. The UI lists Neuronpedia sources for
that model, keeps only compatible residual JB sources such as `0-res-jb`,
`6-res-jb`, and `11-res-jb`, downloads any missing labels, and searches across
those cached layers. Results show the feature id, the layer, and the label.
Manual model/source listing and one-source downloads are still available under
the collapsed `Advanced Source Controls` section at the bottom of the right
pane.

Search works against the local SQLite cache only. It is case-insensitive,
splits the query into words, and returns labels whose description contains
every query word. It is keyword search over cached descriptions, not embedding
or semantic search.

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

Search cached labels across every cached source for the model:

```bash
python steer.py feature-cache search "time phrases" --model-id gpt2-small --limit 10
```

Use `--source` only when you intentionally want to restrict the CLI search to
one source. Leave off both `--model-id` and `--source` to search the whole local
cache:

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

### `serve`

Start the local FastAPI backend:

```bash
python steer.py serve
python steer.py serve --host 127.0.0.1 --port 8000
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

### `doctor`

Check local dependencies, state/cache paths, and backend health:

```bash
python steer.py doctor
python steer.py doctor --skip-server
python steer.py doctor --json
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
python steer.py ui --server-url http://127.0.0.1:8000 --max-tokens 32 --temperature 0
```

Use `--temperature 0` for deterministic demos with the raw GPT-2 backend.

### Web API

When the backend is running, the browser UI and JSON API are available from the
same server:

| Endpoint | Effect |
|----------|--------|
| `/` | Browser UI. |
| `/health` | Active backend model, SAE release, device, and busy status. |
| `/api/model` | Load/switch a TransformerLens model and SAE Lens release. |
| `/api/state` | Read or clear the shared steering state. |
| `/api/state/items` | Set or append an active steering feature. |
| `/api/neuronpedia/models` | List models in the Neuronpedia public exports. |
| `/api/neuronpedia/sources` | List source ids for a Neuronpedia model. |
| `/api/cache/source` | Download labels for one Neuronpedia model/source. |
| `/api/cache/search` | Search cached labels by model/source/query. |

### `feature-cache`

```bash
python steer.py feature-cache models
python steer.py feature-cache sources --model-id gpt2-small
python steer.py feature-cache download --model-id gpt2-small --source 6-res-jb
python steer.py feature-cache download --model-id gpt2-small --all-sources --source-contains res-jb
python steer.py feature-cache search "calendar dates" --model-id gpt2-small
python steer.py feature-cache search "calendar dates" --model-id gpt2-small --json
python steer.py feature-cache show --model-id gpt2-small --source 6-res-jb --feature-id 204
python steer.py feature-cache status
python steer.py feature-cache status --json
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

Start the web server in a tmux session:

```bash
tmux new -s steering './start.sh'
```

Reattach later:

```bash
tmux attach -t steering
```

Recommended usage:

| Session | Purpose |
|---------|---------|
| `steering` | Keep `./start.sh` / `python steer.py serve` running. |

tmux is optional; `./start.sh` can also run directly in any terminal until it is
stopped with `Ctrl+C`.

## Configuration

Environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `STEERING_DEVICE` | `cpu` | Stable default for local Mac demos. Set `auto`, `mps`, or `cuda` explicitly to opt into acceleration. |
| `STEERING_MODEL_NAME` | `gpt2-small` | TransformerLens model name. |
| `STEERING_SAE_RELEASE` | `gpt2-small-res-jb` | SAE Lens release. |
| `STEERING_SAE_ID_TEMPLATE` | `blocks.{layer}.hook_resid_pre` | Maps `--layers` values to SAE Lens ids. |
| `STEERING_STATE_PATH` | `.steering/state.json` | Shared steering state path. |
| `STEERING_FEATURE_CACHE_PATH` | `.steering/feature-cache.sqlite3` | SQLite cache for Neuronpedia labels. |
| `STEERING_SERVER_URL` | `http://127.0.0.1:8000` | CLI target server. |
| `STEERING_SERVER_HOST` | `127.0.0.1` | Host used by `./start.sh` when starting uvicorn. |
| `STEERING_SERVER_PORT` | `8000` | Port used by `./start.sh` when starting uvicorn. |
| `STEERING_PYTHON` | auto-detected | Python interpreter used by `./start.sh` when creating `.venv`. |
| `STEERING_VENV_DIR` | `.venv` | Virtual environment directory created/used by `./start.sh`. |
| `STEERING_CLIENT_TIMEOUT` | `60` | Seconds before a CLI/UI request gives up. |
| `STEERING_GENERATION_LOCK_TIMEOUT` | `30` | Seconds a generation waits for another token compute to finish. |
| `STEERING_NEURONPEDIA_TIMEOUT` | `120` | Seconds before feature API requests give up. |
| `STEERING_NEURONPEDIA_DATASET_TIMEOUT` | `120` | Seconds before public export downloads give up. |

TransformerLens currently warns that MPS may produce incorrect results with
PyTorch 2.11.0, and in this stack it can also leave a generation request stuck
before the first streamed token. CPU is therefore the default. To start the
backend explicitly on CPU:

```bash
source .venv/bin/activate
STEERING_DEVICE=cpu python steer.py serve
```

To experiment with MPS anyway:

```bash
source .venv/bin/activate
STEERING_DEVICE=mps python steer.py serve
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
scripts/check.sh
```

The check script runs unit tests, Python compilation, shell syntax checks,
version smoke checks, and packaging metadata consistency checks. The expanded
manual form is:

```bash
python -m unittest discover -s tests
python -m py_compile steer.py server.py steering/*.py tests/*.py
python steer.py feature --layer 6 --feature-id 204
tmpdir=$(mktemp -d)
python steer.py feature-cache download --model-id gpt2-small --source 6-res-jb --max-files 1 --cache-path "$tmpdir/cache.sqlite3"
python steer.py feature-cache search time --model-id gpt2-small --source 6-res-jb --cache-path "$tmpdir/cache.sqlite3" --limit 3
rm -rf "$tmpdir"
```

Backend smoke check, after starting the server:

```bash
scripts/backend_smoke.sh
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
