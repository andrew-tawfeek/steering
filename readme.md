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

Inside chat:

| Command | Effect |
|---------|--------|
| `/show` | Print active steering state. |
| `/clear` | Clear active steering state. |
| `/health` | Check backend status. |
| `/exit` or `/quit` | Leave chat. |

## tmux Workflow

Existing sessions:

```bash
tmux attach -t steering-backend
tmux attach -t steering-impl
tmux attach -t steering-tests
```

Recommended usage:

| Session | Purpose |
|---------|---------|
| `steering-backend` | Keep `uvicorn server:app --host 127.0.0.1 --port 8000` running. |
| `steering-impl` | Edit code and inspect files. |
| `steering-tests` | Run tests and CLI smoke checks. |

## Configuration

Environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `STEERING_DEVICE` | `auto` | Chooses `mps`, then `cuda`, then `cpu`. |
| `STEERING_MODEL_NAME` | `gpt2-small` | TransformerLens model name. |
| `STEERING_SAE_RELEASE` | `gpt2-small-res-jb` | SAE Lens release. |
| `STEERING_SAE_ID_TEMPLATE` | `blocks.{layer}.hook_resid_pre` | Maps `--layers` values to SAE Lens ids. |
| `STEERING_STATE_PATH` | `.steering/state.json` | Shared steering state path. |
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
```

## Backend Notes

TransformerLens is the first backend because its hook names line up directly
with SAE Lens GPT-2 residual-stream SAEs.

NNsight is a good next backend for Hugging Face models and NDIF-style remote
execution, but it is not required for the current GPT-2/SAE Lens path.
