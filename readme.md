# Steering CLI

A local CLI and server for live SAE feature steering.

The intended runtime is **not Ollama**. Ollama is useful for ordinary local
generation, but it does not expose hidden activations or residual-stream hooks,
so it cannot perform the feature steering described by Neuronpedia SAE pages.

This project uses:

- **TransformerLens** for a hookable local transformer runtime.
- **SAE Lens** for loading pretrained sparse autoencoders and decoder vectors.
- **Neuronpedia** as the feature catalog and metadata source.

## How It Works

1. `server.py` loads a local TransformerLens model, defaulting to `gpt2-small`.
2. The server loads SAE Lens SAEs on demand, defaulting to `gpt2-small-res-jb`.
3. `steer.py update` writes active feature steers into `.steering/state.json`.
4. During generation, the server rereads that state and adds:

```text
strength * SAE.W_dec[feature_id]
```

to the configured residual-stream hook point.

For the default GPT-2 setup, `--layers 6` maps to:

```text
blocks.6.hook_resid_pre
```

and the matching Neuronpedia page is:

```text
https://www.neuronpedia.org/gpt2-small/6-res-jb/<feature-id>
```

## Requirements

- macOS
- Python 3.11 or newer
- Enough disk space for GPT-2 small and SAE downloads

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Apple Silicon, PyTorch should use MPS by default through the configured
backend device. You can force CPU if needed:

```bash
export STEERING_DEVICE=cpu
```

## Run

Terminal A starts the hookable backend:

```bash
source .venv/bin/activate
uvicorn server:app --host 127.0.0.1 --port 8000
```

The first launch downloads model and SAE weights.

Terminal B controls steering:

```bash
source .venv/bin/activate
python3 steer.py health
python3 steer.py feature --layer 6 --feature-id 204
python3 steer.py update --feature-id 204 --strength 30 --layers 6
python3 steer.py generate "The weather today is" --max-tokens 40
python3 steer.py clear
```

Existing tmux lanes for this repo:

```bash
tmux attach -t steering-backend
tmux attach -t steering-impl
tmux attach -t steering-tests
```

## Commands

### Update State

Replace active steering with one feature:

```bash
python3 steer.py update --feature-id 204 --strength 30 --layers 6
```

Stack another feature:

```bash
python3 steer.py update --feature-id 1200 --strength 15 --layers 8 --append
```

Use an explicit SAE Lens id instead of layer shorthand:

```bash
python3 steer.py update \
  --feature-id 204 \
  --strength 30 \
  --sae-id blocks.6.hook_resid_pre
```

### Inspect State

```bash
python3 steer.py show
python3 steer.py show --json
```

### Clear State

```bash
python3 steer.py clear
```

### Look Up a Feature

Use Neuronpedia as the source for feature descriptions:

```bash
python3 steer.py feature --layer 6 --feature-id 204
```

Equivalent explicit Neuronpedia SAE id:

```bash
python3 steer.py feature --model-id gpt2-small --sae-id 6-res-jb --feature-id 204
```

### Generate

Generate through the local TransformerLens server:

```bash
python3 steer.py generate "The weather today is" --max-tokens 40
```

Generation streams by default. Use `--no-stream` to wait for the full response.

### Chat

```bash
python3 steer.py chat
```

Inside chat:

| Command | Effect |
|---------|--------|
| `/show` | Print active steering state. |
| `/clear` | Clear active steering state. |
| `/health` | Check the backend. |
| `/exit` or `/quit` | Leave chat. |

### Ollama Models

This command is only for reference:

```bash
python3 steer.py ollama-models
```

Ollama models are not used for SAE feature steering by this backend.

## Configuration

Environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `STEERING_DEVICE` | `auto` | Torch device. `auto` chooses `mps`, then `cuda`, then `cpu`. |
| `STEERING_MODEL_NAME` | `gpt2-small` | TransformerLens model name. |
| `STEERING_SAE_RELEASE` | `gpt2-small-res-jb` | SAE Lens release name. |
| `STEERING_SAE_ID_TEMPLATE` | `blocks.{layer}.hook_resid_pre` | Maps `--layers` values to SAE Lens ids. |
| `STEERING_STATE_PATH` | `.steering/state.json` | Shared steering state path. |
| `STEERING_SERVER_URL` | `http://127.0.0.1:8000` | CLI target server. |

## State File

By default, state is written to:

```bash
.steering/state.json
```

Example:

```json
{
  "version": 1,
  "updated_at": "2026-04-30T18:14:35+00:00",
  "items": [
    {
      "feature_id": 204,
      "strength": 30.0,
      "layers": [6]
    }
  ]
}
```

Use a custom path by putting `--state-path` before the subcommand:

```bash
python3 steer.py --state-path /tmp/steering.json update --feature-id 204 --strength 30 --layers 6
```

## Testing

Run unit tests:

```bash
python3 -m unittest discover -s tests
```

Run CLI checks:

```bash
python3 steer.py --help
python3 steer.py feature --layer 6 --feature-id 204
```

## Notes on Backend Choices

TransformerLens is the best first backend for this repo because its hook points
line up directly with the SAE Lens GPT-2 residual-stream SAEs.

NNsight is also a strong route, especially for Hugging Face models and remote
NDIF execution. It is a good candidate for a second backend once the
TransformerLens path is stable.
