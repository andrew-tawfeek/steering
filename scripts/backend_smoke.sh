#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python3}"
SERVER_URL="${STEERING_SERVER_URL:-http://127.0.0.1:8000}"
PROMPT="${STEERING_SMOKE_PROMPT:-The weather today is}"
MAX_TOKENS="${STEERING_SMOKE_MAX_TOKENS:-8}"
STATE_PATH="${STEERING_SMOKE_STATE_PATH:-$(mktemp -t steering-smoke-state.XXXXXX.json)}"
created_state_path=0

if [[ -z "${STEERING_SMOKE_STATE_PATH:-}" ]]; then
  created_state_path=1
fi

cleanup() {
  if [[ "$created_state_path" == "1" ]]; then
    rm -f "$STATE_PATH"
  fi
}
trap cleanup EXIT

"$PYTHON_BIN" steer.py doctor --server-url "$SERVER_URL"
"$PYTHON_BIN" steer.py --state-path "$STATE_PATH" clear >/dev/null
"$PYTHON_BIN" steer.py --state-path "$STATE_PATH" generate "$PROMPT" \
  --server-url "$SERVER_URL" \
  --max-tokens "$MAX_TOKENS" \
  --temperature 0 \
  --no-stream >/dev/null
"$PYTHON_BIN" steer.py --state-path "$STATE_PATH" update \
  --feature-id 204 \
  --strength 10 \
  --layers 6 >/dev/null
"$PYTHON_BIN" steer.py --state-path "$STATE_PATH" generate "$PROMPT" \
  --server-url "$SERVER_URL" \
  --max-tokens "$MAX_TOKENS" \
  --temperature 0 \
  --no-stream >/dev/null
"$PYTHON_BIN" steer.py --state-path "$STATE_PATH" clear >/dev/null

printf 'backend_smoke.sh: ok\n'
