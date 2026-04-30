#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  /opt/homebrew/bin/python3.12 -m venv .venv
fi

source .venv/bin/activate

python - <<'PY' || python -m pip install -r requirements.txt
import fastapi
import sae_lens
import textual
import torch
import transformer_lens
import uvicorn
PY

mkdir -p .steering

SERVER_URL="${STEERING_SERVER_URL:-http://127.0.0.1:8000}"
SERVER_LOG="${STEERING_SERVER_LOG:-.steering/server.log}"
SERVER_PID=""

server_ready() {
  python - "$SERVER_URL" <<'PY'
import json
import sys
from urllib import request

try:
    with request.urlopen(sys.argv[1].rstrip("/") + "/health", timeout=1.0) as response:
        json.loads(response.read().decode("utf-8"))
except Exception:
    raise SystemExit(1)
PY
}

if ! server_ready; then
  uvicorn server:app --host 127.0.0.1 --port 8000 >"$SERVER_LOG" 2>&1 &
  SERVER_PID="$!"
  echo "Starting steering backend on $SERVER_URL (log: $SERVER_LOG)"

  for _ in $(seq 1 180); do
    if server_ready; then
      break
    fi
    sleep 1
  done
fi

cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if ! server_ready; then
  echo "Backend did not become ready. Last log lines:" >&2
  tail -n 80 "$SERVER_LOG" >&2 || true
  exit 1
fi

python steer.py ui --server-url "$SERVER_URL"
