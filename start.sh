#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

find_python() {
  if [[ -n "${STEERING_PYTHON:-}" ]]; then
    printf '%s\n' "$STEERING_PYTHON"
    return
  fi

  local candidate
  for candidate in python3.12 /opt/homebrew/bin/python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return
    fi
  done

  echo "No Python interpreter found. Install Python 3.12 or set STEERING_PYTHON=/path/to/python." >&2
  return 1
}

if [[ ! -d ".venv" ]]; then
  "$(find_python)" -m venv .venv
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

SERVER_HOST="${STEERING_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${STEERING_SERVER_PORT:-8000}"
SERVER_URL="${STEERING_SERVER_URL:-http://${SERVER_HOST}:${SERVER_PORT}}"
if [[ -n "${STEERING_SERVER_URL:-}" && ( -z "${STEERING_SERVER_HOST:-}" || -z "${STEERING_SERVER_PORT:-}" ) ]]; then
  parsed_server="$(
    python - "$SERVER_URL" <<'PY'
import sys
from urllib.parse import urlparse

parsed = urlparse(sys.argv[1])
host = parsed.hostname or "127.0.0.1"
if parsed.port is not None:
    port = parsed.port
elif parsed.scheme == "https":
    port = 443
else:
    port = 80
print(host, port)
PY
  )"
  read -r parsed_host parsed_port <<<"$parsed_server"
  SERVER_HOST="${STEERING_SERVER_HOST:-$parsed_host}"
  SERVER_PORT="${STEERING_SERVER_PORT:-$parsed_port}"
fi
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
  python steer.py serve --host "$SERVER_HOST" --port "$SERVER_PORT" >"$SERVER_LOG" 2>&1 &
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
