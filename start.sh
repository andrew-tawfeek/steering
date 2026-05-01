#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${STEERING_VENV_DIR:-.venv}"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=10

find_python() {
  if [[ -n "${STEERING_PYTHON:-}" ]]; then
    printf '%s\n' "$STEERING_PYTHON"
    return
  fi

  local candidate
  for candidate in python3.12 /opt/homebrew/bin/python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return
    fi
  done

  echo "error: no Python interpreter found. Install Python 3.12, or set STEERING_PYTHON=/path/to/python." >&2
  return 1
}

dependency_hash() {
  python - <<'PY'
from hashlib import sha256
from pathlib import Path

digest = sha256()
for name in ("requirements.txt", "pyproject.toml"):
    path = Path(name)
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")
print(digest.hexdigest())
PY
}

dependencies_importable() {
  python - <<'PY'
import importlib

for module_name in (
    "fastapi",
    "textual",
    "torch",
    "transformer_lens",
    "sae_lens",
    "uvicorn",
    "steering",
):
    importlib.import_module(module_name)
PY
}

ensure_python_version() {
  python - "$MIN_PYTHON_MAJOR" "$MIN_PYTHON_MINOR" <<'PY'
import sys

major = int(sys.argv[1])
minor = int(sys.argv[2])
if sys.version_info < (major, minor):
    raise SystemExit(
        f"error: Python {major}.{minor}+ is required; active Python is "
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} at {sys.executable}"
    )
PY
}

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment in $VENV_DIR"
  "$(find_python)" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
ensure_python_version

mkdir -p .steering

STAMP_FILE="$VENV_DIR/.steering-deps.sha256"
CURRENT_HASH="$(dependency_hash)"
INSTALLED_HASH=""
if [[ -f "$STAMP_FILE" ]]; then
  INSTALLED_HASH="$(cat "$STAMP_FILE")"
fi

if [[ "$CURRENT_HASH" != "$INSTALLED_HASH" ]] || ! dependencies_importable >/dev/null 2>&1; then
  echo "Installing Steering dependencies into $VENV_DIR"
  python -m pip install --upgrade pip "setuptools<82" wheel
  python -m pip install -r requirements.txt
  python -m pip install -e .
  printf '%s\n' "$CURRENT_HASH" > "$STAMP_FILE"
else
  echo "Dependencies ready in $VENV_DIR"
fi

SERVER_HOST="${STEERING_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${STEERING_SERVER_PORT:-8000}"
if [[ -n "${STEERING_SERVER_URL:-}" && ( -z "${STEERING_SERVER_HOST:-}" || -z "${STEERING_SERVER_PORT:-}" ) ]]; then
  parsed_server="$(
    python - "$STEERING_SERVER_URL" <<'PY'
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

DISPLAY_HOST="$SERVER_HOST"
DISPLAY_PORT="$SERVER_PORT"
user_args=("$@")
i=0
while (( i < ${#user_args[@]} )); do
  case "${user_args[$i]}" in
    --host)
      if (( i + 1 < ${#user_args[@]} )); then
        DISPLAY_HOST="${user_args[$((i + 1))]}"
        ((i += 1))
      fi
      ;;
    --host=*)
      DISPLAY_HOST="${user_args[$i]#--host=}"
      ;;
    --port)
      if (( i + 1 < ${#user_args[@]} )); then
        DISPLAY_PORT="${user_args[$((i + 1))]}"
        ((i += 1))
      fi
      ;;
    --port=*)
      DISPLAY_PORT="${user_args[$i]#--port=}"
      ;;
  esac
  ((i += 1))
done

SERVER_URL="http://${DISPLAY_HOST}:${DISPLAY_PORT}"
echo "Starting Steering web server at $SERVER_URL"
echo "Open $SERVER_URL/ in your browser. Press Ctrl+C here to stop the server."

serve_args=(serve --host "$SERVER_HOST" --port "$SERVER_PORT")
if [[ $# -gt 0 ]]; then
  serve_args+=("$@")
fi

exec python steer.py "${serve_args[@]}"
