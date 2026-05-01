#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python3}"

"$PYTHON_BIN" -m unittest discover -s tests
"$PYTHON_BIN" -m py_compile steer.py server.py steering/*.py tests/*.py
bash -n start.sh scripts/*.sh
"$PYTHON_BIN" steer.py --version >/dev/null
"$PYTHON_BIN" - <<'PY'
import tomllib
from pathlib import Path

pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
assert pyproject["project"]["scripts"]["steer"] == "steering.cli:main"
assert set(pyproject["project"]["dependencies"]) == {
    line.strip()
    for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
}
PY

printf 'check.sh: ok\n'
