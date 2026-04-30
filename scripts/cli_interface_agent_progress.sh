#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/Users/atawfeek/GitHub/steering"
LOG_FILE="$ROOT_DIR/.steering/cli-interface-agent.log"
HEARTBEAT_SECONDS="${CLI_AGENT_HEARTBEAT_SECONDS:-30}"

cd "$ROOT_DIR"
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

heartbeat() {
  local changed_count
  local codex_summary

  while true; do
    changed_count="$(git status --short | wc -l | tr -d ' ')"
    codex_summary="$(
      ps -axo pid,etime,stat,command |
        grep 'codex exec --cd /Users/atawfeek/GitHub/steering' |
        grep -v grep |
        head -n 1 |
        awk '{print "pid="$1" elapsed="$2" stat="$3}'
    )"
    if [[ -z "$codex_summary" ]]; then
      codex_summary="codex exec not currently visible"
    fi
    printf '[watch] %s heartbeat: %s changed_files=%s\n' \
      "$(timestamp)" "$codex_summary" "$changed_count" >>"$LOG_FILE"
    sleep "$HEARTBEAT_SECONDS"
  done
}

printf 'Live cli-interface-agent progress. Log: %s\n\n' "$LOG_FILE"
heartbeat &
heartbeat_pid="$!"

cleanup() {
  kill "$heartbeat_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT

tail -n 120 -f "$LOG_FILE"
