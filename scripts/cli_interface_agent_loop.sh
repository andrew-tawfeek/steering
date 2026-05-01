#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${STEERING_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BRANCH="${CLI_AGENT_BRANCH:-$(git -C "$ROOT_DIR" branch --show-current 2>/dev/null || printf main)}"
LOG_DIR="$ROOT_DIR/.steering"
LOG_FILE="$LOG_DIR/cli-interface-agent.log"
HEARTBEAT_SECONDS="${CLI_AGENT_HEARTBEAT_SECONDS:-30}"
SLEEP_SECONDS="${CLI_AGENT_SLEEP_SECONDS:-10}"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

emit() {
  printf '%s\n' "$*" | tee -a "$LOG_FILE"
}

prefix_stream() {
  local prefix="$1"
  while IFS= read -r line; do
    emit "[$prefix] $line"
  done
}

summarize_status() {
  local status
  status="$(git status --short)"
  if [[ -z "$status" ]]; then
    emit "[git] status: clean"
  else
    while IFS= read -r line; do
      emit "[git] $line"
    done <<<"$status"
  fi
}

heartbeat_until_done() {
  local pid="$1"
  local started_at="$2"
  local elapsed
  local changed_count

  while kill -0 "$pid" >/dev/null 2>&1; do
    sleep "$HEARTBEAT_SECONDS"
    if kill -0 "$pid" >/dev/null 2>&1; then
      elapsed="$(($(date +%s) - started_at))"
      changed_count="$(git status --short | wc -l | tr -d ' ')"
      emit "[agent] heartbeat: codex exec still running pid=$pid elapsed=${elapsed}s changed_files=$changed_count"
    fi
  done
}

iteration=0

while true; do
  iteration="$((iteration + 1))"
  emit ""
  emit "===== $(timestamp) :: CLI interface loop iteration $iteration ====="

  if ! git switch "$BRANCH" > >(prefix_stream git) 2> >(prefix_stream git); then
    emit "[agent] git switch failed; retrying after ${SLEEP_SECONDS}s"
    sleep "$SLEEP_SECONDS"
    continue
  fi

  summarize_status

  started_at="$(date +%s)"
  emit "[agent] starting codex exec; attach with: tmux attach -t cli-interface-agent"

  codex exec \
    --cd "$ROOT_DIR" \
    --full-auto \
    --sandbox workspace-write \
    "You are working on branch $BRANCH of the steering repo at $ROOT_DIR. Continue developing and hardening the terminal UI launched by start.sh. Focus only on the CLI/TUI interface: left-pane completion workflow, right-pane live feature steering controls, editing/selecting/clearing active steers, accessibility, keyboard navigation, error states, feature-cache workflows, and UI/UX polish. Preserve the existing TransformerLens/SAE backend. Run relevant tests. Do not touch unrelated backend behavior unless needed by the UI." \
    > >(prefix_stream codex) \
    2> >(prefix_stream codex) &
  codex_pid="$!"

  heartbeat_until_done "$codex_pid" "$started_at" &
  heartbeat_pid="$!"

  set +e
  wait "$codex_pid"
  codex_status="$?"
  set -e

  kill "$heartbeat_pid" >/dev/null 2>&1 || true
  wait "$heartbeat_pid" >/dev/null 2>&1 || true

  elapsed="$(($(date +%s) - started_at))"
  emit "[agent] codex exec finished status=$codex_status elapsed=${elapsed}s"
  summarize_status
  emit "[agent] sleeping ${SLEEP_SECONDS}s before next iteration"

  sleep "$SLEEP_SECONDS"
done
