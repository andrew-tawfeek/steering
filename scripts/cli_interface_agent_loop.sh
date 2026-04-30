#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/Users/atawfeek/GitHub/steering"
BRANCH="CLI-interface"
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
    "You are working on branch CLI-interface of the steering repo. Continue developing the terminal UI launched by start.sh. Focus only on the CLI/TUI interface: left-pane chat/model workflow, right-pane live feature steering controls, editing/selecting/clearing active steers, accessibility, keyboard navigation, error states, and UI/UX polish. New handoff: /Users/atawfeek/GitHub/steering2 main now contains a merged feature-cache implementation (merge commit c0a7ca8) with steering/feature_cache.py and steer.py feature-cache commands. Pull that feature search ability into the TUI: let users choose/list sources for a model, download/cache Neuronpedia feature labels, search cached labels, inspect a feature id, and apply a selected feature to the active steer form without leaving the interface. Preserve the existing TransformerLens/SAE backend. Run relevant tests. Do not touch unrelated backend behavior unless needed by the UI." \
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
