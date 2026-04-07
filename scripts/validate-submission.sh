#!/usr/bin/env bash
set -euo pipefail

PING_URL="${1:-}"
REPO_DIR="${2:-$(pwd)}"
DOCKER_BUILD_TIMEOUT=600

if [ -z "$PING_URL" ]; then
  echo "Usage: $0 <ping_url> [repo_dir]" >&2
  exit 1
fi

log() { printf '%s\n' "$*"; }
pass() { printf '[PASS] %s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*"; }

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$secs" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null || true
    wait "$watcher" 2>/dev/null || true
    return "$rc"
  fi
}

log "Step 1/3: Checking HF Space health..."
HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' "$PING_URL/health" || true)
if [ "$HTTP_CODE" = "200" ]; then
  pass "Health endpoint returned 200"
else
  fail "HF Space health check failed with HTTP $HTTP_CODE"
  exit 1
fi

log "Step 2/3: Running docker build..."
if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found"
  exit 1
fi

if run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" >/tmp/incident-commander-build.log 2>&1; then
  pass "Docker build succeeded"
else
  fail "Docker build failed"
  tail -20 /tmp/incident-commander-build.log || true
  exit 1
fi

log "Step 3/3: Running openenv validate..."
if command -v openenv >/dev/null 2>&1; then
  if (cd "$REPO_DIR" && openenv validate); then
    pass "openenv validate passed"
  else
    fail "openenv validate failed"
    exit 1
  fi
else
  fail "openenv command not found"
  exit 1
fi

pass "All checks passed"
