#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BACKEND_DIR="${PROJECT_ROOT}/playground/backend"
FRONTEND_DIR="${PROJECT_ROOT}/playground/frontend"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  echo "Loading environment variables from .env"
  # shellcheck disable=SC2046
  export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]] && ps -p "${BACKEND_PID}" > /dev/null 2>&1; then
    echo "Stopping backend (PID ${BACKEND_PID})"
    kill "${BACKEND_PID}" || true
  fi
}

trap cleanup EXIT INT TERM

wait_for_backend() {
  local attempts=0
  local max_attempts=60

  echo "Waiting for backend to become ready..."
  until curl -sf "http://127.0.0.1:8000/api/health" > /dev/null 2>&1; do
    if ! ps -p "${BACKEND_PID}" > /dev/null 2>&1; then
      echo "Backend process exited unexpectedly. Check logs above for details."
      exit 1
    fi
    attempts=$((attempts + 1))
    if [[ ${attempts} -ge ${max_attempts} ]]; then
      echo "Timed out waiting for backend on http://127.0.0.1:8000/api/health"
      exit 1
    fi
    sleep 2
  done
  echo "Backend is ready."
}

echo "Starting backend (uvicorn)..."
(
  cd "${BACKEND_DIR}"
  export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
  if [[ ! -d ".venv" ]]; then
    python3.11 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
    .venv/bin/pip install -e ../../sap-rpt-1-oss
  fi
  source .venv/bin/activate
  uvicorn playground.backend.main:app --host 0.0.0.0 --port 8000
) &

BACKEND_PID=$!
echo "Backend started with PID ${BACKEND_PID}"

wait_for_backend

echo "Starting frontend (npm run dev)..."
cd "${FRONTEND_DIR}"
if [[ ! -d "node_modules" ]]; then
  npm install
fi
npm run dev -- --host 0.0.0.0


