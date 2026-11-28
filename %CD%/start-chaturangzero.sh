#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UI_DIR="${PROJECT_ROOT}/shared/ui"
HOST="${CHATURANGZERO_HOST:-127.0.0.1}"
PORT="${CHATURANGZERO_PORT:-43147}"

if ! command -v npm >/dev/null 2>&1; then
  echo "Node.js and npm are required to start ChaturangZero."
  echo "Install Node.js, then run this launcher again."
  exit 1
fi

if [[ ! -f "${UI_DIR}/package.json" ]]; then
  echo "Frontend package not found at ${UI_DIR}."
  exit 1
fi

cd "${UI_DIR}"

if [[ ! -d node_modules ]]; then
  echo "Installing frontend dependencies..."
  npm install
fi

echo "Starting ChaturangZero at http://${HOST}:${PORT}/"
echo "Press Ctrl+C to stop."

# Future Python backend startup can be added here before the frontend process.
exec npm run dev -- --host "${HOST}" --port "${PORT}" --strictPort
