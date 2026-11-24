#!/usr/bin/env bash
# Measure the two route variants and 4-worker Uvicorn against the exact same index.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v oha >/dev/null || { echo 'Install oha first: brew install oha'; exit 1; }
mkdir -p docs/lab07_oha
export INDEX_PATH="${INDEX_PATH:-$(pwd)/index.bin}"
export ENABLE_BENCH_ENDPOINTS=1
if [[ ! -f "$INDEX_PATH" ]]; then
  echo "Missing index: $INDEX_PATH. Build one with uv run findex index data/ --out index.bin"; exit 2
fi
start_server(){
  local workers="$1"
  uv run findex serve --host 127.0.0.1 --port 18007 --workers "$workers" > /tmp/findex-lab07-uvicorn.log 2>&1 &
  server_pid=$!
  for _ in $(seq 1 80); do
    if curl -fsS http://127.0.0.1:18007/health > /dev/null; then return 0; fi
    sleep 0.25
  done
  echo 'Server failed to start'; cat /tmp/findex-lab07-uvicorn.log; exit 3
}
stop_server(){
  if [[ -n "${server_pid:-}" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}
trap stop_server EXIT
start_server 1
oha -n 400 -c 20 --no-tui --output-format json -o docs/lab07_oha/one_async.json 'http://127.0.0.1:18007/_bench/async?q=python&k=5'
oha -n 400 -c 20 --no-tui --output-format json -o docs/lab07_oha/one_sync.json 'http://127.0.0.1:18007/_bench/sync?q=python&k=5'
stop_server
start_server 4
oha -n 400 -c 20 --no-tui --output-format json -o docs/lab07_oha/four_workers.json 'http://127.0.0.1:18007/search?q=python&k=5'
stop_server
printf '\nNow measure the public deployment (only your own server):\n'
printf "  oha -n 400 -c 20 --no-tui --output-format json -o docs/lab07_oha/deployed.json 'https://YOUR-URL/search?q=python&k=5'\n"
printf 'After public test: python3 scripts/oha_table.py\n'
