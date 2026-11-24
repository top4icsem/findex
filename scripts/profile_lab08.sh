#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p docs/lab08_profile
QUERY='python AND search AND engine'
uv run python -m findex.profile_lab08 --corpus data/ --index index.bin --query "$QUERY"
if command -v py-spy >/dev/null 2>&1; then
  echo 'Recording py-spy SVGs. macOS may ask for process-inspection permission.'
  py-spy record -o docs/lab08_profile/build_flame.svg -- .venv/bin/python -m findex.cli index data/ --out /tmp/findex_lab08_profile.bin --executor serial --workers 1
  py-spy record -o docs/lab08_profile/search_flame.svg -- .venv/bin/python -m findex.ranking index.bin "$QUERY" --scorer bm25
else
  echo 'py-spy not installed. Install: brew install py-spy; then rerun this script.'
fi
if command -v scalene >/dev/null 2>&1; then
  uv run scalene --cpu-only --outfile docs/lab08_profile/scalene_search.txt -- python -m findex.ranking index.bin "$QUERY" --scorer bm25 || true
else
  echo 'Scalene optional: uv tool run scalene --help (macOS support varies)'
fi
