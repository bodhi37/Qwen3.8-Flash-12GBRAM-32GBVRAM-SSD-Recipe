#!/usr/bin/env bash
# serve.sh — OpenAI-compatible endpoint for Qwen3.8-Flash-Next (qwen4exp MoE)
#             on a 12 GB GPU + ~30 GB RAM, via the Strata orca-port branch.
#
# Measured on: Ryzen 9 9900X · RTX 4070 SUPER 12 GB · 30 GB RAM · SPCC 1 TB NVMe
#   22.6 tok/s decode live (window mean 20.6, median 20.5, p95 25.3) · 389 tok/s
#   fresh prefill, 17k-91k on prefix-cache hits · 131072 context · RSS 24.6 GiB
#
# Override with env vars:  CONFIG=… STRATA_SRC=… PORT=… HOST=… ./serve.sh
# Extra server args pass straight through:  ./serve.sh --api-key foo
set -euo pipefail

cd "$(dirname "$0")"

CONFIG="${CONFIG:-./strata-sc117-iq3s.json}"
STRATA_SRC="${STRATA_SRC:-$HOME/strata}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8126}"

SERVER="$STRATA_SRC/serve/server.py"
VENV_PY="$STRATA_SRC/venv/bin/python"

[ -f "$CONFIG" ] || { echo "serve.sh: config not found: $CONFIG" >&2; exit 1; }
[ -f "$SERVER" ] || { echo "serve.sh: server not found: $SERVER (run ./build.sh)" >&2; exit 1; }
[ -x "$VENV_PY" ] || { echo "serve.sh: venv python not found: $VENV_PY (run setup.sh in $STRATA_SRC)" >&2; exit 1; }

# The checked-in JSONs carry the measured box's absolute paths (/home/USER).
# Materialize a copy with $HOME swapped in so the same layout works elsewhere.
TMP="/tmp/strata-recipe-$(basename "$CONFIG" .json)-$PORT.json"
sed "s|/home/USER|$HOME|g" "$CONFIG" > "$TMP"

# Fail early on the paths that matter instead of mid-load.
missing=0
for p in $(python3 -c "
import json,sys
c=json.load(open('$TMP'))
print(c.get('cwd',''))
print(c.get('tokenizer',''))
for a in c.get('args',[]):
  if a.startswith('/') and ('pack' in a or 'gguf' in a or 'ple' in a or 'profile' in a or '/rt' in a or 'tokenizer' in a):
    print(a)
" 2>/dev/null); do
  [ -e "$p" ] || { echo "serve.sh: missing path: $p" >&2; missing=1; }
done
[ "$missing" = 0 ] || {
  echo "serve.sh: fix the paths above (edit $CONFIG, keeping layout) and retry" >&2
  exit 1
}

# --port/--host on the CLI override the JSON. Server defaults: watchdog 300 s,
# mem-floor governor on; tune via STRATA_WATCHDOG_S / STRATA_MEM_FLOOR_MIB.
exec "$VENV_PY" "$SERVER" \
  --engine strata \
  --config "$TMP" \
  --port "$PORT" --host "$HOST" \
  "$@"
