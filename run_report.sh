#!/usr/bin/env bash
set -euo pipefail

# One-command report generator + local browser view.
# Requires app server running (default: http://127.0.0.1:5060).

APP_BASE_URL="${APP_BASE_URL:-http://127.0.0.1:5060}"
OUT_FILE="${OUT_FILE:-scan_report.html}"
SERVE_PORT="${SERVE_PORT:-8080}"
SYMBOLS="${SYMBOLS:-AAPL,MSFT,NVDA,TSLA,AMZN,META}"
WORKERS="${WORKERS:-1}"
TOP="${TOP:-20}"
DATA_SOURCE="${DATA_SOURCE:-polygon}"
USE_SP500="${USE_SP500:-0}"

if [[ "${DATA_SOURCE}" == "polygon" && -z "${POLYGON_API_KEY:-}" ]]; then
  echo "error: POLYGON_API_KEY is not set." >&2
  echo "set it first: export POLYGON_API_KEY=\"YOUR_REAL_KEY\"" >&2
  exit 1
fi

echo "Checking app at ${APP_BASE_URL}/simple ..."
curl -fsS "${APP_BASE_URL}/simple" >/dev/null

echo "Generating ${OUT_FILE} ..."
if [[ "${USE_SP500}" == "1" ]]; then
  curl -fsS -X POST "${APP_BASE_URL}/simple" \
    -d "use_sp500=on" \
    -d "workers=${WORKERS}" \
    -d "top=${TOP}" \
    -d "data_source=${DATA_SOURCE}" \
    -d "polygon_api_key=${POLYGON_API_KEY:-}" \
    -o "${OUT_FILE}"
else
  curl -fsS -X POST "${APP_BASE_URL}/simple" \
    -d "symbols=${SYMBOLS}" \
    -d "workers=${WORKERS}" \
    -d "top=${TOP}" \
    -d "data_source=${DATA_SOURCE}" \
    -d "polygon_api_key=${POLYGON_API_KEY:-}" \
    -o "${OUT_FILE}"
fi

bytes="$(wc -c <"${OUT_FILE}")"
echo "Saved ${OUT_FILE} (${bytes} bytes)."

echo "Serving current folder on http://127.0.0.1:${SERVE_PORT}/"
echo "Open: http://127.0.0.1:${SERVE_PORT}/${OUT_FILE}"
python3 -m http.server "${SERVE_PORT}"
