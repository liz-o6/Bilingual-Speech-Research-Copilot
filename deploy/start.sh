#!/bin/sh
set -eu
exec python -m uvicorn src.api.main:app --host 0.0.0.0 --port "${PORT:-8080}" --workers 1
