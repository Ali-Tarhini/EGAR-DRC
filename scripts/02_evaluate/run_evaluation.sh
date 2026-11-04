#!/bin/sh
# Usage: run_evaluation.sh [RUN_DIR] [options]; extra options override the defaults below.
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
if [ $# -eq 0 ] || [ "${1#-}" != "$1" ]; then
    set -- "$ROOT/results/egar" "$@"
fi
exec python -m egar_drc.evaluation.pipeline --repository "$ROOT/data/Rule2DRC" "$@"
