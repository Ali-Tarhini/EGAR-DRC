#!/bin/sh
# Extra arguments override the defaults below.
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
exec python -m egar_drc.runtime.runner --flow egar-drc \
    --model-config "$ROOT/model_config.yaml" \
    --repository "$ROOT/data/Rule2DRC" \
    --output-dir "$ROOT/results/egar" \
    --jobs 64 "$@"
