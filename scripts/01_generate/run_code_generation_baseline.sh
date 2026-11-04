#!/bin/sh
# Extra arguments override the defaults below.
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
exec python -m egar_drc.runtime.runner --flow code_generation_baseline \
    --model-config "$ROOT/model_config.yaml" \
    --repository "$ROOT/data/Rule2DRC" \
    --output-dir "$ROOT/results/code_generation" \
    --jobs 64 "$@"
