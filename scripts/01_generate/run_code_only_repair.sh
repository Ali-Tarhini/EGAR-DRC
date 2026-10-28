#!/bin/sh
# Extra arguments override the defaults below.
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
exec python -m egar_drc.runtime.runner --flow code_only_repair \
    --model-config "$ROOT/model_config.yaml" \
    --repository "$ROOT/data/Rule2DRC" \
    --output-dir "$ROOT/results/code_only_repair" \
    --jobs 64 "$@"
