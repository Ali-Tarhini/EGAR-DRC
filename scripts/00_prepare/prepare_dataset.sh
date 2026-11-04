#!/bin/sh
# Extra arguments override the defaults below.
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
exec python -m egar_drc.prepare_dataset --output-dir "$ROOT/data/Rule2DRC" "$@"
