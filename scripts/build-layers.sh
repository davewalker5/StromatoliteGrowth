#!/usr/bin/env bash

if (( $# != 2 )); then
    scriptname=$(basename -- "$0")
    echo Usage: $scriptname INPUT OUTPUT
    exit 1
fi

export PROJECT_ROOT=$( cd "$(dirname "$0")/.." ; pwd -P )
cd "$PROJECT_ROOT"

. venv/bin/activate

python "$PROJECT_ROOT/src/build-layer-surfaces.py" --input "$1" --output "$2"