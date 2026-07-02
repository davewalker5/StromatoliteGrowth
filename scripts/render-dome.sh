#!/usr/bin/env bash

if (( $# != 1 )); then
    scriptname=$(basename -- "$0")
    echo Usage: $scriptname /path/to/layers.npz
    exit 1
fi

export PROJECT_ROOT=$( cd "$(dirname "$0")/.." ; pwd -P )
cd "$PROJECT_ROOT"

. venv/bin/activate

python scripts/render-interactive-dome.py --clean-axes --dark-mode --input "$1"