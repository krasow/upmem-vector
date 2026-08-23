#!/usr/bin/env bash

set -euo pipefail

suite_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
benchmarks_dir="$(cd "${suite_dir}/.." && pwd)"
results_dir="${benchmarks_dir}/results/dynamic"

mkdir -p "${results_dir}"

"${benchmarks_dir}/run.sh" adaptive_image \
    --config "${suite_dir}/benchmark.toml" \
    --default-params --check --resume --runner \
    --csv "${results_dir}/adaptive-image.csv" \
    --state "${results_dir}/adaptive-image.checkpoint.toml" "$@"
python3 "${suite_dir}/plot_adaptive_image.py"
