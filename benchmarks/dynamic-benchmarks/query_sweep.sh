#!/usr/bin/env bash

set -euo pipefail

suite_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
benchmarks_dir="$(cd "${suite_dir}/.." && pwd)"

"${benchmarks_dir}/run.sh" dynamic_query --config "${suite_dir}/query-sweep.toml" \
    --default-params --check --resume --runner \
    --csv "${benchmarks_dir}/results/query-sweep.csv" \
    --state "${benchmarks_dir}/results/query-sweep-state.toml" "$@"
python3 "${suite_dir}/plot_query_sweep.py"
