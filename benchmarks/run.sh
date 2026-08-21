#!/usr/bin/env bash

set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
julia="${JULIA:-julia}"
shared=()
tune=()
runner=()
phase=shared
reset_tune=false

for arg in "$@"; do
    case "${arg}" in
        --tune) phase=tune ;;
        --runner) phase=runner ;;
        --reset|--reset-tune) reset_tune=true ;;
        *)
            case "${phase}" in
                shared) shared+=("${arg}") ;;
                tune) tune+=("${arg}") ;;
                runner) runner+=("${arg}") ;;
            esac
            ;;
    esac
done

if [[ "${reset_tune}" == true ]]; then
    tune+=(--reset)
fi

make -C "${dir}/.." dependencies

echo "== Fusion tuning =="
"${julia}" "${dir}/tune.jl" "${shared[@]}" "${tune[@]}"

echo "== Benchmark suite =="
"${julia}" "${dir}/runner.jl" "${shared[@]}" "${runner[@]}"
