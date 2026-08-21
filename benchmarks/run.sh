#!/usr/bin/env bash

set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
julia="${JULIA:-julia}"
shared=()
tune=()
runner=()
phase=shared
reset_tune=false
default_params=false

for arg in "$@"; do
    case "${arg}" in
        --tune) phase=tune ;;
        --runner) phase=runner ;;
        --reset|--reset-tune) reset_tune=true ;;
        --default-params) default_params=true ;;
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
    if [[ "${default_params}" == true ]]; then
        echo "--reset and --default-params cannot be used together" >&2
        exit 2
    fi
    tune+=(--reset)
fi

make -C "${dir}/.." dependencies

if [[ "${default_params}" == true ]]; then
    echo "== Fusion tuning skipped; using default parameters =="
    runner+=(--no-profile)
else
    echo "== Fusion tuning =="
    "${julia}" "${dir}/tune.jl" "${shared[@]}" "${tune[@]}"
fi

echo "== Benchmark suite =="
"${julia}" "${dir}/runner.jl" "${shared[@]}" "${runner[@]}"
