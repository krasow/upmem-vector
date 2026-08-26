#!/usr/bin/env bash

set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
julia="${JULIA:-julia}"
shared=()
tune=()
runner=()
phase=shared
pending_scope=
reset_run=false
reset_tune=false
resume_run=false
default_params=false

common_options=(-h --help --dpus --elements-per-dpu --warmup --iterations
                --check --verbose --profiles --timeout
                --build-timeout --config)
tune_options=(--passes --lookahead --hfuse-chains --jit-batch --vfuse-ops
              --workspace --checkpoints)
runner_options=(--list --variant --ntrials --skip-setup --generate-only
                --dry-run --state --csv --no-profile --no-load-ref
                --keep-ref-data --help-all)
runner_value_options=(--variant --ntrials --state --csv)

contains() {
    local needle="$1"
    shift
    local item
    for item in "$@"; do
        [[ "${item}" != "${needle}" ]] || return 0
    done
    return 1
}

validate_option() {
    local scope="$1"
    local option="$2"
    contains "${option}" "${common_options[@]}" && return
    case "${scope}" in
        tune) contains "${option}" "${tune_options[@]}" && return ;;
        runner) contains "${option}" "${runner_options[@]}" && return ;;
    esac
    echo "unknown ${scope} option: ${option}" >&2
    exit 2
}

for arg in "$@"; do
    if [[ -n "${pending_scope}" ]]; then
        case "${pending_scope}" in
            tune) tune+=("${arg}") ;;
            runner) runner+=("${arg}") ;;
        esac
        pending_scope=
        continue
    fi
    case "${arg}" in
        --tune) phase=tune ;;
        --runner) phase=runner ;;
        --reset) reset_run=true ;;
        --reset-tune) reset_tune=true ;;
        --resume) resume_run=true ;;
        --default-params) default_params=true ;;
        *)
            if [[ "${phase}" == shared && "${arg}" == -* ]]; then
                if contains "${arg}" "${runner_options[@]}"; then
                    runner+=("${arg}")
                    contains "${arg}" "${runner_value_options[@]}" &&
                        pending_scope=runner
                    continue
                fi
                if contains "${arg}" "${tune_options[@]}"; then
                    tune+=("${arg}")
                    pending_scope=tune
                    continue
                fi
            fi
            [[ "${arg}" != -* ]] || validate_option "${phase}" "${arg}"
            case "${phase}" in
                shared) shared+=("${arg}") ;;
                tune) tune+=("${arg}") ;;
                runner) runner+=("${arg}") ;;
            esac
            ;;
    esac
done

[[ -z "${pending_scope}" ]] || {
    echo "${!#} needs a value" >&2
    exit 2
}

if [[ "${reset_tune}" == true ]]; then
    [[ "${default_params}" == false ]] || {
        echo "--reset-tune cannot be used with --default-params" >&2
        exit 2
    }
    tune+=(--reset)
fi

[[ "${reset_run}" == false ]] || runner+=(--reset)
if [[ "${resume_run}" == true ]]; then
    runner+=(--resume)
    [[ "${reset_tune}" == true ]] || tune+=(--resume)
fi

make -C "${dir}/.." dependencies

if [[ "${default_params}" == true ]]; then
    echo "== Fusion tuning skipped; using default parameters =="
    runner+=(--no-profile)
else
    echo "== Fusion tuning =="
    "${julia}" --project="${dir}" "${dir}/tune.jl" \
        "${shared[@]}" "${tune[@]}"
fi

echo "== Benchmark suite =="
"${julia}" --project="${dir}" "${dir}/runner.jl" \
    "${shared[@]}" "${runner[@]}"
