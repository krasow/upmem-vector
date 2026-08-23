#!/usr/bin/env bash

export UPMEM_NO_OS_WARNING=1

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${BENCHMARK_DIR}/../.localenv" ]]; then
    source "${BENCHMARK_DIR}/../.localenv"
fi

if [[ -z "${UPMEM_HOME:-}" ]]; then
    UPMEM_ENV="${UPMEM_ENV:-/usr/upmem_env.sh}"
    if [[ ! -f "${UPMEM_ENV}" ]]; then
        echo "UPMEM SDK environment not found; set UPMEM_ENV" >&2
        return 1
    fi
    source "${UPMEM_ENV}"
fi

if [[ -z "${SIMPLE_PIM_LIB:-}" ]]; then
    export SIMPLE_PIM_LIB="${BENCHMARK_DIR}/../opt/SimplePIM/lib"
fi
