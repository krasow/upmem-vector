#!/bin/bash
# Same clang-format pass as the hook in .pre-commit-config.yaml.
# For the whitespace hooks as well, use: pre-commit run --all-files
set -euo pipefail

find common dpu host test julia benchmarks/variants -type f \
    \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' \
       -o -name '*.h' -o -name '*.hpp' -o -name '*.inl' \) \
    -not -path '*/build/*' \
    -not -name 'config.h' -not -name 'opcodes.h' \
    -not -name 'opinfo.h' -not -name 'kernelids.h' -not -name 'kernels.h' \
    -print0 | xargs -0 clang-format -i --style=file
