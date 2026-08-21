#!/usr/bin/env bash

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
simplepim="${root}/opt/SimplePIM"
perfetto="${root}/opt/Perfetto"
simplepim_repo="https://github.com/krasow/simple-pim-clone.git"

mkdir -p "${root}/opt"

if [[ -d "${simplepim}/.git" ]]; then
    echo "SimplePIM already installed: ${simplepim}"
elif [[ -e "${simplepim}" ]]; then
    echo "Cannot install SimplePIM: ${simplepim} already exists" >&2
    exit 1
else
    git clone "${simplepim_repo}" "${simplepim}"
fi

if [[ -f "${perfetto}/include/perfetto.h" &&
      -f "${perfetto}/lib/libperfetto.a" ]]; then
    echo "Perfetto already installed: ${perfetto}"
else
    "${root}/scripts/install_perfetto.sh" "${perfetto}"
fi
