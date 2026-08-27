#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${1:-${ROOT}/opt/scc}"
SCC_VERSION="v4.0.0"
BASE_URL="https://github.com/boyter/scc/releases/download/${SCC_VERSION}"
ASSET="scc_Linux_x86_64.tar.gz"

TEMP_DIR=$(mktemp -d)

cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

echo "Installing scc ${SCC_VERSION} (${ASSET}) to ${DEST_DIR}..."

curl -fsSL -o "${TEMP_DIR}/${ASSET}" "${BASE_URL}/${ASSET}"
curl -fsSL -o "${TEMP_DIR}/checksums.txt" "${BASE_URL}/checksums.txt"

# One checksums.txt covers every platform, so keep only our asset's line;
# sha256sum would otherwise fail on the files we did not download.
(cd "$TEMP_DIR" && grep "  ${ASSET}\$" checksums.txt | sha256sum --check -)

mkdir -p "${DEST_DIR}/bin"
tar -xzf "${TEMP_DIR}/${ASSET}" -C "$TEMP_DIR" scc
install -m 755 "${TEMP_DIR}/scc" "${DEST_DIR}/bin/scc"

echo "scc installed successfully to ${DEST_DIR}/bin/scc"
"${DEST_DIR}/bin/scc" --version
