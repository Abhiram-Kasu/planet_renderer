#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TARGET="wasm32-unknown-unknown"
BINARY="target/$TARGET/release/planet_renderer.wasm"
WASM_BINDGEN_VERSION="0.2.129"

if ! rustup target list --installed | rg -q "^${TARGET}$"; then
  echo "Install the WebAssembly target with: rustup target add ${TARGET}" >&2
  exit 1
fi

if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo "Install the matching wasm-bindgen CLI with: cargo install -f wasm-bindgen-cli --version ${WASM_BINDGEN_VERSION} --locked" >&2
  exit 1
fi

INSTALLED_WASM_BINDGEN_VERSION="$(wasm-bindgen --version | awk '{print $2}')"
if [[ "$INSTALLED_WASM_BINDGEN_VERSION" != "$WASM_BINDGEN_VERSION" ]]; then
  echo "This project needs wasm-bindgen-cli ${WASM_BINDGEN_VERSION}; found ${INSTALLED_WASM_BINDGEN_VERSION}." >&2
  echo "Install it with: cargo install -f wasm-bindgen-cli --version ${WASM_BINDGEN_VERSION} --locked" >&2
  exit 1
fi

cargo build --release --target "$TARGET" --bin planet_renderer
mkdir -p web/pkg
wasm-bindgen --target web --out-dir web/pkg "$BINARY"

PORT="${PORT:-8000}"
echo "Serving the WebAssembly app at http://localhost:${PORT}/"
exec python3 -m http.server "$PORT" --directory "$ROOT/web"
