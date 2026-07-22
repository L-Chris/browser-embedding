#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo "wasm-bindgen CLI is required (version 0.2.126)." >&2
  echo "Install it with: cargo install wasm-bindgen-cli --version 0.2.126 --locked" >&2
  exit 1
fi

cargo build --release --target wasm32-unknown-unknown -p browser-embedding-wasm
wasm-bindgen \
  target/wasm32-unknown-unknown/release/browser_embedding_wasm.wasm \
  --target bundler \
  --out-dir packages/browser/generated \
  --out-name browser_embedding_wasm
