#!/usr/bin/env bash
# build.sh — reproduce the exact Strata engine build behind the numbers in README.md.
#
# Engine: bodhi37/strata at orca-port (my fork, not stock upstream). The recipe's
# measurements need that branch: prefix cache across turns, KV streaming,
# mlocked hot tier, and the server-side sampler/MTP path.
#
# Usage:
#   ./build.sh                 # clone/pull into ~/strata and build
#   SRC=/path/to/strata ./build.sh
set -euo pipefail

# orca-port commit the measurements were taken on.
COMMIT=b90510c78e5334554a48c1ccd2b257019e19d896
REPO=https://github.com/bodhi37/strata.git
BRANCH=orca-port

SRC="${SRC:-$HOME/strata}"
BUILD="${BUILD:-$SRC/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-8}"

# --- toolchain checks -------------------------------------------------------
command -v cmake >/dev/null || { echo "build.sh: cmake not found" >&2; exit 1; }
command -v ninja >/dev/null || { echo "build.sh: ninja not found" >&2; exit 1; }
command -v python3 >/dev/null || { echo "build.sh: python3 not found" >&2; exit 1; }

CUDA="${CUDA:-$HOME/deps/cuda-13.3/opt/cuda}"
[ -x "$CUDA/bin/nvcc" ] || [ -x /opt/cuda/bin/nvcc ] || command -v nvcc >/dev/null || {
  echo "build.sh: nvcc not found — expected at $CUDA/bin/nvcc (13.3 tested)" >&2
  exit 1
}
if [ -x "$CUDA/bin/nvcc" ]; then
  export CUDA_PATH="$CUDA"
  export PATH="$CUDA/bin:$PATH"
  export CUDACXX="$CUDA/bin/nvcc"
fi

# --- source -----------------------------------------------------------------
if [ ! -d "$SRC/.git" ]; then
  mkdir -p "$(dirname "$SRC")"
  git clone --branch "$BRANCH" "$REPO" "$SRC"
fi
git -C "$SRC" fetch --quiet origin
git -C "$SRC" checkout --quiet "$COMMIT"

# vendored ggml must be present (STRATA_GGML_DIR below points at it)
[ -d "$SRC/third_party/llama.cpp" ] || {
  echo "build.sh: $SRC/third_party/llama.cpp missing (submodule not checked out?)" >&2
  exit 1
}

# --- configure + build ------------------------------------------------------
# Flags below match the build-engine.sh of the measured build:
#   STRATA_ENABLE_CUDA=ON  STRATA_BUILD_TESTS=OFF  STRATA_WERROR=OFF
#   CMAKE_CUDA_ARCHITECTURES=89 (RTX 4070 SUPER, sm_89)
#   STRATA_GGML_DIR=<src>/third_party/llama.cpp  STRATA_NATIVE_EXPERTS=ON
cmake -G Ninja -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_WERROR=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DSTRATA_GGML_DIR="$SRC/third_party/llama.cpp" \
  -DSTRATA_NATIVE_EXPERTS=ON

cmake --build "$BUILD" --target strata -j "$JOBS"

mkdir -p "$SRC/engine"
cp -f "$BUILD/strata" "$SRC/engine/strata"
chmod +x "$SRC/engine/strata"

# cap_ipc_lock lets pin_hot actually mlock the host tier; `cp` clears file
# xattrs, so re-apply. Needs root once — non-fatal if it fails, but the hot
# tier will then stay reclaimable and decode collapses under memory pressure.
if command -v setcap >/dev/null 2>&1; then
  if ! setcap cap_ipc_lock,cap_sys_nice+ep "$SRC/engine/strata" 2>/dev/null; then
    echo "build.sh: setcap failed (run once with sudo for mlocked hot tier):" >&2
    echo "  sudo setcap cap_ipc_lock,cap_sys_nice+ep $SRC/engine/strata" >&2
  fi
fi
getcap "$SRC/engine/strata" || true

echo
echo "built: $SRC/engine/strata"
echo
echo "next:  ./serve.sh  (expects packs + configs; see README.md section 3)"
