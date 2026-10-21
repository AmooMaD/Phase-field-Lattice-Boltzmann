#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# COOLBM build script (no modules required)
# - Uses CMake on PATH (you have 3.28.3)
# - Enables Release + IPO/LTO + fast math + unrolling + march=native
# - Picks Ninja if available, otherwise Make
# - Lets you override tiles and compiler via env:
#       CXX=clang++ COOLBM_TILE_X=256 COOLBM_TILE_Y=128 ./compile.sh
# -----------------------------------------------------------------------------
set -euo pipefail

# --- knobs you can override via env ------------------------------------------
: "${CXX:=g++}"                         # e.g., CXX=clang++
: "${BUILD_DIR:=build}"
: "${COOLBM_TILE_X:=256}"               # perf-friendly defaults
: "${COOLBM_TILE_Y:=128}"

# --- print tool versions (nice for logs) -------------------------------------
echo "==> Using compiler: ${CXX}"
if command -v cmake >/dev/null 2>&1; then
  echo "==> cmake: $(cmake --version | head -n1)"
elif command -v cmake3 >/dev/null 2>&1; then
  echo "==> cmake3: $(cmake3 --version | head -n1)"
else
  echo "ERROR: cmake not found on PATH." >&2
  exit 1
fi

# --- pick cmake binary --------------------------------------------------------
if command -v cmake3 >/dev/null 2>&1; then
  CMAKE=cmake3
else
  CMAKE=cmake
fi

# --- prepare build dir (clean build) -----------------------------------------
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
/bin/rm -rf *

# --- choose generator: Ninja if available (faster), else Make ----------------
GEN_ARGS=()
BUILD_CMD="make -j"
if command -v ninja >/dev/null 2>&1; then
  GEN_ARGS+=(-G Ninja)
  BUILD_CMD="ninja -v"
fi

# --- configure flags ----------------------------------------------------------
# NOTE: We also set CMP0069 policy to NEW here to avoid touching CMakeLists.txt.
"${CMAKE}" .. \
  "${GEN_ARGS[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_DEFAULT_CMP0069=NEW \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
  -DUSE_TBB=ON \
  -DCOOLBM_TILE_X="${COOLBM_TILE_X}" \
  -DCOOLBM_TILE_Y="${COOLBM_TILE_Y}" \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -ffast-math -funroll-loops -fno-math-errno -fno-trapping-math" \
  -DCMAKE_CUDA_FLAGS_RELEASE="-O3 --use_fast_math"

# If you want verbose compile lines once, uncomment:
# ${BUILD_CMD/ -v/} --verbose -j1

# --- build -------------------------------------------------------------------
eval "${BUILD_CMD}"

echo "==> Run like: COOLBM_THREADS=\$(nproc) ./COOLBM"

