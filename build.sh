#!/usr/bin/env bash
set -euo pipefail

# build.sh - Configure & build the project with CMake into ./build
# Usage:
#   ./build.sh                                 # configure (always) & build default target (all)
#   CMAKE_BUILD_TYPE=RelWithDebInfo ./build.sh  # override build type
#   BUILD_DIR=out ./build.sh                   # override build directory

BUILD_DIR=${BUILD_DIR:-build}
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Debug}

if command -v ninja >/dev/null 2>&1; then
	GENERATOR="-G Ninja"
	echo "[build] Using Ninja generator"
else
	GENERATOR=""
	echo "[build] Using CMake default generator"
fi

echo "[build] Build directory: ${BUILD_DIR}";
echo "[build] Build type: ${CMAKE_BUILD_TYPE}";

mkdir -p "${BUILD_DIR}"

echo "[build] Configuring..."
cmake -S . -B "${BUILD_DIR}" ${GENERATOR} -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" "$@" || {
	echo "[build] Configure failed" >&2; exit 1; }

echo "[build] Building..."
cmake --build "${BUILD_DIR}" --parallel

echo "[build] Done. Run ${BUILD_DIR}/se3_foxglove"
