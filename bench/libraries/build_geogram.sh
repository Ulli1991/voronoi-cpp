#!/bin/bash
# Build geogram (https://github.com/BrunoLevy/geogram, with submodules) at a fixed commit into
# /raven/u/uli/libbench-src/geogram/install, and the driver bench/libraries/geogram.cpp into
# /raven/u/uli/libbench-src/geogram/geogram_bench. The clone (login node: downloads only) is made if missing.
#   sbatch -o /raven/u/uli/libbench-data/logs/build_geogram-%j.log bench/libraries/build_geogram.sh
#SBATCH -J build_geogram
#SBATCH -p general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:45:00
set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cmake

commit=8d4b20af24b12fe43ee2289c9c507e5b4ca7a35d   # main, 2026-11-11 (v1.10.1-261-g8d4b20af)
root=/raven/u/uli/libbench-src/geogram
repo="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
here="$repo/bench/libraries"

if [ ! -d "$root/src/.git" ]; then
    git clone --recurse-submodules https://github.com/BrunoLevy/geogram.git "$root/src"
fi
git -C "$root/src" checkout -q "$commit"
git -C "$root/src" submodule update --init --recursive -q
git -C "$root/src" log -1 --format='# geogram %H %cd' >&2

# geogram's own Release flags for Linux64-gcc (static): -O3 -DNDEBUG -m64 -fopenmp -ffp-contract=off -fPIC
# -fvisibility=hidden -D_FORTIFY_SOURCE=2; no -march. Library only: no graphics, Lua, examples.
rm -rf "$root/build"
cmake -S "$root/src" -B "$root/build" \
    -DCMAKE_C_COMPILER="$(which gcc)" -DCMAKE_CXX_COMPILER="$(which g++)" \
    -DCMAKE_BUILD_TYPE=Release \
    -DVORPALINE_PLATFORM=Linux64-gcc \
    -DGEOGRAM_LIB_ONLY=ON \
    -DGEOGRAM_WITH_GRAPHICS=OFF \
    -DGEOGRAM_WITH_LUA=OFF \
    -DGEOGRAM_WITH_TETGEN=OFF \
    -DGEOGRAM_WITH_TRIANGLE=OFF \
    -DGEOGRAM_WITH_HLBFGS=OFF \
    -DGEOGRAM_WITH_LEGACY_NUMERICS=OFF \
    -DCMAKE_INSTALL_PREFIX="$root/install"
cmake --build "$root/build" -j"${SLURM_CPUS_PER_TASK:-16}"
cmake --install "$root/build"
grep -E "^CMAKE_CXX_FLAGS(_RELEASE)?:" "$root/build/CMakeCache.txt" >&2 || true

inc=$(dirname "$(find "$root/install/include" -path '*/geogram/basic/common.h' | head -1)")/../..
lib=$(dirname "$(find "$root/install" -name 'libgeogram.a' | head -1)")
g++ -std=c++20 -O3 -DNDEBUG -fopenmp -ffp-contract=off -Wall -Wextra -I"$here" -I"$inc" \
    "$here/geogram.cpp" -o "$root/geogram_bench" -L"$lib" -lgeogram -ldl -lpthread
echo "# built $root/geogram_bench" >&2
