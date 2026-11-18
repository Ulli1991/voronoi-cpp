#!/bin/bash
# Compile check of the step levers on a CPU node: voronoi_bench of the levers tree (CUDA, Release, -lineinfo), then
# registers, stack and local memory of every device kernel of the repair and the geometry (cuobjdump).
#   sbatch -o build-jobs/logs/%x-%j.out scripts/levers_build.sh [source]
#SBATCH -J lev_build
#SBATCH -p interactive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake
src="${1:-/raven/u/uli/voronoi-cpp-levers}"
dir="$src/build-check"
mkdir -p "$dir"
cmake -S "$src" -B "$dir" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/u/uli/kokkos-cuda \
    -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper -DCMAKE_CXX_EXTENSIONS=OFF -DBUILD_TESTING=OFF \
    -DCMAKE_CXX_FLAGS=-lineinfo >"$dir/configure.log" 2>&1
cmake --build "$dir" --target voronoi_bench -j8 >"$dir/build.log" 2>&1
rc=$?
echo "build exit $rc"
# A clean worktree root's binary goes to the measuring jobs' cache (scripts/levers_a100.sh), keyed by commit.
cache=/raven/u/uli/voronoi-cpp-levers/build-jobs/levers/cache
if ((rc == 0)) && [[ "$(git -C "$src" rev-parse --show-toplevel 2>/dev/null)" == "$(realpath "$src")" ]] &&
    git -C "$src" diff --quiet HEAD -- src CMakeLists.txt; then
    mkdir -p "$cache"
    cp "$dir/voronoi_bench" "$cache/voronoi_bench_$(git -C "$src" rev-parse --short HEAD)" && echo cached
fi
grep -E "error|warning" "$dir/build.log" | grep -v "^--" | head -40
# Resource usage of the kernels: enclosing function, lambda instance, REG, STACK (local memory per thread).
cuobjdump --dump-resource-usage "$dir/voronoi_bench" 2>/dev/null | cu++filt | grep -E "Function .*(tet_repair|tet_mesh|tet_cells)" |
    sed -E 's/.*ParallelFor<[^(]*((tet_repair|tet_mesh|tet_cells)::[a-z_:]+)[<(].*(instance [0-9]+).*(REG:[0-9]+ STACK:[0-9]+).*/\1 \3 \4/' |
    grep -E "^(tet_repair|tet_mesh|tet_cells)" | sort | uniq
