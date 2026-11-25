#!/bin/bash
# voronoi-cpp's CPU rows of the library benchmark on the tree it runs from (issue #6): OpenMP build of library_bench
# (Kokkos gcc 15, -march=icelake-server, as the tables' rows), then bench/libraries/run_voronoi_cpp_cpu.sh's runs at 72
# threads on one full node (warp rebuild and exact start of the 2.1M and 16.8M point files, the moving trajectory).
#   sbatch -o <out>/voronoi-cpp-cpu-%j.jsonl -e <log> scripts/readme_numbers_cpu.sh <build dir> <library data dir>
#SBATCH -J readme_cpu
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=01:30:00
set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/15 cmake
build="$1"
data="$2"
cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DKokkos_DIR=/u/uli/kokkos-gcc15/lib64/cmake/Kokkos \
    -DCMAKE_CXX_FLAGS=-march=icelake-server -DBUILD_TESTING=OFF >&2
cmake --build "$build" --parallel 72 --target library_bench >&2
export LIBBENCH_COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src bench || echo +local)"
export OMP_PROC_BIND=spread OMP_PLACES=cores OMP_NUM_THREADS=72
lscpu | grep -E "Model name|MHz" >&2
for n in 2097152 16777216; do
    for kind in jittered uniform clustered halo bluenoise; do
        for path in warp exact; do
            "$build/library_bench" build "$data/points/${kind}_${n}.pts" --path=$path --repeats=3 \
                --volumes="$data/volumes/voronoi-cpp-${path}-cpu72_${kind}_${n}.f64"
        done
    done
done
"$build/library_bench" step "$data/points/jittered_2097152_step0.03.trj" \
    --volumes="$data/volumes/voronoi-cpp-step-cpu72_jittered_2097152_step10.f64"
