#!/bin/bash
# One Kokkos with OpenMP and CUDA (docs/ideas.md, "CPU and GPU together"): build cost against the CUDA-only Kokkos on the
# same node, then backends_probe (Mesh<3, OpenMP> against Mesh<3, Cuda>, bits and times; CPU time of the clipping tail).
#   sbatch -o build-overlap/jobs/%x-%j.out scripts/backends_a100.sh
#SBATCH -J backends
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=01:30:00
set -uo pipefail
source /etc/profile.d/modules.sh
repo="$(pwd)"
out="$repo/build-overlap/jobs/$SLURM_JOB_ID"
mkdir -p "$out"
hostname
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.current.sm --format=csv,noheader
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu --format=csv,noheader -l 5 >"$out/clock.csv" &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
configure() {  # configure <dir> <kokkos> <cuda module>
    module purge
    module load gcc/14 "$3" cmake
    cmake -S "$repo" -B "$1" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DCMAKE_PREFIX_PATH="$2" \
        -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper -DCMAKE_CXX_EXTENSIONS=OFF >/dev/null 2>&1
    echo "configure $1 exit $?"
}
timed() {  # timed <label> <dir> <target> <cuda module>: one translation unit, -j1, wall time and binary size
    module purge
    module load gcc/14 "$4" cmake
    local t0=$(date +%s)
    cmake --build "$2" --target "$3" -j1 >"$out/build_$1.log" 2>&1
    local rc=$?
    echo "build $1 exit $rc seconds $(( $(date +%s) - t0 )) bytes $(stat -c %s "$2/$3" 2>/dev/null)"
    grep -m5 -E "error" "$out/build_$1.log"
}
configure "$out/b-cuda" /u/uli/kokkos-cuda cuda/13.0
configure "$out/b-both" /u/uli/kokkos-cuda132-mpi cuda/13.2
# voronoi_bench against the CUDA-only Kokkos (CUDA 13.0) and the OpenMP+CUDA one (CUDA 13.2), side by side.
timed bench_cuda_only "$out/b-cuda" voronoi_bench cuda/13.0 &
timed bench_both "$out/b-both" voronoi_bench cuda/13.2 &
wait
# The probe: a project of its own over the library's headers, against the OpenMP+CUDA Kokkos.
mkdir -p "$out/probe-src"
cat >"$out/probe-src/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.16)
project(backends_probe LANGUAGES CXX)
find_package(Kokkos CONFIG REQUIRED)
add_executable(backends_probe $repo/src/bench/backends_probe.cpp)
target_include_directories(backends_probe PRIVATE $repo/src/neighbour_search $repo/src/voronoi_construct $repo/src/test)
target_link_libraries(backends_probe PRIVATE Kokkos::kokkos)
target_compile_features(backends_probe PRIVATE cxx_std_17)
EOF
module purge
module load gcc/14 cuda/13.2 cmake
cmake -S "$out/probe-src" -B "$out/b-probe" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/u/uli/kokkos-cuda132-mpi \
    -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper -DCMAKE_CXX_EXTENSIONS=OFF >/dev/null 2>&1
timed probe "$out/b-probe" backends_probe cuda/13.2
export OMP_PROC_BIND=spread OMP_PLACES=cores OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Tail counts: cells the warp kernel left at 128³ (measurement build, job 32373756).
for row in "jittered 0" "uniform 33" "clustered 68" "halo 261" "cosmicweb 976"; do
    read -r kind tail <<<"$row"
    echo "== $kind 128"
    timeout 1200 "$out/b-probe/backends_probe" 2097152 "$kind" 0.1 3 "$tail"
done
