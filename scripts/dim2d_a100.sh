#!/bin/bash
# The 2D mesh on one A100; modes build [targets] and mesh (the mesh table at 2^20 and 2^24 cells).
#   sbatch --export=ALL,REPO=$PWD,BUILD=<build dir> -o <logs>/dim2d-%j.out scripts/dim2d_a100.sh <mode> [targets]
#SBATCH -J dim2d
#SBATCH -p gpu1,gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
repo="${REPO:-$PWD}"
build="${BUILD:-$repo/build-2d}"
bin="${BIN:-$build/cuda}"
mode="${1:-build}"
mkdir -p "$build/runs"
hostname
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.applications.graphics,power.limit --format=csv,noheader
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 5 >"$build/runs/clock-${SLURM_JOB_ID:-0}.csv" &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
export KOKKOS_NUM_DEVICES=1 CUDA_LAUNCH_BLOCKING=0
module purge
module load gcc/14 cuda/13.0 cmake hdf5-serial
case "$mode" in
build)
    [[ -f "$build/cuda/CMakeCache.txt" ]] || cmake -S "$repo" -B "$build/cuda" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH=/u/uli/kokkos-cuda -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper | tail -2
    cmake --build "$build/cuda" -j18 --target "${@:2}" 2>&1 | grep -E "warning|error|Error |Built target" | sort | uniq -c | sort -rn | head -30
    ctest --test-dir "$build/cuda" --output-on-failure -R "${TESTS:-mesh2d}" | tail -30
    ;;
mesh)
    for n in 1048576 16777216; do
        for k in jittered random clustered lattice; do
            timeout 100 "$bin/mesh2d_bench" "$k" "$n" --move=0.03
        done
        timeout 100 "$bin/mesh2d_bench" jittered "$n" --move=0.1
        timeout 100 "$bin/mesh2d_bench" jittered "$n" --move=0.3
    done
    ;;
esac
