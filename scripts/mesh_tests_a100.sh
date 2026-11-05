#!/bin/bash
# The mesh's special cases and distorting flows on one A100 (TESTING.md).
# Submit from the repository root with a built CUDA directory and a mode;
# logs must be on a shared file system:
#   sbatch -o <dir>/cases-%j.out scripts/mesh_tests_a100.sh <build-cuda> cases "coincident|few"   # a filter of names
#   sbatch -o <dir>/flows128-%j.out scripts/mesh_tests_a100.sh <build-cuda> flows128
#   sbatch -o <dir>/flows128b-%j.out scripts/mesh_tests_a100.sh <build-cuda> flows128b
#   sbatch -o <dir>/flows256-%j.out scripts/mesh_tests_a100.sh <build-cuda> flows256
#   sbatch -o <dir>/drift-%j.out scripts/mesh_tests_a100.sh <build-cuda> drift
#   sbatch -o <dir>/oracle-%j.out scripts/mesh_tests_a100.sh <build-cuda> oracle
#   sbatch -o <dir>/reach-%j.out scripts/mesh_tests_a100.sh <build-cuda> large
# Each line of a flow is one step (flips, rounds, stuck, fallbacks, M cells/s,
# faces, B/cell, aspect, reach, max faces); each ends in an `ok` or `FAIL` line.
#SBATCH -J mesh_tests
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake
build="${1:-build-cuda}"
mode="${2:-cases}"
export OMP_NUM_THREADS=18
nvidia-smi -L
d="$build/distortion_test"
case "$mode" in
cases)
    # Every case, each in its process, the known ones too.
    for name in $("$build/mesh_cases_test" list | awk '{print $1}' | grep -E "^(${3:-.*})"); do
        timeout -k 5 60 "$build/mesh_cases_test" --known "$name" > /tmp/case.$$ 2>&1
        status=$?
        printf "case=%s exit=%s fail=%s refused=%s\n" "$name" "$status" "$(grep -c '^  FAIL' /tmp/case.$$)" "$(grep -c 'refused:' /tmp/case.$$)"
        grep -E '^  FAIL' /tmp/case.$$ | cut -c1-500
    done
    rm -f /tmp/case.$$
    ;;
repro)
    # Hashes per step on this build, to compare with a CPU build.
    rm -f "$build/repro_cuda_1.txt" "$build/repro_cuda_2.txt"
    MESH_REPRO_DUMP="$build/repro_cuda_1.txt" "$build/mesh_cases_test" repro.dump
    MESH_REPRO_DUMP="$build/repro_cuda_2.txt" "$build/mesh_cases_test" repro.dump
    cmp "$build/repro_cuda_1.txt" "$build/repro_cuda_2.txt" && echo "repro: two CUDA runs agree"
    ;;
flows128)
    # 2M cells. Up to 45 s per flow; checks against a rebuild every 400 steps.
    for flow in shear sawtooth vortex taylor_green turbulence collapse noh; do
        "$d" "$flow" --no-drift --n=2097152 --d=0.3 --steps=3000 --seconds=45 --rebuild-every=400 --print
    done
    ;;
flows128b)
    for flow in expansion pancake counter thin collapse_wide pancake_wide streams; do
        "$d" "$flow" --no-drift --n=2097152 --d=0.3 --steps=3000 --seconds=45 --rebuild-every=400 --print
    done
    ;;
flows256)
    # 16.8M cells, one A100: four flows, rebuild checks on two.
    for flow in turbulence thin shear; do
        "$d" "$flow" --no-drift --n=16777216 --d=0.3 --steps=600 --seconds=150 --rebuild-every=250 --print
    done
    "$build/voronoi_bench" step 16777216 jittered 0.3 6 --memory | grep -E "memory: phase|gpuflips" | cut -c1-200
    ;;
drift)
    # The same flows with this driver's drift to the centroids (the library has none).
    for flow in shear turbulence pancake counter thin; do
        for chi in 0.1 0.5; do
            "$d" "$flow" --drift=$chi --n=2097152 --d=0.3 --steps=3000 --seconds=40 --rebuild-every=800 --print
        done
    done
    ;;
oracle)
    # The full oracle (host, single thread) on the last step of a shorter run.
    for flow in turbulence thin pancake; do
        "$d" "$flow" --no-drift --n=262144 --d=0.3 --steps=400 --seconds=60 --final-oracle --cells=64 --print | tail -3 | cut -c1-700
    done
    ;;
large)
    # Special cases at larger n through the benchmark's verify.
    for spec in "sc@0 0.3" "bcc@0 0.3" "fcc@0 0.3" "shells@1e-9 0.03" "planes@0 0.3" "pairs@1e-6 0.3"; do
        set -- $spec
        "$build/voronoi_bench" step 2097152 "$1" "$2" 3 --verify | grep -E "^start|verify" | cut -c1-220
    done
    for spec in "1 1 0.0625" "1 1 16" "1 2 3"; do
        set -- $spec
        "$build/voronoi_bench" step 2097152 jittered 0.3 3 --verify --box=$1,$2,$3 | grep -E "^start|verify" | cut -c1-220
    done
    ;;
esac
