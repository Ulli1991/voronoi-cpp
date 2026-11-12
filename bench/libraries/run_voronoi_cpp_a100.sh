#!/bin/bash
# voronoi-cpp in the library benchmark on one A100: warp rebuild (device to device and host to host) and exact start on
# every point file (env SIZES, KINDS select a subset; SKIP_STEP=1 skips the trajectory), 1 untimed run
# (peak memory, volumes) + 3 timed; then the moving-mesh trajectory. Records (JSON lines) on stdout.
#   sbatch -o bench/libraries/results/voronoi-cpp-a100-%j.jsonl bench/libraries/run_voronoi_cpp_a100.sh <build> <data>
#SBATCH -J libbench_vcpp
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0
build="$1"
data="$2"
export LIBBENCH_COMMIT="$(git -C "$SLURM_SUBMIT_DIR" rev-parse --short HEAD)$(git -C "$SLURM_SUBMIT_DIR" diff --quiet HEAD -- src bench || echo +local)"
clocks() { nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu --format=csv,noheader | sed 's/^/# clocks: /' >&2; }
nvidia-smi -L >&2
clocks
for n in ${SIZES:-2097152 16777216}; do
    for kind in ${KINDS:-jittered uniform clustered halo bluenoise}; do
        for path in warp warp-host exact; do
            "$build/library_bench" build "$data/points/${kind}_${n}.pts" --path=$path --repeats=3 \
                --volumes="$data/volumes/voronoi-cpp-${path}-a100_${kind}_${n}.f64"
            clocks
        done
    done
done
if [ -z "${SKIP_STEP:-}" ]; then
    "$build/library_bench" step "$data/points/jittered_2097152_step0.03.trj" \
        --volumes="$data/volumes/voronoi-cpp-step-a100_jittered_2097152_step10.f64"
fi
clocks
