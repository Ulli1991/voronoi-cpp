#!/bin/bash
# peclet-voro on one A100 (Kokkos CUDA): cold build (buildTessellation) and resident build (MovingTessellation
# alloc + rebuild) on every point file, 1 warm-up + 1 untimed peak-memory run + 3 timed; then the moving-mesh
# trajectory, frames 1..10 as MovingTessellation::step. Records (JSON lines) on stdout.
#   sbatch -o bench/libraries/results/peclet-voro-a100-%j.jsonl -e /raven/u/uli/libbench-data/logs/peclet-voro-a100-%j.err \
#          bench/libraries/run_peclet_voro_a100.sh /raven/u/uli/libbench-data
#SBATCH -J libbench_peclet
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:30:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0
data="${1:-/raven/u/uli/libbench-data}"
bin=/raven/u/uli/libbench-src/peclet-voro/install/bin/peclet_voro_cuda
export PECLET_VORO_VERSION="v1.0.4-1-gd7371c9 (d7371c9), Kokkos 5.1.1"
clocks() { nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu --format=csv,noheader | sed 's/^/# clocks: /' >&2; }
nvidia-smi -L >&2
clocks
for n in ${SIZES:-2097152 16777216}; do
    for kind in jittered uniform clustered halo; do
        echo "# $kind $n plain" >&2
        "$bin" build "$data/points/${kind}_${n}.pts" --variant=plain --repeats=3 \
            --volumes="$data/volumes/peclet-voro-a100_${kind}_${n}.f64" || echo "# FAILED plain $kind $n: exit $?" >&2
        clocks
        echo "# $kind $n resident" >&2
        "$bin" build "$data/points/${kind}_${n}.pts" --variant=resident --repeats=3 \
            || echo "# FAILED resident $kind $n: exit $?" >&2
        clocks
    done
done
if [ -z "${SKIP_STEP:-}" ]; then
    "$bin" step "$data/points/jittered_2097152_step0.03.trj" \
    --volumes="$data/volumes/peclet-voro-a100_jittered_2097152_step10.f64" || echo "# FAILED step: exit $?" >&2
fi
clocks
