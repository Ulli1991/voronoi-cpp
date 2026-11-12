#!/bin/bash
# Liu et al. 3D-Voronoi-GPU in the library benchmark on one A100: K 90 (PRESET 0), variants cell-centric and
# tet-centric, on jittered, uniform, clustered, halo at 2097152 and 16777216 points. 1 untimed run + 3 timed each.
#   sbatch -o bench/libraries/results/voronoi-gpu-liu-a100-%j.jsonl -e /raven/u/uli/libbench-data/logs/voronoi-gpu-liu-a100-%j.err \
#          bench/libraries/run_voronoi_gpu_liu_a100.sh /raven/u/uli/libbench-data [extra driver options]
#SBATCH -J libbench_liu
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/13 cuda/12.6
data="$1"; shift
bin=/raven/u/uli/libbench-src/voronoi-gpu-liu/install/bin
export OMP_NUM_THREADS=18 OMP_PROC_BIND=spread OMP_PLACES=cores
export LIBBENCH_LIB_VERSION="b2c64f5 + voronoi-gpu-liu.patch"
clocks() { nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu --format=csv,noheader | sed 's/^/# clocks: /' >&2; }
nvidia-smi -L >&2
clocks
for n in ${SIZES:-2097152 16777216}; do
    for kind in ${KINDS:-jittered uniform clustered halo}; do
        "$bin/voronoi_gpu_liu" "$data/points/${kind}_${n}.pts" \
            --volumes="$data/volumes/voronoi-gpu-liu-a100_${kind}_${n}.f64" "$@" || echo "# liu $kind $n failed: exit $?" >&2
        clocks
    done
done
