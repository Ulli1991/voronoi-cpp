#!/bin/bash
# Basselin et al. GPU-Restricted-Power-Diagrams (OpenCL, double) in the library benchmark on one A100, K 70 P 30 V 40
# (README white-noise settings; relaunches with x1.5), on jittered, uniform, clustered, halo at 2097152 and 16777216 points. 1 untimed run + 3 timed each.
#   sbatch -o bench/libraries/results/basselin-a100-%j.jsonl -e /raven/u/uli/libbench-data/logs/basselin-a100-%j.err \
#          bench/libraries/run_basselin_a100.sh /raven/u/uli/libbench-data [extra driver options]
#SBATCH -J libbench_basselin
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
bin=/raven/u/uli/libbench-src/basselin/install/bin
export OMP_NUM_THREADS=18 OMP_PROC_BIND=spread OMP_PLACES=cores
export LIBBENCH_LIB_VERSION="196e7d7"
clocks() { nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu --format=csv,noheader | sed 's/^/# clocks: /' >&2; }
nvidia-smi -L >&2
clocks
for n in ${SIZES:-2097152 16777216}; do
    for kind in ${KINDS:-jittered uniform clustered halo}; do
        "$bin/basselin" "$data/points/${kind}_${n}.pts" \
            --volumes="$data/volumes/basselin-a100_${kind}_${n}.f64" "$@" || echo "# basselin $kind $n failed: exit $?" >&2
        clocks
    done
done
