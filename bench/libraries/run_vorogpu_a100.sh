#!/bin/bash
# VoroGPU in the library benchmark on one A100, on jittered, uniform, clustered, halo: K 190 (PRESET 0, "white noise")
# at 2097152 points, K 90 (PRESET 1, "perturbed grid") at 2097152 and 16777216. K 190 cannot run at 16777216: the
# code indexes the neighbour array with int (K * points: 190 x 21M padded points > 2^31; kn_prepare's allocation size
# overflows and the kernels' index _K_ * seed + v would too). 1 untimed run, warm-up to 2 s, 3 timed runs each.
# Environment KINDS / SIZES (with sbatch --export=ALL,...) restrict the inputs.
#   sbatch -o bench/libraries/results/vorogpu-a100-%j.jsonl -e /raven/u/uli/libbench-data/logs/vorogpu-a100-%j.err \
#          bench/libraries/run_vorogpu_a100.sh /raven/u/uli/libbench-data [extra driver options]
#SBATCH -J libbench_vorogpu
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
bin=/raven/u/uli/libbench-src/vorogpu/install/bin
export OMP_NUM_THREADS=18 OMP_PROC_BIND=spread OMP_PLACES=cores
export LIBBENCH_LIB_VERSION="f4d7e7e + vorogpu.patch"
clocks() { nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu --format=csv,noheader | sed 's/^/# clocks: /' >&2; }
nvidia-smi -L >&2
clocks
for n in ${SIZES:-2097152 16777216}; do
    for kind in ${KINDS:-jittered uniform clustered halo bluenoise}; do
        # The code's three presets (README: white noise K 190, perturbed grid K 90, blue noise K 35); K 190 only at
        # 2097152 (int index overflow at 16.8M padded points).
        for k in 190 90 35; do
            [ "$k" = 190 ] && [ "$n" != 2097152 ] && continue
            "$bin/vorogpu_k$k" "$data/points/${kind}_${n}.pts" --volumes="$data/volumes/vorogpu-k$k-a100_${kind}_${n}.f64" \
                || echo "# vorogpu k$k $kind $n failed: exit $?" >&2
            clocks
        done
    done
done
