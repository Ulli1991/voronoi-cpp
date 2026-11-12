#!/bin/bash
# votess on one A100 through AdaptiveCpp (generic SSCP target, CUDA backend; build_votess.sh): tesellate() on the
# device path, and the same plus our double volumes from its neighbour lists, on every point file (1 untimed run for
# peak memory and volumes + 3 timed per variant). Records (JSON lines) on stdout.
#   sbatch -o bench/libraries/results/votess-a100-%j.jsonl -e /raven/u/uli/libbench-data/logs/votess-a100-%j.err \
#          bench/libraries/run_votess_a100.sh /raven/u/uli/libbench-data   (env SIZES, KINDS select a subset)
# 16777216 does not run on the device path (so SIZES defaults to 2097152): unchunked, votess sizes its SYCL buffers with
# int (n_padded * 32 * 4 > 2^31: AdaptiveCpp assertion, job 32231100) and would need ~2.6 kB/cell, 48 GB > 40 GB; with its
# use_chunking the device path returns wrong neighbour lists (2097152 jittered, chunk 524288: 7378 open cells, volume
# sum 0.9969, 6.5x slower, test job 32231524) and at 16777216 its CPU recompute then fails (std::length_error, 32231213).
#SBATCH -J libbench_votess
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/13 cuda/12.2 boost/1.83
data="${1:-/raven/u/uli/libbench-data}"
V=/raven/u/uli/libbench-src/votess
export LD_LIBRARY_PATH=/mpcdf/soft/SLE_15/packages/x86_64/clang/18.1.8/lib:$V/install/acpp/lib:${LD_LIBRARY_PATH:-}
# No OpenMP binding: it would pin the main thread to one core and votess's host std::threads with it (run_votess_cpu.sh).
export OMP_NUM_THREADS=18 OMP_PROC_BIND=false OMP_DYNAMIC=false ACPP_VISIBILITY_MASK=cuda
export VOTESS_VERSION="f0f805f (main 2025-07-15), AdaptiveCpp v25.02.0 generic SSCP, clang 18.1.8, CUDA 12.2"
clocks() { nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu --format=csv,noheader | sed 's/^/# clocks: /' >&2; }
nvidia-smi -L >&2
"$V/install/acpp/bin/acpp-info" -l >&2
clocks
declare -A shell=([jittered]=4 [uniform]=4 [clustered]=8 [halo]=8)
for n in ${SIZES:-2097152}; do
    for kind in ${KINDS:-jittered uniform clustered halo}; do
        "$V/install/bin/votess_acpp" "$data/points/${kind}_${n}.pts" --device=gpu --shell=${shell[$kind]} --repeats=3 \
            --volumes="$data/volumes/votess-a100_${kind}_${n}.f64" || echo "# FAILED $kind $n: exit $?" >&2
        clocks
    done
done
