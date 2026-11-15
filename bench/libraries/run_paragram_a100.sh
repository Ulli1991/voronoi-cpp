#!/bin/bash
# Paragram on one A100: voronoi_diagram on the padded point set, and the same plus our volumes from its adjacency,
# on every point file (1 warm-up + 1 untimed peak-memory run + 3 timed per variant). Shell width in mean spacings per
# input (checked a posteriori: 2 max R <= w, in the records' periodic field). Records (JSON lines) on stdout.
#   sbatch -o bench/libraries/results/paragram-a100-%j.jsonl -e /raven/u/uli/libbench-data/logs/paragram-a100-%j.err \
#          bench/libraries/run_paragram_a100.sh /raven/u/uli/libbench-data   (env SIZES, KINDS select a subset)
#SBATCH -J libbench_paragram
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/12.8
data="${1:-/raven/u/uli/libbench-data}"
P=/raven/u/uli/libbench-src/paragram
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -n "${SLURM_SUBMIT_DIR:-}" ] && here="$SLURM_SUBMIT_DIR/bench/libraries"
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PATH="$P/venv/bin:$PATH" TORCH_EXTENSIONS_DIR="$P/torch_ext" TORCH_CUDA_ARCH_LIST=8.0 PYTHONPATH="$P/paragram-src"
clocks() { nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu --format=csv,noheader | sed 's/^/# clocks: /' >&2; }
nvidia-smi -L >&2
clocks
declare -A shell=([bluenoise]=4 [jittered]=4 [uniform]=5 [clustered]=8 [halo]=8 [cosmicweb]=8 [fccsolid]=4 [surface]=28)  # uniform 5: 4 failed the check at 16.8M (job 32230657)
for n in ${SIZES:-2097152 16777216}; do
    for kind in ${KINDS:-jittered uniform clustered halo bluenoise}; do
        w=${shell[$kind]}
        # halo at 16.8M: a status-0 cell of R 0.040 failed 2R <= w at 8 spacings (job 32230796); 24 spacings = 0.094.
        [ "$kind" = halo ] && [ "$n" = 16777216 ] && w=24
        [ "$kind" = surface ] && [ "$n" = 16777216 ] && w=56  # empty balls of radius about 0.1 box: 2R in spacings
        "$P/venv/bin/python" "$here/paragram.py" "$data/points/${kind}_${n}.pts" --shell=$w --repeats=3 \
            --volumes="$data/volumes/paragram-a100_${kind}_${n}.f64" || echo "# FAILED $kind $n: exit $?" >&2
        clocks
    done
done
