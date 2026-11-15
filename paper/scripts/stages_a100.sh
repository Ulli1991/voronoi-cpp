#!/bin/bash
# Time per stage and per kernel (voronoi_bench --kernels), and long runs with and without re-sort, on one A100:
#   sbatch -o paper/data/raw/stages-%j.txt paper/scripts/stages_a100.sh <bin> kernels|long128|long256
#SBATCH -J paper_stages
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0
bin="$1" mode="$2"
echo "# bin=$bin host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.applications.graphics --format=csv,noheader | sed 's/^/# /'
mkdir -p paper/data/clocks
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.applications.graphics,utilization.gpu,temperature.gpu,power.draw \
    --format=csv,noheader -l 1 >"paper/data/clocks/clock-$SLURM_JOB_ID.csv" &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
case "$mode" in
kernels)  # 128³, six inputs: rebuild kernels, and the step at three spacings with its kernels
    for kind in jittered uniform clustered halo lattice wave; do
        echo "== rebuild $kind 2097152"
        "$bin/voronoi_bench" rebuild 2097152 "$kind" 5 --no-check --kernels
        for d in 0.03 0.1 0.3; do
            echo "== step $kind 2097152 $d"
            "$bin/voronoi_bench" step 2097152 "$kind" "$d" 10 --kernels | grep -v "^verify"
        done
    done
    ;;
long128 | long256)  # 500 steps at 0.1 spacings, jittered: no re-sort, and a Hilbert re-sort every 50 updates
    n=2097152; [[ "$mode" == long256 ]] && n=16777216
    for flags in "" "--every=50"; do
        echo "== long $n flags=$flags"
        "$bin/voronoi_bench" step "$n" jittered 0.1 ${STEPS:-500} $flags | grep -E "^(#|start|mesh|layout|gpuflips|final)"
    done
    ;;
esac
