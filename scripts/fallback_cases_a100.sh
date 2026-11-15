#!/bin/bash
# Steps that reach the host fallback and the rebuild, 128³ on one A100 (voronoi_bench step, 12 steps each):
#   sbatch -o build-cuda/gpu-run/fallback-<part>-%j.txt scripts/fallback_cases_a100.sh <bin> flow|jumps|inputs
#SBATCH -J fallback_cases
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
bin="$1" part="$2" n=2097152 steps=${STEPS:-12}
echo "# bin=$bin host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.applications.graphics --format=csv,noheader | sed 's/^/# /'
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu --format=csv,noheader -l 1 >"${bin}/clock-$SLURM_JOB_ID.csv" &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
run() {  # case label, input, spacings, flags
    echo "== case $1 $2 $n $3 ${*:4}"
    timeout 240 "$bin/voronoi_bench" step "$n" "$2" "$3" "$steps" "${@:4}" | grep -E "^(#|motion|gpuflips|refused|final)"
}
case "$part" in
flow)  # the smooth flow at large steps, and the fallback policies at 1 spacing
    for d in 0.5 1 2; do run "flow-$d" jittered "$d"; done
    run flow-1 uniform 1
    run flow-0.3 clustered 0.3
    run flow-0.5 clustered 0.5
    run flow-0.3 halo 0.3
    run flow-1-host jittered 1 --fallback=host
    run flow-1-rebuild jittered 1 --fallback=rebuild
    ;;
jumps)  # flow at 0.1 spacings plus a share of points jumping 4 spacings each step
    for f in 1e-6 5e-6 1e-5 1e-4 1e-3 1e-2; do run "jump-$f" jittered 0.1 --jump-share=$f --jump-spacings=4; done
    run jump-1e-3-host jittered 0.1 --jump-share=1e-3 --jump-spacings=4 --fallback=host
    ;;
inputs)  # the surface, cosmic web, lattice and wave inputs at large steps
    for kind in surface cosmicweb; do for d in 0.3 1; do run "flow-$d" "$kind" "$d"; done; done
    for kind in lattice wave; do for d in 0.5 1; do run "flow-$d" "$kind" "$d"; done; done
    ;;
esac
