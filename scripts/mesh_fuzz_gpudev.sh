#!/bin/bash
# Long CPU fuzz of the mesh in 15-minute chunks, for when `small` is at its QOS
# limit: one array task of 18 one-thread workers on the CPUs of an A100 job (the
# GPU stays idle). Each worker has its own results file and resumes from it, so
# resubmitting the same command continues from the seed counter:
#   sbatch --array=0-1 -o <out>/logs/dev-%A_%a.out scripts/mesh_fuzz_gpudev.sh <build-omp> <out> [first-seed] [seconds]
# Add -p gpu1 on the sbatch line when gpudev is at its job limit.
#SBATCH -J mesh_fuzz
#SBATCH -p gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/15
build="$1"; out="$2"; base="${3:-0}"; seconds="${4:-820}"
task="${SLURM_ARRAY_TASK_ID:-0}"
workers=18
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false
mkdir -p "$out/results"
for w in $(seq 0 $((workers - 1))); do
    # Seeds first, first + stride, ... : no two workers of any task share one.
    first=$((base + 1 + task * workers + w))
    "$build/mesh_fuzz_test" run --first="$first" --stride=$((workers * 64)) --count=100000000 \
        --results="$out/results/t${task}_w${w}.txt" --seconds="$seconds" --timeout=30 &
done
wait
