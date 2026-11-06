#!/bin/bash
# Long CPU fuzz of the mesh (TESTING.md), one array task of 36 one-thread
# workers. Each worker has its own results file and resumes from it, so
# resubmitting the same array continues where the last job stopped:
#   sbatch --array=0-3 -o <out>/logs/cpu-%A_%a.out scripts/mesh_fuzz_cpu.sh <build-omp> <out> [first-seed] [seconds]
#SBATCH -J mesh_fuzz
#SBATCH -p small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=60G
#SBATCH --time=01:00:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/15
build="$1"; out="$2"; base="${3:-0}"; seconds="${4:-3300}"
task="${SLURM_ARRAY_TASK_ID:-0}"
workers=36
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false
mkdir -p "$out/results"
for w in $(seq 0 $((workers - 1))); do
    # Seeds first, first + stride, ... : no two workers of any task share one.
    first=$((base + 1 + task * workers + w))
    "$build/mesh_fuzz_test" run --first="$first" --stride=$((workers * 64)) --count=100000000 \
        --results="$out/results/t${task}_w${w}.txt" --seconds="$seconds" --timeout=120 &
done
wait
