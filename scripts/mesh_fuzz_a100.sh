#!/bin/bash
# Mesh fuzz at 1e5 to 1e7 particles on the four A100 of one node, 15-minute chunks: one CUDA worker per GPU, profiles
# large (GPU 0), huge (1) and giant (2, 3). Each worker resumes from its results file, so resubmitting continues:
#   sbatch --array=0-3 -o <out>/logs/a100-%A_%a.out scripts/mesh_fuzz_a100.sh <build-cuda> <out> [seconds]
# Add -p gpu1 on the sbatch line when gpudev is at its job limit.
#SBATCH -J mesh_fuzz_a100
#SBATCH -p gpudev
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=480G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0
build="$1"; out="$2"; seconds="${3:-600}"
task="${SLURM_ARRAY_TASK_ID:-0}"
export OMP_NUM_THREADS=18 OMP_PROC_BIND=false
mkdir -p "$out/results"
worker() {  # gpu, profile digit, first seed offset, stride, per-case timeout
    CUDA_VISIBLE_DEVICES=$1 "$build/mesh_fuzz_test" run --first=$(($2 * 1000000000000 + $3)) --stride="$4" \
        --count=100000000 --results="$out/results/t${task}_g$1.txt" --seconds="$seconds" --timeout="$5" &
}
worker 0 3 $((1 + task)) 4 120
worker 1 4 $((1 + task)) 4 280
worker 2 7 $((1 + 2 * task)) 8 280
worker 3 7 $((2 + 2 * task)) 8 280
wait
