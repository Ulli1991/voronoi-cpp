#!/bin/bash
# One voronoi_bench run of a built binary on one A100, its whole output (for failures):
#   sbatch -o build-jobs/logs/%x-%j.out scripts/levers_raw.sh <binary> <bench args...>
#SBATCH -J lev_raw
#SBATCH -p gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60G
#SBATCH --time=00:10:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0
"$@" 2>&1 | grep -v "^kernel" | tail -40
