#!/bin/bash
# Build every CUDA target on the node, then ctest (the fuzz smoke included) on one A100:
#   sbatch -o build-jobs/%x-%j.out scripts/ctest_a100.sh <build-cuda>
#SBATCH -J ctest_a100
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=02:00:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake
build="$1"
hostname
nvidia-smi --query-gpu=name,clocks.applications.graphics --format=csv,noheader
cmake --build "$build" --parallel 18 2>&1 | grep -E " error|Error " | head -20
ctest --test-dir "$build" --output-on-failure 2>&1 | tail -40
