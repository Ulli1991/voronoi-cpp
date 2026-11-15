#!/bin/bash
# Build CUDA targets on a general node (no GPU needed), off the shared login node:
#   sbatch -o build-jobs/%x-%j.out scripts/build_cuda_general.sh <build-cuda> [target...]
#SBATCH -J build_cuda
#SBATCH -p general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=120G
#SBATCH --time=01:30:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake
build="$1"
shift
targets=()
for t in "$@"; do targets+=(--target "$t"); done
cmake --build "$build" --parallel 36 "${targets[@]}" 2>&1 | grep -E " error|Error |warning #" | grep -v mesh_order | head -30
echo "build done: $?"
