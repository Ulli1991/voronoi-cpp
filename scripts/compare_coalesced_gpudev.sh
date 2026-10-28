#!/bin/bash
#SBATCH --job-name=voronoi-coalesced
#SBATCH --partition=gpudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
#SBATCH --output=build-table-validation/coalesced-%j.out

# Ask the user before submitting. Once submitted this runs independently of
# the interactive agent. Fresh processes release GPU allocations between sizes.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
scaling_root="$PWD/build-gpu-scaling/${SLURM_JOB_ID:?}"
gcc_gpu=/mpcdf/soft/SLE_15/packages/x86_64/gcc/14.1.0
cuda_root=/mpcdf/soft/SLE_15/packages/x86_64/cuda/13.0.1
export PATH="$cuda_root/bin:$gcc_gpu/bin:$PATH"
export LD_LIBRARY_PATH="$gcc_gpu/lib64:$cuda_root/lib64:${LD_LIBRARY_PATH:-}"
export NVCC_WRAPPER_DEFAULT_COMPILER="$gcc_gpu/bin/g++"
export OMP_NUM_THREADS=1
hostname
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.free --format=csv
cmake -S . -B "$scaling_root/build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper \
    -DKokkos_DIR=/u/uli/kokkos-cuda/lib64/cmake/Kokkos
cmake --build "$scaling_root/build" --target construction_scaling -j 4
for side in 32 128; do
  for policy in relative_early relative_early_coalesced relative_early_fixed32; do
    srun "$scaling_root/build/construction_scaling" "$side" 5 "$policy" "$scaling_root/comparison.csv"
  done
done
