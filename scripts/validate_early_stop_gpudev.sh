#!/bin/bash
#SBATCH --job-name=voronoi-early-stop
#SBATCH --partition=gpudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
#SBATCH --output=build-table-validation/early-stop-%j.out
# Obtain user approval before submitting this job.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
validation_root="$PWD/build-early-stop/${SLURM_JOB_ID:?}"
gcc_gpu=/mpcdf/soft/SLE_15/packages/x86_64/gcc/14.1.0
cuda_root=/mpcdf/soft/SLE_15/packages/x86_64/cuda/13.0.1
export PATH="$cuda_root/bin:$gcc_gpu/bin:$PATH"
export LD_LIBRARY_PATH="$gcc_gpu/lib64:$cuda_root/lib64:${LD_LIBRARY_PATH:-}"
export NVCC_WRAPPER_DEFAULT_COMPILER="$gcc_gpu/bin/g++"
export OMP_NUM_THREADS=1
cmake -S . -B "$validation_root/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper \
  -DKokkos_DIR=/u/uli/kokkos-cuda/lib64/cmake/Kokkos -DVORONOI_CUDA_RESOURCE_REPORT=ON
cmake --build "$validation_root/build" --target construction_early_stress_check table_mesh_check construction_scaling -j 4
srun "$validation_root/build/construction_early_stress_check"
for distribution in lattice disordered regular random gradient; do
  for policy in relative relative_early; do
    srun "$validation_root/build/table_mesh_check" "$validation_root/$distribution-$policy.csv" 12 team_bitonic cached "$distribution" 64 "$policy"
  done
  python3 python/compare_faces.py "$validation_root/$distribution-relative.csv" "$validation_root/$distribution-relative_early.csv"
  python3 python/compare_faces.py "$validation_root/$distribution-relative.csv.hydro.csv" "$validation_root/$distribution-relative_early.csv.hydro.csv"
done
for side in 32 128 195; do
  for policy in relative relative_early; do
    srun "$validation_root/build/construction_scaling" "$side" 5 "$policy" "$validation_root/scaling.csv"
  done
done
